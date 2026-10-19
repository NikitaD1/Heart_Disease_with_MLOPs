"""Train, tune, compare and package the heart-disease classifiers with MLflow tracking.

Run:  python -m heart.train            (full hyper-parameter grids)
      python -m heart.train --quick    (tiny grids, used by unit tests)

What it does
1. Stratified 80/20 train/test split of the cleaned data (fixed seed).
2. For each candidate model, GridSearchCV with 5-fold stratified CV, scoring
   accuracy / precision / recall / F1 / ROC-AUC and refitting on ROC-AUC.
3. MLflow: one parent run per model (best params, CV metrics, test metrics,
   plots, cv_results.csv, the fitted model) and one nested child run per
   grid candidate (params + CV metrics).
4. Selects the champion by mean CV ROC-AUC (never by test score), saves it as
   joblib + MLflow model + ONNX, writes metadata, and registers it in the
   MLflow Model Registry with the alias "champion".
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heart.config import (
    CLEAN_FILE, CV_FOLDS, FEATURES, FIGURES_DIR, MLFLOW_EXPERIMENT, MLFLOW_MODEL_DIR,
    MODEL_FILE, MODEL_META_FILE, MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT, RANDOM_STATE,
    REPORTS_DIR, TARGET, TEST_SIZE,
)
from heart.data import file_sha256, load_clean, split_xy
from heart.evaluation import (
    classification_metrics, feature_importance, plot_confusion, plot_importance,
    plot_model_comparison, plot_roc_pr,
)
from heart.features import build_pipeline

log = logging.getLogger(__name__)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
# zero_division=0: heavily regularised candidates can predict a single class in a
# fold; score them 0 instead of emitting UndefinedMetricWarning noise.
SCORERS = {"accuracy": "accuracy", "recall": "recall", "roc_auc": "roc_auc",
           "precision": make_scorer(precision_score, zero_division=0),
           "f1": make_scorer(f1_score, zero_division=0)}
REGISTERED_MODEL = "heart-disease-classifier"


def candidate_models(quick: bool = False) -> dict[str, tuple]:
    """name -> (estimator, param_grid, scale_numeric)."""
    if quick:
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=2000, solver="liblinear"),
                {"model__C": [0.1, 1.0]}, True),
            "random_forest": (
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                {"model__n_estimators": [50], "model__max_depth": [4]}, False),
        }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, solver="liblinear"),
            {"model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
             "model__penalty": ["l1", "l2"],
             "model__class_weight": [None, "balanced"]},
            True),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [200, 500],
             "model__max_depth": [None, 4, 8],
             "model__min_samples_leaf": [1, 3, 5],
             "model__max_features": ["sqrt", 0.5]},
            False),  # trees do not need scaling
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200],
             "model__learning_rate": [0.03, 0.1],
             "model__max_depth": [2, 3]},
            False),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("GITHUB_SHA", "unknown")[:7]


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("alembic").setLevel(logging.WARNING)
    log.disabled = False


def setup_mlflow() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT,
                                 artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return uri


def _summarise_cv(search: GridSearchCV) -> dict[str, float]:
    i = search.best_index_
    res = search.cv_results_
    out = {}
    for m in SCORING:
        out[f"cv_{m}_mean"] = float(res[f"mean_test_{m}"][i])
        out[f"cv_{m}_std"] = float(res[f"std_test_{m}"][i])
    return out


def train_one(name, estimator, grid, scale, X_train, y_train, X_test, y_test, tags) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(build_pipeline(estimator, scale_numeric=scale), grid, cv=cv,
                          scoring=SCORERS, refit="roc_auc", n_jobs=-1, return_train_score=True)

    with mlflow.start_run(run_name=name, tags={**tags, "model_type": name}) as parent:
        t0 = time.time()
        search.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        # One nested run per hyper-parameter combination -> every run is tracked.
        res = pd.DataFrame(search.cv_results_)
        for _, row in res.iterrows():
            with mlflow.start_run(run_name=f"{name}-cand-{int(row['rank_test_roc_auc'])}",
                                  nested=True, tags={**tags, "model_type": name,
                                                     "stage": "grid-candidate"}):
                mlflow.log_params({k.replace("model__", ""): v for k, v in row["params"].items()})
                mlflow.log_metrics({f"cv_{m}_mean": row[f"mean_test_{m}"] for m in SCORING})
                mlflow.log_metrics({f"cv_{m}_std": row[f"std_test_{m}"] for m in SCORING})
                mlflow.log_metric("mean_fit_time_s", row["mean_fit_time"])

        best = search.best_estimator_
        cv_summary = _summarise_cv(search)
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        test_metrics = {f"test_{k}": v for k, v in
                        classification_metrics(y_test, y_pred, y_proba).items()}
        gap = cv_summary["cv_roc_auc_mean"] - float(
            search.cv_results_["mean_train_roc_auc"][search.best_index_])

        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_params({"cv_folds": CV_FOLDS, "n_candidates": len(res),
                           "scale_numeric": scale, "n_train": len(X_train),
                           "n_test": len(X_test)})
        mlflow.log_metrics({**cv_summary, **test_metrics, "fit_seconds": fit_seconds,
                            "cv_minus_train_roc_auc": gap})

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            res.drop(columns=["params"]).to_csv(tmp / "cv_results.csv", index=False)
            plot_confusion(best, X_test, y_test, tmp / "confusion_matrix.png",
                           f"{name} - test confusion matrix")
            plot_roc_pr(best, X_test, y_test, tmp / "roc_pr_curves.png", f"{name} - test set")
            plot_importance(best, tmp / "feature_importance.png", f"{name} - feature importance")
            feature_importance(best).to_csv(tmp / "feature_importance.csv", header=["importance"])
            mlflow.log_artifacts(str(tmp), artifact_path="evaluation")

        signature = infer_signature(X_test, best.predict_proba(X_test))
        mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                 input_example=X_test.head(3))
        log.info("%-20s CV ROC-AUC %.3f ± %.3f | test ROC-AUC %.3f | %d candidates",
                 name, cv_summary["cv_roc_auc_mean"], cv_summary["cv_roc_auc_std"],
                 test_metrics["test_roc_auc"], len(res))
        return {"name": name, "run_id": parent.info.run_id, "model": best,
                "best_params": search.best_params_, "cv": cv_summary, "test": test_metrics,
                "n_candidates": len(res)}


def export_onnx(pipeline, X_sample: pd.DataFrame, path: Path) -> bool:
    """Best-effort ONNX export of the full pipeline (preprocessing included).

    Each feature becomes a named float input of shape [N, 1], matching the
    column names the ColumnTransformer selects. The exported graph is checked
    against scikit-learn before it is kept.
    """
    try:
        import numpy as np
        import onnxruntime as rt
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_types = [(c, FloatTensorType([None, 1])) for c in X_sample.columns]
        onx = convert_sklearn(pipeline, initial_types=initial_types, target_opset=17,
                              options={id(pipeline.named_steps["model"]): {"zipmap": False}})
        sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        feeds = {c: X_sample[[c]].to_numpy(dtype=np.float32) for c in X_sample.columns}
        onnx_proba = sess.run(None, feeds)[1][:, 1]
        skl_proba = pipeline.predict_proba(X_sample)[:, 1]
        max_diff = float(np.max(np.abs(onnx_proba - skl_proba)))
        if max_diff > 1e-3:
            raise ValueError(f"ONNX/sklearn mismatch (max diff {max_diff:.2e})")
        path.write_bytes(onx.SerializeToString())
        log.info("ONNX export verified (max |diff| = %.2e) -> %s", max_diff, path)
        return True
    except Exception as exc:  # noqa: BLE001 - ONNX is an optional extra format
        log.warning("ONNX export skipped: %s", exc)
        return False


def main(quick: bool = False) -> dict:
    _configure_logging()
    df = load_clean()
    X, y = split_xy(df)
    X = X.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(PROCESSED_DIR / "test.csv", index=False)

    uri = setup_mlflow()
    _configure_logging()  # MLflow's DB migration (alembic) resets logging on a fresh store
    tags = {"git_commit": _git_commit(), "data_sha256": file_sha256(CLEAN_FILE)[:16],
            "dataset": "uci-heart-disease-cleveland", "quick_mode": str(quick)}
    log.info("MLflow tracking URI: %s", uri)

    results = [train_one(n, est, grid, scale, X_train, y_train, X_test, y_test, tags)
               for n, (est, grid, scale) in candidate_models(quick).items()]

    summary = pd.DataFrame([{"model": r["name"], **r["cv"], **r["test"],
                             "n_candidates": r["n_candidates"], "run_id": r["run_id"]}
                            for r in results]).set_index("model")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(REPORTS_DIR / "model_comparison.csv")
    plot_model_comparison(summary, FIGURES_DIR / "07_model_comparison.png")

    champion = max(results, key=lambda r: r["cv"]["cv_roc_auc_mean"])
    model = champion["model"]
    plot_confusion(model, X_test, y_test, FIGURES_DIR / "08_champion_confusion_matrix.png",
                   f"Champion ({champion['name']}) - test set")
    plot_roc_pr(model, X_test, y_test, FIGURES_DIR / "09_champion_roc_pr.png",
                f"Champion ({champion['name']}) - test set")
    plot_importance(model, FIGURES_DIR / "10_champion_feature_importance.png",
                    f"Champion ({champion['name']}) - top features")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    if MLFLOW_MODEL_DIR.exists():
        import shutil
        shutil.rmtree(MLFLOW_MODEL_DIR)
    mlflow.sklearn.save_model(model, str(MLFLOW_MODEL_DIR),
                              signature=infer_signature(X_test, model.predict_proba(X_test)),
                              input_example=X_test.head(3))
    onnx_ok = export_onnx(model, X_test, MODELS_DIR / "heart_model.onnx")

    version = "unregistered"
    try:
        mv = mlflow.register_model(f"runs:/{champion['run_id']}/model", REGISTERED_MODEL)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", mv.version)
        client.set_model_version_tag(REGISTERED_MODEL, mv.version, "cv_roc_auc",
                                     f"{champion['cv']['cv_roc_auc_mean']:.4f}")
        version = str(mv.version)
    except Exception as exc:  # noqa: BLE001 - file-store backends have no registry
        log.warning("Model registry not available: %s", exc)

    metadata = {
        "model_name": REGISTERED_MODEL,
        "model_version": version,
        "algorithm": champion["name"],
        "best_params": {k.replace("model__", ""): v for k, v in champion["best_params"].items()},
        "selection_rule": "highest mean 5-fold CV ROC-AUC on the training split",
        "cv_metrics": champion["cv"],
        "test_metrics": champion["test"],
        "decision_threshold": 0.5,
        "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mlflow_run_id": champion["run_id"],
        "git_commit": tags["git_commit"],
        "data_sha256": file_sha256(CLEAN_FILE),
        "sklearn_version": sklearn.__version__,
        "onnx_exported": onnx_ok,
        "quick_mode": quick,
    }
    MODEL_META_FILE.write_text(json.dumps(metadata, indent=2, default=str))
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"champion": champion["name"], "cv": champion["cv"], "test": champion["test"]},
        indent=2))
    log.info("Champion: %s (CV ROC-AUC %.3f) -> %s", champion["name"],
             champion["cv"]["cv_roc_auc_mean"], MODEL_FILE)
    print(summary[["cv_roc_auc_mean", "cv_accuracy_mean", "cv_recall_mean",
                   "test_roc_auc", "test_accuracy", "test_recall"]].round(3).to_string())
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heart-disease models")
    parser.add_argument("--quick", action="store_true", help="tiny grids for smoke tests")
    main(quick=parser.parse_args().quick)
