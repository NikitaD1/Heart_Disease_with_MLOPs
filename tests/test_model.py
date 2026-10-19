"""Unit tests for model training, evaluation, inference and packaging."""
import json

import joblib
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from heart.config import FEATURES, MODEL_FILE, TARGET
from heart.evaluation import classification_metrics, feature_importance
from heart.predict import HeartModel


def test_pipeline_beats_majority_baseline(heart_df, trained_pipeline):
    X, y = heart_df[FEATURES].astype(float), heart_df[TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc = cross_val_score(trained_pipeline, X, y, cv=cv, scoring="roc_auc")
    assert auc.mean() > 0.85, f"CV ROC-AUC too low: {auc.mean():.3f}"


def test_probabilities_are_valid(heart_df, trained_pipeline):
    proba = trained_pipeline.predict_proba(heart_df[FEATURES].astype(float))
    assert proba.shape == (len(heart_df), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_classification_metrics_keys_and_ranges():
    m = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert m["accuracy"] == 0.75 and m["precision"] == 1.0 and m["recall"] == 0.5
    assert m["roc_auc"] == 1.0


def test_feature_importance_covers_all_features(trained_pipeline):
    imp = feature_importance(trained_pipeline)
    assert len(imp) == 22 and (imp >= 0).all()


def test_heartmodel_predict_records(model_files, patient, high_risk_patient):
    model = HeartModel(*model_files)
    out = model.predict_records([patient, {**patient, "ca": None, "thal": None},
                                 high_risk_patient])
    assert len(out) == 3
    # both classes must be exercised, otherwise a confidence bug in one branch hides
    assert {r["prediction"] for r in out} == {0, 1}
    for r in out:
        assert r["prediction"] in (0, 1)
        assert 0.5 <= r["confidence"] <= 1.0
        assert 0.0 <= r["probability_disease"] <= 1.0
        expected = r["probability_disease"] if r["prediction"] else 1 - r["probability_disease"]
        assert r["confidence"] == pytest.approx(expected, abs=1e-4)
    assert model.version == "test"


def test_heartmodel_rejects_missing_features(model_files, patient):
    model = HeartModel(*model_files)
    bad = {k: v for k, v in patient.items() if k != "chol"}
    with pytest.raises(ValueError, match="chol"):
        model.predict_records([bad])


def test_heartmodel_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        HeartModel(tmp_path / "nope.joblib", None)


def test_high_and_low_risk_patients_are_ordered(model_files, patient):
    """Sanity check on learned direction: an asymptomatic patient with exercise
    angina, 3 blocked vessels and a reversible defect should score higher."""
    model = HeartModel(*model_files)
    low = {**patient, "cp": 3, "exang": 0, "ca": 0, "thal": 3, "oldpeak": 0.0, "thalach": 180}
    high = {**patient, "cp": 4, "exang": 1, "ca": 3, "thal": 7, "oldpeak": 3.5, "thalach": 110}
    p_low, p_high = (r["probability_disease"] for r in model.predict_records([low, high]))
    assert p_high > p_low


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="no packaged model yet")
def test_packaged_model_reloads_and_performs(heart_df):
    pipe = joblib.load(MODEL_FILE)
    proba = pipe.predict_proba(heart_df[FEATURES].astype(float))[:, 1]
    m = classification_metrics(heart_df[TARGET], (proba >= 0.5).astype(int), proba)
    assert m["roc_auc"] > 0.85
    meta = json.loads(MODEL_FILE.with_name("model_metadata.json").read_text())
    assert meta["features"] == FEATURES


@pytest.mark.slow
def test_training_entrypoint_end_to_end(tmp_path, monkeypatch):
    """Runs heart.train.main(quick=True) with every output redirected to tmp_path."""
    import heart.train as train

    for name, sub in [("MODELS_DIR", "models"), ("REPORTS_DIR", "reports"),
                      ("FIGURES_DIR", "reports/figures"), ("PROCESSED_DIR", "processed"),
                      ("PROJECT_ROOT", "")]:
        monkeypatch.setattr(train, name, tmp_path / sub)
    monkeypatch.setattr(train, "MODEL_FILE", tmp_path / "models/heart_model.joblib")
    monkeypatch.setattr(train, "MODEL_META_FILE", tmp_path / "models/model_metadata.json")
    monkeypatch.setattr(train, "MLFLOW_MODEL_DIR", tmp_path / "models/mlflow_model")
    monkeypatch.setattr(train, "MLFLOW_EXPERIMENT", "pytest-quick")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    meta = train.main(quick=True)
    assert (tmp_path / "models/heart_model.joblib").exists()
    assert (tmp_path / "models/mlflow_model/MLmodel").exists()
    assert meta["algorithm"] in {"logistic_regression", "random_forest"}
    assert meta["test_metrics"]["test_roc_auc"] > 0.8
    assert (tmp_path / "reports/model_comparison.csv").exists()
