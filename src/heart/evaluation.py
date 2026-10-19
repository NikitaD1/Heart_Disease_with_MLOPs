"""Metric computation and evaluation plots shared by training and the notebooks."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from heart.features import feature_names  # noqa: E402


def classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion(model, X, y, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=["No disease", "Disease"],
                                          cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_pr(model, X, y, path: Path, title: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    RocCurveDisplay.from_estimator(model, X, y, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("ROC curve")
    PrecisionRecallDisplay.from_estimator(model, X, y, ax=axes[1])
    axes[1].set_title("Precision-recall curve")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def feature_importance(pipeline) -> pd.Series:
    """Coefficients (|value|) for linear models, impurity importance for trees."""
    est = pipeline.named_steps["model"]
    names = feature_names(pipeline)
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_).ravel()
    else:
        return pd.Series(dtype=float)
    return pd.Series(values, index=names).sort_values(ascending=False)


def plot_importance(pipeline, path: Path, title: str, top: int = 15) -> Path:
    imp = feature_importance(pipeline).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(imp.index, imp.values, color="#2E7D9A")
    est = pipeline.named_steps["model"]
    ax.set_xlabel("Importance" if hasattr(est, "feature_importances_") else "|coefficient|")
    ax.set_title(title)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> Path:
    """summary: index = model name, columns = cv_<metric>_mean / cv_<metric>_std."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.8 / len(summary)
    colors = ["#2E7D9A", "#D1495B", "#EDAE49", "#66A182"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, row) in enumerate(summary.iterrows()):
        means = [row[f"cv_{m}_mean"] for m in metrics]
        stds = [row[f"cv_{m}_std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=name,
               color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(summary) - 1) / 2, metrics)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("5-fold CV score (mean ± std)")
    ax.set_title("Model comparison on the training set (best hyper-parameters)")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
