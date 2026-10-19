"""Exploratory data analysis: produces the figures used in the notebook and report.

Run:  python -m heart.eda
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart.config import (  # noqa: E402
    CATEGORICAL_FEATURES, FIGURES_DIR, NUMERIC_FEATURES, TARGET,
)
from heart.data import load_clean, load_raw  # noqa: E402

log = logging.getLogger(__name__)

PALETTE = {0: "#2E7D9A", 1: "#D1495B"}
LABELS = {0: "No disease", 1: "Disease"}
NICE = {
    "age": "Age (years)", "trestbps": "Resting BP (mm Hg)", "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate", "oldpeak": "ST depression", "ca": "Major vessels (0-3)",
    "cp": "Chest pain type", "sex": "Sex (1 = male)", "exang": "Exercise-induced angina",
    "restecg": "Resting ECG", "slope": "ST slope", "thal": "Thalassemia",
}

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} ({v / len(df):.0%})",
                ha="center", fontsize=10)
    ax.set_ylabel("Patients")
    ax.set_title("Class balance of the target")
    ax.set_ylim(0, counts.max() * 1.18)
    return _save(fig, "01_class_balance.png", out_dir)


def plot_missing(raw: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    miss = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(miss.index, miss.values, color=["#D1495B" if v else "#B0BEC5" for v in miss.values])
    for i, v in enumerate(miss.values):
        if v:
            ax.text(i, v + 0.1, str(v), ha="center")
    ax.set_ylabel("Missing values")
    ax.set_title("Missing values per raw column ('?' in the UCI file)")
    plt.xticks(rotation=45)
    return _save(fig, "02_missing_values.png", out_dir)


def plot_numeric_histograms(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        for t in (0, 1):
            sns.histplot(df.loc[df[TARGET] == t, col].dropna(), ax=ax, color=PALETTE[t],
                         label=LABELS[t], kde=(col != "ca"), stat="density", alpha=0.45,
                         bins=15, discrete=(col == "ca"))
        ax.set_title(NICE[col])
        ax.set_xlabel("")
    axes[0, 0].legend()
    fig.suptitle("Distribution of numeric features by diagnosis", fontsize=14)
    fig.tight_layout()
    return _save(fig, "03_numeric_histograms.png", out_dir)


def plot_correlation(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.75}, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Pearson correlation matrix (incl. target)")
    return _save(fig, "04_correlation_heatmap.png", out_dir)


def plot_categorical(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    cols = CATEGORICAL_FEATURES + ["sex", "exang"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        rate = df.groupby(col)[TARGET].agg(["mean", "count"]).reset_index()
        ax.bar(range(len(rate)), rate["mean"], color="#D1495B", alpha=0.85)
        for i, (m, n) in enumerate(zip(rate["mean"], rate["count"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_xticks(range(len(rate)), rate[col].astype(int).astype(str))
        ax.set_ylim(0, 1.1)
        ax.set_title(NICE.get(col, col))
        ax.set_ylabel("Disease rate")
    fig.suptitle("Disease rate by category level", fontsize=14)
    fig.tight_layout()
    return _save(fig, "05_categorical_disease_rate.png", out_dir)


def plot_boxplots(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8))
    for ax, col in zip(axes, ["age", "trestbps", "chol", "thalach", "oldpeak"]):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=PALETTE, ax=ax,
                    legend=False)
        ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_xlabel("Disease")
        ax.set_title(NICE[col])
        ax.set_ylabel("")
    fig.suptitle("Outlier check: numeric features by diagnosis", fontsize=13)
    fig.tight_layout()
    return _save(fig, "06_boxplots.png", out_dir)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe().T.round(2)


def run_all(out_dir: Path = FIGURES_DIR) -> list[Path]:
    raw, df = load_raw(), load_clean()
    paths = [
        plot_class_balance(df, out_dir), plot_missing(raw, out_dir),
        plot_numeric_histograms(df, out_dir), plot_correlation(df, out_dir),
        plot_categorical(df, out_dir), plot_boxplots(df, out_dir),
    ]
    summary_table(df).to_csv(out_dir / "summary_statistics.csv")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for p in run_all():
        print("saved", p)
