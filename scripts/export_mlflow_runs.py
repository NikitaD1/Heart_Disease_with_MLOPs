#!/usr/bin/env python
"""Export the MLflow tracking store to CSV (for the report / CI summary).

    python scripts/export_mlflow_runs.py             # writes reports/mlflow_runs.csv
    python scripts/export_mlflow_runs.py --markdown  # also prints a markdown table
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlflow  # noqa: E402

from heart.config import MLFLOW_EXPERIMENT, REPORTS_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    default_uri = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_uri))
    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT])
    if runs.empty:
        print("No runs found", file=sys.stderr)
        return 1
    keep = [c for c in runs.columns if c.startswith(("params.", "metrics.cv_", "metrics.test_"))]
    cols = ["run_id", "tags.mlflow.runName", "tags.model_type", "tags.stage", "start_time"] + keep
    out = runs[[c for c in cols if c in runs.columns]]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS_DIR / "mlflow_runs.csv", index=False)
    parents = out[out["tags.stage"].isna()] if "tags.stage" in out else out
    print(f"{len(out)} runs exported ({len(parents)} model runs, "
          f"{len(out) - len(parents)} grid-candidate runs) -> reports/mlflow_runs.csv",
          file=sys.stderr)
    if args.markdown:
        cols = ["tags.mlflow.runName", "metrics.cv_roc_auc_mean", "metrics.cv_accuracy_mean",
                "metrics.cv_recall_mean", "metrics.test_roc_auc", "metrics.test_accuracy"]
        print("| model | CV ROC-AUC | CV accuracy | CV recall | test ROC-AUC | test accuracy |")
        print("|---|---|---|---|---|---|")
        for _, r in parents[cols].iterrows():
            print("| " + " | ".join([str(r[cols[0]])] + [f"{r[c]:.3f}" for c in cols[1:]]) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
