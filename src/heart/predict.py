"""Inference helpers shared by the CLI, the notebook and the FastAPI service.

CLI:
    python -m heart.predict --json '{"age": 63, "sex": 1, ...}'
    python -m heart.predict --csv data/processed/test.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import joblib
import pandas as pd

from heart.config import FEATURES, MODEL_FILE, MODEL_META_FILE

DEFAULT_THRESHOLD = 0.5


class HeartModel:
    """Thin wrapper: loads the saved pipeline and turns records into predictions."""

    def __init__(self, model_path: Path | str = MODEL_FILE,
                 meta_path: Path | str | None = MODEL_META_FILE):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file {self.model_path} not found. Run `python -m heart.train` first.")
        self.pipeline = joblib.load(self.model_path)
        self.metadata: dict = {}
        if meta_path and Path(meta_path).exists():
            self.metadata = json.loads(Path(meta_path).read_text())
        self.threshold = float(self.metadata.get("decision_threshold", DEFAULT_THRESHOLD))

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "local"))

    @staticmethod
    def to_frame(records: Iterable[Mapping]) -> pd.DataFrame:
        df = pd.DataFrame(list(records))
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return df[FEATURES].astype(float)

    def predict_records(self, records: Iterable[Mapping]) -> list[dict]:
        X = self.to_frame(records)
        proba = self.pipeline.predict_proba(X)[:, 1]
        out = []
        for p in proba:
            label = int(p >= self.threshold)
            out.append({
                "prediction": label,
                "label": "heart disease" if label else "no heart disease",
                "probability_disease": round(float(p), 4),
                "confidence": round(float(p if label else 1 - p), 4),
            })
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart-disease risk")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="one patient as a JSON object, or a list of objects")
    group.add_argument("--csv", help="CSV file with the 13 feature columns")
    parser.add_argument("--model", default=str(MODEL_FILE))
    args = parser.parse_args()

    model = HeartModel(args.model)
    if args.json:
        payload = json.loads(args.json)
        records = payload if isinstance(payload, list) else [payload]
        print(json.dumps(model.predict_records(records), indent=2))
    else:
        df = pd.read_csv(args.csv)
        preds = pd.DataFrame(model.predict_records(df.to_dict("records")))
        print(pd.concat([df.reset_index(drop=True), preds], axis=1).head(20).to_string())


if __name__ == "__main__":
    main()
