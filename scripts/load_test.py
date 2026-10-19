#!/usr/bin/env python
"""Generate realistic traffic against the API (to populate monitoring dashboards).

    python scripts/load_test.py --url http://localhost:8000 --seconds 120 --rps 5

Sends patients sampled from the held-out test split, plus ~5% malformed
requests (to exercise 422 handling) and occasional batch calls.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "sex", "fbs", "exang",
            "cp", "restecg", "slope", "thal"]
INT_FIELDS = {"sex", "fbs", "exang", "cp", "restecg", "slope", "thal", "ca"}


def load_patients() -> list[dict]:
    df = pd.read_csv(ROOT / "data/processed/test.csv")[FEATURES]
    out = []
    for rec in df.to_dict("records"):
        out.append({k: (None if pd.isna(v) else int(v) if k in INT_FIELDS else float(v))
                    for k, v in rec.items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--rps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    patients = load_patients()
    session = requests.Session()
    stats = {"ok": 0, "client_error": 0, "server_error": 0, "conn_error": 0}
    end = time.time() + args.seconds
    while time.time() < end:
        roll = rng.random()
        try:
            if roll < 0.05:      # malformed request
                bad = dict(rng.choice(patients), cp=9)
                r = session.post(f"{args.url}/predict", json=bad, timeout=5)
            elif roll < 0.10:    # batch request
                r = session.post(f"{args.url}/predict/batch",
                                 json={"instances": rng.sample(patients, 5)}, timeout=5)
            elif roll < 0.13:
                r = session.get(f"{args.url}/health", timeout=5)
            else:
                r = session.post(f"{args.url}/predict", json=rng.choice(patients), timeout=5)
            if r.status_code < 400:
                stats["ok"] += 1
            elif r.status_code < 500:
                stats["client_error"] += 1
            else:
                stats["server_error"] += 1
        except requests.RequestException:
            stats["conn_error"] += 1
        time.sleep(rng.expovariate(args.rps))
    print(json.dumps(stats))
    return 1 if stats["server_error"] or stats["conn_error"] else 0


if __name__ == "__main__":
    sys.exit(main())
