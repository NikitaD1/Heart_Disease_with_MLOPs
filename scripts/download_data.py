#!/usr/bin/env python
"""Download the UCI Heart Disease (Cleveland) dataset and build the cleaned CSV.

Usage:
    python scripts/download_data.py            # download + clean
    python scripts/download_data.py --no-clean # download only

Outputs:
    data/raw/heart_cleveland_raw.csv   raw 14 UCI columns, '?' -> NaN
    data/raw/SOURCE.txt                which URL was used + sha256
    data/processed/heart_clean.csv     cleaned features + binary target
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart.data import build_clean_dataset, download_raw  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-clean", action="store_true", help="skip cleaning step")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df, source = download_raw()
    print(f"Downloaded {len(df)} rows x {df.shape[1]} cols from {source}")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
    if not args.no_clean:
        clean = build_clean_dataset()
        print(f"Cleaned dataset: {clean.shape}; class balance:\n"
              f"{clean['target'].value_counts().to_string()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
