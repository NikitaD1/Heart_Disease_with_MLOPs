"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset.

Acquisition order:
1. Official UCI archive zip  -> processed.cleveland.data
2. Official UCI direct file URL
3. Public GitHub mirror of the same 303 Cleveland records (ISLR "Heart.csv"),
   converted back to the UCI column names and numeric codes.

All three produce the same raw table: the 14 UCI columns with '?' turned into NaN.
"""
from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from heart.config import (
    BINARY_FEATURES, CATEGORY_LEVELS, CLEAN_FILE, FEATURES, RAW_FILE, TARGET,
    UCI_COLUMNS, VALID_RANGES,
)

log = logging.getLogger(__name__)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
UCI_FILE_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "heart-disease/processed.cleveland.data")
MIRROR_URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv"

# ISLR mirror uses text labels; map them back to the UCI numeric codes.
_MIRROR_MAPS = {
    "ChestPain": {"typical": 1, "nontypical": 2, "nonanginal": 3, "asymptomatic": 4},
    "Thal": {"normal": 3, "fixed": 6, "reversable": 7},
    "AHD": {"No": 0, "Yes": 1},
}
_MIRROR_RENAME = {
    "Age": "age", "Sex": "sex", "ChestPain": "cp", "RestBP": "trestbps",
    "Chol": "chol", "Fbs": "fbs", "RestECG": "restecg", "MaxHR": "thalach",
    "ExAng": "exang", "Oldpeak": "oldpeak", "Slope": "slope", "Ca": "ca",
    "Thal": "thal", "AHD": "num",
}


def parse_uci_text(text: str) -> pd.DataFrame:
    """Parse the comma-separated processed.cleveland.data content."""
    df = pd.read_csv(io.StringIO(text), header=None, names=UCI_COLUMNS, na_values="?")
    return df


def parse_mirror_csv(text: str) -> pd.DataFrame:
    """Convert the ISLR Heart.csv mirror into the UCI layout."""
    df = pd.read_csv(io.StringIO(text), index_col=0, na_values="NA")
    for col, mapping in _MIRROR_MAPS.items():
        df[col] = df[col].map(mapping)
    df = df.rename(columns=_MIRROR_RENAME)[UCI_COLUMNS].reset_index(drop=True)
    return df


def _fetch(url: str, timeout: int = 30) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def download_raw(dest: Path = RAW_FILE) -> tuple[pd.DataFrame, str]:
    """Download the dataset, trying the official source first. Returns (df, source)."""
    attempts = []
    try:
        blob = _fetch(UCI_ZIP_URL)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            text = zf.read("processed.cleveland.data").decode("latin-1")
        df, source = parse_uci_text(text), UCI_ZIP_URL
    except Exception as exc:  # noqa: BLE001 - try the next source
        attempts.append(f"{UCI_ZIP_URL}: {exc}")
        try:
            df = parse_uci_text(_fetch(UCI_FILE_URL).decode("latin-1"))
            source = UCI_FILE_URL
        except Exception as exc2:  # noqa: BLE001
            attempts.append(f"{UCI_FILE_URL}: {exc2}")
            try:
                df = parse_mirror_csv(_fetch(MIRROR_URL).decode("utf-8"))
                source = MIRROR_URL
            except Exception as exc3:  # noqa: BLE001
                attempts.append(f"{MIRROR_URL}: {exc3}")
                raise RuntimeError("All download sources failed:\n" + "\n".join(attempts))
    for a in attempts:
        log.warning("Source unavailable, falling back -> %s", a)

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    (dest.parent / "SOURCE.txt").write_text(
        f"source: {source}\nrows: {len(df)}\nsha256: {file_sha256(dest)}\n"
    )
    log.info("Saved %d rows from %s to %s", len(df), source, dest)
    return df, source


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw table.

    * coerce every column to numeric ('?' and junk become NaN)
    * drop exact duplicate rows
    * drop rows with an unknown target
    * set physiologically impossible values and unknown category codes to NaN
      (they are then imputed inside the model pipeline, fitted on training
      folds only, which avoids leakage from the test set)
    * binarise the target: num 0 -> 0 (absence), num 1-4 -> 1 (presence)
    """
    out = df.copy()
    missing = [c for c in UCI_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")
    out = out[UCI_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out = out.drop_duplicates().dropna(subset=["num"]).reset_index(drop=True)

    for col, (lo, hi) in VALID_RANGES.items():
        bad = ~out[col].between(lo, hi) & out[col].notna()
        out.loc[bad, col] = np.nan
    for col, levels in CATEGORY_LEVELS.items():
        bad = ~out[col].isin(levels) & out[col].notna()
        out.loc[bad, col] = np.nan
    for col in BINARY_FEATURES:
        bad = ~out[col].isin([0, 1]) & out[col].notna()
        out.loc[bad, col] = np.nan

    out[TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])
    return out[FEATURES + [TARGET]]


def build_clean_dataset(raw_path: Path = RAW_FILE, out_path: Path = CLEAN_FILE) -> pd.DataFrame:
    cleaned = clean(load_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    log.info("Clean dataset: %d rows, %d missing cells -> %s",
             len(cleaned), int(cleaned.isna().sum().sum()), out_path)
    return cleaned


def load_clean(path: Path = CLEAN_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        return build_clean_dataset(out_path=Path(path))
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]
