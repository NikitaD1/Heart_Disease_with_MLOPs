"""Shared pytest fixtures. Tests use the committed cleaned dataset (no network)."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart.config import CLEAN_FILE, FEATURES, TARGET
from heart.data import clean, parse_uci_text
from heart.features import build_pipeline

# Five genuine rows of processed.cleveland.data (row 5 has '?' for ca and thal)
UCI_SAMPLE = """63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1
37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0
53.0,0.0,3.0,128.0,216.0,0.0,2.0,115.0,0.0,0.0,1.0,?,?,0
"""


@pytest.fixture
def uci_text() -> str:
    return UCI_SAMPLE


@pytest.fixture
def raw_sample(uci_text) -> pd.DataFrame:
    return parse_uci_text(uci_text)


@pytest.fixture
def clean_sample(raw_sample) -> pd.DataFrame:
    return clean(raw_sample)


@pytest.fixture(scope="session")
def heart_df() -> pd.DataFrame:
    if not CLEAN_FILE.exists():
        pytest.skip("cleaned dataset not present - run scripts/download_data.py")
    return pd.read_csv(CLEAN_FILE)


@pytest.fixture(scope="session")
def trained_pipeline(heart_df):
    X, y = heart_df[FEATURES].astype(float), heart_df[TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=2000, solver="liblinear"))
    return pipe.fit(X, y)


@pytest.fixture(scope="session")
def model_files(trained_pipeline, tmp_path_factory):
    """A saved model + metadata pair in a temp dir (never touches models/)."""
    d = tmp_path_factory.mktemp("model")
    joblib.dump(trained_pipeline, d / "model.joblib")
    (d / "meta.json").write_text(json.dumps(
        {"model_version": "test", "algorithm": "logistic_regression",
         "decision_threshold": 0.5}))
    return d / "model.joblib", d / "meta.json"


@pytest.fixture
def patient() -> dict:
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6}


@pytest.fixture
def high_risk_patient() -> dict:
    """Row 2 of the UCI file (num = 2): asymptomatic chest pain, exercise angina, 3 vessels."""
    return {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
            "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2,
            "ca": 3, "thal": 3}
