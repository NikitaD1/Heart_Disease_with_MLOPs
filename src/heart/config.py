"""Central configuration: paths, column definitions and constants.

Every script and the API import from here so that the feature list used for
training is guaranteed to be the same one used at inference time.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "heart_cleveland_raw.csv"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MODEL_FILE = MODELS_DIR / "heart_model.joblib"
MODEL_META_FILE = MODELS_DIR / "model_metadata.json"
ONNX_FILE = MODELS_DIR / "heart_model.onnx"
MLFLOW_MODEL_DIR = MODELS_DIR / "mlflow_model"

# The 13 predictive attributes of the UCI "processed.cleveland.data" file,
# in the order they appear in the original file, followed by the target.
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Binary flags are passed through unchanged (already 0/1).
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Nominal codes with no natural order are one-hot encoded.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Valid categorical codes (from the UCI data dictionary). Used for validation
# in cleaning, in the API schema and for fixing the one-hot categories so the
# encoder never sees an "unknown" level at serving time.
CATEGORY_LEVELS = {
    "cp": [1, 2, 3, 4],        # 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
    "restecg": [0, 1, 2],      # 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "slope": [1, 2, 3],        # 1 upsloping, 2 flat, 3 downsloping
    "thal": [3, 6, 7],         # 3 normal, 6 fixed defect, 7 reversible defect
}

# Plausible physiological ranges used to reject corrupt rows / bad API input.
VALID_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (80, 700),
    "thalach": (40, 250),
    "oldpeak": (0.0, 10.0),
    "ca": (0, 3),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MLFLOW_EXPERIMENT = "heart-disease-classification"
