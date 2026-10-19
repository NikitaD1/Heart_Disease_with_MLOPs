"""Feature engineering: a single sklearn ColumnTransformer used for training AND serving.

Because the transformer is saved inside the model Pipeline, the exact imputation
values, scaling statistics and one-hot categories learned on the training data
are reapplied to every prediction request (no training-serving skew).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.config import (
    BINARY_FEATURES, CATEGORICAL_FEATURES, CATEGORY_LEVELS, NUMERIC_FEATURES,
)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Numeric: median impute (+ standard scale). Binary: mode impute.
    Categorical: mode impute + one-hot with the fixed UCI category levels."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=[[float(v) for v in CATEGORY_LEVELS[c]] for c in CATEGORICAL_FEATURES],
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, scale_numeric: bool = True) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
        ("model", estimator),
    ])


def feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
