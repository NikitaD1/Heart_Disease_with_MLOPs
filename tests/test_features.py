"""Unit tests for the preprocessing (feature engineering) pipeline."""
import numpy as np
import pandas as pd

from heart.config import FEATURES
from heart.features import build_preprocessor

N_OUT = 6 + 3 + (4 + 3 + 3 + 3)  # numeric + binary + one-hot levels


def test_output_shape_and_names(heart_df):
    pre = build_preprocessor().fit(heart_df[FEATURES])
    Xt = pre.transform(heart_df[FEATURES])
    assert Xt.shape == (len(heart_df), N_OUT)
    names = list(pre.get_feature_names_out())
    assert "cp_4.0" in names and "thal_7.0" in names and "age" in names


def test_missing_values_are_imputed(heart_df):
    assert heart_df[FEATURES].isna().sum().sum() > 0
    Xt = build_preprocessor().fit_transform(heart_df[FEATURES])
    assert not np.isnan(Xt).any()


def test_numeric_features_are_standardised(heart_df):
    Xt = build_preprocessor().fit_transform(heart_df[FEATURES])
    numeric = Xt[:, :6]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-6)


def test_scaling_can_be_disabled_for_trees(heart_df):
    Xt = build_preprocessor(scale_numeric=False).fit_transform(heart_df[FEATURES])
    assert Xt[:, 0].max() > 50  # raw age kept


def test_one_hot_rows_sum_to_one_per_category(heart_df):
    Xt = build_preprocessor().fit_transform(heart_df[FEATURES])
    cp_block = Xt[:, 9:13]
    assert np.allclose(cp_block.sum(axis=1), 1)


def test_unseen_category_does_not_crash(heart_df, patient):
    pre = build_preprocessor().fit(heart_df[FEATURES])
    row = pd.DataFrame([{**patient, "thal": 5}])[FEATURES].astype(float)
    out = pre.transform(row)
    assert out.shape == (1, N_OUT)
    assert out[0, -3:].sum() == 0  # unknown thal -> all-zero one-hot block


def test_transform_is_deterministic(heart_df):
    pre = build_preprocessor().fit(heart_df[FEATURES])
    a = pre.transform(heart_df[FEATURES].head(10))
    b = pre.transform(heart_df[FEATURES].head(10))
    assert np.array_equal(a, b)
