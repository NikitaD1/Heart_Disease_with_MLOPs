"""Unit tests for data acquisition parsing and cleaning."""
import numpy as np
import pandas as pd
import pytest

from heart.config import FEATURES, TARGET, UCI_COLUMNS
from heart.data import clean, parse_mirror_csv


def test_parse_uci_text_columns_and_missing(raw_sample):
    assert list(raw_sample.columns) == UCI_COLUMNS
    assert len(raw_sample) == 5
    # '?' in the original file becomes NaN
    assert raw_sample.loc[4, ["ca", "thal"]].isna().all()


def test_mirror_is_converted_to_uci_codes():
    text = (",Age,Sex,ChestPain,RestBP,Chol,Fbs,RestECG,MaxHR,ExAng,Oldpeak,Slope,Ca,Thal,AHD\n"
            "1,63,1,typical,145,233,1,2,150,0,2.3,3,0,fixed,No\n"
            "2,67,1,asymptomatic,160,286,0,2,108,1,1.5,2,NA,normal,Yes\n")
    df = parse_mirror_csv(text)
    assert list(df.columns) == UCI_COLUMNS
    assert df.loc[0, "cp"] == 1 and df.loc[0, "thal"] == 6 and df.loc[0, "num"] == 0
    assert df.loc[1, "cp"] == 4 and df.loc[1, "thal"] == 3 and df.loc[1, "num"] == 1
    assert np.isnan(df.loc[1, "ca"])


def test_clean_binarises_target(clean_sample):
    # num values 0,2,1,0,0 -> 0,1,1,0,0
    assert clean_sample[TARGET].tolist() == [0, 1, 1, 0, 0]
    assert set(clean_sample.columns) == set(FEATURES + [TARGET])
    assert "num" not in clean_sample.columns


def test_clean_keeps_missing_for_pipeline_imputation(clean_sample):
    assert clean_sample[["ca", "thal"]].isna().sum().sum() == 2


def test_clean_drops_duplicates_and_unlabelled_rows(raw_sample):
    dup = pd.concat([raw_sample, raw_sample.iloc[[0]]], ignore_index=True)
    dup.loc[len(dup)] = [50, 1, 2, 120, 200, 0, 0, 160, 0, 0.5, 1, 0, 3, np.nan]
    out = clean(dup)
    assert len(out) == 5


def test_clean_invalidates_impossible_values(raw_sample):
    bad = raw_sample.copy()
    bad.loc[0, "chol"] = 5000      # impossible cholesterol
    bad.loc[1, "cp"] = 9           # unknown chest-pain code
    bad.loc[2, "sex"] = 3          # invalid binary flag
    out = clean(bad)
    assert np.isnan(out.loc[0, "chol"])
    assert np.isnan(out.loc[1, "cp"])
    assert np.isnan(out.loc[2, "sex"])


def test_clean_rejects_missing_columns(raw_sample):
    with pytest.raises(ValueError, match="missing columns"):
        clean(raw_sample.drop(columns=["chol"]))


def test_clean_coerces_non_numeric(raw_sample):
    bad = raw_sample.astype(object)
    bad.loc[0, "trestbps"] = "abc"
    assert np.isnan(clean(bad).loc[0, "trestbps"])


def test_committed_clean_dataset(heart_df):
    assert heart_df.shape == (303, 14)
    assert heart_df[TARGET].isin([0, 1]).all()
    assert heart_df[TARGET].notna().all()
    assert heart_df.duplicated().sum() == 0
    # Cleveland: 164 without disease, 139 with disease
    assert heart_df[TARGET].value_counts().to_dict() == {0: 164, 1: 139}


def test_download_falls_back_to_mirror(monkeypatch, tmp_path):
    """If both UCI URLs fail, the mirror is used and SOURCE.txt records it."""
    import heart.data as data

    mirror = (",Age,Sex,ChestPain,RestBP,Chol,Fbs,RestECG,MaxHR,ExAng,Oldpeak,Slope,Ca,Thal,AHD\n"
              "1,63,1,typical,145,233,1,2,150,0,2.3,3,0,fixed,No\n")

    def fake_fetch(url, timeout=30):
        if url == data.MIRROR_URL:
            return mirror.encode()
        raise ConnectionError("blocked")

    monkeypatch.setattr(data, "_fetch", fake_fetch)
    df, source = data.download_raw(tmp_path / "raw.csv")
    assert source == data.MIRROR_URL and len(df) == 1
    assert "sha256" in (tmp_path / "SOURCE.txt").read_text()


def test_download_raises_when_all_sources_fail(monkeypatch, tmp_path):
    import heart.data as data

    def fail(url, timeout=30):
        raise ConnectionError("offline")

    monkeypatch.setattr(data, "_fetch", fail)
    with pytest.raises(RuntimeError, match="All download sources failed"):
        data.download_raw(tmp_path / "raw.csv")


def test_eda_figures_are_generated(heart_df, tmp_path):
    from heart import eda

    paths = [eda.plot_class_balance(heart_df, tmp_path), eda.plot_correlation(heart_df, tmp_path),
             eda.plot_numeric_histograms(heart_df, tmp_path)]
    assert all(p.exists() and p.stat().st_size > 10_000 for p in paths)
