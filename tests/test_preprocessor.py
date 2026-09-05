"""Tests for cleaning, feature engineering and the fitted transform pipeline."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.data.preprocessor import (
    DAYS_EMPLOYED_ANOMALY,
    MISSING_CATEGORY,
    CreditPreprocessor,
    clean_applications,
    engineer_features,
)


# ----------------------------------------------------------------- cleaning --
def test_days_employed_sentinel_is_replaced(joined_dataset: pd.DataFrame) -> None:
    """The 365243 sentinel becomes NaN plus an explicit flag."""
    cleaned = clean_applications(joined_dataset)
    assert (cleaned["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).sum() == 0
    assert cleaned["DAYS_EMPLOYED_ANOMALY"].sum() > 0
    # The flag must mark exactly the rows that carried the sentinel.
    original = joined_dataset["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
    assert (cleaned["DAYS_EMPLOYED_ANOMALY"] == original.astype(int)).all()


def test_cleaning_does_not_mutate_input(joined_dataset: pd.DataFrame) -> None:
    before = joined_dataset["DAYS_EMPLOYED"].copy()
    clean_applications(joined_dataset)
    pd.testing.assert_series_equal(before, joined_dataset["DAYS_EMPLOYED"])


def test_xna_gender_becomes_null() -> None:
    frame = pd.DataFrame({"CODE_GENDER": ["M", "F", "XNA"], "DAYS_BIRTH": [-1, -2, -3]})
    assert clean_applications(frame)["CODE_GENDER"].isna().sum() == 1


def test_positive_day_offsets_are_nulled() -> None:
    frame = pd.DataFrame({"DAYS_REGISTRATION": [-100.0, 250.0], "DAYS_BIRTH": [-9000, -9000]})
    assert clean_applications(frame)["DAYS_REGISTRATION"].isna().sum() == 1


def test_overdue_days_survive_cleaning() -> None:
    """Days *past due* are positive durations and must not be nulled.

    Regression test: the naive "every DAYS_ column must be negative" rule
    silently deleted these columns and the anomaly flag.
    """
    frame = pd.DataFrame(
        {
            "BUREAU_DAYS_OVERDUE_MAX": [0.0, 45.0, 180.0],
            "DAYS_EMPLOYED": [-100, DAYS_EMPLOYED_ANOMALY, -300],
        }
    )
    cleaned = clean_applications(frame)
    assert cleaned["BUREAU_DAYS_OVERDUE_MAX"].notna().all()
    assert cleaned["BUREAU_DAYS_OVERDUE_MAX"].max() == 180.0
    assert cleaned["DAYS_EMPLOYED_ANOMALY"].tolist() == [0, 1, 0]


# -------------------------------------------------------------- engineering --
def test_ratio_features_are_correct() -> None:
    frame = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100_000.0],
            "AMT_CREDIT": [400_000.0],
            "AMT_ANNUITY": [20_000.0],
            "AMT_GOODS_PRICE": [360_000.0],
            "DAYS_BIRTH": [-14610],
            "DAYS_EMPLOYED": [-3652.5],
        }
    )
    out = engineer_features(frame)
    assert out["CREDIT_TO_INCOME_RATIO"].iloc[0] == pytest.approx(4.0)
    assert out["ANNUITY_TO_INCOME_RATIO"].iloc[0] == pytest.approx(0.2)
    assert out["CREDIT_TO_ANNUITY_RATIO"].iloc[0] == pytest.approx(20.0)
    assert out["PAYMENT_RATE"].iloc[0] == pytest.approx(0.05)
    assert out["GOODS_TO_CREDIT_RATIO"].iloc[0] == pytest.approx(0.9)
    assert out["AGE_YEARS"].iloc[0] == pytest.approx(40.0, abs=0.1)
    assert out["EMPLOYED_TO_AGE_RATIO"].iloc[0] == pytest.approx(0.25, abs=0.01)


def test_ext_source_aggregates_survive_partial_missingness() -> None:
    """A row with only one external score still gets a usable mean."""
    frame = pd.DataFrame(
        {
            "EXT_SOURCE_1": [0.2, np.nan],
            "EXT_SOURCE_2": [0.4, np.nan],
            "EXT_SOURCE_3": [0.6, 0.5],
        }
    )
    out = engineer_features(frame)
    assert out["EXT_SOURCE_MEAN"].iloc[0] == pytest.approx(0.4)
    assert out["EXT_SOURCE_MEAN"].iloc[1] == pytest.approx(0.5)
    assert out["EXT_SOURCE_MIN"].iloc[0] == pytest.approx(0.2)
    assert out["EXT_SOURCE_COUNT"].tolist() == [3, 1]


def test_zero_income_does_not_produce_infinity() -> None:
    """Division guards keep inf out of the matrix; LogReg cannot handle it."""
    frame = pd.DataFrame({"AMT_INCOME_TOTAL": [0.0], "AMT_CREDIT": [100_000.0]})
    value = engineer_features(frame)["CREDIT_TO_INCOME_RATIO"].iloc[0]
    assert np.isnan(value)


# --------------------------------------------------------------- transformer --
def test_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        CreditPreprocessor().transform(pd.DataFrame({"AMT_CREDIT": [1.0]}))


def test_output_schema_is_stable(fitted_preprocessor, feature_matrix) -> None:
    assert list(feature_matrix.columns) == fitted_preprocessor.feature_names_
    assert feature_matrix.shape[1] == fitted_preprocessor.n_features_in_


def test_no_target_or_id_leaks_into_features(feature_matrix: pd.DataFrame) -> None:
    assert "TARGET" not in feature_matrix.columns
    assert "SK_ID_CURR" not in feature_matrix.columns


def test_key_engineered_features_are_retained(feature_matrix: pd.DataFrame) -> None:
    """Regression guard: these must not be dropped as 'constant'."""
    for column in (
        "EXT_SOURCE_MEAN", "CREDIT_TO_INCOME_RATIO", "PAYMENT_RATE",
        "AGE_YEARS", "DAYS_EMPLOYED_ANOMALY", "BUREAU_DAYS_OVERDUE_MAX",
    ):
        assert column in feature_matrix.columns, f"{column} was dropped"
    assert feature_matrix["DAYS_EMPLOYED_ANOMALY"].nunique() == 2


def test_numeric_block_has_no_infinities(fitted_preprocessor, feature_matrix) -> None:
    values = feature_matrix[fitted_preprocessor.numeric_features_].to_numpy(dtype=float)
    assert np.all(np.isfinite(values) | np.isnan(values))


def test_categoricals_are_category_dtype(fitted_preprocessor, feature_matrix) -> None:
    """Tree models consume category dtype natively; LogReg one-hots it later."""
    for column in fitted_preprocessor.categorical_features_:
        assert isinstance(feature_matrix[column].dtype, pd.CategoricalDtype)
        assert not feature_matrix[column].isna().any(), "categoricals use MISSING, not NaN"


def test_missing_categoricals_become_explicit_level(fitted_preprocessor, joined_dataset) -> None:
    frame = joined_dataset.assign(OCCUPATION_TYPE=np.nan)
    assert (fitted_preprocessor.transform(frame)["OCCUPATION_TYPE"] == MISSING_CATEGORY).all()


def test_unseen_category_collapses_to_missing(fitted_preprocessor, joined_dataset) -> None:
    """An unseen level must not silently shift the encoding at inference."""
    frame = joined_dataset.iloc[[0]].copy()
    frame["NAME_INCOME_TYPE"] = "Astronaut"
    assert fitted_preprocessor.transform(frame)["NAME_INCOME_TYPE"].iloc[0] == MISSING_CATEGORY


def test_single_row_with_missing_columns_transforms(fitted_preprocessor, joined_dataset) -> None:
    """The UI submits a partial applicant form; that must still transform."""
    partial = joined_dataset.iloc[[0]][
        ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "EXT_SOURCE_2"]
    ]
    out = fitted_preprocessor.transform(partial)
    assert out.shape == (1, fitted_preprocessor.n_features_in_)
    assert list(out.columns) == fitted_preprocessor.feature_names_


def test_preprocessor_is_picklable(fitted_preprocessor, joined_dataset, feature_matrix) -> None:
    """The artifact is saved with the model and reloaded at inference time."""
    restored = pickle.loads(pickle.dumps(fitted_preprocessor))
    pd.testing.assert_frame_equal(restored.transform(joined_dataset), feature_matrix)


def test_transform_is_deterministic(fitted_preprocessor, joined_dataset, feature_matrix) -> None:
    pd.testing.assert_frame_equal(fitted_preprocessor.transform(joined_dataset), feature_matrix)


def test_categorical_indices_match_names(fitted_preprocessor) -> None:
    """CatBoost consumes positional indices; they must agree with the names."""
    names = [fitted_preprocessor.feature_names_[i]
             for i in fitted_preprocessor.categorical_indices_]
    assert names == fitted_preprocessor.categorical_features_
