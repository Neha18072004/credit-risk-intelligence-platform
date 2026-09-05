"""Fixture integrity tests.

These lock in the properties the rest of the pipeline depends on: the real
Home Credit schema, the ~8% default rate, the DAYS_EMPLOYED sentinel, genuine
missingness, and applicants who have no bureau history at all.
"""

from __future__ import annotations

import pandas as pd

from src.data.generate_sample import DAYS_EMPLOYED_ANOMALY


def test_application_train_shape(application_train: pd.DataFrame) -> None:
    """The fixture reproduces the real 122-column application schema."""
    assert application_train.shape[1] == 122
    assert len(application_train) >= 1000
    assert application_train["SK_ID_CURR"].is_unique


def test_required_columns_present(application_train: pd.DataFrame) -> None:
    """Columns the downstream feature engineering relies on all exist."""
    required = {
        "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "NAME_CONTRACT_TYPE", "NAME_EDUCATION_TYPE", "CODE_GENDER",
    }
    assert required.issubset(application_train.columns)


def test_target_is_imbalanced(application_train: pd.DataFrame) -> None:
    """The class balance sits near the real ~8% default rate."""
    rate = application_train["TARGET"].mean()
    assert set(application_train["TARGET"].unique()) == {0, 1}
    assert 0.05 < rate < 0.12, f"default rate {rate:.3f} outside the expected band"


def test_days_employed_anomaly_present(application_train: pd.DataFrame) -> None:
    """The 365243 sentinel exists and attaches to pensioners, as in reality."""
    anomalous = application_train["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
    assert anomalous.mean() > 0.10
    income_types = set(application_train.loc[anomalous, "NAME_INCOME_TYPE"].unique())
    assert income_types <= {"Pensioner", "Unemployed"}


def test_days_columns_are_negative(application_train: pd.DataFrame) -> None:
    """DAYS_* offsets are negative, except for the known sentinel."""
    assert (application_train["DAYS_BIRTH"] < 0).all()
    real_tenure = application_train.loc[
        application_train["DAYS_EMPLOYED"] != DAYS_EMPLOYED_ANOMALY, "DAYS_EMPLOYED"
    ]
    assert (real_tenure < 0).all()


def test_missingness_is_realistic(application_train: pd.DataFrame) -> None:
    """EXT_SOURCE_1 and OCCUPATION_TYPE carry their real-world null rates."""
    assert 0.45 < application_train["EXT_SOURCE_1"].isna().mean() < 0.70
    assert 0.20 < application_train["OCCUPATION_TYPE"].isna().mean() < 0.40


def test_ext_source_carries_signal(application_train: pd.DataFrame) -> None:
    """Low external scores must default more often, or the bake-off is noise."""
    quintile = pd.qcut(application_train["EXT_SOURCE_2"], 5, labels=False)
    by_quintile = application_train.groupby(quintile, observed=True)["TARGET"].mean()
    assert by_quintile.iloc[0] > by_quintile.iloc[-1]


def test_bureau_schema_and_coverage(
    bureau: pd.DataFrame, application_train: pd.DataFrame
) -> None:
    """Bureau has the real 17 columns and only partial applicant coverage."""
    assert bureau.shape[1] == 17
    assert bureau["SK_ID_BUREAU"].is_unique
    assert set(bureau["CREDIT_ACTIVE"].unique()) <= {"Active", "Closed", "Sold", "Bad debt"}

    covered = application_train["SK_ID_CURR"].isin(bureau["SK_ID_CURR"])
    assert 0.0 < (~covered).mean() < 0.30, "some applicants must lack bureau history"


def test_test_split_has_no_target(sample_dir) -> None:
    """application_test mirrors train minus the label."""
    test = pd.read_csv(sample_dir / "application_test.csv")
    assert "TARGET" not in test.columns
    assert test.shape[1] == 121
