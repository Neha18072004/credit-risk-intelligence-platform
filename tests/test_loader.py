"""Tests for dataset loading, bureau aggregation and joining."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import (
    BUREAU_PREFIX,
    ID_COLUMN,
    aggregate_bureau,
    build_dataset,
    dataset_summary,
    split_features_target,
)


def test_aggregate_bureau_is_one_row_per_applicant(bureau: pd.DataFrame) -> None:
    aggregated = aggregate_bureau(bureau)
    assert aggregated[ID_COLUMN].is_unique
    assert len(aggregated) == bureau[ID_COLUMN].nunique()


def test_aggregate_bureau_counts_are_consistent(bureau: pd.DataFrame) -> None:
    """Active + closed must reconcile with the total loan count."""
    aggregated = aggregate_bureau(bureau).set_index(ID_COLUMN)
    totals = aggregated[f"{BUREAU_PREFIX}ACTIVE_COUNT"] + aggregated[f"{BUREAU_PREFIX}CLOSED_COUNT"]
    assert (totals <= aggregated[f"{BUREAU_PREFIX}LOAN_COUNT"]).all()

    expected = bureau.groupby(ID_COLUMN).size()
    assert (aggregated[f"{BUREAU_PREFIX}LOAN_COUNT"] == expected).all()


def test_aggregate_bureau_ratios_are_bounded(bureau: pd.DataFrame) -> None:
    aggregated = aggregate_bureau(bureau)
    active_ratio = aggregated[f"{BUREAU_PREFIX}ACTIVE_RATIO"].dropna()
    assert active_ratio.between(0.0, 1.0).all()
    assert set(aggregated[f"{BUREAU_PREFIX}HAS_OVERDUE"].unique()) <= {0, 1}


def test_aggregate_bureau_handles_empty_input() -> None:
    """An empty bureau table must not crash the pipeline."""
    result = aggregate_bureau(pd.DataFrame(columns=["SK_ID_CURR", "SK_ID_BUREAU"]))
    assert result.empty


def test_build_dataset_left_joins(joined_dataset: pd.DataFrame,
                                  application_train: pd.DataFrame) -> None:
    """The join must preserve every applicant, including thin-file ones."""
    assert len(joined_dataset) == len(application_train)
    assert joined_dataset[ID_COLUMN].is_unique
    # Some applicants genuinely have no bureau history.
    assert (joined_dataset[f"{BUREAU_PREFIX}HAS_HISTORY"] == 0).any()
    assert (joined_dataset[f"{BUREAU_PREFIX}LOAN_COUNT"] == 0).any()


def test_thin_file_applicants_have_null_bureau_metrics(joined_dataset: pd.DataFrame) -> None:
    """No-history applicants keep NaN metrics -- that absence is real signal."""
    thin = joined_dataset[joined_dataset[f"{BUREAU_PREFIX}HAS_HISTORY"] == 0]
    assert len(thin) > 0
    assert thin[f"{BUREAU_PREFIX}DEBT_TOTAL"].isna().all()


def test_build_dataset_without_bureau() -> None:
    """The bureau block can be switched off to measure its contribution."""
    frame = build_dataset("train", include_bureau=False)
    assert not any(column.startswith(BUREAU_PREFIX) for column in frame.columns)


def test_split_features_target_drops_id_and_label(joined_dataset: pd.DataFrame) -> None:
    features, labels = split_features_target(joined_dataset)
    assert "TARGET" not in features.columns
    assert ID_COLUMN not in features.columns
    assert labels is not None and len(labels) == len(features)


def test_split_features_target_on_unlabelled_data() -> None:
    features, labels = split_features_target(build_dataset("test"))
    assert labels is None


def test_dataset_summary_reports_imbalance(joined_dataset: pd.DataFrame) -> None:
    summary = dataset_summary(joined_dataset)
    assert summary["n_rows"] == len(joined_dataset)
    assert 0.05 < summary["default_rate"] < 0.12
    assert summary["imbalance_ratio"] > 5
    assert summary["n_bureau_features"] > 20


def test_invalid_split_rejected() -> None:
    with pytest.raises(ValueError, match="train.*test"):
        from src.data.loader import load_applications

        load_applications("validation")


# ------------------------------------------- previous applications --------
def test_aggregate_previous_application_computes_refusal_rate() -> None:
    """The refusal rate is the headline feature: an applicant this lender has
    already declined is a different proposition from a first-time applicant."""
    from src.data.loader import PREV_PREFIX, aggregate_previous_application

    previous = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 1, 2, 2],
            "SK_ID_PREV": [10, 11, 12, 13, 20, 21],
            "NAME_CONTRACT_STATUS": [
                "Refused", "Refused", "Approved", "Canceled", "Approved", "Approved",
            ],
            "AMT_APPLICATION": [100.0, 100.0, 200.0, 100.0, 500.0, 500.0],
            "AMT_CREDIT": [0.0, 0.0, 150.0, 0.0, 500.0, 400.0],
            "AMT_ANNUITY": [0.0, 0.0, 20.0, 0.0, 50.0, 40.0],
            "AMT_DOWN_PAYMENT": [0.0] * 6,
            "RATE_DOWN_PAYMENT": [0.0] * 6,
            "DAYS_DECISION": [-100, -200, -300, -400, -50, -60],
            "CNT_PAYMENT": [12.0] * 6,
            "NAME_YIELD_GROUP": ["middle"] * 6,
        }
    )
    result = aggregate_previous_application(previous).set_index("SK_ID_CURR")

    assert result.loc[1, f"{PREV_PREFIX}COUNT"] == 4
    assert result.loc[1, f"{PREV_PREFIX}REFUSED_COUNT"] == 2
    assert result.loc[1, f"{PREV_PREFIX}REFUSED_RATE"] == pytest.approx(0.5)
    assert result.loc[1, f"{PREV_PREFIX}EVER_REFUSED"] == 1
    assert result.loc[2, f"{PREV_PREFIX}REFUSED_RATE"] == pytest.approx(0.0)
    assert result.loc[2, f"{PREV_PREFIX}EVER_REFUSED"] == 0


def test_previous_application_credit_to_application_ratio() -> None:
    """Below 1 means underwriting granted less than was asked for."""
    from src.data.loader import PREV_PREFIX, aggregate_previous_application

    previous = pd.DataFrame(
        {
            "SK_ID_CURR": [1], "SK_ID_PREV": [10], "NAME_CONTRACT_STATUS": ["Approved"],
            "AMT_APPLICATION": [1000.0], "AMT_CREDIT": [750.0], "AMT_ANNUITY": [80.0],
            "AMT_DOWN_PAYMENT": [0.0], "RATE_DOWN_PAYMENT": [0.0],
            "DAYS_DECISION": [-100], "CNT_PAYMENT": [12.0], "NAME_YIELD_GROUP": ["middle"],
        }
    )
    result = aggregate_previous_application(previous)
    assert result[f"{PREV_PREFIX}CREDIT_TO_APPLICATION"].iloc[0] == pytest.approx(0.75)


# ------------------------------------------------ repayment behaviour -----
def test_aggregate_installments_measures_lateness_and_underpayment() -> None:
    """Days past due and payment ratio are the two quantities that carry the
    repayment-behaviour signal, so their arithmetic is pinned down here."""
    from src.data.loader import INST_PREFIX, aggregate_installments

    # DAYS_* are negative offsets from application, so entry minus instalment
    # is still positive when the payment landed late.
    installments = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2, 2],
            "DAYS_INSTALMENT": [-300.0, -270.0, -240.0, -100.0, -70.0],
            "DAYS_ENTRY_PAYMENT": [-290.0, -270.0, -250.0, -100.0, -70.0],  # +10, 0, early
            "AMT_INSTALMENT": [100.0, 100.0, 100.0, 50.0, 50.0],
            "AMT_PAYMENT": [100.0, 60.0, 100.0, 50.0, 50.0],  # one underpayment
        }
    )
    result = aggregate_installments(installments).set_index("SK_ID_CURR")

    assert result.loc[1, f"{INST_PREFIX}COUNT"] == 3
    assert result.loc[1, f"{INST_PREFIX}DPD_MAX"] == pytest.approx(10.0)
    assert result.loc[1, f"{INST_PREFIX}LATE_COUNT"] == 1
    assert result.loc[1, f"{INST_PREFIX}LATE_RATE"] == pytest.approx(1 / 3)
    assert result.loc[1, f"{INST_PREFIX}EVER_LATE"] == 1
    assert result.loc[1, f"{INST_PREFIX}UNDERPAID_COUNT"] == 1
    assert result.loc[1, f"{INST_PREFIX}SHORTFALL_SUM"] == pytest.approx(40.0)

    # Applicant 2 paid everything on time and in full.
    assert result.loc[2, f"{INST_PREFIX}EVER_LATE"] == 0
    assert result.loc[2, f"{INST_PREFIX}LATE_RATE"] == pytest.approx(0.0)
    assert result.loc[2, f"{INST_PREFIX}PAYMENT_RATIO_MEAN"] == pytest.approx(1.0)


def test_early_payment_is_not_counted_as_late() -> None:
    """Paying early must not register as days past due."""
    from src.data.loader import INST_PREFIX, aggregate_installments

    installments = pd.DataFrame(
        {
            "SK_ID_CURR": [1], "DAYS_INSTALMENT": [-100.0], "DAYS_ENTRY_PAYMENT": [-120.0],
            "AMT_INSTALMENT": [100.0], "AMT_PAYMENT": [100.0],
        }
    )
    result = aggregate_installments(installments)
    assert result[f"{INST_PREFIX}DPD_MAX"].iloc[0] == 0.0
    assert result[f"{INST_PREFIX}EVER_LATE"].iloc[0] == 0


def test_empty_auxiliary_tables_are_handled() -> None:
    from src.data.loader import aggregate_installments, aggregate_previous_application

    assert aggregate_previous_application(pd.DataFrame()).empty
    assert aggregate_installments(pd.DataFrame()).empty


# ------------------------------------------------------- composition ------
def test_blocks_can_be_switched_off_independently() -> None:
    """Each block is a flag so its contribution can be measured, not assumed."""
    from src.data.loader import build_dataset

    minimal = build_dataset(
        "train", include_bureau=False, include_previous=False, include_installments=False
    )
    assert not any(
        column.startswith(("BUREAU_", "PREV_", "INST_")) for column in minimal.columns
    )

    with_bureau = build_dataset(
        "train", include_bureau=True, include_previous=False, include_installments=False
    )
    assert any(column.startswith("BUREAU_") for column in with_bureau.columns)
    assert not any(column.startswith(("PREV_", "INST_")) for column in with_bureau.columns)


def test_sample_fixtures_cover_every_table(sample_dir) -> None:
    """The committed fixtures must exercise the same blocks as the real data,
    or an evaluator running the default sample mode sees a lesser platform."""
    for name in (
        "application_train", "application_test", "bureau",
        "previous_application", "installments_payments",
    ):
        assert (sample_dir / f"{name}.csv").exists(), f"missing fixture {name}.csv"


def test_repayment_features_present_in_sample_mode(joined_dataset) -> None:
    from src.data.loader import INST_PREFIX, PREV_PREFIX

    assert f"{INST_PREFIX}LATE_RATE" in joined_dataset.columns
    assert f"{PREV_PREFIX}REFUSED_RATE" in joined_dataset.columns
    # And they must carry signal, not just exist.
    late = joined_dataset[joined_dataset[f"{INST_PREFIX}EVER_LATE"] == 1]["TARGET"].mean()
    ontime = joined_dataset[joined_dataset[f"{INST_PREFIX}EVER_LATE"] == 0]["TARGET"].mean()
    assert late > ontime, "late payers must default more, or the fixture is noise"
