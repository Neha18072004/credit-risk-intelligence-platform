"""Tests for dataset loading, bureau aggregation and joining."""

from __future__ import annotations

import numpy as np
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
