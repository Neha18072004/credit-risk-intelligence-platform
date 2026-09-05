"""Unit tests for the shared helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.helpers import (
    categorize_features,
    domain_of,
    missing_value_report,
    probability_to_score,
    safe_divide,
)


def test_safe_divide_handles_zero_and_nan() -> None:
    """Division never produces inf, which would break LogisticRegression."""
    result = safe_divide(pd.Series([10.0, 5.0, np.nan]), pd.Series([2.0, 0.0, 4.0]))
    assert result.iloc[0] == 5.0
    assert np.isnan(result.iloc[1])  # 5/0 -> NaN, not inf
    assert np.isnan(result.iloc[2])
    assert np.isfinite(result.dropna()).all()


def test_missing_value_report_orders_worst_first() -> None:
    frame = pd.DataFrame({"full": [1, 2, 3], "half": [1, None, None]})
    report = missing_value_report(frame)
    assert report.iloc[0]["column"] == "half"
    assert report.iloc[0]["pct_missing"] == 66.67


def test_categorize_features_buckets() -> None:
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "DAYS_BIRTH": [-1000, -2000],
            "FLAG_OWN_CAR_BIN": [0, 1],
            "AMT_INCOME_TOTAL": [1.0, 2.5],
            "NAME_CONTRACT_TYPE": ["a", "b"],
        }
    )
    buckets = categorize_features(frame)
    assert buckets["identifier"] == ["SK_ID_CURR"]
    assert buckets["date_like"] == ["DAYS_BIRTH"]
    assert buckets["binary"] == ["FLAG_OWN_CAR_BIN"]
    assert buckets["categorical"] == ["NAME_CONTRACT_TYPE"]
    assert "TARGET" not in sum(buckets.values(), [])


def test_domain_of_known_columns() -> None:
    assert domain_of("EXT_SOURCE_1") == "external_scores"
    assert domain_of("AMT_CREDIT") == "financials"
    assert domain_of("FLAG_DOCUMENT_3") == "documents"
    assert domain_of("BUREAU_ACTIVE_COUNT") == "credit_history"
    assert domain_of("SK_ID_CURR") == "identifier"


def test_probability_to_score_is_monotonic_and_bounded() -> None:
    """Higher probability must mean a higher (riskier) score."""
    low, high = probability_to_score(0.02), probability_to_score(0.6)
    assert low < high
    assert probability_to_score(0.0) == 0.0
    assert probability_to_score(1.0) > 0
