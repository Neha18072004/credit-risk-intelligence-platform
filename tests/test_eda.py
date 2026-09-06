"""Tests for the EDA layer: analysis correctness and chart-safety guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from notebooks import eda
from src.utils.viz import MAX_ORDINAL_BANDS, RISK_BAND_COLORS, SERIES, ordinal_ramp


# ------------------------------------------------------------- statistics --
def test_wilson_interval_brackets_the_estimate() -> None:
    low, high = eda.wilson_interval(np.array([10, 1]), np.array([100, 10]))
    assert low[0] < 10.0 < high[0]
    assert low[1] < 10.0 < high[1]
    # A smaller sample must produce a wider interval at the same rate.
    assert (high[1] - low[1]) > (high[0] - low[0])


def test_wilson_interval_stays_within_bounds() -> None:
    """The normal approximation goes negative here; Wilson must not."""
    low, high = eda.wilson_interval(np.array([0, 50]), np.array([50, 50]))
    assert low.min() >= 0.0
    assert high.max() <= 100.0


def test_default_rate_by_segment_computes_correct_rates() -> None:
    # Segment "a": 10 defaults in 50 (20%). Segment "b": 20 in 50 (40%).
    frame = pd.DataFrame(
        {
            "TARGET": [1] * 10 + [0] * 40 + [1] * 20 + [0] * 30,
            "seg": ["a"] * 50 + ["b"] * 50,
        }
    )
    table = eda.default_rate_by_segment(frame, frame["seg"]).set_index("segment")
    assert table.loc["a", "default_rate"] == pytest.approx(20.0)
    assert table.loc["b", "default_rate"] == pytest.approx(40.0)
    assert table.loc["a", "n"] == 50 and table.loc["a", "n_default"] == 10
    assert (table["ci_low"] < table["default_rate"]).all()
    assert (table["ci_high"] > table["default_rate"]).all()


def test_default_rate_by_segment_drops_thin_segments() -> None:
    """A rate computed on a handful of rows is noise and must not be plotted."""
    frame = pd.DataFrame({"TARGET": [1] * 5 + [0] * 95, "seg": ["tiny"] * 5 + ["big"] * 95})
    table = eda.default_rate_by_segment(frame, frame["seg"], min_count=50)
    assert table["segment"].tolist() == ["big"]


def test_separated_detects_overlap() -> None:
    overlapping = pd.DataFrame(
        {"default_rate": [5.0, 7.0], "ci_low": [3.0, 5.0], "ci_high": [8.0, 10.0]}
    )
    distinct = pd.DataFrame(
        {"default_rate": [5.0, 20.0], "ci_low": [4.0, 18.0], "ci_high": [6.0, 22.0]}
    )
    assert eda._separated(overlapping) is False
    assert eda._separated(distinct) is True


# ------------------------------------------------------------- structure --
def test_dataset_overview_reports_target_stats(joined_dataset: pd.DataFrame) -> None:
    summary = eda.dataset_overview(joined_dataset)
    assert summary["n_rows"] == len(joined_dataset)
    assert 0.05 < summary["default_rate"] < 0.12
    assert summary["imbalance_ratio"] > 5


def test_feature_catalog_covers_every_column(joined_dataset: pd.DataFrame) -> None:
    catalog = eda.feature_catalog(joined_dataset)
    assert len(catalog) == joined_dataset.shape[1]
    assert set(catalog["column"]) == set(joined_dataset.columns)
    assert catalog["type_bucket"].notna().all()
    assert catalog["domain"].notna().all()


def test_data_quality_finds_the_employment_sentinel(joined_dataset: pd.DataFrame) -> None:
    findings = eda.data_quality_findings(joined_dataset)
    issues = [f["issue"] for f in findings]
    assert any("365243" in issue for issue in issues)
    sentinel = next(f for f in findings if "365243" in f["issue"])
    assert sentinel["severity"] == "high"
    assert sentinel["n_affected"] > 0
    # Every finding must state what was actually done about it.
    assert all(f["treatment"] for f in findings)


def test_data_quality_flags_thin_files(joined_dataset: pd.DataFrame) -> None:
    issues = [f["issue"] for f in eda.data_quality_findings(joined_dataset)]
    assert any("no external credit history" in issue for issue in issues)


# ---------------------------------------------------------------- charts --
def test_ordinal_ramp_is_single_hue_and_capped() -> None:
    """Ordered bands use one hue; beyond the cap the bands stop being separable."""
    ramp = ordinal_ramp(5)
    assert len(ramp) == 5
    assert len(set(ramp)) == 5
    with pytest.raises(ValueError, match="bands"):
        ordinal_ramp(MAX_ORDINAL_BANDS + 1)


def test_categorical_slots_are_fixed_order() -> None:
    """Colour follows identity, not rank, so the slot order must be stable."""
    assert SERIES[0] == "#2a78d6"
    assert len(set(SERIES)) == len(SERIES)


def test_risk_band_colors_are_distinct() -> None:
    assert set(RISK_BAND_COLORS) == {"Low", "Medium", "High"}
    assert len(set(RISK_BAND_COLORS.values())) == 3


# ------------------------------------------------------------- insights ---
@pytest.mark.parametrize("builder", eda.INSIGHT_BUILDERS, ids=lambda b: b.__name__)
def test_every_insight_builds(builder, joined_dataset: pd.DataFrame) -> None:
    """Each insight must produce a non-trivial takeaway and a table."""
    insight = builder(joined_dataset, False)  # save=False: no disk writes in tests
    assert insight.key and insight.title
    assert len(insight.takeaway) > 80, "a takeaway must actually explain something"
    assert isinstance(insight.table, pd.DataFrame)
    assert insight.figure_path is None


def test_insight_count_meets_requirement(joined_dataset: pd.DataFrame) -> None:
    """The brief asks for at least five business insights."""
    assert len(eda.INSIGHT_BUILDERS) >= 5


def test_external_score_insight_finds_the_gradient(joined_dataset: pd.DataFrame) -> None:
    """The strongest signal in the data must show up as a downward gradient."""
    insight = eda.insight_external_scores(joined_dataset, False)
    rates = insight.table["default_rate"]
    assert rates.iloc[0] > rates.iloc[-1]
    assert eda._separated(insight.table), "extreme bands should separate clearly"


def test_credit_history_insight_orders_arrears_correctly(joined_dataset: pd.DataFrame) -> None:
    """Prior arrears must read as higher risk, as it does in the real data."""
    insight = eda.insight_credit_history(joined_dataset, False)
    profile = insight.table[insight.table["view"] == "profile"].set_index("segment")
    assert profile.loc["History with arrears", "default_rate"] > profile.loc[
        "History, no arrears", "default_rate"
    ]


def test_run_eda_assembles_a_full_report(joined_dataset: pd.DataFrame) -> None:
    report = eda.run_eda(joined_dataset, save=False)
    assert len(report.insights) == len(eda.INSIGHT_BUILDERS)
    assert not report.feature_catalog.empty
    assert not report.missingness.empty
    assert report.quality_findings

    index = report.insight_index()
    assert len(index) == len(report.insights)
    assert index["takeaway"].str.len().min() > 80
