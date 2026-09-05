"""Exploratory data analysis for the Home Credit application portfolio.

This module is the single source of truth for the EDA. ``eda.ipynb`` imports and
calls it rather than duplicating the logic, so the notebook, the figures written
into ``reports/figures/`` and the Streamlit EDA tab can never drift apart.

It delivers four things:

1. **Dataset summary** -- shape, target distribution, class imbalance.
2. **Feature categorization** -- by storage type and by business domain.
3. **Data-quality analysis** -- per-column missingness, the ``DAYS_EMPLOYED``
   sentinel, outliers and suspicious values.
4. **Business insights** -- six findings, each a chart plus a plain-English
   takeaway a credit analyst could act on.

Run the whole thing from the command line::

    python -m notebooks.eda
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # figures are written to disk, never displayed interactively

from src.data.loader import BUREAU_PREFIX, build_dataset, dataset_summary
from src.data.preprocessor import DAYS_EMPLOYED_ANOMALY, clean_applications, engineer_features
from src.utils.config import settings
from src.utils.helpers import categorize_features, domain_of, missing_value_report, write_json
from src.utils.logger import get_logger
from src.utils.viz import (
    INK_SECONDARY,
    SERIES,
    add_reference_line,
    apply_theme,
    label_bars,
    new_figure,
    ordinal_ramp,
    rate_and_volume_panels,
    save_figure,
    style_axes,
)

logger = get_logger(__name__)
apply_theme()

TARGET: Final[str] = "TARGET"

# Five bands is the ceiling the validated ordinal colour ramp can keep visually
# separable, and it maps naturally onto quintiles.
N_BANDS: Final[int] = 5


@dataclass(frozen=True)
class Insight:
    """One business finding: a chart, a table and a plain-English takeaway."""

    key: str
    title: str
    takeaway: str
    table: pd.DataFrame
    figure_path: Path | None = None

    def describe(self) -> str:
        """Render the insight as readable text for logs and the README."""
        return f"{self.title}\n  {self.takeaway}"


@dataclass
class EDAReport:
    """Everything the EDA produces, ready for the notebook, README and UI."""

    summary: dict[str, object]
    feature_catalog: pd.DataFrame
    missingness: pd.DataFrame
    quality_findings: list[dict[str, object]]
    insights: list[Insight] = field(default_factory=list)

    def insight_index(self) -> pd.DataFrame:
        """Tabular index of the insights, for the UI and the README."""
        return pd.DataFrame(
            [
                {
                    "key": insight.key,
                    "title": insight.title,
                    "takeaway": insight.takeaway,
                    "figure": insight.figure_path.name if insight.figure_path else None,
                }
                for insight in self.insights
            ]
        )


# --------------------------------------------------------------------------- #
# 1. Dataset summary
# --------------------------------------------------------------------------- #
def dataset_overview(frame: pd.DataFrame) -> dict[str, object]:
    """Headline shape and target statistics.

    Args:
        frame: The joined application + bureau dataset.

    Returns:
        A dictionary of summary statistics, including the class imbalance ratio
        that the model's class weighting has to counteract.
    """
    summary = dict(dataset_summary(frame))
    summary["n_applicants"] = int(frame["SK_ID_CURR"].nunique()) if "SK_ID_CURR" in frame else len(frame)
    logger.info(
        "Dataset: %s rows x %s cols | default rate %.2f%% | imbalance %.1f:1",
        f"{summary['n_rows']:,}", summary["n_columns"],
        100 * float(summary.get("default_rate", 0)), summary.get("imbalance_ratio", 0),
    )
    return summary


# --------------------------------------------------------------------------- #
# 2. Feature categorization
# --------------------------------------------------------------------------- #
def feature_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """Catalogue every column by storage type and business domain.

    Two independent groupings, because they answer different questions. The
    storage type decides how a column is *encoded*; the business domain decides
    who *owns* it and how a finding about it gets acted on.

    Args:
        frame: The joined dataset.

    Returns:
        One row per column with ``column``, ``type_bucket``, ``domain``,
        ``pct_missing`` and ``n_unique``.
    """
    buckets = categorize_features(frame, target=TARGET)
    type_of: dict[str, str] = {
        column: bucket for bucket, columns in buckets.items() for column in columns
    }
    missing = missing_value_report(frame).set_index("column")

    rows = [
        {
            "column": column,
            "type_bucket": type_of.get(column, "target" if column == TARGET else "other"),
            "domain": domain_of(column),
            "pct_missing": float(missing.loc[column, "pct_missing"]),
            "n_unique": int(missing.loc[column, "n_unique"]),
        }
        for column in frame.columns
    ]
    catalog = pd.DataFrame(rows)
    logger.info(
        "Feature catalog: %s",
        catalog["type_bucket"].value_counts().to_dict(),
    )
    return catalog


# --------------------------------------------------------------------------- #
# 3. Data quality
# --------------------------------------------------------------------------- #
def data_quality_findings(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Enumerate concrete data-quality defects, each with its remedy.

    Every finding here is acted on in :mod:`src.data.preprocessor`; this
    function is what justifies that code existing.

    Args:
        frame: The raw joined dataset.

    Returns:
        A list of findings, each with a severity, the affected volume and the
        treatment applied downstream.
    """
    findings: list[dict[str, object]] = []
    n_rows = len(frame)

    # --- the headline anomaly ---
    if "DAYS_EMPLOYED" in frame.columns:
        anomalous = int((frame["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).sum())
        if anomalous:
            findings.append(
                {
                    "issue": "DAYS_EMPLOYED sentinel value 365243",
                    "severity": "high",
                    "n_affected": anomalous,
                    "pct_affected": round(100 * anomalous / n_rows, 2),
                    "detail": (
                        "365243 days is ~1000 years of employment. It is a sentinel "
                        "meaning 'no employment record', which attaches almost "
                        "entirely to pensioners."
                    ),
                    "treatment": "Nulled and replaced with an explicit DAYS_EMPLOYED_ANOMALY flag.",
                }
            )

    # --- undefined categorical level ---
    if "CODE_GENDER" in frame.columns:
        xna = int((frame["CODE_GENDER"] == "XNA").sum())
        if xna:
            findings.append(
                {
                    "issue": "CODE_GENDER = 'XNA'",
                    "severity": "low",
                    "n_affected": xna,
                    "pct_affected": round(100 * xna / n_rows, 2),
                    "detail": "An undefined placeholder level, not a real category.",
                    "treatment": "Mapped to null rather than treated as a third gender.",
                }
            )

    # --- heavy missingness ---
    missing = missing_value_report(frame)
    severe = missing[missing["pct_missing"] > 50]
    if not severe.empty:
        findings.append(
            {
                "issue": "Columns more than 50% missing",
                "severity": "medium",
                "n_affected": int(len(severe)),
                "pct_affected": round(100 * len(severe) / frame.shape[1], 2),
                "detail": (
                    f"{len(severe)} columns exceed 50% missing, led by "
                    f"{severe.iloc[0]['column']} at {severe.iloc[0]['pct_missing']}%. "
                    "Most are building-block property attributes."
                ),
                "treatment": (
                    "Retained un-imputed for the tree models, which branch on "
                    "missingness directly; the linear baseline imputes inside its "
                    "own CV fold."
                ),
            }
        )

    # --- implausible values ---
    if "AMT_INCOME_TOTAL" in frame.columns:
        income = frame["AMT_INCOME_TOTAL"].dropna()
        # Tukey's rule on a heavily right-skewed variable flags the tail that
        # would otherwise dominate a linear model's scaling.
        q1, q3 = income.quantile([0.25, 0.75])
        upper = q3 + 3.0 * (q3 - q1)
        extreme = int((income > upper).sum())
        if extreme:
            findings.append(
                {
                    "issue": "Extreme income outliers",
                    "severity": "medium",
                    "n_affected": extreme,
                    "pct_affected": round(100 * extreme / n_rows, 2),
                    "detail": (
                        f"{extreme} applicants report income above "
                        f"{upper:,.0f} (Q3 + 3*IQR). Income is strongly "
                        "right-skewed, so the tail is long but genuine."
                    ),
                    "treatment": (
                        "Left in place -- tree models are rank-based and immune. "
                        "Ratio features (credit-to-income) also neutralise scale."
                    ),
                }
            )

    if "CNT_CHILDREN" in frame.columns and "CNT_FAM_MEMBERS" in frame.columns:
        inconsistent = int((frame["CNT_CHILDREN"] > frame["CNT_FAM_MEMBERS"]).sum())
        if inconsistent:
            findings.append(
                {
                    "issue": "More children than family members",
                    "severity": "low",
                    "n_affected": inconsistent,
                    "pct_affected": round(100 * inconsistent / n_rows, 2),
                    "detail": "Internally inconsistent household counts.",
                    "treatment": "Flagged; the derived CHILDREN_RATIO absorbs the effect.",
                }
            )

    # --- thin credit files ---
    history_column = f"{BUREAU_PREFIX}HAS_HISTORY"
    if history_column in frame.columns:
        thin = int((frame[history_column] == 0).sum())
        if thin:
            findings.append(
                {
                    "issue": "Applicants with no external credit history",
                    "severity": "medium",
                    "n_affected": thin,
                    "pct_affected": round(100 * thin / n_rows, 2),
                    "detail": (
                        "These applicants have no bureau record at all, so every "
                        "BUREAU_* metric is null. This is a thin file, not a "
                        "data defect."
                    ),
                    "treatment": (
                        "Nulls preserved and marked with BUREAU_HAS_HISTORY so the "
                        "model can price the absence of history."
                    ),
                }
            )

    logger.info("Data quality: %d findings", len(findings))
    return findings


# --------------------------------------------------------------------------- #
# Shared analysis helper
# --------------------------------------------------------------------------- #
def default_rate_by_segment(
    frame: pd.DataFrame, segment: pd.Series, min_count: int = 20
) -> pd.DataFrame:
    """Compute the default rate within each level of ``segment``.

    Args:
        frame: Dataset containing ``TARGET``.
        segment: A categorical or binned series aligned to ``frame``.
        min_count: Levels with fewer rows than this are dropped -- a default
            rate computed on a handful of applicants is noise, and plotting it
            beside a real one invites a false conclusion.

    Returns:
        Columns ``segment``, ``n``, ``n_default``, ``default_rate`` (percent).
    """
    grouped = frame.groupby(segment, observed=True)[TARGET].agg(["count", "sum"])
    grouped.columns = ["n", "n_default"]
    grouped["default_rate"] = 100 * grouped["n_default"] / grouped["n"]
    filtered = grouped[grouped["n"] >= min_count].reset_index()
    filtered.columns = ["segment", "n", "n_default", "default_rate"]
    return filtered


def _quantile_bands(values: pd.Series, n: int = N_BANDS, label: str = "") -> pd.Series:
    """Bin a numeric column into ``n`` quantile bands with readable edge labels."""
    try:
        bands, edges = pd.qcut(values, n, retbins=True, duplicates="drop")
    except ValueError:  # pragma: no cover - degenerate distribution
        return pd.Series(index=values.index, dtype="object")
    names = [f"{edges[i]:,.2f}-{edges[i + 1]:,.2f}" for i in range(len(edges) - 1)]
    return pd.Series(
        pd.Categorical(bands.cat.rename_categories(names), categories=names, ordered=True),
        index=values.index,
        name=label or values.name,
    )
