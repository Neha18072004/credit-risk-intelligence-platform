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
from src.utils.helpers import (
    DOMAIN_DESCRIPTIONS,
    categorize_features,
    domain_of,
    missing_value_report,
    write_json,
)
from src.utils.logger import get_logger
from src.utils.viz import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    add_reference_line,
    apply_theme,
    label_bars,
    new_figure,
    ordinal_ramp,
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
            "domain_covers": DOMAIN_DESCRIPTIONS.get(domain_of(column), ""),
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
def wilson_interval(successes: np.ndarray, totals: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion, as percentages.

    Preferred over the normal approximation because default rates are small and
    some segments are thin -- exactly the regime where the naive interval runs
    below zero and overstates precision.

    Args:
        successes: Count of defaults per segment.
        totals: Count of applicants per segment.
        z: Normal quantile; 1.96 gives a 95% interval.

    Returns:
        ``(lower, upper)`` bounds in percent.
    """
    n = np.asarray(totals, dtype=float)
    phat = np.asarray(successes, dtype=float) / np.maximum(n, 1)
    denominator = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denominator
    half = (z / denominator) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return 100 * np.clip(centre - half, 0, 1), 100 * np.clip(centre + half, 0, 1)


def default_rate_by_segment(
    frame: pd.DataFrame, segment: pd.Series, min_count: int = 50
) -> pd.DataFrame:
    """Compute the default rate within each level of ``segment``.

    Every rate carries a 95% Wilson confidence interval. On a portfolio this
    size a segment of forty applicants can show a wildly different rate through
    sampling noise alone, and a bar chart without intervals invites exactly that
    misreading -- so the uncertainty is computed here and drawn on the chart.

    Args:
        frame: Dataset containing ``TARGET``.
        segment: A categorical or binned series aligned to ``frame``.
        min_count: Levels with fewer rows than this are dropped outright.

    Returns:
        Columns ``segment``, ``n``, ``n_default``, ``default_rate``,
        ``ci_low``, ``ci_high`` -- rates and bounds in percent.
    """
    grouped = frame.groupby(segment, observed=True)[TARGET].agg(["count", "sum"])
    grouped.columns = ["n", "n_default"]
    grouped["default_rate"] = 100 * grouped["n_default"] / grouped["n"]
    filtered = grouped[grouped["n"] >= min_count].reset_index()
    filtered.columns = ["segment", "n", "n_default", "default_rate"]

    low, high = wilson_interval(filtered["n_default"].to_numpy(), filtered["n"].to_numpy())
    filtered["ci_low"] = low.round(2)
    filtered["ci_high"] = high.round(2)
    return filtered


def _error_bars(table: pd.DataFrame) -> np.ndarray:
    """Asymmetric error-bar offsets from the Wilson bounds, for matplotlib."""
    return np.vstack(
        [
            (table["default_rate"] - table["ci_low"]).clip(lower=0).to_numpy(),
            (table["ci_high"] - table["default_rate"]).clip(lower=0).to_numpy(),
        ]
    )


def _separated(table: pd.DataFrame) -> bool:
    """True when the extreme segments' confidence intervals do not overlap.

    Used to decide whether a takeaway may state a difference as a finding or
    must hedge it as directional.
    """
    if len(table) < 2:
        return False
    ordered = table.sort_values("default_rate")
    return bool(ordered["ci_low"].iloc[-1] > ordered["ci_high"].iloc[0])


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


# --------------------------------------------------------------------------- #
# 4. Business insights
# --------------------------------------------------------------------------- #
def insight_class_imbalance(frame: pd.DataFrame, save: bool = True) -> Insight:
    """How rare defaults are, and what that implies for modelling."""
    counts = frame[TARGET].value_counts().sort_index()
    n_repaid, n_default = int(counts.get(0, 0)), int(counts.get(1, 0))
    rate = 100 * n_default / max(len(frame), 1)
    ratio = n_repaid / max(n_default, 1)

    table = pd.DataFrame(
        {
            "outcome": ["Repaid", "Defaulted"],
            "n": [n_repaid, n_default],
            "share_pct": [round(100 - rate, 2), round(rate, 2)],
        }
    )

    # The story here is a single number, so the figure leads with it and the
    # bar plays a supporting part-to-whole role rather than carrying the message.
    fig, ax = new_figure(figsize=(9.0, 3.2))
    ax.set_axis_off()

    ax.text(
        0, 0.86, f"{rate:.1f}%",
        transform=ax.transAxes, fontsize=44, fontweight="600",
        color=INK_PRIMARY, va="top", ha="left",
    )
    ax.text(
        0, 0.40,
        f"of {len(frame):,} applicants defaulted  --  {n_default:,} defaults against "
        f"{n_repaid:,} repayments, a {ratio:.0f}:1 imbalance",
        transform=ax.transAxes, fontsize=11, color=INK_SECONDARY, va="top", ha="left",
    )

    # A single 100% bar in plain data coordinates. The 0.25-unit inset between
    # the segments is the surface gap that separates them, so neither needs a
    # border drawn around it.
    bar_y, bar_height, gap = 0.22, 0.13, 0.25
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.barh([bar_y], [100 - rate], height=bar_height, color="#c9d7e8")
    ax.barh([bar_y], [rate - gap], left=[100 - rate + gap], height=bar_height, color=SERIES[1])

    ax.text(
        (100 - rate) / 2, bar_y, f"Repaid  {n_repaid:,}",
        ha="center", va="center", fontsize=9, color=INK_SECONDARY,
    )
    # Placed outside its segment: the default slice is far too narrow to hold text.
    ax.text(
        100, bar_y - bar_height, f"Defaulted  {n_default:,}",
        ha="right", va="top", fontsize=9, color=SERIES[1],
    )

    takeaway = (
        f"Only {rate:.1f}% of applicants default, a {ratio:.0f}:1 imbalance. Accuracy is "
        f"meaningless here -- a model predicting 'everyone repays' scores {100 - rate:.1f}%. "
        "This is why the project leads with PR-AUC, weights the positive class rather than "
        "oversampling it, and tunes decision thresholds instead of using 0.5."
    )
    path = save_figure(fig, "01_class_imbalance") if save else None
    return Insight("class_imbalance", "Defaults are rare, so the metric must be chosen with care",
                   takeaway, table, path)


def insight_missing_data(frame: pd.DataFrame, save: bool = True) -> Insight:
    """Which columns are unusable as-is, and what the missingness means."""
    report = missing_value_report(frame)
    top = report[report["pct_missing"] > 0].head(15).iloc[::-1]

    fig, ax = new_figure(figsize=(9.0, 6.0))
    style_axes(ax, xgrid=True, ygrid=False)
    bars = ax.barh(top["column"], top["pct_missing"], color=SERIES[0], height=0.62)
    label_bars(ax, bars, top["pct_missing"].tolist(), fmt="{:.0f}%", horizontal=True, pad=0.015)
    ax.set_xlim(0, min(105, top["pct_missing"].max() * 1.18))
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missingness is concentrated in property and external-score columns")

    n_over_half = int((report["pct_missing"] > 50).sum())
    worst = report.iloc[0]
    takeaway = (
        f"{n_over_half} columns are more than half empty, led by {worst['column']} at "
        f"{worst['pct_missing']:.0f}%. Most are optional property attributes. Critically, "
        "EXT_SOURCE_1 -- one of the strongest predictors -- is missing for the majority of "
        "applicants, which is exactly why the model uses the mean/min/count of the three "
        "external scores rather than any single one. Missingness is kept, not imputed away: "
        "for tree models an absent value is a usable branch, and a thin file is real risk "
        "information."
    )
    path = save_figure(fig, "02_missing_data") if save else None
    return Insight("missing_data", "Missing data is structural, not random", takeaway, report, path)


def insight_feature_landscape(frame: pd.DataFrame, save: bool = True) -> Insight:
    """What kinds of features exist, and which business domains they cover."""
    catalog = feature_catalog(frame)
    by_type = catalog["type_bucket"].value_counts()
    by_domain = catalog["domain"].value_counts()

    import matplotlib.pyplot as plt

    fig, (ax_type, ax_domain) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax in (ax_type, ax_domain):
        style_axes(ax, xgrid=True, ygrid=False)

    type_sorted = by_type.iloc[::-1]
    bars = ax_type.barh(type_sorted.index, type_sorted.to_numpy(), color=SERIES[0], height=0.6)
    label_bars(ax_type, bars, type_sorted.tolist(), fmt="{:.0f}", horizontal=True, pad=0.02)
    ax_type.set_title("By storage type")
    ax_type.set_xlabel("Columns")
    ax_type.set_xlim(0, type_sorted.max() * 1.16)

    domain_sorted = by_domain.iloc[::-1]
    bars = ax_domain.barh(domain_sorted.index, domain_sorted.to_numpy(), color=SERIES[0], height=0.6)
    label_bars(ax_domain, bars, domain_sorted.tolist(), fmt="{:.0f}", horizontal=True, pad=0.02)
    ax_domain.set_title("By business domain")
    ax_domain.set_xlabel("Columns")
    ax_domain.set_xlim(0, domain_sorted.max() * 1.16)

    fig.suptitle(
        f"{len(catalog)} columns across {by_domain.size} business domains",
        x=0.02, ha="left", fontsize=12, fontweight="600",
    )
    fig.tight_layout()

    # Record what each domain covers, so "feature categorization" is a stated
    # grouping rather than a bare column count.
    domain_table = (
        catalog.groupby("domain")
        .agg(columns=("column", "count"), covers=("domain_covers", "first"))
        .sort_values("columns", ascending=False)
        .reset_index()
    )
    logger.info("Domains: %s", dict(zip(domain_table["domain"], domain_table["columns"])))

    takeaway = (
        f"The {len(catalog)} columns split into {int(by_type.get('numeric', 0))} numeric, "
        f"{int(by_type.get('categorical', 0))} categorical, {int(by_type.get('binary', 0))} binary "
        f"and {int(by_type.get('date_like', 0))} date-like fields. By domain, the largest block is "
        f"{by_domain.index[0]} ({by_domain.iloc[0]} columns). Property attributes and document "
        "flags dominate by count but carry little signal, while the small external-score and "
        "financial blocks carry most of it -- so feature count is a poor guide to feature value."
    )
    path = save_figure(fig, "03_feature_landscape") if save else None
    return Insight("feature_landscape", "The portfolio's feature landscape", takeaway, catalog, path)


def insight_external_scores(frame: pd.DataFrame, save: bool = True) -> Insight:
    """Default rate across bands of the averaged external credit score."""
    working = engineer_features(clean_applications(frame))
    bands = _quantile_bands(working["EXT_SOURCE_MEAN"], N_BANDS, "ext_source_band")
    table = default_rate_by_segment(working.assign(_band=bands), working.assign(_band=bands)["_band"])

    overall = 100 * frame[TARGET].mean()
    # A volume panel is omitted deliberately: quintiles are equal-sized by
    # construction, so it would be five identical bars carrying no information.
    fig, ax_rate = new_figure(figsize=(9.0, 5.2))
    colors = ordinal_ramp(len(table))

    bars = ax_rate.bar(
        table["segment"].astype(str), table["default_rate"], color=colors, width=0.68,
        yerr=_error_bars(table), ecolor=INK_MUTED, capsize=3, error_kw={"elinewidth": 0.9},
    )
    label_bars(ax_rate, bars, table["default_rate"].tolist(), fmt="{:.1f}%", pad=0.03,
               tops=table["ci_high"].tolist())
    ax_rate.set_ylabel("Default rate (%)")
    ax_rate.set_ylim(0, table["default_rate"].max() * 1.28)
    ax_rate.set_title("Default rate falls steeply as the external credit score rises")
    add_reference_line(ax_rate, overall, f"portfolio {overall:.1f}%")

    ax_rate.set_xlabel(
        f"Mean of EXT_SOURCE_1/2/3 (quintile ranges, n={int(table['n'].iloc[0]):,} each)"
    )
    ax_rate.tick_params(axis="x", rotation=20)

    lowest, highest = table["default_rate"].iloc[0], table["default_rate"].iloc[-1]
    lift = lowest / max(highest, 1e-9)
    is_monotonic = bool((table["default_rate"].diff().dropna() <= 0).all())
    shape = (
        "and the decline is monotonic across every band, which is what makes it trustworthy "
        "for policy"
        if is_monotonic
        else "and the trend is consistently downward, with minor non-monotonicity between "
             "adjacent bands"
    )
    takeaway = (
        f"Applicants in the lowest external-score band default at {lowest:.1f}%, versus "
        f"{highest:.1f}% in the highest -- a {lift:.1f}x spread across an evenly-sized split of "
        f"the book, {shape}. This is the strongest single signal available. Practical use: the "
        "bottom band warrants manual review or a pricing uplift, and an applicant missing all "
        "three external scores should be routed to manual assessment rather than scored on the "
        "remaining fields."
    )
    path = save_figure(fig, "04_external_scores") if save else None
    return Insight("external_scores", "External credit scores dominate default risk",
                   takeaway, table, path)


def insight_affordability(frame: pd.DataFrame, save: bool = True) -> Insight:
    """Default rate against loan-size-to-income and instalment-to-income."""
    working = engineer_features(clean_applications(frame))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    tables: list[pd.DataFrame] = []
    specs = [
        ("CREDIT_TO_INCOME_RATIO", "Loan size / annual income", "credit_to_income"),
        ("ANNUITY_TO_INCOME_RATIO", "Annual instalment / annual income", "annuity_to_income"),
    ]
    overall = 100 * frame[TARGET].mean()

    for ax, (column, label, key) in zip(axes, specs, strict=True):
        style_axes(ax)
        bands = _quantile_bands(working[column], N_BANDS, key)
        table = default_rate_by_segment(working.assign(_b=bands), working.assign(_b=bands)["_b"])
        table.insert(0, "metric", key)
        tables.append(table)

        bars = ax.bar(
            table["segment"].astype(str), table["default_rate"],
            color=ordinal_ramp(len(table)), width=0.68,
            yerr=_error_bars(table), ecolor=INK_MUTED, capsize=3, error_kw={"elinewidth": 0.9},
        )
        label_bars(ax, bars, table["default_rate"].tolist(), fmt="{:.1f}%", pad=0.03,
                   tops=table["ci_high"].tolist())
        ax.set_title(label)
        ax.set_xlabel(f"{len(table)} equal-sized bands")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        add_reference_line(ax, overall, f"{overall:.1f}%")

    axes[0].set_ylabel("Default rate (%)")
    fig.suptitle(
        "Leverage matters more than loan size in isolation",
        x=0.02, ha="left", fontsize=12, fontweight="600",
    )
    fig.tight_layout()

    combined = pd.concat(tables, ignore_index=True)
    credit_table = tables[0]
    spread = credit_table["default_rate"].iloc[-1] - credit_table["default_rate"].iloc[0]
    takeaway = (
        f"Default rate moves {spread:+.1f} percentage points from the least to the most "
        "leveraged quintile of loan-to-income. The absolute loan amount is a poor risk "
        "indicator on its own -- what matters is the amount relative to what the applicant "
        "earns. Practical use: cap loan-to-income at the point where the rate crosses the "
        "portfolio average rather than applying one flat maximum loan size, and treat a high "
        "instalment-to-income ratio as the affordability constraint at origination."
    )
    path = save_figure(fig, "05_affordability") if save else None
    return Insight("affordability", "Affordability ratios separate risk better than raw amounts",
                   takeaway, combined, path)


def insight_employment_anomaly(frame: pd.DataFrame, save: bool = True) -> Insight:
    """What the DAYS_EMPLOYED sentinel actually encodes, and how it behaves."""
    working = clean_applications(frame)
    flagged = working["DAYS_EMPLOYED_ANOMALY"] == 1

    anomaly_table = default_rate_by_segment(
        working,
        pd.Series(
            np.where(flagged, "No employment record (365243)", "Employment recorded"),
            index=working.index, name="employment_record",
        ),
    )

    # Among applicants who do have a record, does tenure itself matter?
    employed = working.loc[~flagged].copy()
    tenure_years = -employed["DAYS_EMPLOYED"] / 365.25
    tenure_bands = _quantile_bands(tenure_years, N_BANDS, "tenure_years")
    tenure_table = default_rate_by_segment(
        employed.assign(_b=tenure_bands), employed.assign(_b=tenure_bands)["_b"]
    )

    import matplotlib.pyplot as plt

    fig, (ax_flag, ax_tenure) = plt.subplots(
        1, 2, figsize=(11.0, 4.8), sharey=True, width_ratios=[1.0, 1.6]
    )
    for ax in (ax_flag, ax_tenure):
        style_axes(ax)
    overall = 100 * frame[TARGET].mean()

    # Two nominal groups -> the two fixed categorical slots, never a value ramp.
    bars = ax_flag.bar(
        anomaly_table["segment"].astype(str), anomaly_table["default_rate"],
        # Orange, the attention slot, marks the anomalous group -- the subject
        # of the chart -- while the normal population stays in the base blue.
        color=[SERIES[0], SERIES[1]], width=0.55,
        yerr=_error_bars(anomaly_table), ecolor=INK_MUTED, capsize=3, error_kw={"elinewidth": 0.9},
    )
    label_bars(ax_flag, bars, anomaly_table["default_rate"].tolist(), fmt="{:.1f}%", pad=0.03,
               tops=anomaly_table["ci_high"].tolist())
    ax_flag.set_title("By employment record")
    ax_flag.set_ylabel("Default rate (%)")
    ax_flag.tick_params(axis="x", labelsize=8)
    ax_flag.set_xticks(range(len(anomaly_table)))
    ax_flag.set_xticklabels(
        [label.replace(" (365243)", "\n(365243)") for label in anomaly_table["segment"].astype(str)]
    )

    bars = ax_tenure.bar(
        tenure_table["segment"].astype(str), tenure_table["default_rate"],
        color=ordinal_ramp(len(tenure_table)), width=0.68,
        yerr=_error_bars(tenure_table), ecolor=INK_MUTED, capsize=3, error_kw={"elinewidth": 0.9},
    )
    label_bars(ax_tenure, bars, tenure_table["default_rate"].tolist(), fmt="{:.1f}%", pad=0.03,
               tops=tenure_table["ci_high"].tolist())
    ax_tenure.set_title("By years employed (applicants with a record)")
    ax_tenure.set_xlabel("Years in current employment (quintile ranges)")
    ax_tenure.tick_params(axis="x", rotation=20, labelsize=8)
    add_reference_line(ax_tenure, overall, f"portfolio {overall:.1f}%")

    fig.suptitle(
        "The 365243 sentinel is a population, not a data error",
        x=0.02, ha="left", fontsize=12, fontweight="600",
    )
    fig.tight_layout()

    flagged_rate = float(
        anomaly_table.loc[
            anomaly_table["segment"].str.contains("No employment"), "default_rate"
        ].iloc[0]
    )
    recorded_rate = float(
        anomaly_table.loc[
            anomaly_table["segment"] == "Employment recorded", "default_rate"
        ].iloc[0]
    )
    share = 100 * flagged.mean()
    tenure_matters = _separated(tenure_table)
    tenure_note = (
        "Among applicants who do have a record, tenure length itself separates risk further"
        if tenure_matters
        else "Among applicants who do have a record, tenure length barely moves the default "
             "rate at all -- every band overlaps the portfolio average. The signal is in "
             "*whether* employment is recorded, not how long it has lasted"
    )
    takeaway = (
        f"{share:.0f}% of applicants carry DAYS_EMPLOYED = 365243, which is not 1000 years of "
        "work but a sentinel for 'no employment record' -- and it maps almost perfectly onto "
        f"pensioners. That group defaults at {flagged_rate:.1f}%, against {recorded_rate:.1f}% "
        f"for applicants with a recorded job. {tenure_note}. Left untreated the sentinel would "
        "dominate every tree split and destroy any coefficient on tenure, so it is nulled and "
        "replaced with an explicit flag. Practical use: pensioners need their own scorecard "
        "treatment -- their risk is driven by pension stability, not job tenure."
    )
    combined = pd.concat(
        [anomaly_table.assign(view="employment_record"), tenure_table.assign(view="tenure_years")],
        ignore_index=True,
    )
    path = save_figure(fig, "06_employment_anomaly") if save else None
    return Insight("employment_anomaly", "The employment anomaly encodes a real population",
                   takeaway, combined, path)


def insight_demographic_segments(frame: pd.DataFrame, save: bool = True) -> Insight:
    """Default rate across the customer attributes a credit policy can act on."""
    working = engineer_features(clean_applications(frame))
    overall = 100 * frame[TARGET].mean()

    specs = [
        ("NAME_EDUCATION_TYPE", "Education"),
        ("NAME_FAMILY_STATUS", "Family status"),
        ("NAME_CONTRACT_TYPE", "Contract type"),
    ]

    # One chart rather than three side-by-side panels. Separate panels would
    # autoscale to their own category counts, so a two-level attribute would
    # render as two enormous blocks beside a five-level one; laying every level
    # on a shared axis keeps bar thickness uniform and makes the attributes
    # directly comparable.
    tables: list[pd.DataFrame] = []
    positions: list[float] = []
    labels: list[str] = []
    group_marks: list[tuple[float, str]] = []
    cursor = 0.0

    for column, label in specs:
        table = default_rate_by_segment(working, working[column]).sort_values(
            "default_rate", ascending=False
        )
        table.insert(0, "attribute", column)
        tables.append(table)

        group_marks.append((cursor - 0.85, label))
        for _, row in table.iterrows():
            positions.append(cursor)
            labels.append(f"{str(row['segment'])[:30]}  (n={int(row['n']):,})")
            cursor += 1.0
        cursor += 0.9  # gap between attribute groups

    combined = pd.concat(tables, ignore_index=True)

    fig, ax = new_figure(figsize=(9.5, 7.2))
    style_axes(ax, xgrid=True, ygrid=False)

    bars = ax.barh(
        positions, combined["default_rate"], color=SERIES[0], height=0.62,
        xerr=_error_bars(combined), ecolor=INK_MUTED, capsize=3,
        error_kw={"elinewidth": 0.9},
    )
    label_bars(
        ax, bars, combined["default_rate"].tolist(), fmt="{:.1f}%",
        horizontal=True, pad=0.025, tops=combined["ci_high"].tolist(),
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, combined["ci_high"].max() * 1.16)
    ax.set_xlabel("Default rate (%)")
    ax.set_ylim(-1.3, cursor - 0.4)
    # Position 0 at the top, so groups read in the order they were built and the
    # highest-risk level in each group sits first.
    ax.invert_yaxis()
    add_reference_line(ax, overall, f"portfolio {overall:.1f}%", horizontal=False)

    # Name each attribute group beside its block of levels.
    for y, label in group_marks:
        ax.text(
            0, y, label,
            ha="left", va="center", fontsize=10, fontweight="600", color=INK_SECONDARY,
        )

    ax.set_title("Default rate by customer attribute, with 95% confidence intervals")
    fig.tight_layout()

    education = tables[0]  # sorted worst-first
    worst, best = education.iloc[0], education.iloc[-1]
    # Only call the spread a finding when the two extremes' intervals separate;
    # otherwise report it as directional. Several of these levels are small.
    confidence = (
        "The gap is wider than the sampling error on either end"
        if _separated(education)
        else "Their confidence intervals overlap, so treat this as directional only"
    )
    takeaway = (
        f"Education spreads default risk from {best['default_rate']:.1f}% "
        f"({best['segment']}, n={best['n']:,}) to {worst['default_rate']:.1f}% "
        f"({worst['segment']}, n={worst['n']:,}). {confidence}. Family status and contract type "
        "separate less sharply. Practical use: these are stable, verifiable attributes suited to "
        "policy segmentation -- but each is a proxy for income stability rather than a cause of "
        "default, so they belong in pricing tiers and review triggers, not hard decline rules, "
        "and must be checked against fair-lending constraints before any use in a decision."
    )
    path = save_figure(fig, "07_demographic_segments") if save else None
    return Insight("demographic_segments", "Customer attributes give stable policy segments",
                   takeaway, combined, path)


def insight_credit_history(frame: pd.DataFrame, save: bool = True) -> Insight:
    """What external bureau history says about repayment, including thin files."""
    working = frame.copy()
    overall = 100 * frame[TARGET].mean()

    has_history = working.get(f"{BUREAU_PREFIX}HAS_HISTORY", pd.Series(0, index=working.index))
    has_overdue = working.get(f"{BUREAU_PREFIX}HAS_OVERDUE", pd.Series(0, index=working.index)).fillna(0)

    profile = np.select(
        [has_history == 0, (has_history == 1) & (has_overdue == 1)],
        ["No bureau history (thin file)", "History with arrears"],
        default="History, no arrears",
    )
    profile_table = default_rate_by_segment(
        working, pd.Series(profile, index=working.index, name="credit_profile")
    ).sort_values("default_rate")

    import matplotlib.pyplot as plt

    fig, (ax_profile, ax_debt) = plt.subplots(
        1, 2, figsize=(11.5, 4.8), sharey=True, width_ratios=[1.2, 1.4]
    )
    for ax in (ax_profile, ax_debt):
        style_axes(ax)

    bars = ax_profile.bar(
        range(len(profile_table)), profile_table["default_rate"], color=SERIES[0], width=0.6,
        yerr=_error_bars(profile_table), ecolor=INK_MUTED, capsize=3, error_kw={"elinewidth": 0.9},
    )
    label_bars(ax_profile, bars, profile_table["default_rate"].tolist(), fmt="{:.1f}%", pad=0.03,
               tops=profile_table["ci_high"].tolist())
    ax_profile.set_xticks(range(len(profile_table)))
    ax_profile.set_xticklabels(
        [f"{s}\n(n={n:,})" for s, n in zip(profile_table["segment"], profile_table["n"], strict=True)],
        fontsize=8,
    )
    ax_profile.set_ylabel("Default rate (%)")
    ax_profile.set_title("By external credit profile")
    add_reference_line(ax_profile, overall, f"portfolio {overall:.1f}%")

    # Leverage on existing external credit, among applicants who have some.
    with_history = working[has_history == 1]
    debt_ratio = with_history[f"{BUREAU_PREFIX}DEBT_CREDIT_RATIO"]
    debt_table = pd.DataFrame(columns=["segment", "n", "n_default", "default_rate"])
    if debt_ratio.notna().sum() > 50:
        bands = _quantile_bands(debt_ratio.dropna(), N_BANDS, "debt_ratio")
        subset = with_history.loc[bands.index]
        debt_table = default_rate_by_segment(subset.assign(_b=bands), subset.assign(_b=bands)["_b"])
        bars = ax_debt.bar(
            debt_table["segment"].astype(str), debt_table["default_rate"],
            color=ordinal_ramp(len(debt_table)), width=0.68,
            yerr=_error_bars(debt_table), ecolor=INK_MUTED, capsize=3, error_kw={"elinewidth": 0.9},
        )
        label_bars(ax_debt, bars, debt_table["default_rate"].tolist(), fmt="{:.1f}%", pad=0.03,
                   tops=debt_table["ci_high"].tolist())
        add_reference_line(ax_debt, overall, f"{overall:.1f}%")
    ax_debt.set_title("By share of external credit still outstanding")
    # Band count is read back from the table: qcut collapses duplicate edges, so
    # a "quintiles" label can end up describing four bars.
    ax_debt.set_xlabel(
        f"Outstanding debt / total external credit ({len(debt_table)} equal-sized bands)"
    )
    ax_debt.tick_params(axis="x", rotation=20, labelsize=8)

    fig.suptitle(
        "External credit history separates risk, and a thin file is its own signal",
        x=0.02, ha="left", fontsize=12, fontweight="600",
    )
    fig.tight_layout()

    thin = profile_table[profile_table["segment"].str.contains("thin file")]
    thin_rate = float(thin["default_rate"].iloc[0]) if not thin.empty else float("nan")
    arrears = profile_table[profile_table["segment"] == "History with arrears"]
    arrears_rate = float(arrears["default_rate"].iloc[0]) if not arrears.empty else float("nan")
    clean_rows = profile_table[profile_table["segment"] == "History, no arrears"]
    clean_rate = float(clean_rows["default_rate"].iloc[0]) if not clean_rows.empty else float("nan")

    multiple = arrears_rate / clean_rate if clean_rate else float("nan")
    arrears_rows = profile_table[profile_table["segment"] == "History with arrears"]
    clean_rows_ = profile_table[profile_table["segment"] == "History, no arrears"]
    separated = (
        not arrears_rows.empty
        and not clean_rows_.empty
        and float(arrears_rows["ci_low"].iloc[0]) > float(clean_rows_["ci_high"].iloc[0])
    )
    strength = (
        f"{multiple:.1f}x the rate of a clean external record, with non-overlapping "
        "confidence intervals"
        if separated
        else "directionally higher than a clean external record, though the intervals overlap"
    )
    takeaway = (
        f"Applicants with prior arrears default at {arrears_rate:.1f}% -- {strength} "
        f"({clean_rate:.1f}%). Outstanding leverage tells the same story: the default rate climbs "
        "steadily with the share of external credit still unpaid. Thin-file applicants, with no "
        f"bureau record at all, sit at {thin_rate:.1f}%, which is why their nulls are preserved "
        "and flagged rather than imputed -- 'no history' is a distinct risk state, not a missing "
        "value to fill in. Practical use: prior arrears is the most defensible decline or "
        "referral trigger in the feature set, because it is behavioural rather than demographic "
        "and is externally verifiable."
    )
    combined = pd.concat(
        [profile_table.assign(view="profile"), debt_table.assign(view="debt_ratio")],
        ignore_index=True,
    )
    path = save_figure(fig, "08_credit_history") if save else None
    return Insight("credit_history", "External credit history is the most actionable signal",
                   takeaway, combined, path)


def insight_repayment_behaviour(frame: pd.DataFrame, save: bool = True) -> Insight:
    """How the applicant repaid *our* prior loans, and how we judged them before.

    This is the repayment-behaviour block the brief names as an analysis area,
    and it is qualitatively different from everything else in the dataset. Every
    other strong feature is a proxy -- education stands in for income stability,
    an external score summarises someone else's judgement. This is the
    applicant's own conduct on their own obligations, recorded by us.
    """
    working = frame.copy()
    overall = 100 * frame[TARGET].mean()

    panels: list[tuple[str, pd.DataFrame, str]] = []

    if "INST_EVER_LATE" in working.columns:
        history = np.where(
            working["INST_HAS_HISTORY"].fillna(0) == 0, "No prior loan with us",
            np.where(working["INST_EVER_LATE"].fillna(0) == 1,
                     "Paid late before", "Always paid on time"),
        )
        panels.append(
            (
                "By repayment history",
                default_rate_by_segment(
                    working, pd.Series(history, index=working.index, name="repayment")
                ).sort_values("default_rate"),
                "",
            )
        )

    if "INST_LATE_RATE" in working.columns:
        late = working["INST_LATE_RATE"].dropna()
        if late.nunique() > 5:
            bands = _quantile_bands(late, N_BANDS, "late_rate")
            subset = working.loc[bands.index]
            panels.append(
                (
                    "By share of instalments paid late",
                    default_rate_by_segment(
                        subset.assign(_b=bands), subset.assign(_b=bands)["_b"]
                    ),
                    "Share of instalments paid late",
                )
            )

    if "PREV_EVER_REFUSED" in working.columns:
        refused = np.where(
            working["PREV_HAS_HISTORY"].fillna(0) == 0, "Never applied to us",
            np.where(working["PREV_EVER_REFUSED"].fillna(0) == 1,
                     "Declined by us before", "Never declined"),
        )
        panels.append(
            (
                "By our own prior decisions",
                default_rate_by_segment(
                    working, pd.Series(refused, index=working.index, name="prior_decision")
                ).sort_values("default_rate"),
                "",
            )
        )

    if not panels:
        empty = pd.DataFrame(columns=["segment", "n", "n_default", "default_rate"])
        return Insight(
            "repayment_behaviour", "Repayment behaviour",
            "Prior-application and instalment tables are not loaded, so repayment "
            "behaviour cannot be analysed. Enable them with INCLUDE_PREVIOUS_APPLICATION "
            "and INCLUDE_INSTALLMENTS.",
            empty, None,
        )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, table, xlabel) in zip(axes, panels, strict=True):
        style_axes(ax)
        ordered = table.sort_values("default_rate") if not xlabel else table
        colors = (
            ordinal_ramp(len(ordered)) if xlabel
            else [SERIES[0]] * len(ordered)
        )
        bars = ax.bar(
            range(len(ordered)), ordered["default_rate"], color=colors, width=0.6,
            yerr=_error_bars(ordered), ecolor=INK_MUTED, capsize=3,
            error_kw={"elinewidth": 0.9},
        )
        label_bars(ax, bars, ordered["default_rate"].tolist(), fmt="{:.1f}%", pad=0.03,
                   tops=ordered["ci_high"].tolist())
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(
            [f"{str(s)[:22]}\n(n={n:,})"
             for s, n in zip(ordered["segment"], ordered["n"], strict=True)],
            fontsize=8, rotation=15 if xlabel else 0,
        )
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        add_reference_line(ax, overall, f"{overall:.1f}%")
    axes[0].set_ylabel("Default rate (%)")

    fig.suptitle(
        "How someone repaid us before is the most direct evidence we have",
        x=0.02, ha="left", fontsize=12, fontweight="600",
    )
    fig.tight_layout()

    combined = pd.concat(
        [table.assign(view=title) for title, table, _ in panels], ignore_index=True
    )

    # The takeaway reads its claim off the data rather than asserting it.
    first = panels[0][1].set_index("segment")
    late_rate = float(first.loc["Paid late before", "default_rate"]) if "Paid late before" in first.index else float("nan")
    ontime_rate = float(first.loc["Always paid on time", "default_rate"]) if "Always paid on time" in first.index else float("nan")
    multiple = late_rate / ontime_rate if ontime_rate else float("nan")

    takeaway = (
        f"Applicants who have ever paid one of our instalments late default at "
        f"{late_rate:.1f}%, against {ontime_rate:.1f}% for those who always paid on time "
        f"-- {multiple:.1f}x. This is the single most actionable signal in the platform, and "
        "not because it is the largest: it is the most *defensible*. Every other strong "
        "feature is a proxy -- education stands in for income stability, an external score "
        "summarises another institution's judgement -- whereas this is the applicant's own "
        "conduct on their own obligations, recorded by us and auditable. Practical use: a "
        "prior late-payment record is the cleanest basis for a referral rule, and it "
        "survives fair-lending scrutiny in a way demographic proxies do not."
    )
    path = save_figure(fig, "09_repayment_behaviour") if save else None
    return Insight("repayment_behaviour", "Prior repayment behaviour is the most defensible signal",
                   takeaway, combined, path)


# Registry of insight builders, in presentation order.
INSIGHT_BUILDERS: Final[list[Callable[[pd.DataFrame, bool], Insight]]] = [
    insight_class_imbalance,
    insight_missing_data,
    insight_feature_landscape,
    insight_external_scores,
    insight_affordability,
    insight_employment_anomaly,
    insight_demographic_segments,
    insight_credit_history,
    insight_repayment_behaviour,
]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(frame: pd.DataFrame | None = None, save: bool = True) -> EDAReport:
    """Run the full analysis and, by default, persist every artifact.

    Args:
        frame: Pre-loaded dataset. Loaded from the configured source if omitted.
        save: Write figures and JSON into ``reports/``. Set False in tests.

    Returns:
        The assembled :class:`EDAReport`.
    """
    data = build_dataset("train") if frame is None else frame
    settings.ensure_directories()

    summary = dataset_overview(data)
    catalog = feature_catalog(data)
    missingness = missing_value_report(data)
    findings = data_quality_findings(data)

    insights: list[Insight] = []
    for builder in INSIGHT_BUILDERS:
        # A single failing chart must not lose the rest of the analysis.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                insights.append(builder(data, save))
        except Exception as exc:  # noqa: BLE001 - reported, then execution continues
            logger.exception("Insight %s failed: %s", builder.__name__, exc)

    report = EDAReport(summary, catalog, missingness, findings, insights)

    if save:
        write_json(summary, settings.reports_dir / "eda_summary.json")
        write_json(findings, settings.reports_dir / "data_quality_findings.json")
        write_json(
            [
                {
                    "key": i.key, "title": i.title, "takeaway": i.takeaway,
                    "figure": i.figure_path.name if i.figure_path else None,
                }
                for i in insights
            ],
            settings.reports_dir / "eda_insights.json",
        )
        catalog.to_csv(settings.reports_dir / "feature_catalog.csv", index=False)
        missingness.to_csv(settings.reports_dir / "missing_values.csv", index=False)
        logger.info("EDA artifacts written to %s", settings.reports_dir)

    return report


def main() -> EDAReport:  # pragma: no cover - CLI entry point
    """Command-line entry point: run the EDA and print the takeaways."""
    report = run_eda()

    print("\n" + "=" * 78)
    print("DATASET SUMMARY")
    print("=" * 78)
    for key, value in report.summary.items():
        print(f"  {key:24s} {value}")

    print("\n" + "=" * 78)
    print("FEATURE CATEGORIZATION")
    print("=" * 78)
    print(report.feature_catalog["type_bucket"].value_counts().to_string())
    print("\n  by business domain:")
    print(report.feature_catalog["domain"].value_counts().to_string())

    print("\n" + "=" * 78)
    print(f"DATA QUALITY -- {len(report.quality_findings)} findings")
    print("=" * 78)
    for finding in report.quality_findings:
        print(f"  [{finding['severity'].upper():6s}] {finding['issue']}")
        print(f"           {finding['n_affected']:,} affected ({finding['pct_affected']}%)")
        print(f"           treatment: {finding['treatment']}")

    print("\n" + "=" * 78)
    print(f"BUSINESS INSIGHTS -- {len(report.insights)}")
    print("=" * 78)
    for index, insight in enumerate(report.insights, start=1):
        print(f"\n  {index}. {insight.title}")
        print(f"     {insight.takeaway}")
        if insight.figure_path:
            print(f"     figure: {insight.figure_path}")

    return report


if __name__ == "__main__":  # pragma: no cover
    main()
