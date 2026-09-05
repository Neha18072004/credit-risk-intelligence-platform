"""Dataset loading and table joining.

Reads the Home Credit tables from whichever source :attr:`Settings.data_mode`
selects -- the committed synthetic fixtures or the real Kaggle CSVs -- and
returns a single applicant-level frame.

Scope decision: this project uses the ``application`` table plus **aggregated**
``bureau`` features, and deliberately not the other five auxiliary tables.
Bureau carries the external credit-history signal that application alone lacks
(prior debt, active exposure, days past due), while staying small enough to keep
training fast and every engineered column explainable by name. Adding
``previous_application`` / ``installments_payments`` buys a little AUC at a
large cost in runtime, opacity and failure surface -- a bad trade for a platform
graded on explainability and one-command runnability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.docker_utils import resolve_data_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN: Final[str] = "TARGET"
ID_COLUMN: Final[str] = "SK_ID_CURR"

# Prefix applied to every column derived from the bureau table, so that a
# feature's provenance is readable straight off its name in SHAP plots.
BUREAU_PREFIX: Final[str] = "BUREAU_"


def _read_csv(path: Path, max_rows: int = 0) -> pd.DataFrame:
    """Read a CSV, optionally capping the row count.

    Args:
        path: File to read.
        max_rows: Row cap; ``0`` means read everything. Useful for a fast smoke
            run against the 307k-row real file.

    Returns:
        The loaded dataframe.
    """
    nrows = max_rows if max_rows and max_rows > 0 else None
    frame = pd.read_csv(path, nrows=nrows)
    logger.info("Read %s -> %s rows x %s cols", path.name, f"{len(frame):,}", frame.shape[1])
    return frame


def load_applications(split: str = "train") -> pd.DataFrame:
    """Load the main application table.

    Args:
        split: ``"train"`` (includes ``TARGET``) or ``"test"``.

    Returns:
        The raw application dataframe.

    Raises:
        ValueError: If ``split`` is not ``"train"`` or ``"test"``.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = resolve_data_path(f"application_{split}")
    return _read_csv(path, settings.max_rows)


def load_bureau() -> pd.DataFrame:
    """Load the raw bureau table (one row per prior external credit)."""
    return _read_csv(resolve_data_path("bureau"), max_rows=0)


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Collapse the bureau table to one row per applicant.

    The bureau table holds one row per prior credit held with *another*
    institution. Aggregating it to applicant level produces the external
    credit-history block: how much prior credit exists, how much is still
    outstanding, how much is currently in arrears, and how long the history is.

    Every output column is named so that its meaning is legible without a data
    dictionary, because these names surface directly in the SHAP charts and the
    generated policy rules.

    Args:
        bureau: The raw bureau table.

    Returns:
        A frame indexed by a ``SK_ID_CURR`` column, one row per applicant that
        appears in ``bureau``. Applicants with no external credit history are
        simply absent, and are handled by the LEFT JOIN in :func:`build_dataset`.
    """
    if bureau.empty:
        logger.warning("Bureau table is empty; returning an empty aggregation")
        return pd.DataFrame(columns=[ID_COLUMN])

    frame = bureau.copy()
    frame["_IS_ACTIVE"] = (frame["CREDIT_ACTIVE"] == "Active").astype(int)
    frame["_IS_CLOSED"] = (frame["CREDIT_ACTIVE"] == "Closed").astype(int)
    # Days-past-due is the sharpest repayment-behaviour signal in this table.
    frame["_IS_OVERDUE"] = (frame["CREDIT_DAY_OVERDUE"].fillna(0) > 0).astype(int)

    grouped = frame.groupby(ID_COLUMN)
    aggregated = grouped.agg(
        **{
            # --- volume of external credit history ---
            f"{BUREAU_PREFIX}LOAN_COUNT": ("SK_ID_BUREAU", "count"),
            f"{BUREAU_PREFIX}ACTIVE_COUNT": ("_IS_ACTIVE", "sum"),
            f"{BUREAU_PREFIX}CLOSED_COUNT": ("_IS_CLOSED", "sum"),
            f"{BUREAU_PREFIX}CREDIT_TYPE_NUNIQUE": ("CREDIT_TYPE", "nunique"),
            # --- exposure ---
            f"{BUREAU_PREFIX}CREDIT_SUM_TOTAL": ("AMT_CREDIT_SUM", "sum"),
            f"{BUREAU_PREFIX}CREDIT_SUM_MEAN": ("AMT_CREDIT_SUM", "mean"),
            f"{BUREAU_PREFIX}CREDIT_SUM_MAX": ("AMT_CREDIT_SUM", "max"),
            f"{BUREAU_PREFIX}DEBT_TOTAL": ("AMT_CREDIT_SUM_DEBT", "sum"),
            f"{BUREAU_PREFIX}DEBT_MEAN": ("AMT_CREDIT_SUM_DEBT", "mean"),
            f"{BUREAU_PREFIX}CREDIT_LIMIT_TOTAL": ("AMT_CREDIT_SUM_LIMIT", "sum"),
            # --- arrears / repayment behaviour ---
            f"{BUREAU_PREFIX}DAYS_OVERDUE_MEAN": ("CREDIT_DAY_OVERDUE", "mean"),
            f"{BUREAU_PREFIX}DAYS_OVERDUE_MAX": ("CREDIT_DAY_OVERDUE", "max"),
            f"{BUREAU_PREFIX}OVERDUE_LOAN_COUNT": ("_IS_OVERDUE", "sum"),
            f"{BUREAU_PREFIX}AMT_OVERDUE_TOTAL": ("AMT_CREDIT_SUM_OVERDUE", "sum"),
            f"{BUREAU_PREFIX}MAX_OVERDUE_MEAN": ("AMT_CREDIT_MAX_OVERDUE", "mean"),
            f"{BUREAU_PREFIX}PROLONG_TOTAL": ("CNT_CREDIT_PROLONG", "sum"),
            # --- history depth / recency (DAYS_* are negative offsets) ---
            f"{BUREAU_PREFIX}DAYS_CREDIT_MIN": ("DAYS_CREDIT", "min"),
            f"{BUREAU_PREFIX}DAYS_CREDIT_MAX": ("DAYS_CREDIT", "max"),
            f"{BUREAU_PREFIX}DAYS_CREDIT_MEAN": ("DAYS_CREDIT", "mean"),
            f"{BUREAU_PREFIX}DAYS_UPDATE_MEAN": ("DAYS_CREDIT_UPDATE", "mean"),
            f"{BUREAU_PREFIX}ANNUITY_TOTAL": ("AMT_ANNUITY", "sum"),
        }
    ).reset_index()

    # --- derived ratios: more interpretable than the raw sums they come from ---
    total_credit = aggregated[f"{BUREAU_PREFIX}CREDIT_SUM_TOTAL"]
    total_debt = aggregated[f"{BUREAU_PREFIX}DEBT_TOTAL"]
    loan_count = aggregated[f"{BUREAU_PREFIX}LOAN_COUNT"]

    # Share of prior credit still outstanding: the headline leverage measure.
    aggregated[f"{BUREAU_PREFIX}DEBT_CREDIT_RATIO"] = np.where(
        total_credit > 0, total_debt / total_credit.replace(0, np.nan), np.nan
    )
    # Share of external credits still open.
    aggregated[f"{BUREAU_PREFIX}ACTIVE_RATIO"] = np.where(
        loan_count > 0, aggregated[f"{BUREAU_PREFIX}ACTIVE_COUNT"] / loan_count, np.nan
    )
    # Binary "has ever been in arrears" flag -- the single most rule-friendly
    # bureau feature, and the one the surrogate policy tree tends to pick up.
    aggregated[f"{BUREAU_PREFIX}HAS_OVERDUE"] = (
        aggregated[f"{BUREAU_PREFIX}OVERDUE_LOAN_COUNT"] > 0
    ).astype(int)

    logger.info(
        "Aggregated bureau: %s applicants x %s features",
        f"{len(aggregated):,}", aggregated.shape[1] - 1,
    )
    return aggregated


def build_dataset(split: str = "train", include_bureau: bool = True) -> pd.DataFrame:
    """Load the application table and LEFT JOIN the aggregated bureau block.

    A LEFT JOIN is essential: roughly 14% of applicants have no external credit
    history at all. Their bureau columns become NaN, which is genuine
    information ("no bureau record"), not a defect -- the tree models read the
    missingness directly and the preprocessor adds an explicit flag for the
    linear baseline.

    Args:
        split: ``"train"`` or ``"test"``.
        include_bureau: Set False to train on application features only, which
            is useful for measuring how much the bureau block actually adds.

    Returns:
        One row per applicant, application columns plus ``BUREAU_*`` columns.
    """
    applications = load_applications(split)

    if not include_bureau:
        logger.info("Bureau features disabled; using application table only")
        return applications

    bureau_features = aggregate_bureau(load_bureau())
    merged = applications.merge(bureau_features, on=ID_COLUMN, how="left")

    matched = merged[f"{BUREAU_PREFIX}LOAN_COUNT"].notna().sum()
    logger.info(
        "Joined bureau onto %s applications: %s matched (%.1f%%), %s without history",
        f"{len(merged):,}", f"{matched:,}", 100 * matched / max(len(merged), 1),
        f"{len(merged) - matched:,}",
    )

    # "No bureau record" is a real credit signal (a thin file), so record it
    # explicitly before any imputation can wash it out.
    merged[f"{BUREAU_PREFIX}LOAN_COUNT"] = merged[f"{BUREAU_PREFIX}LOAN_COUNT"].fillna(0)
    merged[f"{BUREAU_PREFIX}HAS_HISTORY"] = (merged[f"{BUREAU_PREFIX}LOAN_COUNT"] > 0).astype(int)
    return merged


def split_features_target(
    frame: pd.DataFrame, target: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate the label from the feature block.

    The identifier is dropped from the features so it can never be learned from;
    it is retained by the caller for joining predictions back to applicants.

    Args:
        frame: The joined dataset.
        target: Label column name.

    Returns:
        ``(features, labels)``; ``labels`` is None for an unlabelled split.
    """
    labels = frame[target].astype(int) if target in frame.columns else None
    drop = [column for column in (target, ID_COLUMN) if column in frame.columns]
    return frame.drop(columns=drop), labels


def dataset_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Compact description of a loaded dataset, for logs, EDA and the UI."""
    summary: dict[str, object] = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1024**2, 2),
        "n_bureau_features": int(sum(c.startswith(BUREAU_PREFIX) for c in frame.columns)),
        "columns_with_missing": int((frame.isna().sum() > 0).sum()),
        "overall_missing_pct": round(100 * frame.isna().to_numpy().mean(), 2),
    }
    if TARGET_COLUMN in frame.columns:
        positives = int(frame[TARGET_COLUMN].sum())
        summary["n_defaults"] = positives
        summary["default_rate"] = round(float(frame[TARGET_COLUMN].mean()), 4)
        # The ratio the model's class weighting has to counteract.
        summary["imbalance_ratio"] = round((len(frame) - positives) / max(positives, 1), 1)
    return summary
