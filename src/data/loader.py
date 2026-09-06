"""Dataset loading and table joining.

Reads the Home Credit tables from whichever source :attr:`Settings.data_mode`
selects -- the committed synthetic fixtures or the real Kaggle CSVs -- and
returns a single applicant-level frame.

Table scope, and why each one is here:

* ``application`` -- the applicant and the loan being applied for.
* ``bureau`` -- credit held with *other* institutions: prior debt, active
  exposure, arrears.
* ``previous_application`` -- prior applications to *this* lender, including
  the ones that were refused. An applicant who has been declined three times
  before is a different proposition from a first-time applicant, and nothing in
  the application table says so.
* ``installments_payments`` -- how prior loans were actually repaid: payment
  date against due date, amount paid against amount due. This is the genuine
  *repayment behaviour* table, and behavioural evidence is far more defensible
  as a decline reason than demographic evidence.

``bureau_balance``, ``POS_CASH_balance`` and ``credit_card_balance`` are
available but excluded by default: each is a monthly panel that is largely
redundant with the aggregates above, and each costs real runtime. Every table
here is behind a config flag so its contribution can be *measured* rather than
assumed -- see ``settings.include_*``. Nothing is used merely because it exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.helpers import safe_divide
from src.utils.docker_utils import resolve_data_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN: Final[str] = "TARGET"
ID_COLUMN: Final[str] = "SK_ID_CURR"

# Prefix applied to every column derived from the bureau table, so that a
# feature's provenance is readable straight off its name in SHAP plots.
BUREAU_PREFIX: Final[str] = "BUREAU_"
PREV_PREFIX: Final[str] = "PREV_"
INST_PREFIX: Final[str] = "INST_"


def _read_csv(
    path: Path,
    max_rows: int = 0,
    usecols: list[str] | None = None,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read a CSV, optionally capping rows, columns and dtypes.

    Args:
        path: File to read.
        max_rows: Row cap; ``0`` means read everything. Useful for a fast smoke
            run against the 307k-row real file.
        usecols: Read only these columns. The auxiliary tables are millions of
            rows wide, so reading 37 columns to use 9 wastes real memory.
        dtype: Explicit dtypes, used to hold the largest table in float32.

    Returns:
        The loaded dataframe.
    """
    nrows = max_rows if max_rows and max_rows > 0 else None
    frame = pd.read_csv(path, nrows=nrows, usecols=usecols, dtype=dtype)
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


def _join_optional(
    merged: pd.DataFrame, table: str, aggregate: Any, prefix: str, marker: str
) -> pd.DataFrame:
    """Aggregate an optional table and LEFT JOIN it, tolerating its absence.

    The committed sample fixtures cover only application and bureau, so a
    missing auxiliary table is a normal condition rather than an error: the
    features are skipped and the run continues.

    Args:
        merged: The frame built so far.
        table: Logical table name, for path resolution.
        aggregate: Callable taking the raw frame and returning applicant-level
            features.
        prefix: Column prefix, used for the coverage flag.
        marker: Column whose presence indicates a matched applicant.

    Returns:
        ``merged`` with the new block joined on, or unchanged if the table is
        not available.
    """
    try:
        path = resolve_data_path(table)
    except (FileNotFoundError, KeyError):
        logger.info("Optional table %s not present; skipping its features", table)
        return merged

    raw = _read_csv(
        path, max_rows=0,
        usecols=_PREV_COLUMNS if table == "previous_application" else _INSTALMENT_COLUMNS,
        dtype=_INSTALMENT_DTYPES if table == "installments_payments" else None,
    )
    features = aggregate(raw)
    if features.empty:
        return merged

    merged = merged.merge(features, on=ID_COLUMN, how="left")
    matched = merged[marker].notna().sum()
    logger.info(
        "Joined %s: %s of %s applicants matched (%.1f%%)",
        table, f"{matched:,}", f"{len(merged):,}", 100 * matched / max(len(merged), 1),
    )
    # Absence of history is itself informative, so it is recorded explicitly
    # rather than left to be inferred from a wall of nulls.
    merged[f"{prefix}HAS_HISTORY"] = merged[marker].notna().astype(int)
    return merged


def build_dataset(
    split: str = "train",
    include_bureau: bool | None = None,
    include_previous: bool | None = None,
    include_installments: bool | None = None,
) -> pd.DataFrame:
    """Load the application table and LEFT JOIN every enabled auxiliary block.

    LEFT JOINs throughout: a large minority of applicants have no external
    credit history, no prior application, or no instalment record. Their columns
    become NaN, which is genuine information -- a thin file -- not a defect. The
    tree models read the missingness directly and each block carries an explicit
    ``*_HAS_HISTORY`` flag.

    Each block can be switched off independently, which is how its contribution
    to model performance is measured rather than assumed.

    Args:
        split: ``"train"`` or ``"test"``.
        include_bureau: Override the configured setting for bureau features.
        include_previous: Override for previous-application features.
        include_installments: Override for instalment repayment features.

    Returns:
        One row per applicant.
    """
    applications = load_applications(split)
    merged = applications

    use_bureau = settings.include_bureau if include_bureau is None else include_bureau
    if use_bureau:
        bureau_features = aggregate_bureau(load_bureau())
        merged = merged.merge(bureau_features, on=ID_COLUMN, how="left")
        matched = merged[f"{BUREAU_PREFIX}LOAN_COUNT"].notna().sum()
        logger.info(
            "Joined bureau onto %s applications: %s matched (%.1f%%), %s without history",
            f"{len(merged):,}", f"{matched:,}", 100 * matched / max(len(merged), 1),
            f"{len(merged) - matched:,}",
        )
        merged[f"{BUREAU_PREFIX}LOAN_COUNT"] = merged[f"{BUREAU_PREFIX}LOAN_COUNT"].fillna(0)
        merged[f"{BUREAU_PREFIX}HAS_HISTORY"] = (
            merged[f"{BUREAU_PREFIX}LOAN_COUNT"] > 0
        ).astype(int)

    use_previous = (
        settings.include_previous_application if include_previous is None else include_previous
    )
    if use_previous:
        merged = _join_optional(
            merged, "previous_application", aggregate_previous_application,
            PREV_PREFIX, f"{PREV_PREFIX}COUNT",
        )

    use_installments = (
        settings.include_installments if include_installments is None else include_installments
    )
    if use_installments:
        merged = _join_optional(
            merged, "installments_payments", aggregate_installments,
            INST_PREFIX, f"{INST_PREFIX}COUNT",
        )

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


# --------------------------------------------------------------------------- #
# Previous applications with this lender
# --------------------------------------------------------------------------- #
# Only the columns the aggregation uses are read. previous_application is 1.7M
# rows and installments_payments 13.6M, so reading all 37 columns to use 9 of
# them costs memory for nothing.
_PREV_COLUMNS: Final[list[str]] = [
    "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_APPLICATION",
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_DOWN_PAYMENT", "RATE_DOWN_PAYMENT",
    "DAYS_DECISION", "CNT_PAYMENT", "NAME_YIELD_GROUP",
]


def load_previous_application() -> pd.DataFrame:
    """Load prior applications made to this lender."""
    return _read_csv(resolve_data_path("previous_application"), max_rows=0, usecols=_PREV_COLUMNS)


def aggregate_previous_application(previous: pd.DataFrame) -> pd.DataFrame:
    """Collapse prior applications to one row per applicant.

    The headline feature is the **refusal rate**. An applicant this lender has
    already declined several times is a materially different proposition from a
    first-time applicant, and the application table contains no trace of that
    history.

    The second is the **credit-to-application ratio**: how much the lender
    actually granted against how much was asked for. A consistent gap means
    previous underwriting judged the request unaffordable, which is a prior
    credit assessment expressed as a number.

    Args:
        previous: Raw ``previous_application`` table.

    Returns:
        One row per applicant that appears in the table.
    """
    if previous.empty:
        return pd.DataFrame(columns=[ID_COLUMN])

    frame = previous.copy()
    status = frame["NAME_CONTRACT_STATUS"]
    frame["_APPROVED"] = (status == "Approved").astype(int)
    frame["_REFUSED"] = (status == "Refused").astype(int)
    frame["_CANCELLED"] = status.isin(["Canceled", "Unused offer"]).astype(int)

    aggregated = frame.groupby(ID_COLUMN).agg(
        **{
            f"{PREV_PREFIX}COUNT": ("SK_ID_PREV", "count"),
            f"{PREV_PREFIX}APPROVED_COUNT": ("_APPROVED", "sum"),
            f"{PREV_PREFIX}REFUSED_COUNT": ("_REFUSED", "sum"),
            f"{PREV_PREFIX}CANCELLED_COUNT": ("_CANCELLED", "sum"),
            f"{PREV_PREFIX}AMT_APPLICATION_MEAN": ("AMT_APPLICATION", "mean"),
            f"{PREV_PREFIX}AMT_APPLICATION_MAX": ("AMT_APPLICATION", "max"),
            f"{PREV_PREFIX}AMT_CREDIT_MEAN": ("AMT_CREDIT", "mean"),
            f"{PREV_PREFIX}AMT_ANNUITY_MEAN": ("AMT_ANNUITY", "mean"),
            f"{PREV_PREFIX}DOWN_PAYMENT_RATE_MEAN": ("RATE_DOWN_PAYMENT", "mean"),
            f"{PREV_PREFIX}DAYS_DECISION_MAX": ("DAYS_DECISION", "max"),
            f"{PREV_PREFIX}DAYS_DECISION_MIN": ("DAYS_DECISION", "min"),
            f"{PREV_PREFIX}CNT_PAYMENT_MEAN": ("CNT_PAYMENT", "mean"),
        }
    ).reset_index()

    count = aggregated[f"{PREV_PREFIX}COUNT"]
    aggregated[f"{PREV_PREFIX}REFUSED_RATE"] = safe_divide(
        aggregated[f"{PREV_PREFIX}REFUSED_COUNT"], count
    )
    aggregated[f"{PREV_PREFIX}APPROVED_RATE"] = safe_divide(
        aggregated[f"{PREV_PREFIX}APPROVED_COUNT"], count
    )
    # Below 1 means the lender granted less than was requested.
    aggregated[f"{PREV_PREFIX}CREDIT_TO_APPLICATION"] = safe_divide(
        aggregated[f"{PREV_PREFIX}AMT_CREDIT_MEAN"],
        aggregated[f"{PREV_PREFIX}AMT_APPLICATION_MEAN"],
    )
    aggregated[f"{PREV_PREFIX}EVER_REFUSED"] = (
        aggregated[f"{PREV_PREFIX}REFUSED_COUNT"] > 0
    ).astype(int)

    logger.info(
        "Aggregated previous applications: %s applicants x %s features",
        f"{len(aggregated):,}", aggregated.shape[1] - 1,
    )
    return aggregated


# --------------------------------------------------------------------------- #
# Instalment repayment behaviour
# --------------------------------------------------------------------------- #
_INSTALMENT_COLUMNS: Final[list[str]] = [
    "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT", "AMT_PAYMENT",
]

# 13.6M rows: float32 halves the memory of the payment columns with no
# meaningful loss of precision on amounts and day offsets.
_INSTALMENT_DTYPES: Final[dict[str, str]] = {
    "SK_ID_CURR": "int32",
    "DAYS_INSTALMENT": "float32",
    "DAYS_ENTRY_PAYMENT": "float32",
    "AMT_INSTALMENT": "float32",
    "AMT_PAYMENT": "float32",
}


def load_installments() -> pd.DataFrame:
    """Load the instalment payment history."""
    return _read_csv(
        resolve_data_path("installments_payments"), max_rows=0,
        usecols=_INSTALMENT_COLUMNS, dtype=_INSTALMENT_DTYPES,
    )


def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """Collapse instalment history into repayment-behaviour features.

    This is the table the brief's "repayment behaviour" actually refers to. Each
    row is one scheduled instalment with the date it was due and the date it was
    paid, so two derived quantities carry the signal:

    * **Days past due** -- ``DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT``. Positive
      means the payment landed late. Both columns are negative offsets from the
      application date, so the subtraction is still the right way round.
    * **Payment ratio** -- ``AMT_PAYMENT / AMT_INSTALMENT``. Below 1 means the
      applicant paid less than was owed that period.

    These are the most defensible features in the whole dataset for a decline
    decision: they are behavioural rather than demographic, they concern the
    applicant's own conduct rather than a proxy for it, and they are verifiable
    from the lender's own records.

    Args:
        installments: Raw ``installments_payments`` table.

    Returns:
        One row per applicant that appears in the table.
    """
    if installments.empty:
        return pd.DataFrame(columns=[ID_COLUMN])

    frame = installments
    days_past_due = (frame["DAYS_ENTRY_PAYMENT"] - frame["DAYS_INSTALMENT"]).clip(lower=0)
    payment_ratio = safe_divide(frame["AMT_PAYMENT"], frame["AMT_INSTALMENT"])
    shortfall = (frame["AMT_INSTALMENT"] - frame["AMT_PAYMENT"]).clip(lower=0)

    working = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN],
            "_DPD": days_past_due,
            "_LATE": (days_past_due > 0).astype("int8"),
            "_RATIO": payment_ratio,
            "_UNDERPAID": (payment_ratio < 0.999).astype("int8"),
            "_SHORTFALL": shortfall,
            "_AMT_PAYMENT": frame["AMT_PAYMENT"],
            "_DAYS_INSTALMENT": frame["DAYS_INSTALMENT"],
        }
    )

    aggregated = working.groupby(ID_COLUMN).agg(
        **{
            f"{INST_PREFIX}COUNT": ("_DPD", "size"),
            f"{INST_PREFIX}DPD_MEAN": ("_DPD", "mean"),
            f"{INST_PREFIX}DPD_MAX": ("_DPD", "max"),
            f"{INST_PREFIX}DPD_SUM": ("_DPD", "sum"),
            f"{INST_PREFIX}LATE_COUNT": ("_LATE", "sum"),
            f"{INST_PREFIX}PAYMENT_RATIO_MEAN": ("_RATIO", "mean"),
            f"{INST_PREFIX}PAYMENT_RATIO_MIN": ("_RATIO", "min"),
            f"{INST_PREFIX}UNDERPAID_COUNT": ("_UNDERPAID", "sum"),
            f"{INST_PREFIX}SHORTFALL_SUM": ("_SHORTFALL", "sum"),
            f"{INST_PREFIX}AMT_PAYMENT_SUM": ("_AMT_PAYMENT", "sum"),
            f"{INST_PREFIX}DAYS_LAST_INSTALMENT": ("_DAYS_INSTALMENT", "max"),
        }
    ).reset_index()

    count = aggregated[f"{INST_PREFIX}COUNT"]
    # Rates rather than counts: someone with 200 instalments and 10 late
    # payments is in better standing than someone with 12 and 10.
    aggregated[f"{INST_PREFIX}LATE_RATE"] = safe_divide(
        aggregated[f"{INST_PREFIX}LATE_COUNT"], count
    )
    aggregated[f"{INST_PREFIX}UNDERPAID_RATE"] = safe_divide(
        aggregated[f"{INST_PREFIX}UNDERPAID_COUNT"], count
    )
    aggregated[f"{INST_PREFIX}EVER_LATE"] = (
        aggregated[f"{INST_PREFIX}LATE_COUNT"] > 0
    ).astype(int)

    logger.info(
        "Aggregated instalments: %s applicants x %s features from %s payment records",
        f"{len(aggregated):,}", aggregated.shape[1] - 1, f"{len(frame):,}",
    )
    return aggregated
