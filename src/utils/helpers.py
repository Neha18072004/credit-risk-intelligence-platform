"""Small shared helpers used across the data, ML and UI layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    fill: float = np.nan,
) -> pd.Series | np.ndarray:
    """Element-wise division that never raises and never yields +/-inf.

    Ratio features such as credit-to-income are central to this model, and the
    raw data contains zero and missing denominators.  Rather than let ``inf``
    leak into the feature matrix (LightGBM tolerates it, LogisticRegression does
    not), invalid results collapse to ``fill``.

    Args:
        numerator: Values to divide.
        denominator: Values to divide by.
        fill: Value substituted wherever the result is undefined.

    Returns:
        The element-wise ratio, with undefined entries replaced by ``fill``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.asarray(numerator, dtype=float) / np.asarray(denominator, dtype=float)
    result = np.where(np.isfinite(result), result, fill)
    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index, name=numerator.name)
    return result


def missing_value_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-column missingness, ordered worst-first.

    Args:
        frame: Any dataframe.

    Returns:
        A dataframe with ``column``, ``dtype``, ``n_missing``, ``pct_missing``
        and ``n_unique``, sorted by descending missingness.
    """
    total = len(frame)
    report = pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "n_missing": frame.isna().sum().to_numpy(),
            "n_unique": [frame[col].nunique(dropna=True) for col in frame.columns],
        }
    )
    report["pct_missing"] = (100.0 * report["n_missing"] / max(total, 1)).round(2)
    ordered = report[["column", "dtype", "n_missing", "pct_missing", "n_unique"]]
    return ordered.sort_values("pct_missing", ascending=False).reset_index(drop=True)


def _is_binary(series: pd.Series) -> bool:
    """True when a numeric column only ever takes the values 0 and 1.

    Deliberately stricter than "has two distinct values": on a small slice an
    ordinary continuous column can happen to show only two values, and calling
    that binary would mis-route it away from the numeric feature pipeline.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna().unique()
    return len(values) > 0 and set(values).issubset({0, 1})


def categorize_features(frame: pd.DataFrame, target: str = "TARGET") -> dict[str, list[str]]:
    """Bucket columns into numeric / categorical / binary / identifier / date-like.

    ``date-like`` covers the Home Credit ``DAYS_*`` columns, which are integer
    offsets from the application date rather than true timestamps.

    Args:
        frame: The dataframe to inspect.
        target: Name of the label column, excluded from every bucket.

    Returns:
        Mapping of bucket name to the list of column names in it.
    """
    buckets: dict[str, list[str]] = {
        "identifier": [], "date_like": [], "binary": [], "categorical": [], "numeric": [],
    }
    for column in frame.columns:
        if column == target:
            continue
        if column.startswith("SK_ID"):
            buckets["identifier"].append(column)
        elif column.startswith("DAYS_"):
            buckets["date_like"].append(column)
        elif _is_binary(frame[column]):
            buckets["binary"].append(column)
        elif pd.api.types.is_numeric_dtype(frame[column]):
            buckets["numeric"].append(column)
        else:
            buckets["categorical"].append(column)
    return buckets


def domain_of(column: str) -> str:
    """Map a raw column name to a business domain, for EDA grouping.

    Returns one of: ``demographics``, ``financials``, ``credit_history``,
    ``external_scores``, ``property``, ``documents``, ``application_process``,
    ``identifier`` or ``other``.
    """
    if column.startswith("SK_ID"):
        return "identifier"
    if column.startswith("EXT_SOURCE"):
        return "external_scores"
    if column.startswith("FLAG_DOCUMENT"):
        return "documents"
    if column.startswith(("BUREAU_", "AMT_REQ_CREDIT_BUREAU")):
        return "credit_history"
    if column.startswith("AMT_") or column in {"CNT_CREDIT_PROLONG"}:
        return "financials"
    if column.startswith(("CODE_GENDER", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_BIRTH",
                          "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE",
                          "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE")):
        return "demographics"
    if any(column.startswith(prefix) for prefix in (
        "APARTMENTS", "BASEMENTAREA", "YEARS_", "COMMONAREA", "ELEVATORS", "ENTRANCES",
        "FLOORS", "LANDAREA", "LIVING", "NONLIVING", "TOTALAREA", "HOUSETYPE",
        "WALLSMATERIAL", "EMERGENCYSTATE", "FONDKAPREMONT",
    )):
        return "property"
    if column.startswith(("WEEKDAY_APPR", "HOUR_APPR", "NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE")):
        return "application_process"
    if column.startswith(("REG_", "LIVE_", "REGION_", "FLAG_")) or column.startswith("DAYS_"):
        return "demographics"
    return "other"


def write_json(payload: Any, path: Path) -> Path:
    """Serialise ``payload`` to ``path`` as pretty JSON, creating parent dirs.

    NumPy scalars and arrays are converted to native Python types so that
    metrics dictionaries produced by scikit-learn serialise cleanly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")

    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    logger.debug("Wrote JSON -> %s", path)
    return path


def read_json(path: Path) -> Any:
    """Load JSON from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def probability_to_score(probability: float | np.ndarray) -> float | np.ndarray:
    """Convert a calibrated default probability into a risk score.

    The score is oriented so that **higher means riskier**, on a 0..
    ``settings.risk_score_scale`` range.  Keeping the orientation explicit here
    stops the UI and the rules layer from disagreeing about direction.

    Args:
        probability: Calibrated P(default), in [0, 1].

    Returns:
        The risk score on the configured scale.
    """
    clipped = np.clip(probability, 0.0, 1.0)
    scaled = clipped * settings.risk_score_scale
    return float(np.round(scaled, 1)) if np.isscalar(probability) else np.round(scaled, 1)


def format_currency(value: float | None) -> str:
    """Render a monetary amount compactly for the UI (e.g. ``1.2M``)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:,.2f}{suffix}"
    return f"{value:,.0f}"


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Yield ``items`` in lists of at most ``size`` elements."""
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
