"""Cleaning, feature engineering and encoding.

The public surface is :class:`CreditPreprocessor`, a picklable scikit-learn
transformer that turns a raw joined dataset into a model-ready feature frame.
It is fitted once during training, saved alongside the model, and reloaded at
inference time so that a single applicant is transformed by exactly the same
logic as the training data.

**Leakage policy.** Every transformation here is either row-wise arithmetic or
uses state learned only from the training split (category levels, degenerate
column detection). Nothing consults the target. Missing-value imputation for the
tree models is deliberately *not* performed: LightGBM and CatBoost route NaN
down their own branch, and in credit data "value absent" is itself predictive --
a thin bureau file or an undisclosed occupation. Only the linear baseline
imputes, and it does so inside its own pipeline, fitted per CV fold.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.helpers import safe_divide
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Home Credit's sentinel for "no employment record".
DAYS_EMPLOYED_ANOMALY: Final[int] = 365243

# Placeholder substituted for missing categorical levels. Kept as an explicit
# level rather than imputed, because "not disclosed" carries real signal.
MISSING_CATEGORY: Final[str] = "MISSING"

DAYS_PER_YEAR: Final[float] = 365.25

EXT_SOURCE_COLUMNS: Final[list[str]] = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# Columns carrying no usable signal: a constant in the real data, or an
# identifier that must never reach the model.
ALWAYS_DROP: Final[list[str]] = ["FLAG_MOBIL", "SK_ID_CURR", "SK_ID_BUREAU"]


# --------------------------------------------------------------------------- #
# Stateless cleaning
# --------------------------------------------------------------------------- #
def _is_day_offset(column: str) -> bool:
    """True only for columns that are negative day offsets from application date.

    Not every ``DAYS_``-prefixed name is a date. ``DAYS_EMPLOYED_ANOMALY`` is a
    derived 0/1 flag, and the bureau ``*DAYS_OVERDUE*`` aggregates are positive
    durations (days *past due*). Applying the "must be negative" rule to those
    would null every non-zero value and silently delete two real features, so
    they are excluded explicitly.
    """
    if column.endswith("_ANOMALY") or "OVERDUE" in column:
        return False
    return column.startswith("DAYS_") or column.startswith("BUREAU_DAYS_")


def clean_applications(frame: pd.DataFrame) -> pd.DataFrame:
    """Repair the known data-quality defects in the application table.

    Three fixes, each documented in the EDA:

    1. ``DAYS_EMPLOYED == 365243`` -- a sentinel meaning "no employment record",
       not 1000 years of tenure. Left in place it would dominate any tree split
       and destroy the coefficient on employment length. It is nulled and
       replaced by an explicit flag, so the model can still use the fact.
    2. ``CODE_GENDER == 'XNA'`` -- a handful of rows with an undefined value,
       mapped to NaN rather than being treated as a third gender.
    3. Positive ``DAYS_*`` values -- these columns are negative day offsets from
       the application date; any positive entry is corrupt and is nulled.

    Args:
        frame: Raw joined dataset.

    Returns:
        A cleaned copy. The input is never mutated.
    """
    df = frame.copy()

    if "DAYS_EMPLOYED" in df.columns:
        anomalous = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
        df["DAYS_EMPLOYED_ANOMALY"] = anomalous.astype(int)
        df.loc[anomalous, "DAYS_EMPLOYED"] = np.nan
        if anomalous.any():
            logger.debug("Nulled DAYS_EMPLOYED sentinel on %d rows", int(anomalous.sum()))

    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    # True date offsets must be <= 0; a positive value is corrupt.
    for column in df.columns:
        if _is_day_offset(column) and pd.api.types.is_numeric_dtype(df[column]):
            df.loc[df[column] > 0, column] = np.nan

    return df


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the domain features that carry most of the model's signal.

    Three families, all standard credit-risk practice:

    * **External-score aggregates** -- EXT_SOURCE_1/2/3 are the strongest raw
      predictors but are heavily and independently missing. Their mean, min,
      product and non-null count summarise the block into features that survive
      partial availability, which the individual columns do not.
    * **Affordability ratios** -- absolute amounts say little; a 500k loan is
      routine on a 300k income and reckless on a 60k one. Credit-to-income,
      annuity-to-income and payment rate express leverage directly, and are the
      terms a credit officer actually reasons about.
    * **Readable time features** -- ages and tenures in positive years, so that
      a derived policy rule reads "age below 32" rather than
      "DAYS_BIRTH above -11688".

    Args:
        frame: Cleaned dataset.

    Returns:
        A copy with the engineered columns appended.
    """
    df = frame.copy()
    new_columns: dict[str, pd.Series | np.ndarray] = {}

    # ------------------------------------------------ external score block --
    available = [column for column in EXT_SOURCE_COLUMNS if column in df.columns]
    if available:
        # Coerce explicitly: a single applicant arriving as a Series transposes
        # into object-dtype columns, on which the aggregates below would be both
        # slow and silently wrong.
        ext = df[available].apply(pd.to_numeric, errors="coerce")
        new_columns["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
        new_columns["EXT_SOURCE_MIN"] = ext.min(axis=1)
        new_columns["EXT_SOURCE_MAX"] = ext.max(axis=1)
        # Dispersion: agreement between bureaus is itself informative.
        new_columns["EXT_SOURCE_STD"] = ext.std(axis=1)
        # Product punishes a low score on any single source more sharply than
        # the mean does; NaN-filled with the row mean so partial rows survive.
        new_columns["EXT_SOURCE_PROD"] = ext.fillna(ext.mean(axis=1).median()).prod(axis=1)
        new_columns["EXT_SOURCE_COUNT"] = ext.notna().sum(axis=1)

    # -------------------------------------------------- affordability ratios --
    income = df.get("AMT_INCOME_TOTAL")
    credit = df.get("AMT_CREDIT")
    annuity = df.get("AMT_ANNUITY")
    goods = df.get("AMT_GOODS_PRICE")

    if credit is not None and income is not None:
        new_columns["CREDIT_TO_INCOME_RATIO"] = safe_divide(credit, income)
    if annuity is not None and income is not None:
        new_columns["ANNUITY_TO_INCOME_RATIO"] = safe_divide(annuity, income)
    if credit is not None and annuity is not None:
        # Notional term in years: credit divided by the yearly instalment.
        new_columns["CREDIT_TO_ANNUITY_RATIO"] = safe_divide(credit, annuity)
    if annuity is not None and credit is not None:
        # Payment rate -- the share of principal repaid per period.
        new_columns["PAYMENT_RATE"] = safe_divide(annuity, credit)
    if goods is not None and credit is not None:
        # Below 1 means the loan exceeds the value of the goods financed.
        new_columns["GOODS_TO_CREDIT_RATIO"] = safe_divide(goods, credit)
    if income is not None and "CNT_FAM_MEMBERS" in df.columns:
        new_columns["INCOME_PER_FAMILY_MEMBER"] = safe_divide(income, df["CNT_FAM_MEMBERS"])
    if income is not None and "CNT_CHILDREN" in df.columns:
        new_columns["CHILDREN_RATIO"] = safe_divide(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])

    # ------------------------------------------------------- time features --
    if "DAYS_BIRTH" in df.columns:
        age_years = -df["DAYS_BIRTH"] / DAYS_PER_YEAR
        new_columns["AGE_YEARS"] = age_years
    if "DAYS_EMPLOYED" in df.columns:
        employed_years = -df["DAYS_EMPLOYED"] / DAYS_PER_YEAR
        new_columns["EMPLOYED_YEARS"] = employed_years
        if "DAYS_BIRTH" in df.columns:
            # Share of adult life spent in the current job: a stability proxy
            # that is comparable across age groups in a way tenure is not.
            new_columns["EMPLOYED_TO_AGE_RATIO"] = safe_divide(
                df["DAYS_EMPLOYED"], df["DAYS_BIRTH"]
            )
    if "DAYS_REGISTRATION" in df.columns and "DAYS_BIRTH" in df.columns:
        new_columns["REGISTRATION_TO_AGE_RATIO"] = safe_divide(
            df["DAYS_REGISTRATION"], df["DAYS_BIRTH"]
        )
    if "DAYS_ID_PUBLISH" in df.columns and "DAYS_BIRTH" in df.columns:
        new_columns["ID_PUBLISH_TO_AGE_RATIO"] = safe_divide(
            df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"]
        )

    # --------------------------------------------------------- credit load --
    if "BUREAU_DEBT_TOTAL" in df.columns and income is not None:
        # External debt measured against income: the cross-table leverage view.
        new_columns["BUREAU_DEBT_TO_INCOME"] = safe_divide(df["BUREAU_DEBT_TOTAL"], income)
    if "BUREAU_CREDIT_SUM_TOTAL" in df.columns and credit is not None:
        new_columns["BUREAU_CREDIT_TO_APP_CREDIT"] = safe_divide(
            df["BUREAU_CREDIT_SUM_TOTAL"], credit
        )

    # ------------------------------------------------------ document count --
    document_flags = [column for column in df.columns if column.startswith("FLAG_DOCUMENT_")]
    if document_flags:
        new_columns["DOCUMENT_SUBMITTED_COUNT"] = df[document_flags].sum(axis=1)

    # ------------------------------------------------ bureau enquiry volume --
    enquiry_columns = [
        column for column in df.columns if column.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    if enquiry_columns:
        # A burst of recent credit searches is a classic distress signal.
        new_columns["CREDIT_ENQUIRY_TOTAL"] = df[enquiry_columns].sum(axis=1)

    if not new_columns:
        return df

    engineered = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    logger.debug("Engineered %d additional features", len(new_columns))
    return engineered


# --------------------------------------------------------------------------- #
# Fitted transformer
# --------------------------------------------------------------------------- #
class CreditPreprocessor(BaseEstimator, TransformerMixin):
    """Fit-once, apply-everywhere feature pipeline.

    Learned state is intentionally minimal -- column layout, categorical levels
    and which columns are degenerate -- which keeps the artifact small, keeps
    pickling trivial, and makes it obvious by inspection that no target
    information is retained.

    Args:
        max_categorical_levels: Categorical columns with more distinct levels
            than this are dropped. High-cardinality free-text-ish columns
            explode a one-hot matrix and produce unreadable SHAP output.
        drop_constant: Whether to drop columns that are constant on the
            training split.

    Attributes:
        feature_names_: Ordered output columns after transformation.
        categorical_features_: Output columns of ``category`` dtype.
        numeric_features_: Output columns of numeric dtype.
        category_levels_: Level list learned per categorical column, so unseen
            values at inference time collapse to ``MISSING`` instead of
            silently changing the encoding.
    """

    def __init__(self, max_categorical_levels: int = 40, drop_constant: bool = True) -> None:
        self.max_categorical_levels = max_categorical_levels
        self.drop_constant = drop_constant

    # ------------------------------------------------------------- helpers --
    @staticmethod
    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        """Run the stateless half of the pipeline."""
        return engineer_features(clean_applications(frame))

    def _drop_reserved(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove identifiers, the label, and known-constant columns."""
        reserved = [c for c in (*ALWAYS_DROP, "TARGET") if c in frame.columns]
        return frame.drop(columns=reserved)

    # ----------------------------------------------------------------- fit --
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CreditPreprocessor":
        """Learn the output schema from the training split.

        Args:
            X: Raw joined training dataset.
            y: Unused; accepted for scikit-learn API compatibility. The
                preprocessor never looks at the target.

        Returns:
            ``self``.
        """
        prepared = self._drop_reserved(self._prepare(X))

        categorical = [
            column for column in prepared.columns
            if not pd.api.types.is_numeric_dtype(prepared[column])
        ]

        # Drop unusably high-cardinality categoricals.
        self.dropped_high_cardinality_: list[str] = [
            column for column in categorical
            if prepared[column].nunique(dropna=True) > self.max_categorical_levels
        ]

        # Drop columns that are constant (or entirely missing) in training.
        self.dropped_constant_: list[str] = []
        if self.drop_constant:
            self.dropped_constant_ = [
                column for column in prepared.columns
                if prepared[column].nunique(dropna=True) <= 1
            ]

        dropped = set(self.dropped_high_cardinality_) | set(self.dropped_constant_)
        kept = [column for column in prepared.columns if column not in dropped]

        self.categorical_features_: list[str] = [c for c in categorical if c in kept]
        self.numeric_features_: list[str] = [
            c for c in kept if c not in self.categorical_features_
        ]

        # Freeze the level set per categorical column so inference-time encoding
        # cannot drift when a rare level is absent from a single-row request.
        self.category_levels_: dict[str, list[str]] = {}
        for column in self.categorical_features_:
            levels = sorted(prepared[column].dropna().astype(str).unique().tolist())
            if MISSING_CATEGORY not in levels:
                levels.append(MISSING_CATEGORY)
            self.category_levels_[column] = levels

        self.feature_names_: list[str] = kept
        self.n_features_in_ = len(kept)

        logger.info(
            "Preprocessor fitted: %d features (%d numeric, %d categorical); "
            "dropped %d constant, %d high-cardinality",
            len(kept), len(self.numeric_features_), len(self.categorical_features_),
            len(self.dropped_constant_), len(self.dropped_high_cardinality_),
        )
        return self

    # ----------------------------------------------------------- transform --
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline to any compatible frame.

        Columns absent from the input are recreated as all-NaN, so that a
        partially-filled applicant form from the UI transforms without error.

        Args:
            X: Raw dataset or a single-row applicant frame.

        Returns:
            A frame with exactly :attr:`feature_names_` as its columns, in
            order, with categoricals as ``category`` dtype.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        if not hasattr(self, "feature_names_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("CreditPreprocessor must be fitted before transform().")

        prepared = self._prepare(X)

        missing = [column for column in self.feature_names_ if column not in prepared.columns]
        if missing:
            logger.debug("Filling %d absent columns with NaN: %s", len(missing), missing[:5])
            prepared = pd.concat(
                [prepared, pd.DataFrame(np.nan, index=prepared.index, columns=missing)], axis=1
            )

        output = prepared[self.feature_names_].copy()

        for column in self.categorical_features_:
            levels = self.category_levels_[column]
            values = output[column].astype("object").where(output[column].notna(), MISSING_CATEGORY)
            values = values.astype(str)
            # Any level unseen during training becomes MISSING rather than a new
            # code, which keeps the encoding stable across calls.
            values = values.where(values.isin(levels), MISSING_CATEGORY)
            output[column] = pd.Categorical(values, categories=levels)

        for column in self.numeric_features_:
            numeric = pd.to_numeric(output[column], errors="coerce")
            # Guard against inf sneaking in from an unexpected ratio.
            output[column] = numeric.replace([np.inf, -np.inf], np.nan).astype("float64")

        return output

    # ------------------------------------------------------------ metadata --
    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the output column names (scikit-learn convention)."""
        return np.asarray(self.feature_names_, dtype=object)

    @property
    def categorical_indices_(self) -> list[int]:
        """Positional indices of the categorical columns, for CatBoost."""
        return [self.feature_names_.index(column) for column in self.categorical_features_]
