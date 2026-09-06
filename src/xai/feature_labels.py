"""Plain-English names and phrasing for model features.

A credit decision has to be explainable to the applicant, not only to the data
scientist. ``BUREAU_DEBT_CREDIT_RATIO = 0.59`` is not an explanation; "59% of
their existing external credit is still unpaid" is.

This module is the single vocabulary shared by both explanation surfaces --
the SHAP contributions in :mod:`src.xai.shap_explainer` and the policy rules in
:mod:`src.xai.rules`. Keeping one mapping means the two never describe the same
feature differently, which would undermine both.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

# Readable names for the features that actually surface in explanations.
FEATURE_LABELS: Final[dict[str, str]] = {
    # --- external credit scores ---
    "EXT_SOURCE_MEAN": "average external credit score",
    "EXT_SOURCE_MIN": "lowest external credit score",
    "EXT_SOURCE_MAX": "highest external credit score",
    "EXT_SOURCE_STD": "spread between external credit scores",
    "EXT_SOURCE_PROD": "combined external credit score",
    "EXT_SOURCE_COUNT": "number of external scores available",
    "EXT_SOURCE_1": "external credit score 1",
    "EXT_SOURCE_2": "external credit score 2",
    "EXT_SOURCE_3": "external credit score 3",
    # --- affordability ---
    "CREDIT_TO_INCOME_RATIO": "loan-to-income ratio",
    "ANNUITY_TO_INCOME_RATIO": "instalment-to-income ratio",
    "CREDIT_TO_ANNUITY_RATIO": "implied loan term in years",
    "PAYMENT_RATE": "payment rate",
    "GOODS_TO_CREDIT_RATIO": "goods-value-to-loan ratio",
    "INCOME_PER_FAMILY_MEMBER": "income per household member",
    "CHILDREN_RATIO": "share of household who are children",
    # --- money ---
    "AMT_INCOME_TOTAL": "annual income",
    "AMT_CREDIT": "loan amount",
    "AMT_ANNUITY": "annual instalment",
    "AMT_GOODS_PRICE": "value of the goods financed",
    # --- time and stability ---
    "AGE_YEARS": "age",
    "EMPLOYED_YEARS": "years in current employment",
    "EMPLOYED_TO_AGE_RATIO": "share of adult life in current job",
    "DAYS_BIRTH": "age",
    "DAYS_EMPLOYED": "employment tenure",
    "DAYS_EMPLOYED_ANOMALY": "no employment record (pensioner or unemployed)",
    "DAYS_REGISTRATION": "time since registration",
    "DAYS_ID_PUBLISH": "time since identity document issued",
    "DAYS_LAST_PHONE_CHANGE": "time since last phone change",
    "REGISTRATION_TO_AGE_RATIO": "registration age relative to age",
    "ID_PUBLISH_TO_AGE_RATIO": "identity-document age relative to age",
    # --- external credit history ---
    "BUREAU_HAS_OVERDUE": "has prior arrears on external credit",
    "BUREAU_HAS_HISTORY": "has any external credit history",
    "BUREAU_DEBT_CREDIT_RATIO": "share of external credit still unpaid",
    "BUREAU_DEBT_TO_INCOME": "external debt relative to income",
    "BUREAU_DEBT_TOTAL": "total outstanding external debt",
    "BUREAU_DEBT_MEAN": "average outstanding debt per external credit",
    "BUREAU_CREDIT_SUM_TOTAL": "total external credit ever taken",
    "BUREAU_CREDIT_SUM_MEAN": "average size of external credits",
    "BUREAU_CREDIT_SUM_MAX": "largest external credit",
    "BUREAU_LOAN_COUNT": "number of prior external credits",
    "BUREAU_ACTIVE_COUNT": "number of external credits still open",
    "BUREAU_CLOSED_COUNT": "number of external credits repaid",
    "BUREAU_ACTIVE_RATIO": "share of external credits still open",
    "BUREAU_DAYS_OVERDUE_MAX": "worst days past due on external credit",
    "BUREAU_DAYS_OVERDUE_MEAN": "average days past due on external credit",
    "BUREAU_OVERDUE_LOAN_COUNT": "number of external credits in arrears",
    "BUREAU_CREDIT_TO_APP_CREDIT": "external credit relative to this loan",
    "BUREAU_DAYS_CREDIT_MIN": "age of oldest external credit",
    "BUREAU_DAYS_CREDIT_MAX": "recency of newest external credit",
    "BUREAU_CREDIT_TYPE_NUNIQUE": "variety of external credit types",
    # --- application and profile ---
    "CREDIT_ENQUIRY_TOTAL": "recent credit bureau enquiries",
    "DOCUMENT_SUBMITTED_COUNT": "number of documents submitted",
    "NAME_EDUCATION_TYPE": "education level",
    "NAME_INCOME_TYPE": "employment type",
    "NAME_FAMILY_STATUS": "family status",
    "NAME_HOUSING_TYPE": "housing situation",
    "NAME_CONTRACT_TYPE": "loan type",
    "OCCUPATION_TYPE": "occupation",
    "ORGANIZATION_TYPE": "employer type",
    "CODE_GENDER": "gender",
    "CNT_CHILDREN": "number of children",
    "CNT_FAM_MEMBERS": "household size",
    "REGION_RATING_CLIENT": "region risk rating",
    "REGION_POPULATION_RELATIVE": "region population density",
    "OBS_30_CNT_SOCIAL_CIRCLE": "contacts observed in social circle",
    "DEF_30_CNT_SOCIAL_CIRCLE": "contacts in default in social circle",
    "DEF_60_CNT_SOCIAL_CIRCLE": "contacts in long-term default in social circle",
    "FLAG_OWN_CAR": "owns a car",
    "FLAG_OWN_REALTY": "owns property",
    "FLAG_EMP_PHONE": "provided an employer phone number",
    "FLAG_PHONE": "provided a home phone number",
    "FLAG_EMAIL": "provided an email address",
}

# Features whose value is a 0/1 flag, so "= 1" should read as "yes".
_BINARY_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "DAYS_EMPLOYED_ANOMALY", "BUREAU_HAS_OVERDUE", "BUREAU_HAS_HISTORY",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_EMP_PHONE", "FLAG_PHONE", "FLAG_EMAIL",
    }
)

# Features denominated in money, shown with thousands separators and no decimals.
_MONEY_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "INCOME_PER_FAMILY_MEMBER", "BUREAU_DEBT_TOTAL", "BUREAU_DEBT_MEAN",
        "BUREAU_CREDIT_SUM_TOTAL", "BUREAU_CREDIT_SUM_MEAN", "BUREAU_CREDIT_SUM_MAX",
    }
)

# Features that are naturally read as a percentage.
_SHARE_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "BUREAU_DEBT_CREDIT_RATIO", "BUREAU_ACTIVE_RATIO", "EMPLOYED_TO_AGE_RATIO",
        "CHILDREN_RATIO", "PAYMENT_RATE", "GOODS_TO_CREDIT_RATIO",
        "ANNUITY_TO_INCOME_RATIO",
    }
)


def humanise(feature: str) -> str:
    """Return a plain-English label for a feature name.

    Args:
        feature: Raw column name, or a ``"COLUMN = level"`` one-hot indicator.

    Returns:
        A readable label. Unknown names degrade to a lower-cased, de-underscored
        form rather than raising, so a new feature is merely ugly, not broken.
    """
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if " = " in feature:  # one-hot indicator: "NAME_EDUCATION_TYPE = Higher education"
        column, _, level = feature.partition(" = ")
        label = FEATURE_LABELS.get(column.strip(), column.strip().replace("_", " ").lower())
        return f"{label} is {level.strip()}"
    return feature.replace("_", " ").lower()


def format_value(feature: str, value: Any) -> str:
    """Render a feature's value the way a person would say it.

    Args:
        feature: Raw column name.
        value: The applicant's value for it.

    Returns:
        A formatted string, e.g. ``"147,000"``, ``"59%"``, ``"yes"``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not provided"
    if feature in _BINARY_FEATURES:
        try:
            return "yes" if float(value) >= 0.5 else "no"
        except (TypeError, ValueError):
            return str(value)
    if not isinstance(value, (int, float, np.number)):
        return str(value)

    numeric = float(value)
    if feature in _MONEY_FEATURES:
        return f"{numeric:,.0f}"
    if feature in _SHARE_FEATURES:
        return f"{100 * numeric:.0f}%"
    if feature == "AGE_YEARS":
        return f"{numeric:.0f} years"
    if feature == "EMPLOYED_YEARS":
        return f"{numeric:.1f} years"
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    return f"{numeric:,.3g}"
