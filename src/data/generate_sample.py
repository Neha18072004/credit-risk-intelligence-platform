"""Generate synthetic Home Credit fixtures.

The real Home Credit Default Risk CSVs are Kaggle-gated and far too large to
commit, so this module fabricates a small stand-in that is **schema-identical**
to the real thing:

* ``application_train.csv`` -- all 122 real columns, including ``TARGET``
* ``application_test.csv``  -- the same 121 columns minus ``TARGET``
* ``bureau.csv``            -- all 17 real columns, several rows per applicant

The fixtures deliberately reproduce the quirks the EDA and preprocessing layers
must cope with:

* a ~8% positive class rate, matching the real default rate;
* the notorious ``DAYS_EMPLOYED == 365243`` sentinel, attached (as in the real
  data) to pensioners and the unemployed;
* heavy, *column-specific* missingness -- ``EXT_SOURCE_1`` ~56% null, the
  building-block columns ~50-70% null, ``OCCUPATION_TYPE`` ~31% null;
* genuine signal, so that a model trained on the fixture actually learns
  something and the three-way bake-off is a meaningful comparison.

Run it with::

    python -m src.data.generate_sample
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Fixture size.  Small enough to commit, large enough for 5-fold stratified CV
# to be stable at an ~8% positive rate.
# --------------------------------------------------------------------------- #
N_TRAIN: int = 1500
N_TEST: int = 300
TARGET_DEFAULT_RATE: float = 0.081  # real Home Credit rate is 8.07%

# The sentinel Home Credit uses for "no employment record".  Roughly 18% of the
# real rows carry it; it maps almost perfectly onto pensioners.
DAYS_EMPLOYED_ANOMALY: int = 365243

# Real categorical level frequencies, rounded.  Keeping these faithful means the
# EDA charts built on the fixture tell the same story as on the real data.
_CATEGORICAL_LEVELS: dict[str, tuple[list[str], list[float]]] = {
    "NAME_CONTRACT_TYPE": (["Cash loans", "Revolving loans"], [0.905, 0.095]),
    "CODE_GENDER": (["F", "M", "XNA"], [0.658, 0.341, 0.001]),
    "FLAG_OWN_CAR": (["N", "Y"], [0.66, 0.34]),
    "FLAG_OWN_REALTY": (["Y", "N"], [0.694, 0.306]),
    "NAME_TYPE_SUITE": (
        ["Unaccompanied", "Family", "Spouse, partner", "Children",
         "Other_B", "Other_A", "Group of people"],
        [0.810, 0.131, 0.037, 0.011, 0.006, 0.003, 0.002],
    ),
    "NAME_INCOME_TYPE": (
        ["Working", "Commercial associate", "Pensioner", "State servant",
         "Unemployed", "Student", "Businessman", "Maternity leave"],
        [0.516, 0.233, 0.180, 0.070, 0.0007, 0.0003, 0.0006, 0.0004],
    ),
    "NAME_EDUCATION_TYPE": (
        ["Secondary / secondary special", "Higher education",
         "Incomplete higher", "Lower secondary", "Academic degree"],
        [0.710, 0.243, 0.033, 0.0135, 0.0005],
    ),
    "NAME_FAMILY_STATUS": (
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        [0.639, 0.148, 0.097, 0.064, 0.052],
    ),
    "NAME_HOUSING_TYPE": (
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        [0.887, 0.048, 0.036, 0.016, 0.0085, 0.0045],
    ),
    "OCCUPATION_TYPE": (
        ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
         "High skill tech staff", "Accountants", "Medicine staff",
         "Security staff", "Cooking staff", "Cleaning staff",
         "Private service staff", "Low-skill Laborers", "Waiters/barmen staff",
         "Secretaries", "Realty agents", "HR staff", "IT staff"],
        [0.261, 0.152, 0.130, 0.101, 0.088, 0.054, 0.048, 0.041,
         0.032, 0.026, 0.021, 0.012, 0.010, 0.006, 0.006, 0.004, 0.004, 0.004],
    ),
    "WEEKDAY_APPR_PROCESS_START": (
        ["TUESDAY", "WEDNESDAY", "MONDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
        [0.170, 0.167, 0.166, 0.162, 0.162, 0.109, 0.064],
    ),
    "ORGANIZATION_TYPE": (
        ["Business Entity Type 3", "XNA", "Self-employed", "Other",
         "Medicine", "Government", "Business Entity Type 2", "School",
         "Trade: type 7", "Kindergarten", "Construction", "Business Entity Type 1",
         "Transport: type 4", "Trade: type 3", "Industry: type 9",
         "Industry: type 3", "Security", "Housing", "Military", "Bank"],
        [0.220, 0.180, 0.125, 0.054, 0.037, 0.034, 0.034, 0.029,
         0.026, 0.022, 0.021, 0.019, 0.017, 0.011, 0.011,
         0.011, 0.011, 0.010, 0.009, 0.009],
    ),
    "FONDKAPREMONT_MODE": (
        ["reg oper account", "reg oper spec account", "not specified", "org spec account"],
        [0.735, 0.126, 0.077, 0.062],
    ),
    "HOUSETYPE_MODE": (
        ["block of flats", "specific housing", "terraced house"],
        [0.985, 0.010, 0.005],
    ),
    "WALLSMATERIAL_MODE": (
        ["Panel", "Stone, brick", "Block", "Wooden", "Mixed", "Monolithic", "Others"],
        [0.428, 0.400, 0.062, 0.037, 0.033, 0.026, 0.014],
    ),
    "EMERGENCYSTATE_MODE": (["No", "Yes"], [0.993, 0.007]),
}

# The 14 "building block" stems, each present as _AVG / _MODE / _MEDI.
_BUILDING_STEMS: list[str] = [
    "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
    "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
    "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA",
]

# Per-column null rates observed in the real application table.
_MISSING_RATES: dict[str, float] = {
    "EXT_SOURCE_1": 0.563,
    "EXT_SOURCE_2": 0.002,
    "EXT_SOURCE_3": 0.198,
    "OWN_CAR_AGE": 0.660,
    "OCCUPATION_TYPE": 0.313,
    "NAME_TYPE_SUITE": 0.004,
    "AMT_ANNUITY": 0.004,
    "AMT_GOODS_PRICE": 0.009,
    "CNT_FAM_MEMBERS": 0.001,
    "TOTALAREA_MODE": 0.482,
    "FONDKAPREMONT_MODE": 0.684,
    "HOUSETYPE_MODE": 0.501,
    "WALLSMATERIAL_MODE": 0.508,
    "EMERGENCYSTATE_MODE": 0.474,
    "OBS_30_CNT_SOCIAL_CIRCLE": 0.003,
    "DEF_30_CNT_SOCIAL_CIRCLE": 0.003,
    "OBS_60_CNT_SOCIAL_CIRCLE": 0.003,
    "DEF_60_CNT_SOCIAL_CIRCLE": 0.003,
    "DAYS_LAST_PHONE_CHANGE": 0.001,
    "AMT_REQ_CREDIT_BUREAU_HOUR": 0.135,
    "AMT_REQ_CREDIT_BUREAU_DAY": 0.135,
    "AMT_REQ_CREDIT_BUREAU_WEEK": 0.135,
    "AMT_REQ_CREDIT_BUREAU_MON": 0.135,
    "AMT_REQ_CREDIT_BUREAU_QRT": 0.135,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 0.135,
}


def _draw(rng: np.random.Generator, column: str, size: int) -> np.ndarray:
    """Sample ``size`` values for a categorical column from its real frequencies."""
    levels, probs = _CATEGORICAL_LEVELS[column]
    weights = np.asarray(probs, dtype=float)
    return rng.choice(levels, size=size, p=weights / weights.sum())


def _apply_missing(
    frame: pd.DataFrame, rng: np.random.Generator, rates: dict[str, float]
) -> pd.DataFrame:
    """Null out a random share of each named column, in place-ish.

    Applied *after* the target is drawn so that missingness never leaks label
    information -- exactly the assumption the preprocessing layer relies on.
    """
    for column, rate in rates.items():
        if column not in frame.columns or rate <= 0:
            continue
        mask = rng.random(len(frame)) < rate
        frame.loc[mask, column] = np.nan
    return frame


def _build_applications(rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
    """Construct the raw application feature block (no TARGET, no missingness)."""
    df = pd.DataFrame(index=pd.RangeIndex(n_rows))

    # ---------------------------------------------------------- identifiers --
    df["NAME_CONTRACT_TYPE"] = _draw(rng, "NAME_CONTRACT_TYPE", n_rows)
    df["CODE_GENDER"] = _draw(rng, "CODE_GENDER", n_rows)
    df["FLAG_OWN_CAR"] = _draw(rng, "FLAG_OWN_CAR", n_rows)
    df["FLAG_OWN_REALTY"] = _draw(rng, "FLAG_OWN_REALTY", n_rows)

    # ------------------------------------------------------------ household --
    df["CNT_CHILDREN"] = rng.choice(
        [0, 1, 2, 3, 4, 5], size=n_rows, p=[0.700, 0.190, 0.087, 0.019, 0.003, 0.001]
    )

    # ------------------------------------------------------------ financials --
    # Income is strongly right-skewed in the real data; a lognormal reproduces
    # both the median (~147k) and the long tail.
    df["AMT_INCOME_TOTAL"] = np.round(
        rng.lognormal(mean=np.log(147_000), sigma=0.55, size=n_rows) / 4500
    ) * 4500
    # Credit is anchored to income but with its own dispersion.
    credit_multiple = rng.lognormal(mean=np.log(3.9), sigma=0.55, size=n_rows)
    df["AMT_CREDIT"] = np.round(
        np.clip(df["AMT_INCOME_TOTAL"] * credit_multiple, 45_000, 4_050_000) / 4500
    ) * 4500
    # Goods price sits just below credit for cash loans.
    df["AMT_GOODS_PRICE"] = np.round(
        df["AMT_CREDIT"] * rng.uniform(0.78, 0.98, n_rows) / 4500
    ) * 4500
    # Annuity implies a 10-40 year notional amortisation.
    df["AMT_ANNUITY"] = np.round(
        df["AMT_CREDIT"] / rng.uniform(10, 40, n_rows) / 500
    ) * 500

    df["NAME_TYPE_SUITE"] = _draw(rng, "NAME_TYPE_SUITE", n_rows)
    df["NAME_INCOME_TYPE"] = _draw(rng, "NAME_INCOME_TYPE", n_rows)
    df["NAME_EDUCATION_TYPE"] = _draw(rng, "NAME_EDUCATION_TYPE", n_rows)
    df["NAME_FAMILY_STATUS"] = _draw(rng, "NAME_FAMILY_STATUS", n_rows)
    df["NAME_HOUSING_TYPE"] = _draw(rng, "NAME_HOUSING_TYPE", n_rows)
    df["REGION_POPULATION_RELATIVE"] = np.round(
        rng.lognormal(mean=np.log(0.0187), sigma=0.62, size=n_rows).clip(0.00029, 0.0725), 6
    )

    # ------------------------------------------------------------- day cols --
    # Home Credit stores every date as a NEGATIVE day offset from application.
    is_pensioner = df["NAME_INCOME_TYPE"].isin(["Pensioner"]).to_numpy()
    age_years = np.where(
        is_pensioner,
        rng.uniform(55, 69, n_rows),
        rng.uniform(21, 60, n_rows),
    )
    df["DAYS_BIRTH"] = -np.round(age_years * 365.25).astype(int)

    # Employment tenure, bounded so nobody is employed before they were born.
    max_tenure_days = (-df["DAYS_BIRTH"].to_numpy() - 18 * 365.25).clip(min=30)
    tenure = rng.lognormal(mean=np.log(1800), sigma=1.0, size=n_rows)
    df["DAYS_EMPLOYED"] = -np.minimum(tenure, max_tenure_days).round().astype(int)

    # The famous anomaly: pensioners and the unemployed carry the 365243
    # sentinel rather than a real tenure.  Reproduced verbatim.
    unemployed = df["NAME_INCOME_TYPE"].isin(["Unemployed"]).to_numpy()
    df.loc[is_pensioner | unemployed, "DAYS_EMPLOYED"] = DAYS_EMPLOYED_ANOMALY

    df["DAYS_REGISTRATION"] = -np.round(
        rng.uniform(0, 1.0, n_rows) ** 1.4 * (-df["DAYS_BIRTH"].to_numpy() - 6570)
    ).astype(float)
    df["DAYS_ID_PUBLISH"] = -rng.integers(0, 6000, n_rows)

    owns_car = (df["FLAG_OWN_CAR"] == "Y").to_numpy()
    df["OWN_CAR_AGE"] = np.where(owns_car, rng.integers(0, 35, n_rows).astype(float), np.nan)

    # ------------------------------------------------------------- contact ---
    df["FLAG_MOBIL"] = 1
    df["FLAG_EMP_PHONE"] = np.where(df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY, 0, 1)
    df["FLAG_WORK_PHONE"] = rng.binomial(1, 0.199, n_rows)
    df["FLAG_CONT_MOBILE"] = rng.binomial(1, 0.998, n_rows)
    df["FLAG_PHONE"] = rng.binomial(1, 0.281, n_rows)
    df["FLAG_EMAIL"] = rng.binomial(1, 0.057, n_rows)

    df["OCCUPATION_TYPE"] = _draw(rng, "OCCUPATION_TYPE", n_rows)
    df["CNT_FAM_MEMBERS"] = (df["CNT_CHILDREN"] + rng.choice([1, 2], n_rows, p=[0.28, 0.72])).astype(float)

    df["REGION_RATING_CLIENT"] = rng.choice([1, 2, 3], n_rows, p=[0.109, 0.739, 0.152])
    df["REGION_RATING_CLIENT_W_CITY"] = np.clip(
        df["REGION_RATING_CLIENT"] + rng.choice([-1, 0, 1], n_rows, p=[0.03, 0.94, 0.03]), 1, 3
    )
    df["WEEKDAY_APPR_PROCESS_START"] = _draw(rng, "WEEKDAY_APPR_PROCESS_START", n_rows)
    df["HOUR_APPR_PROCESS_START"] = np.clip(
        np.round(rng.normal(12.06, 3.27, n_rows)), 0, 23
    ).astype(int)

    # ------------------------------------------------- region mismatch flags --
    df["REG_REGION_NOT_LIVE_REGION"] = rng.binomial(1, 0.015, n_rows)
    df["REG_REGION_NOT_WORK_REGION"] = rng.binomial(1, 0.051, n_rows)
    df["LIVE_REGION_NOT_WORK_REGION"] = rng.binomial(1, 0.041, n_rows)
    df["REG_CITY_NOT_LIVE_CITY"] = rng.binomial(1, 0.078, n_rows)
    df["REG_CITY_NOT_WORK_CITY"] = rng.binomial(1, 0.230, n_rows)
    df["LIVE_CITY_NOT_WORK_CITY"] = rng.binomial(1, 0.179, n_rows)
    df["ORGANIZATION_TYPE"] = _draw(rng, "ORGANIZATION_TYPE", n_rows)

    # ------------------------------------------------------- external scores --
    # EXT_SOURCE_* are the single most predictive block in the real dataset.
    # Drawn here from a shared latent "creditworthiness" factor so that they are
    # correlated with each other, exactly as they are in reality.
    latent = rng.normal(0.0, 1.0, n_rows)
    for idx, (loc, scale, weight) in enumerate(
        [(0.502, 0.211, 0.62), (0.514, 0.191, 0.55), (0.511, 0.195, 0.58)], start=1
    ):
        noise = rng.normal(0.0, 1.0, n_rows)
        blended = weight * latent + np.sqrt(max(1.0 - weight**2, 1e-9)) * noise
        df[f"EXT_SOURCE_{idx}"] = np.clip(loc + blended * scale, 0.0005, 0.9995).round(6)

    # ------------------------------------------------------- building block --
    # ~50-70% missing in reality; generated from a per-row building quality
    # factor so the AVG/MODE/MEDI triplets stay mutually consistent.
    quality = rng.beta(2.0, 4.0, n_rows)
    for stem in _BUILDING_STEMS:
        base = np.clip(quality * rng.uniform(0.6, 1.4, n_rows), 0.0, 1.0)
        for suffix in ("AVG", "MODE", "MEDI"):
            jitter = rng.normal(0.0, 0.02, n_rows)
            df[f"{stem}_{suffix}"] = np.clip(base + jitter, 0.0, 1.0).round(4)

    df["FONDKAPREMONT_MODE"] = _draw(rng, "FONDKAPREMONT_MODE", n_rows)
    df["HOUSETYPE_MODE"] = _draw(rng, "HOUSETYPE_MODE", n_rows)
    df["TOTALAREA_MODE"] = np.clip(quality * rng.uniform(0.5, 1.5, n_rows), 0, 1).round(4)
    df["WALLSMATERIAL_MODE"] = _draw(rng, "WALLSMATERIAL_MODE", n_rows)
    df["EMERGENCYSTATE_MODE"] = _draw(rng, "EMERGENCYSTATE_MODE", n_rows)

    # -------------------------------------------------------- social circle --
    df["OBS_30_CNT_SOCIAL_CIRCLE"] = rng.poisson(1.42, n_rows).astype(float)
    df["DEF_30_CNT_SOCIAL_CIRCLE"] = rng.poisson(0.143, n_rows).astype(float)
    df["OBS_60_CNT_SOCIAL_CIRCLE"] = df["OBS_30_CNT_SOCIAL_CIRCLE"]
    df["DEF_60_CNT_SOCIAL_CIRCLE"] = rng.poisson(0.100, n_rows).astype(float)
    df["DAYS_LAST_PHONE_CHANGE"] = -rng.integers(0, 4200, n_rows).astype(float)

    # ------------------------------------------------------------ documents --
    # FLAG_DOCUMENT_3 is supplied by ~71% of applicants; the rest are rare.
    # Built as one block and concatenated: assigning 26 columns one at a time
    # fragments the frame and makes pandas complain.
    doc_rates = {3: 0.710, 6: 0.088, 8: 0.082, 16: 0.010, 18: 0.008}
    tail_block = {
        f"FLAG_DOCUMENT_{doc}": rng.binomial(1, doc_rates.get(doc, 0.0008), n_rows)
        for doc in range(2, 22)
    }

    # ------------------------------------------------- bureau enquiry counts --
    for window, rate in (
        ("HOUR", 0.006), ("DAY", 0.007), ("WEEK", 0.034),
        ("MON", 0.267), ("QRT", 0.265), ("YEAR", 1.900),
    ):
        tail_block[f"AMT_REQ_CREDIT_BUREAU_{window}"] = rng.poisson(rate, n_rows).astype(float)

    return pd.concat([df, pd.DataFrame(tail_block, index=df.index)], axis=1)


def _draw_target(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Sample ``TARGET`` from a logistic model over the generated features.

    Signal is injected through the same drivers that dominate the real dataset --
    external scores first, then age, leverage and education -- so a model fitted
    on the fixture recovers a believable feature ranking rather than noise.
    The intercept is solved numerically so the realised default rate lands on
    :data:`TARGET_DEFAULT_RATE`.
    """

    def z(values: np.ndarray) -> np.ndarray:
        """Standardise, treating NaN as the column mean."""
        arr = np.asarray(values, dtype=float)
        mean = np.nanmean(arr)
        std = np.nanstd(arr) or 1.0
        return np.nan_to_num((arr - mean) / std, nan=0.0)

    ext_mean = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1).to_numpy()
    age_years = -df["DAYS_BIRTH"].to_numpy() / 365.25
    credit_to_income = df["AMT_CREDIT"].to_numpy() / df["AMT_INCOME_TOTAL"].to_numpy()
    annuity_to_income = df["AMT_ANNUITY"].to_numpy() / df["AMT_INCOME_TOTAL"].to_numpy()

    # Negative weight => higher external score means lower default risk.
    logit = (
        -1.15 * z(ext_mean)
        - 0.34 * z(age_years)
        + 0.30 * z(credit_to_income)
        + 0.22 * z(annuity_to_income)
        + 0.18 * z(df["DAYS_ID_PUBLISH"].to_numpy())
        + 0.16 * z(df["REGION_RATING_CLIENT"].to_numpy())
        + 0.12 * z(df["DEF_30_CNT_SOCIAL_CIRCLE"].to_numpy())
        + 0.30 * (df["NAME_EDUCATION_TYPE"] == "Lower secondary").to_numpy()
        - 0.28 * (df["NAME_EDUCATION_TYPE"] == "Higher education").to_numpy()
        + 0.20 * (df["CODE_GENDER"] == "M").to_numpy()
        + 0.18 * (df["NAME_INCOME_TYPE"] == "Working").to_numpy()
        - 0.25 * (df["NAME_INCOME_TYPE"] == "Pensioner").to_numpy()
        + 0.14 * (df["NAME_CONTRACT_TYPE"] == "Revolving loans").to_numpy()
        + rng.normal(0.0, 0.45, len(df))  # irreducible noise -> realistic AUC
    )

    # Bisect on the intercept until the mean probability matches the target rate.
    low, high = -12.0, 6.0
    for _ in range(80):
        mid = (low + high) / 2.0
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(logit + mid)))))
        if rate < TARGET_DEFAULT_RATE:
            low = mid
        else:
            high = mid
    probability = 1.0 / (1.0 + np.exp(-(logit + (low + high) / 2.0)))
    return rng.binomial(1, probability).astype(int)


def _build_bureau(rng: np.random.Generator, applicant_ids: np.ndarray) -> pd.DataFrame:
    """Build the bureau table: 0-8 prior external credits per applicant.

    Roughly 14% of real applicants have no bureau history at all, which is why
    the loader must LEFT JOIN and tolerate nulls.
    """
    records: list[dict[str, object]] = []
    next_bureau_id = 5_000_000

    credit_counts = rng.poisson(2.9, len(applicant_ids)).clip(0, 8)
    has_history = rng.random(len(applicant_ids)) > 0.14
    credit_counts = np.where(has_history, credit_counts, 0)

    credit_types = [
        "Consumer credit", "Credit card", "Car loan", "Mortgage",
        "Microloan", "Loan for business development", "Another type of loan",
    ]
    credit_type_probs = [0.596, 0.256, 0.063, 0.036, 0.026, 0.012, 0.011]

    for applicant_id, count in zip(applicant_ids, credit_counts, strict=True):
        for _ in range(int(count)):
            days_credit = int(-rng.integers(1, 2900))
            is_active = rng.random() < 0.36
            duration = int(rng.integers(180, 2200))

            amt_sum = float(np.round(rng.lognormal(np.log(180_000), 1.1) / 1000) * 1000)
            if is_active:
                # Active loans still carry debt; closed ones are (almost) repaid.
                amt_debt = float(np.round(amt_sum * rng.uniform(0.05, 0.95) / 1000) * 1000)
                days_enddate_fact = np.nan
                days_credit_enddate = float(days_credit + duration)
            else:
                amt_debt = 0.0
                days_enddate_fact = float(days_credit + duration)
                days_credit_enddate = float(days_credit + duration)

            # Overdue is rare but is the strongest bureau-side risk signal.
            overdue_days = int(rng.integers(1, 180)) if rng.random() < 0.035 else 0
            max_overdue = (
                float(np.round(rng.lognormal(np.log(4_000), 1.3)))
                if rng.random() < 0.28
                else np.nan
            )

            records.append(
                {
                    "SK_ID_CURR": int(applicant_id),
                    "SK_ID_BUREAU": next_bureau_id,
                    "CREDIT_ACTIVE": "Active" if is_active else "Closed",
                    "CREDIT_CURRENCY": "currency 1",
                    "DAYS_CREDIT": days_credit,
                    "CREDIT_DAY_OVERDUE": overdue_days,
                    "DAYS_CREDIT_ENDDATE": days_credit_enddate,
                    "DAYS_ENDDATE_FACT": days_enddate_fact,
                    "AMT_CREDIT_MAX_OVERDUE": max_overdue,
                    "CNT_CREDIT_PROLONG": int(rng.random() < 0.008),
                    "AMT_CREDIT_SUM": amt_sum,
                    "AMT_CREDIT_SUM_DEBT": amt_debt,
                    "AMT_CREDIT_SUM_LIMIT": float(np.round(amt_sum * rng.uniform(0, 0.4) / 1000) * 1000),
                    "AMT_CREDIT_SUM_OVERDUE": float(overdue_days > 0) * np.round(amt_sum * 0.05),
                    "CREDIT_TYPE": str(rng.choice(credit_types, p=credit_type_probs)),
                    "DAYS_CREDIT_UPDATE": int(-rng.integers(0, 2500)),
                    "AMT_ANNUITY": (
                        float(np.round(amt_sum / rng.uniform(12, 60) / 100) * 100)
                        if rng.random() < 0.25
                        else np.nan
                    ),
                }
            )
            next_bureau_id += 1

    return pd.DataFrame.from_records(records)


def generate(output_dir: Path | None = None, seed: int | None = None) -> dict[str, Path]:
    """Generate all three fixture CSVs and write them to ``output_dir``.

    Args:
        output_dir: Destination directory.  Defaults to ``settings.sample_data_dir``.
        seed: RNG seed.  Defaults to ``settings.random_seed`` so the committed
            fixtures are byte-for-byte reproducible.

    Returns:
        Mapping of logical table name -> path written.
    """
    destination = Path(output_dir) if output_dir else settings.sample_data_dir
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed if seed is not None else settings.random_seed)

    total = N_TRAIN + N_TEST
    logger.info("Generating %d synthetic applications (%d train / %d test)", total, N_TRAIN, N_TEST)

    combined = _build_applications(rng, total)
    ids = pd.DataFrame(
        {"SK_ID_CURR": np.arange(100_001, 100_001 + total, dtype=int)}, index=combined.index
    )
    combined = pd.concat([ids, combined], axis=1)

    target = _draw_target(combined, rng)
    combined = _apply_missing(combined, rng, _MISSING_RATES)

    train = combined.iloc[:N_TRAIN].copy()
    train.insert(1, "TARGET", target[:N_TRAIN])
    test = combined.iloc[N_TRAIN:].copy().reset_index(drop=True)

    bureau = _build_bureau(rng, combined["SK_ID_CURR"].to_numpy())

    paths = {
        "application_train": destination / "application_train.csv",
        "application_test": destination / "application_test.csv",
        "bureau": destination / "bureau.csv",
    }
    train.to_csv(paths["application_train"], index=False)
    test.to_csv(paths["application_test"], index=False)
    bureau.to_csv(paths["bureau"], index=False)

    logger.info(
        "Wrote application_train=%s rows x %s cols (default rate %.2f%%)",
        f"{len(train):,}", train.shape[1], 100 * train["TARGET"].mean(),
    )
    logger.info("Wrote application_test=%s rows x %s cols", f"{len(test):,}", test.shape[1])
    logger.info(
        "Wrote bureau=%s rows x %s cols covering %s applicants",
        f"{len(bureau):,}", bureau.shape[1], f"{bureau['SK_ID_CURR'].nunique():,}",
    )
    anomaly_rate = 100 * (train["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).mean()
    logger.info("DAYS_EMPLOYED == %d anomaly present in %.1f%% of train rows",
                DAYS_EMPLOYED_ANOMALY, anomaly_rate)
    return paths


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    generate()
