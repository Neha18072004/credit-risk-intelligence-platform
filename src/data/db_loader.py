"""Populate the analytics database from the CSV source files.

Builds the curated tables described in ``sql/schema.sql``. The loader is
deliberately backend-agnostic -- it runs against PostgreSQL (the default, and
what docker-compose brings up) and against the SQLite fallback -- because the
talk-to-data tests must be runnable without a database server.

Run it with::

    python -m src.data.db_loader            # load application + bureau
    python -m src.data.db_loader --predict  # also score and load predictions
"""

from __future__ import annotations

import argparse
from typing import Final

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

from src.data.loader import build_dataset, load_bureau
from src.data.preprocessor import clean_applications, engineer_features
from src.utils.config import settings
from src.utils.docker_utils import get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Curated application columns. Chosen because analysts actually ask about them,
# and because a narrow table keeps the schema prompt small -- every column here
# costs tokens on every chat turn and widens the surface for a hallucinated
# column name.
APPLICATION_COLUMNS: Final[dict[str, str]] = {
    "SK_ID_CURR": "sk_id_curr",
    "TARGET": "target",
    "NAME_CONTRACT_TYPE": "name_contract_type",
    "CODE_GENDER": "code_gender",
    "FLAG_OWN_CAR": "flag_own_car",
    "FLAG_OWN_REALTY": "flag_own_realty",
    "CNT_CHILDREN": "cnt_children",
    "CNT_FAM_MEMBERS": "cnt_fam_members",
    "AMT_INCOME_TOTAL": "amt_income_total",
    "AMT_CREDIT": "amt_credit",
    "AMT_ANNUITY": "amt_annuity",
    "AMT_GOODS_PRICE": "amt_goods_price",
    "NAME_INCOME_TYPE": "name_income_type",
    "NAME_EDUCATION_TYPE": "name_education_type",
    "NAME_FAMILY_STATUS": "name_family_status",
    "NAME_HOUSING_TYPE": "name_housing_type",
    "OCCUPATION_TYPE": "occupation_type",
    "ORGANIZATION_TYPE": "organization_type",
    "REGION_RATING_CLIENT": "region_rating_client",
    "AGE_YEARS": "age_years",
    "EMPLOYED_YEARS": "employed_years",
    "DAYS_EMPLOYED_ANOMALY": "days_employed_anomaly",
    "EXT_SOURCE_1": "ext_source_1",
    "EXT_SOURCE_2": "ext_source_2",
    "EXT_SOURCE_3": "ext_source_3",
    "EXT_SOURCE_MEAN": "ext_source_mean",
    "CREDIT_TO_INCOME_RATIO": "credit_to_income_ratio",
    "ANNUITY_TO_INCOME_RATIO": "annuity_to_income_ratio",
    "PAYMENT_RATE": "payment_rate",
    "CREDIT_ENQUIRY_TOTAL": "credit_enquiry_total",
}

BUREAU_COLUMNS: Final[dict[str, str]] = {
    "SK_ID_BUREAU": "sk_id_bureau",
    "SK_ID_CURR": "sk_id_curr",
    "CREDIT_ACTIVE": "credit_active",
    "CREDIT_TYPE": "credit_type",
    "DAYS_CREDIT": "days_credit",
    "CREDIT_DAY_OVERDUE": "credit_day_overdue",
    "AMT_CREDIT_SUM": "amt_credit_sum",
    "AMT_CREDIT_SUM_DEBT": "amt_credit_sum_debt",
    "AMT_CREDIT_SUM_OVERDUE": "amt_credit_sum_overdue",
    "CNT_CREDIT_PROLONG": "cnt_credit_prolong",
}

BUREAU_SUMMARY_COLUMNS: Final[dict[str, str]] = {
    "SK_ID_CURR": "sk_id_curr",
    "BUREAU_LOAN_COUNT": "bureau_loan_count",
    "BUREAU_ACTIVE_COUNT": "bureau_active_count",
    "BUREAU_CLOSED_COUNT": "bureau_closed_count",
    "BUREAU_CREDIT_SUM_TOTAL": "bureau_credit_sum_total",
    "BUREAU_DEBT_TOTAL": "bureau_debt_total",
    "BUREAU_DEBT_CREDIT_RATIO": "bureau_debt_credit_ratio",
    "BUREAU_DAYS_OVERDUE_MAX": "bureau_days_overdue_max",
    "BUREAU_OVERDUE_LOAN_COUNT": "bureau_overdue_loan_count",
    "BUREAU_HAS_OVERDUE": "bureau_has_overdue",
    "BUREAU_HAS_HISTORY": "bureau_has_history",
}

TABLE_NAMES: Final[tuple[str, ...]] = (
    "applications", "bureau", "bureau_summary", "predictions",
)


def _select_and_rename(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Project a frame onto the curated columns, renaming to snake_case.

    Columns absent from the source become NULL rather than raising, so a
    partially-populated real dataset still loads.
    """
    available = {source: target for source, target in mapping.items() if source in frame.columns}
    missing = set(mapping) - set(available)
    if missing:
        logger.warning("Source is missing %d expected columns: %s", len(missing), sorted(missing))

    projected = frame[list(available)].rename(columns=available)
    for source in missing:
        projected[mapping[source]] = None
    return projected[list(mapping.values())]


def build_tables(include_predictions: bool = False) -> dict[str, pd.DataFrame]:
    """Assemble every analytics table in memory.

    Args:
        include_predictions: Also score the applicants with the trained model.
            Skipped silently if no model has been trained yet.

    Returns:
        Mapping of table name to dataframe.
    """
    joined = build_dataset("train")
    # Engineered columns (AGE_YEARS, EXT_SOURCE_MEAN, ratios) are materialised
    # into the table so that a question like "average age of defaulters" maps
    # onto a real column instead of requiring the LLM to invent the arithmetic.
    enriched = engineer_features(clean_applications(joined))

    # bureau_summary is projected from the *joined* frame, not from
    # aggregate_bureau alone, so that thin-file applicants appear with
    # bureau_has_history = 0 instead of being absent. Otherwise a question like
    # "how many applicants have no credit history?" would be unanswerable from
    # the table that most obviously ought to answer it.
    tables: dict[str, pd.DataFrame] = {
        "applications": _select_and_rename(enriched, APPLICATION_COLUMNS),
        "bureau": _select_and_rename(load_bureau(), BUREAU_COLUMNS),
        "bureau_summary": _select_and_rename(joined, BUREAU_SUMMARY_COLUMNS),
    }

    if include_predictions:
        tables["predictions"] = _build_predictions(joined)

    for name, frame in tables.items():
        logger.info("Prepared %-16s %s rows x %s cols", name, f"{len(frame):,}", frame.shape[1])
    return tables


def _build_predictions(joined: pd.DataFrame) -> pd.DataFrame:
    """Score every applicant with the trained model, if one exists."""
    from src.ml.predict import artifacts_exist, score_frame

    if not artifacts_exist():
        logger.warning("No trained model found; predictions table will be empty")
        return pd.DataFrame(
            columns=["sk_id_curr", "probability_of_default", "risk_score", "risk_band", "decision"]
        )

    scored = score_frame(joined)
    scored = scored.rename(columns={"SK_ID_CURR": "sk_id_curr"})
    scored["risk_band"] = scored["risk_band"].astype(str)
    return scored[
        ["sk_id_curr", "probability_of_default", "risk_score", "risk_band", "decision"]
    ]


def create_schema(engine: Engine) -> None:
    """Create the analytics tables, dropping any previous version.

    The DDL in ``sql/schema.sql`` is PostgreSQL-flavoured. Rather than maintain
    a second SQLite dialect of it, the tables are created from the same column
    definitions via pandas on the SQLite fallback path -- the fallback exists so
    tests can run without a server, not as a supported deployment target.
    """
    if engine.dialect.name == "sqlite":
        logger.debug("SQLite backend: tables are created by the write step")
        return

    from src.utils.config import PROJECT_ROOT

    statements = (PROJECT_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    with engine.begin() as connection:
        connection.execute(text(statements))
    logger.info("Created schema (%s)", engine.dialect.name)


def write_tables(tables: dict[str, pd.DataFrame], engine: Engine, chunk_size: int = 1000) -> None:
    """Write each dataframe into its table.

    Args:
        tables: Output of :func:`build_tables`.
        engine: Target database engine.
        chunk_size: Rows per INSERT batch.
    """
    for name, frame in tables.items():
        # Pandas maps NaN to NULL, but numpy NaN in an integer-typed column
        # raises on insert, so normalise first.
        cleaned = frame.replace({np.nan: None})
        cleaned.to_sql(
            name, engine, if_exists="replace" if engine.dialect.name == "sqlite" else "append",
            index=False, chunksize=chunk_size, method="multi",
        )
        logger.info("Loaded %-16s %s rows", name, f"{len(cleaned):,}")


def grant_readonly_access(engine: Engine) -> None:
    """Re-grant SELECT to the read-only role on the freshly created tables.

    The compose init script sets default privileges on first start, which covers
    tables created afterwards. This is the belt-and-braces pass: the loader
    DROPs and recreates tables on every run, and running the grant here means
    the read-only role works even against a database that was provisioned
    without the init script -- an externally managed instance, for example.

    Failures are logged, not raised: an insufficiently privileged app role is a
    deployment choice, not a reason to abort the load.
    """
    if engine.dialect.name != "postgresql":
        return

    role = settings.postgres_readonly_user
    try:
        with engine.begin() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role}
            ).scalar()
            if not exists:
                logger.warning(
                    "Read-only role %r does not exist; the chat feature will fall back to "
                    "the application role. Create it with docker/init-readonly.sh.", role
                )
                return
            for statement in (
                f'GRANT USAGE ON SCHEMA public TO "{role}"',
                f'GRANT SELECT ON ALL TABLES IN SCHEMA public TO "{role}"',
            ):
                connection.execute(text(statement))
        logger.info("Granted SELECT on all tables to %r", role)
    except SQLAlchemyError as error:
        logger.warning("Could not grant read-only access to %r: %s", role, error)


def load_database(include_predictions: bool = True, engine: Engine | None = None) -> dict[str, int]:
    """Create the schema and load every table.

    Args:
        include_predictions: Score applicants and populate ``predictions``.
        engine: Target engine; the configured one is used if omitted.

    Returns:
        Row counts per table.
    """
    target = engine or get_engine()
    tables = build_tables(include_predictions=include_predictions)
    create_schema(target)
    write_tables(tables, target)
    grant_readonly_access(target)

    counts = {name: len(frame) for name, frame in tables.items()}
    logger.info("Database load complete: %s", counts)
    return counts


def table_row_counts(engine: Engine | None = None) -> dict[str, int]:
    """Return the current row count of every analytics table that exists."""
    target = engine or get_engine()
    existing = set(inspect(target).get_table_names())
    counts: dict[str, int] = {}
    with target.connect() as connection:
        for name in TABLE_NAMES:
            if name in existing:
                counts[name] = int(
                    connection.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one()
                )
    return counts


def main() -> None:  # pragma: no cover - CLI entry point
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Load the credit-risk analytics database.")
    parser.add_argument(
        "--no-predictions", action="store_true",
        help="Skip scoring applicants (use when no model has been trained yet).",
    )
    arguments = parser.parse_args()

    counts = load_database(include_predictions=not arguments.no_predictions)
    print("\nLoaded:")
    for name, count in counts.items():
        print(f"  {name:18s} {count:>8,} rows")


if __name__ == "__main__":  # pragma: no cover
    main()
