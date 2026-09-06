"""Runtime environment helpers: data-path resolution and DB connectivity.

These utilities exist because the app must behave identically in three places --
a developer's laptop, a Docker container, and CI -- where the data directory and
the database host differ.  Everything that differs is resolved here rather than
being sprinkled through the codebase.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.config import DataMode, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Filenames are identical in sample and real mode -- only the directory changes,
# which is what lets one code path serve both.
TABLE_FILENAMES: dict[str, str] = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
}

# Tables the pipeline can run without. Their absence is reported and the
# corresponding features are skipped, rather than aborting the run -- the
# committed sample fixtures only cover application and bureau.
OPTIONAL_TABLES: frozenset[str] = frozenset({"previous_application", "installments_payments"})


def running_in_docker() -> bool:
    """Best-effort detection of whether we are inside a container.

    Used only for log messages and for choosing a sensible default Postgres
    host; never for behavioural branching that would make local and container
    runs diverge.
    """
    if os.getenv("RUNNING_IN_DOCKER", "").lower() in {"1", "true", "yes"}:
        return True
    if Path("/.dockerenv").exists():
        return True
    try:
        return "docker" in Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False


def resolve_data_path(table: str) -> Path:
    """Return the CSV path for a logical table under the active data mode.

    Args:
        table: One of the keys of :data:`TABLE_FILENAMES`.

    Returns:
        Absolute path to the CSV.

    Raises:
        KeyError: If ``table`` is not a known table.
        FileNotFoundError: If the file is absent, with a message that tells the
            user exactly how to fix it for the mode they are in.
    """
    if table not in TABLE_FILENAMES:
        raise KeyError(f"Unknown table {table!r}. Known tables: {sorted(TABLE_FILENAMES)}")

    path = settings.active_data_dir / TABLE_FILENAMES[table]
    if not path.exists():
        if settings.data_mode is DataMode.SAMPLE:
            hint = "Regenerate the fixtures with: python -m src.data.generate_sample"
        else:
            hint = (
                f"Place the Home Credit CSVs in {settings.data_dir} "
                "(or set DATA_MODE=sample to use the bundled fixtures)."
            )
        raise FileNotFoundError(f"Missing {table} at {path}. {hint}")
    return path


def data_availability() -> dict[str, bool]:
    """Report which of the known tables are present, without raising."""
    availability: dict[str, bool] = {}
    for table, filename in TABLE_FILENAMES.items():
        availability[table] = (settings.active_data_dir / filename).exists()
    return availability


def host_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to ``host:port`` succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def postgres_available() -> bool:
    """Check whether the configured Postgres server is accepting connections."""
    if settings.use_sqlite_fallback:
        return False
    return host_reachable(settings.postgres_host, settings.postgres_port)


def get_engine(readonly: bool = False, **kwargs: object) -> Engine:
    """Create a SQLAlchemy engine for the application or read-only role.

    Args:
        readonly: If True, connect as the restricted role used by the NL->SQL
            feature.  This is defence in depth -- the SQL validator is the first
            line, and the database privileges are the second.
        **kwargs: Extra keyword arguments forwarded to ``create_engine``.

    Returns:
        A configured :class:`~sqlalchemy.engine.Engine`.
    """
    url = settings.readonly_database_url if readonly else settings.database_url
    options: dict[str, object] = {"pool_pre_ping": True, "future": True}

    if url.startswith("postgresql"):
        # Enforce the statement timeout server-side so a pathological generated
        # query cannot pin a connection open.
        options["connect_args"] = {
            "connect_timeout": 10,
            "options": f"-c statement_timeout={settings.sql_timeout_seconds * 1000}",
        }
    options.update(kwargs)
    logger.debug("Creating %s engine (readonly=%s)", url.split("://", 1)[0], readonly)
    return create_engine(url, **options)


@contextmanager
def engine_scope(readonly: bool = False) -> Iterator[Engine]:
    """Context manager that disposes the engine's pool on exit."""
    engine = get_engine(readonly=readonly)
    try:
        yield engine
    finally:
        engine.dispose()


def check_database_connection(readonly: bool = False) -> tuple[bool, str]:
    """Ping the database and report the outcome without raising.

    Returns:
        ``(ok, message)`` -- the message is safe to show in the UI and never
        contains the password, since only the exception's own text is included.
    """
    try:
        with engine_scope(readonly=readonly) as engine, engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        backend = "SQLite" if settings.use_sqlite_fallback else "PostgreSQL"
        return True, f"{backend} connection OK (readonly={readonly})"
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        return False, f"Database unavailable: {type(exc).__name__}: {exc}"


def describe_runtime() -> dict[str, object]:
    """Snapshot of the effective runtime configuration, for logs and the UI."""
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "in_docker": running_in_docker(),
        "data_mode": settings.data_mode.value,
        "active_data_dir": str(settings.active_data_dir),
        "data_files_present": data_availability(),
        "db_backend": "sqlite" if settings.use_sqlite_fallback else "postgresql",
        "db_host": settings.postgres_host,
        "db_port": settings.postgres_port,
        "llm_provider": settings.llm_provider.value,
        "llm_enabled": settings.llm_enabled,
    }
