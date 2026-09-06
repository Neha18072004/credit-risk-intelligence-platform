"""Safe SQL execution and grounded natural-language summarisation.

Execution happens on a **read-only database connection** with a server-side
statement timeout. That is the second of three defences: the validator refuses
unsafe SQL, the database role cannot write even if asked, and the timeout bounds
the damage a pathological-but-valid query could do.

Summarisation is where hallucination is easiest and most damaging -- a fluent
sentence containing an invented number is worse than no answer. Two controls
apply: the model receives only the returned rows and is instructed to use
nothing else, and if no model is available the runner still answers, using a
deterministic summary generated from the rows themselves.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Final

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.talk_to_data.llm_client import LLMClient, LLMUnavailableError, get_llm_client
from src.talk_to_data.prompt_templates import build_summary_prompt
from src.talk_to_data.sql_validator import SQLValidator
from src.utils.docker_utils import get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Rows rendered into the summarisation prompt. Beyond this the table is
# truncated: a long result costs tokens without improving the summary, and the
# full result is always shown to the user in the UI regardless.
MAX_ROWS_IN_PROMPT: int = 25


@dataclass
class QueryResult:
    """Outcome of executing one validated statement."""

    success: bool
    sql: str = ""
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    row_count: int = 0
    answer: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    truncated: bool = False
    summary_source: str = ""  # "llm" or "deterministic"

    def __bool__(self) -> bool:
        return self.success


def fetch_schema(engine: Engine | None = None) -> dict[str, set[str]]:
    """Read the live table and column names from the database.

    The whitelist is built from the real schema rather than a hardcoded list, so
    a table added or a column renamed is reflected immediately and cannot drift
    into a false rejection or, worse, a false acceptance.

    Args:
        engine: Engine to inspect; the read-only one is used by default.

    Returns:
        Mapping of lower-cased table name to its set of lower-cased columns.
    """
    target = engine or get_engine(readonly=True)
    inspector = inspect(target)
    schema: dict[str, set[str]] = {}
    for table in inspector.get_table_names():
        schema[table.lower()] = {
            column["name"].lower() for column in inspector.get_columns(table)
        }
    logger.debug("Live schema: %d tables", len(schema))
    return schema


def execute_sql(sql: str, engine: Engine | None = None) -> QueryResult:
    """Execute an already-validated statement on the read-only connection.

    Args:
        sql: SQL that has passed :class:`~src.talk_to_data.sql_validator.SQLValidator`.
        engine: Engine override; the read-only engine is used by default.

    Returns:
        A :class:`QueryResult`. Database errors are captured rather than raised,
        because a failed query is a normal conversational outcome that the UI
        should render as a message.
    """
    target = engine or get_engine(readonly=True)
    started = time.perf_counter()
    try:
        with target.connect() as connection:
            frame = pd.read_sql_query(text(sql), connection)
    except SQLAlchemyError as error:
        elapsed = time.perf_counter() - started
        # The driver's message can carry connection details; keep only the
        # database's own text.
        message = str(getattr(error, "orig", error)).strip().splitlines()[0]
        logger.warning("Query failed after %.2fs: %s", elapsed, message)
        return QueryResult(success=False, sql=sql, error=message, elapsed_seconds=elapsed)

    elapsed = time.perf_counter() - started
    logger.info("Query returned %d rows in %.3fs", len(frame), elapsed)
    return QueryResult(
        success=True, sql=sql, rows=frame, row_count=len(frame), elapsed_seconds=elapsed
    )


def _format_cell(value: Any) -> str:
    """Render one value for the prompt table.

    Floats are trimmed to four significant figures. Handing the model 14 decimal
    places invites it to quote them back verbatim, which reads as false
    precision in an answer meant for a business reader.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NULL"
    if isinstance(value, (int, bool)) or (hasattr(value, "dtype") and "int" in str(getattr(value, "dtype", ""))):
        return f"{value:,}" if isinstance(value, int) and abs(value) >= 1000 else str(value)
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)


def rows_to_markdown(frame: pd.DataFrame, max_rows: int = MAX_ROWS_IN_PROMPT) -> tuple[str, bool]:
    """Render a result set as a markdown table for the summarisation prompt.

    Written by hand rather than via ``DataFrame.to_markdown`` so the project
    does not take a dependency on ``tabulate`` for one call, and so float
    formatting stays under this module's control.

    Returns:
        ``(markdown, truncated)``.
    """
    if frame.empty:
        return "(no rows returned)", False

    truncated = len(frame) > max_rows
    shown = frame.head(max_rows)

    headers = [str(column) for column in shown.columns]
    body = [[_format_cell(value) for value in row] for row in shown.itertuples(index=False)]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in body)) if body else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths)) + " |",
        "|-" + "-|-".join("-" * width for width in widths) + "-|",
    ]
    lines += [
        "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"
        for row in body
    ]

    table = "\n".join(lines)
    if truncated:
        table += f"\n\n(showing the first {max_rows} of {len(frame)} rows)"
    return table, truncated


def deterministic_summary(question: str, frame: pd.DataFrame) -> str:
    """Describe a result set without a model.

    Used when no LLM is available, and as the fallback when one errors. It is
    deliberately plain: it reports shape and the leading values and claims
    nothing beyond them, so the feature degrades into something still useful
    rather than into an error page.
    """
    if frame.empty:
        return "The query ran successfully but returned no rows."

    if frame.shape == (1, 1):
        value = frame.iloc[0, 0]
        rendered = f"{value:,.4g}" if isinstance(value, (int, float)) else value
        return f"{frame.columns[0].replace('_', ' ')}: {rendered}."

    parts = [
        f"Returned {len(frame):,} row{'s' if len(frame) != 1 else ''} "
        f"with {frame.shape[1]} column{'s' if frame.shape[1] != 1 else ''} "
        f"({', '.join(frame.columns[:6])}"
        f"{', ...' if frame.shape[1] > 6 else ''})."
    ]
    if len(frame) > 1:
        first = frame.iloc[0]
        highlights = ", ".join(
            f"{column} = {first[column]:,.4g}"
            if isinstance(first[column], (int, float)) else f"{column} = {first[column]}"
            for column in frame.columns[:3]
        )
        parts.append(f"The first row shows {highlights}.")
    parts.append("Full results are shown in the table below.")
    return " ".join(parts)


# Numbers this small are ordinals and counts a summary may legitimately use
# ("the top 3 groups"), so they are not treated as unsupported figures.
_TRIVIAL_NUMBER_CEILING: Final[float] = 10.0

# Relative tolerance when matching a quoted figure against the data. Generous
# enough to allow sensible re-rounding (8.53 -> 8.5), tight enough that an
# invented number does not slip through.
_MATCH_RELATIVE_TOLERANCE: Final[float] = 0.02

_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> list[float]:
    """Extract every numeric literal from a block of prose."""
    found: list[float] = []
    for match in _NUMBER_PATTERN.finditer(text):
        try:
            found.append(float(match.group().replace(",", "")))
        except ValueError:  # pragma: no cover - regex already constrains this
            continue
    return found


def _supported_values(frame: pd.DataFrame) -> set[float]:
    """Every number a grounded summary is allowed to quote.

    That is the numeric cell values, plus any numbers embedded in *text* cells,
    plus the row count, plus a rounded form of each so that re-rounding 8.53 to
    8.5 is not flagged as invention.

    Text cells matter more than they look. A banded query returns labels like
    ``"1. Very low (<0.3)"``, and a summary quoting that boundary is perfectly
    grounded -- scanning only the numeric columns flagged those as fabricated.
    """
    values: set[float] = {float(len(frame))}

    def remember(number: float) -> None:
        values.add(number)
        values.add(round(number, 1))
        values.add(round(number))

    for column in frame.columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        for value in numeric.dropna():
            remember(float(value))
        # Numbers inside category labels are part of the result too.
        for value in series[numeric.isna()].dropna().astype(str):
            for embedded in _numbers_in(value):
                remember(embedded)
    return values


# Phrases that assert an empty result. If a summary uses one while rows exist,
# it is contradicting the data it was given.
_EMPTY_CLAIM_PATTERNS: Final[tuple[str, ...]] = (
    "no rows", "no results", "no data", "no records", "no applicants match",
    "nothing matched", "did not return any", "returned none", "empty result",
)


def verify_summary_grounding(answer: str, frame: pd.DataFrame) -> tuple[bool, list[float]]:
    """Check that every figure quoted in a summary actually occurs in the data.

    This is the last line of hallucination control, and the only one that
    inspects the model's *prose* rather than its SQL. A fluent sentence carrying
    an invented statistic is the most damaging failure this feature can produce,
    because it looks exactly like a correct answer -- and it is not something
    the SQL validator can catch, since the query itself was perfectly valid.

    Observed in testing: a small local model answered a question about default
    rates by computing "approximately 0.08% of the total sample size
    (2,854 + 50 + 153 + 942 = 3,999)". Every input was real; the arithmetic and
    the conclusion were invented. Both fabricated figures are caught here.

    Args:
        answer: The model's natural-language summary.
        frame: The rows it was given.

    Returns:
        ``(is_grounded, unsupported_numbers)``.
    """
    if frame.empty:
        return True, []

    # A summary claiming the result was empty when it was not is ungrounded even
    # though it quotes no figures at all. Observed in testing: a model answered
    # "No rows matched." over a two-row result.
    lowered = answer.lower()
    if any(phrase in lowered for phrase in _EMPTY_CLAIM_PATTERNS):
        logger.warning("Summary claimed an empty result over %d rows", len(frame))
        return False, []

    supported = _supported_values(frame)
    unsupported: list[float] = []

    for number in _numbers_in(answer):
        if abs(number) <= _TRIVIAL_NUMBER_CEILING and float(number).is_integer():
            continue
        tolerance = max(abs(number) * _MATCH_RELATIVE_TOLERANCE, 0.01)
        if not any(abs(number - candidate) <= tolerance for candidate in supported):
            unsupported.append(number)

    return not unsupported, unsupported


def summarise_result(
    question: str,
    sql: str,
    frame: pd.DataFrame,
    client: LLMClient | None = None,
) -> tuple[str, str]:
    """Produce a natural-language answer grounded strictly in the returned rows.

    Args:
        question: The original question.
        sql: The SQL that produced the rows.
        frame: The result set.
        client: LLM client; the configured one is used by default.

    Returns:
        ``(answer, source)`` where source is ``"llm"`` or ``"deterministic"``.
    """
    if frame.empty:
        return (
            "The query ran successfully but no rows matched those criteria.",
            "deterministic",
        )

    llm = client or get_llm_client()
    available, _ = llm.is_available()
    if not available:
        return deterministic_summary(question, frame), "deterministic"

    markdown, _ = rows_to_markdown(frame)
    system_prompt, user_prompt = build_summary_prompt(question, sql, markdown, len(frame))
    try:
        response = llm.complete(system_prompt, user_prompt)
    except (LLMUnavailableError, Exception) as error:  # noqa: BLE001
        logger.warning("Summarisation failed, using deterministic fallback: %s", error)
        return deterministic_summary(question, frame), "deterministic"

    answer = response.text.strip()
    if not answer:
        return deterministic_summary(question, frame), "deterministic"

    grounded, unsupported = verify_summary_grounding(answer, frame)
    if not grounded:
        logger.warning(
            "Summary quoted %d figure(s) absent from the result set (%s); "
            "discarding it for the deterministic summary",
            len(unsupported), ", ".join(f"{value:g}" for value in unsupported[:5]),
        )
        return (
            f"{deterministic_summary(question, frame)} "
            "(The generated summary was discarded because it referenced figures "
            "that do not appear in the result.)",
            "deterministic_after_ungrounded",
        )
    return answer, "llm"


def run_validated_query(
    sql: str,
    question: str,
    validator: SQLValidator,
    engine: Engine | None = None,
    summarise: bool = True,
    client: LLMClient | None = None,
) -> QueryResult:
    """Validate, execute and summarise in one call.

    Args:
        sql: Candidate SQL, typically LLM-generated.
        question: The question it answers, used for the summary.
        validator: The configured validator.
        engine: Engine override.
        summarise: Generate a natural-language answer.
        client: LLM client override.

    Returns:
        The complete :class:`QueryResult`.
    """
    validation = validator.validate(sql)
    if not validation.is_valid:
        return QueryResult(success=False, sql=sql, error=validation.reason)

    result = execute_sql(validation.sql, engine=engine)
    if not result.success:
        return result

    if summarise:
        result.answer, result.summary_source = summarise_result(
            question, validation.sql, result.rows, client=client
        )
    _, result.truncated = rows_to_markdown(result.rows)
    return result
