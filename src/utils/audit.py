"""Append-only audit log for scoring decisions and data queries.

The brief lists "satisfy audit and regulatory requirements" as one of the
business needs this platform addresses, and in lending that is not a soft goal.
A credit decision has to be reconstructable long after it was made: which model
version scored the applicant, on what inputs, what it returned, and which
reasons were given. The same applies to the chat -- every question a user asks
becomes SQL against customer records, so what ran and what came back has to be
recoverable.

Design choices worth stating:

* **Append-only JSONL.** One self-describing record per line, never rewritten.
  A log you can edit is not an audit log, and JSONL survives partial writes:
  a truncated final line costs one record rather than the whole file.
* **Failures are recorded, not just successes.** A rejected query and a refused
  question are exactly the events an auditor asks about.
* **Logging never breaks the caller.** A full disk must not stop a lender
  scoring applications, so every write is guarded and a failure is reported to
  the application log instead of raised.
* **No raw applicant records.** Entries hold identifiers, model outputs and the
  reasons given -- not the underlying personal data, which already lives in the
  database under its own controls.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Event kinds. Kept as constants so a reader can enumerate what the log contains
# without grepping the codebase.
EVENT_PREDICTION: str = "prediction"
EVENT_QUERY: str = "nl_query"

# Serialised writes: Streamlit serves concurrent sessions, and interleaved
# partial lines would corrupt records rather than merely reorder them.
_WRITE_LOCK = threading.Lock()


def audit_log_path() -> Path:
    """Location of the audit log."""
    return settings.reports_dir / "audit_log.jsonl"


def _write(record: dict[str, Any]) -> None:
    """Append one record, never raising."""
    try:
        path = audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, ensure_ascii=False)
        with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as error:  # pragma: no cover - disk failure path
        logger.warning("Could not write audit record: %s", error)


def _now() -> str:
    """Current UTC timestamp in ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def record_prediction(
    applicant_id: object,
    probability: float,
    risk_score: float,
    risk_band: str,
    decision: str,
    model_name: str,
    threshold: float,
    reasons: list[dict[str, Any]] | None = None,
    actor: str | None = None,
) -> None:
    """Log a scoring decision.

    Args:
        applicant_id: Identifier of the applicant scored.
        probability: Calibrated probability of default.
        risk_score: Score on the configured scale.
        risk_band: Assigned band.
        decision: Recommended action.
        model_name: Which model produced it.
        threshold: The decision threshold in force at the time.
        reasons: Top contributions, so the explanation given is recoverable.
        actor: Who or what requested the score.
    """
    _write(
        {
            "timestamp": _now(),
            "event": EVENT_PREDICTION,
            "actor": actor or os.getenv("APP_ACTOR", "ui"),
            "applicant_id": applicant_id,
            "model": model_name,
            "probability_of_default": round(float(probability), 6),
            "risk_score": float(risk_score),
            "risk_band": risk_band,
            "decision": decision,
            "decision_threshold": float(threshold),
            # The reasons matter as much as the score: an adverse decision has
            # to be explainable after the fact, not only at the moment it is made.
            "reasons": [
                {
                    "feature": reason.get("feature"),
                    "label": reason.get("label"),
                    "value": reason.get("display_value", reason.get("value")),
                    "direction": reason.get("direction"),
                    "contribution": reason.get("contribution"),
                }
                for reason in (reasons or [])[:5]
            ],
        }
    )


def record_query(
    question: str,
    sql: str,
    success: bool,
    row_count: int = 0,
    error: str = "",
    refused: bool = False,
    tables: list[str] | None = None,
    elapsed_seconds: float = 0.0,
    tokens: int = 0,
    summary_source: str = "",
    prompt_version: str = "",
    actor: str | None = None,
) -> None:
    """Log a natural-language question and the SQL it became.

    Rejections and refusals are logged too: a validator refusal is a security
    event, and a refusal to answer is evidence the grounding controls worked.
    """
    _write(
        {
            "timestamp": _now(),
            "event": EVENT_QUERY,
            "actor": actor or os.getenv("APP_ACTOR", "ui"),
            "question": question,
            "sql": sql,
            "success": bool(success),
            "refused": bool(refused),
            "row_count": int(row_count),
            "error": error,
            "tables": tables or [],
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "tokens": int(tokens),
            "summary_source": summary_source,
            "prompt_version": prompt_version,
        }
    )


def read_events(limit: int = 200, event: str | None = None) -> list[dict[str, Any]]:
    """Read the most recent audit records, newest first.

    Args:
        limit: Maximum records to return.
        event: Restrict to one event kind.

    Returns:
        Parsed records. Malformed lines are skipped rather than raising, so a
        truncated write cannot make the whole log unreadable.
    """
    path = audit_log_path()
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event is None or record.get("event") == event:
                    records.append(record)
    except OSError as error:  # pragma: no cover
        logger.warning("Could not read audit log: %s", error)
        return []

    return list(reversed(records))[:limit]


def iter_events() -> Iterator[dict[str, Any]]:
    """Stream every record in write order, for bulk export."""
    path = audit_log_path()
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def summarise() -> dict[str, Any]:
    """Counts and health of the audit log, for the UI."""
    predictions = 0
    queries = 0
    refused = 0
    rejected = 0
    first: str | None = None
    last: str | None = None

    for record in iter_events():
        timestamp = record.get("timestamp")
        if timestamp:
            first = first or timestamp
            last = timestamp
        if record.get("event") == EVENT_PREDICTION:
            predictions += 1
        elif record.get("event") == EVENT_QUERY:
            queries += 1
            if record.get("success"):
                continue
            # Each unanswered query counts once. A refusal is the model
            # declining to answer; a rejection is the validator or the database
            # refusing to run what it produced. They are different events and an
            # auditor cares about the difference.
            if record.get("refused"):
                refused += 1
            else:
                rejected += 1

    return {
        "predictions_logged": predictions,
        "queries_logged": queries,
        "queries_answered": queries - refused - rejected,
        "queries_refused": refused,
        "queries_rejected": rejected,
        "first_event": first,
        "last_event": last,
        "path": str(audit_log_path()),
    }
