"""Tests for the audit trail.

The brief names "satisfy audit and regulatory requirements" as a business need,
which in lending means a decision must be reconstructable after the fact --
including the reasons that were given for it.
"""

from __future__ import annotations

import json

import pytest

from src.utils import audit


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    """Point the audit log at a temporary file for each test."""
    monkeypatch.setattr(audit, "audit_log_path", lambda: tmp_path / "audit.jsonl")
    return tmp_path / "audit.jsonl"


def test_prediction_is_recorded_with_its_reasons(isolated_log) -> None:
    """The reasons matter as much as the score: an adverse decision must be
    explainable later, not only at the moment it is made."""
    audit.record_prediction(
        applicant_id=100001, probability=0.36, risk_score=360.0, risk_band="High",
        decision="Refer for review", model_name="catboost", threshold=0.24,
        reasons=[
            {
                "feature": "EXT_SOURCE_MEAN", "label": "average external credit score",
                "display_value": "0.26", "direction": "increases risk", "contribution": 0.79,
            }
        ],
    )
    events = audit.read_events()
    assert len(events) == 1
    record = events[0]
    assert record["event"] == audit.EVENT_PREDICTION
    assert record["applicant_id"] == 100001
    assert record["risk_band"] == "High"
    assert record["decision_threshold"] == 0.24
    assert record["reasons"][0]["label"] == "average external credit score"
    assert record["timestamp"]


def test_query_success_and_failure_are_both_recorded(isolated_log) -> None:
    """A rejected query is a security event; a refusal is evidence the controls
    worked. Logging only successes would hide exactly what an auditor wants."""
    audit.record_query(question="q1", sql="SELECT 1", success=True, row_count=5, tables=["applications"])
    audit.record_query(question="q2", sql="", success=False, error="Unknown column", refused=False)
    audit.record_query(question="q3", sql="", success=False, refused=True, error="no such column")

    events = audit.read_events(event=audit.EVENT_QUERY)
    assert len(events) == 3
    summary = audit.summarise()
    # Each unanswered query counts once, and a refusal is distinguished from a
    # rejection: the model declining is not the same event as the validator
    # blocking what it produced.
    assert summary["queries_answered"] == 1
    assert summary["queries_rejected"] == 1
    assert summary["queries_refused"] == 1


def test_log_is_append_only(isolated_log) -> None:
    for index in range(5):
        audit.record_query(question=f"q{index}", sql="SELECT 1", success=True)
    lines = isolated_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    # Every line stands alone, so a truncated write costs one record, not the file.
    for line in lines:
        assert json.loads(line)["event"] == audit.EVENT_QUERY


def test_events_are_returned_newest_first(isolated_log) -> None:
    for index in range(3):
        audit.record_query(question=f"q{index}", sql="SELECT 1", success=True)
    assert [event["question"] for event in audit.read_events()] == ["q2", "q1", "q0"]


def test_malformed_line_does_not_break_reading(isolated_log) -> None:
    """A truncated final write must not make the whole log unreadable."""
    audit.record_query(question="good", sql="SELECT 1", success=True)
    with isolated_log.open("a", encoding="utf-8") as handle:
        handle.write('{"partial": tru\n')
    events = audit.read_events()
    assert len(events) == 1
    assert events[0]["question"] == "good"


def test_logging_never_raises(monkeypatch, tmp_path) -> None:
    """A full disk must not stop a lender scoring applications."""
    monkeypatch.setattr(audit, "audit_log_path", lambda: tmp_path / "nope" / "x.jsonl")
    monkeypatch.setattr(
        audit.Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("full"))
    )
    audit.record_query(question="q", sql="SELECT 1", success=True)  # must not raise


def test_summary_counts_by_event_kind(isolated_log) -> None:
    audit.record_prediction(
        applicant_id=1, probability=0.1, risk_score=100, risk_band="Low",
        decision="Approve", model_name="m", threshold=0.2,
    )
    audit.record_query(question="q", sql="SELECT 1", success=True)
    summary = audit.summarise()
    assert summary["predictions_logged"] == 1
    assert summary["queries_logged"] == 1
    assert summary["first_event"] and summary["last_event"]


def test_prediction_path_writes_an_audit_record(trained_artifacts, joined_dataset, isolated_log) -> None:
    """Scoring an individual applicant audits by default."""
    from src.ml.predict import predict_applicant

    predict_applicant(joined_dataset.iloc[[0]], top_n=5)
    events = audit.read_events(event=audit.EVENT_PREDICTION)
    assert len(events) == 1
    assert events[0]["reasons"], "the explanation given must be recoverable"


def test_batch_scoring_does_not_audit_per_row(trained_artifacts, joined_dataset, isolated_log) -> None:
    """A bulk run is a different kind of event and must not flood the log."""
    from src.ml.predict import score_frame

    score_frame(joined_dataset.head(50))
    assert audit.read_events(event=audit.EVENT_PREDICTION) == []
