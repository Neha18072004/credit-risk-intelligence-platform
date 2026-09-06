"""End-to-end tests for the talk-to-data pipeline.

Runs entirely against SQLite and a scripted LLM, so the whole feature is
verifiable with no database server and no model runtime.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.talk_to_data.memory import ConversationMemory
from src.talk_to_data.nl_to_sql import TalkToData
from src.talk_to_data.prompt_templates import (
    FEW_SHOT_EXAMPLES,
    PROMPT_VERSION,
    build_sql_prompt,
    build_summary_prompt,
)
from src.talk_to_data.query_runner import (
    deterministic_summary,
    execute_sql,
    rows_to_markdown,
    summarise_result,
    verify_summary_grounding,
)
from src.talk_to_data.sql_validator import SQLValidator


# --------------------------------------------------------------- prompts --
def test_prompt_is_versioned_and_has_enough_examples() -> None:
    """The brief asks for at least five worked query patterns."""
    assert PROMPT_VERSION
    assert len(FEW_SHOT_EXAMPLES) >= 5
    for example in FEW_SHOT_EXAMPLES:
        assert example.question and example.sql
        assert example.teaches, "every example must justify its token cost"


def test_prompt_states_the_hard_rules() -> None:
    system, user = build_sql_prompt("test question")
    lowered = system.lower()
    assert "one select" in lowered
    assert "cannot_answer" in lowered
    assert "never invent" in lowered
    assert "test question" in user


def test_prompt_stays_within_a_token_budget() -> None:
    """Schema context is re-sent every turn, so its size is a real cost."""
    system, user = build_sql_prompt("What is the default rate?")
    approx_tokens = (len(system) + len(user)) // 4
    assert approx_tokens < 2500, f"prompt too large: ~{approx_tokens} tokens"


def test_few_shot_limit_reduces_prompt_size() -> None:
    _, full = build_sql_prompt("q")
    _, trimmed = build_sql_prompt("q", few_shot_limit=3)
    assert len(trimmed) < len(full)


def test_history_is_only_included_when_present() -> None:
    _, without = build_sql_prompt("q")
    _, with_history = build_sql_prompt("q", history="Q: earlier question")
    assert "RECENT CONVERSATION" not in without
    assert "RECENT CONVERSATION" in with_history


def test_summary_prompt_forbids_invention() -> None:
    system, user = build_summary_prompt("q", "SELECT 1", "| a |\n|---|\n| 1 |", 1)
    assert "only" in system.lower()
    assert "never" in system.lower()
    assert "SELECT 1" in user


# ---------------------------------------------------------------- memory --
def test_memory_evicts_oldest_turns() -> None:
    memory = ConversationMemory(max_turns=3)
    for index in range(5):
        memory.record(f"question {index}", sql=f"SELECT {index}")
    assert len(memory) == 3
    assert memory.total_questions_asked == 5
    assert "question 4" in memory.render_context()
    assert "question 0" not in memory.render_context()


def test_memory_context_is_empty_on_first_turn() -> None:
    assert ConversationMemory().render_context() == ""


def test_memory_keeps_failures_marked() -> None:
    memory = ConversationMemory()
    memory.record("bad question", succeeded=False, error="Unknown column: fico")
    context = memory.render_context()
    assert "failed" in context
    assert "fico" in context
    assert memory.last_successful() is None


def test_memory_truncates_long_sql() -> None:
    memory = ConversationMemory()
    memory.record("q", sql="SELECT " + "x," * 500 + "y FROM applications")
    assert "..." in memory.render_context()


def test_memory_context_stays_bounded() -> None:
    """History is re-sent each turn, so it must not grow without limit."""
    memory = ConversationMemory(max_turns=6)
    for index in range(50):
        memory.record(f"question {index}", sql="SELECT * FROM applications WHERE x = 1")
    assert memory.stats()["approx_context_tokens"] < 500


def test_memory_can_be_disabled() -> None:
    memory = ConversationMemory(max_turns=0)
    memory.record("q")
    assert len(memory) == 0
    assert memory.render_context() == ""


# ------------------------------------------------------ grounding checks --
def test_grounding_accepts_figures_present_in_the_data() -> None:
    frame = pd.DataFrame({"band": ["Low", "High"], "rate": [2.57, 36.13]})
    grounded, unsupported = verify_summary_grounding(
        "Low risk applicants default at 2.57% and High at 36.13%.", frame
    )
    assert grounded and not unsupported


def test_grounding_allows_sensible_rerounding() -> None:
    frame = pd.DataFrame({"rate": [8.53]})
    assert verify_summary_grounding("The rate is 8.5%.", frame)[0]


def test_grounding_flags_invented_figures() -> None:
    """The failure this control exists for: a fluent, fabricated statistic."""
    frame = pd.DataFrame({"education": ["Lower secondary"], "rate": [16.0]})
    grounded, unsupported = verify_summary_grounding(
        "Lower secondary defaults at 16.0%, above the industry average of 4.2%.", frame
    )
    assert not grounded
    assert 4.2 in unsupported


def test_grounding_flags_invented_arithmetic() -> None:
    frame = pd.DataFrame({"applicants": [50, 2854, 153, 942]})
    grounded, unsupported = verify_summary_grounding(
        "This represents 0.08% of the total sample (2,854 + 50 + 153 + 942 = 3,999).", frame
    )
    assert not grounded
    assert 3999 in unsupported


def test_grounding_accepts_numbers_inside_category_labels() -> None:
    """Band boundaries live in label text, and quoting them is grounded.

    Regression test: scanning only numeric columns flagged a correct summary of
    a banded query as fabricated.
    """
    frame = pd.DataFrame(
        {
            "score_band": ["1. Very low (<0.3)", "4. High (>=0.6)"],
            "default_rate_pct": [30.12, 2.34],
        }
    )
    grounded, unsupported = verify_summary_grounding(
        "Default risk falls from 30.12% below 0.3 to 2.34% at 0.6 and above.", frame
    )
    assert grounded, f"wrongly flagged {unsupported}"


def test_grounding_rejects_a_false_empty_claim() -> None:
    """A summary can contradict the data without quoting a single number.

    Regression test: a model answered "No rows matched." over a two-row result.
    """
    frame = pd.DataFrame({"arrears": [0, 1], "rate": [7.69, 13.28]})
    assert not verify_summary_grounding("No rows matched.", frame)[0]
    assert not verify_summary_grounding("The query returned no data.", frame)[0]
    assert verify_summary_grounding("Arrears default at 13.28% versus 7.69%.", frame)[0]


def test_grounding_allows_small_ordinals() -> None:
    frame = pd.DataFrame({"rate": [16.0]})
    assert verify_summary_grounding("The top 3 groups, worst at 16.0%.", frame)[0]


def test_deterministic_summary_never_invents() -> None:
    frame = pd.DataFrame({"band": ["Low", "High"], "rate": [2.5, 36.1]})
    summary = deterministic_summary("q", frame)
    assert "2 rows" in summary
    assert verify_summary_grounding(summary, frame)[0]


def test_empty_result_summary_is_honest() -> None:
    answer, source = summarise_result("q", "SELECT 1", pd.DataFrame())
    assert "no rows" in answer.lower()
    assert source == "deterministic"


def test_rows_to_markdown_truncates_long_results() -> None:
    frame = pd.DataFrame({"a": range(100)})
    table, truncated = rows_to_markdown(frame, max_rows=10)
    assert truncated
    assert "first 10 of 100" in table


# --------------------------------------------------------------- runtime --
def test_fetch_schema_reads_live_tables(live_schema) -> None:
    assert {"applications", "bureau", "bureau_summary", "predictions"} <= set(live_schema)
    assert "target" in live_schema["applications"]
    assert "risk_band" in live_schema["predictions"]


def test_execute_sql_returns_rows(analytics_db) -> None:
    result = execute_sql("SELECT COUNT(*) AS n FROM applications", engine=analytics_db)
    assert result.success
    assert result.rows.iloc[0]["n"] > 0


def test_execute_sql_captures_errors_without_raising(analytics_db) -> None:
    result = execute_sql("SELECT * FROM does_not_exist", engine=analytics_db)
    assert not result.success
    assert result.error


# ----------------------------------------------- end-to-end query patterns --
QUERY_PATTERNS = [
    (
        "overall default rate",
        "SELECT ROUND(AVG(target) * 100, 2) AS default_rate_pct, COUNT(*) AS total "
        "FROM applications WHERE target IS NOT NULL",
    ),
    (
        "rate by segment",
        "SELECT name_education_type, COUNT(*) AS applicants, "
        "ROUND(AVG(target) * 100, 2) AS default_rate_pct FROM applications "
        "WHERE target IS NOT NULL GROUP BY name_education_type HAVING COUNT(*) >= 50",
    ),
    (
        "two-group comparison",
        "SELECT target, COUNT(*) AS n, AVG(amt_income_total) AS avg_income "
        "FROM applications WHERE target IS NOT NULL GROUP BY target",
    ),
    (
        "join to model output",
        "SELECT p.risk_band, COUNT(*) AS n, AVG(a.target) AS actual_rate "
        "FROM predictions p JOIN applications a ON a.sk_id_curr = p.sk_id_curr "
        "GROUP BY p.risk_band",
    ),
    (
        "bureau join",
        "SELECT b.bureau_has_overdue, COUNT(*) AS n, AVG(a.target) AS rate "
        "FROM applications a JOIN bureau_summary b ON b.sk_id_curr = a.sk_id_curr "
        "WHERE b.bureau_has_history = 1 GROUP BY b.bureau_has_overdue",
    ),
    (
        "row-level ranking",
        "SELECT sk_id_curr, risk_score, risk_band FROM predictions "
        "ORDER BY risk_score DESC LIMIT 10",
    ),
    (
        "data quality",
        "SELECT COUNT(*) AS total, COUNT(ext_source_1) AS ext1_present FROM applications",
    ),
]


@pytest.mark.parametrize("label,sql", QUERY_PATTERNS, ids=[p[0] for p in QUERY_PATTERNS])
def test_query_patterns_validate_and_execute(label, sql, live_schema, analytics_db) -> None:
    """The acceptance criterion: every supported pattern runs and returns rows."""
    validator = SQLValidator(live_schema, max_rows=200)
    validation = validator.validate(sql)
    assert validation.is_valid, f"{label}: {validation.reason}"

    result = execute_sql(validation.sql, engine=analytics_db)
    assert result.success, f"{label}: {result.error}"
    assert result.row_count > 0, f"{label} returned no rows"


def test_few_shot_examples_are_all_valid_sql(live_schema) -> None:
    """A broken example would teach the model to produce broken SQL."""
    validator = SQLValidator(live_schema, max_rows=200)
    for example in FEW_SHOT_EXAMPLES:
        result = validator.validate(example.sql)
        assert result.is_valid, f"few-shot '{example.question}' is invalid: {result.reason}"


def test_few_shot_examples_avoid_round_on_uncast_floats() -> None:
    """ROUND(double precision, int) does not exist in PostgreSQL.

    Regression test for a real failure: one worked example rounded
    AVG(probability_of_default) -- a DOUBLE PRECISION column -- to two decimal
    places without a ::numeric cast. It parsed cleanly, validated cleanly, ran
    fine on SQLite, and failed on the actual database. The model had faithfully
    imitated the broken pattern.

    The columns below are FLOAT in the schema; rounding any of them to decimal
    places requires an explicit cast.
    """
    float_columns = (
        "probability_of_default", "risk_score", "ext_source_mean", "ext_source_1",
        "ext_source_2", "ext_source_3", "amt_income_total", "amt_credit",
        "amt_annuity", "credit_to_income_ratio", "annuity_to_income_ratio",
        "bureau_debt_credit_ratio", "age_years", "employed_years",
    )
    pattern = re.compile(r"ROUND\s*\(([^()]*(?:\([^()]*\))?[^()]*),\s*-?\d+\s*\)", re.IGNORECASE)

    for example in FEW_SHOT_EXAMPLES:
        for expression in pattern.findall(example.sql):
            if "::numeric" in expression.lower():
                continue
            offending = [column for column in float_columns if column in expression.lower()]
            assert not offending, (
                f"few-shot '{example.question}' rounds float column(s) {offending} "
                f"without a ::numeric cast: ROUND({expression.strip()}, n)"
            )


@pytest.mark.parametrize("example", FEW_SHOT_EXAMPLES, ids=lambda e: e.question[:40])
def test_few_shot_examples_actually_execute(example, live_schema, analytics_db) -> None:
    """Executing each example, not merely parsing it.

    The earlier version only validated syntax, which is why a statement that
    PostgreSQL refuses sat in the prompt undetected.
    """
    validator = SQLValidator(live_schema, max_rows=200)
    validation = validator.validate(example.sql)
    assert validation.is_valid, validation.reason

    result = execute_sql(validation.sql, engine=analytics_db)
    assert result.success, f"'{example.question}' failed to execute: {result.error}"


# --------------------------------------------------------- orchestration --
def _interface(fake_llm, responses, live_schema, monkeypatch, analytics_db):
    """Build a TalkToData wired to a scripted model and the test database."""
    from src.talk_to_data import nl_to_sql as module

    monkeypatch.setattr(module, "execute_sql", lambda sql: execute_sql(sql, engine=analytics_db))
    return TalkToData(
        client=fake_llm(responses),
        validator=SQLValidator(live_schema, max_rows=200),
        memory=ConversationMemory(max_turns=4),
    )


def test_ask_returns_answer_and_rows(fake_llm, live_schema, monkeypatch, analytics_db) -> None:
    interface = _interface(
        fake_llm,
        ["SELECT COUNT(*) AS n FROM applications"],
        live_schema, monkeypatch, analytics_db,
    )
    result = interface.ask("How many applicants are there?")
    assert result.success
    assert result.row_count == 1
    assert result.sql
    assert result.answer


def test_ask_repairs_after_a_rejection(fake_llm, live_schema, monkeypatch, analytics_db) -> None:
    """A hallucinated column is fed back once, and the model gets one retry."""
    client = fake_llm(
        [
            "SELECT credit_score FROM applications",          # hallucinated
            "SELECT COUNT(*) AS n FROM applications",         # repaired
        ]
    )
    from src.talk_to_data import nl_to_sql as module

    monkeypatch.setattr(module, "execute_sql", lambda sql: execute_sql(sql, engine=analytics_db))
    interface = TalkToData(
        client=client, validator=SQLValidator(live_schema, max_rows=200),
        memory=ConversationMemory(),
    )
    result = interface.ask("How many applicants?")
    assert result.success
    assert result.repair_attempted
    assert len(client.generation_calls) == 2
    # The rejection reason must actually reach the model.
    assert "credit_score" in client.generation_calls[1][1]


def test_ask_repairs_after_a_database_error(
    fake_llm, live_schema, monkeypatch, analytics_db
) -> None:
    """A query can be valid to the parser and still be refused by the server.

    Regression test: PostgreSQL rejected ROUND(double precision, int) on a
    statement that passed every validation check, and the repair loop -- which
    only retried on validation failure -- had no recovery path.
    """
    client = fake_llm(
        [
            "SELECT AVG(ext_source_mean) AS m FROM applications",   # server refuses
            "SELECT AVG(ext_source_mean) AS m FROM applications",   # repaired
        ]
    )
    from src.talk_to_data import nl_to_sql as module
    from src.talk_to_data.query_runner import QueryResult

    # The first execution is failed deliberately. Crafting SQL that PostgreSQL
    # refuses but SQLite accepts would make the test depend on dialect quirks;
    # what is under test is the recovery path, not the specific error.
    calls = {"n": 0}

    def flaky_execute(sql: str) -> QueryResult:
        calls["n"] += 1
        if calls["n"] == 1:
            return QueryResult(
                success=False, sql=sql,
                error="function round(double precision, integer) does not exist",
            )
        return execute_sql(sql, engine=analytics_db)

    monkeypatch.setattr(module, "execute_sql", flaky_execute)
    interface = TalkToData(
        client=client, validator=SQLValidator(live_schema, max_rows=200),
        memory=ConversationMemory(),
    )
    result = interface.ask("How many applicants?")
    assert result.success
    assert result.repair_attempted
    assert len(client.generation_calls) == 2
    # The database's own message must reach the model.
    assert "database rejected it" in client.generation_calls[1][1]
    assert "round(double precision, integer)" in client.generation_calls[1][1]


def test_ask_gives_up_after_one_repair(fake_llm, live_schema, monkeypatch, analytics_db) -> None:
    client = fake_llm(["SELECT bad_one FROM applications", "SELECT bad_two FROM applications"])
    interface = TalkToData(
        client=client, validator=SQLValidator(live_schema, max_rows=200),
        memory=ConversationMemory(),
    )
    result = interface.ask("something impossible")
    assert not result.success
    assert len(client.generation_calls) == 2, "must not loop indefinitely"


def test_ask_honours_model_refusal(fake_llm, live_schema, monkeypatch, analytics_db) -> None:
    """An admitted gap is preferred over an invented answer."""
    interface = _interface(
        fake_llm,
        ["CANNOT_ANSWER: the database has no information about credit card balances"],
        live_schema, monkeypatch, analytics_db,
    )
    result = interface.ask("What is the average credit card balance?")
    assert not result.success
    assert result.refused
    assert "credit card balances" in result.error
    assert "can't answer" in result.answer.lower()


def test_ask_rejects_a_destructive_generation(fake_llm, live_schema, monkeypatch, analytics_db) -> None:
    interface = _interface(
        fake_llm,
        ["DROP TABLE applications", "DELETE FROM applications"],
        live_schema, monkeypatch, analytics_db,
    )
    result = interface.ask("delete everything")
    assert not result.success
    assert not result.rows.shape[0]


def test_ask_degrades_when_no_model_is_available(fake_llm, live_schema) -> None:
    """Without an LLM the feature explains itself instead of crashing."""
    interface = TalkToData(
        client=fake_llm([], available=False),
        validator=SQLValidator(live_schema, max_rows=200),
    )
    available, message = interface.is_available()
    assert not available and message

    result = interface.ask("anything")
    assert not result.success
    assert result.error


def test_conversation_memory_reaches_the_prompt(
    fake_llm, live_schema, monkeypatch, analytics_db
) -> None:
    """Follow-ups need the earlier turn, or 'and for women?' is meaningless."""
    client = fake_llm(
        [
            "SELECT COUNT(*) AS n FROM applications",
            "SELECT COUNT(*) AS n FROM applications WHERE code_gender = 'F'",
        ]
    )
    from src.talk_to_data import nl_to_sql as module

    monkeypatch.setattr(module, "execute_sql", lambda sql: execute_sql(sql, engine=analytics_db))
    interface = TalkToData(
        client=client, validator=SQLValidator(live_schema, max_rows=200),
        memory=ConversationMemory(max_turns=4),
    )
    interface.ask("How many applicants are there?")
    interface.ask("And for women?")

    second_prompt = client.generation_calls[1][1]
    assert "RECENT CONVERSATION" in second_prompt
    assert "How many applicants are there?" in second_prompt
    assert len(interface.memory) == 2


def test_token_usage_is_reported(fake_llm, live_schema, monkeypatch, analytics_db) -> None:
    interface = _interface(
        fake_llm, ["SELECT COUNT(*) AS n FROM applications"],
        live_schema, monkeypatch, analytics_db,
    )
    result = interface.ask("count them")
    assert result.prompt_tokens > 0
    payload = result.to_dict()
    assert payload["tokens"]["total"] > 0
    assert payload["prompt_version"] == PROMPT_VERSION
