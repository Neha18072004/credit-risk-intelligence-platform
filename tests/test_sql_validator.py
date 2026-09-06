"""Security and grounding tests for the SQL validator.

The validator is the control that makes it acceptable for a language model to
write SQL against this database, so these tests are the most important in the
suite. Every one of them asserts that something unsafe is *refused*.
"""

from __future__ import annotations

import pytest

from src.talk_to_data.sql_validator import SQLValidator

SCHEMA = {
    "applications": {
        "sk_id_curr", "target", "amt_income_total", "amt_credit", "amt_annuity",
        "name_education_type", "code_gender", "ext_source_mean", "age_years",
        "name_contract_type", "name_income_type",
    },
    "predictions": {
        "sk_id_curr", "probability_of_default", "risk_score", "risk_band", "decision"
    },
    "bureau_summary": {
        "sk_id_curr", "bureau_has_overdue", "bureau_debt_total", "bureau_has_history"
    },
}


@pytest.fixture
def validator() -> SQLValidator:
    return SQLValidator(SCHEMA, max_rows=200)


# ------------------------------------------------------- write operations --
@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM applications WHERE target = 1",
        "UPDATE applications SET target = 0",
        "INSERT INTO applications (sk_id_curr) VALUES (1)",
        "DROP TABLE predictions",
        "CREATE TABLE evil (a int)",
        "ALTER TABLE applications ADD COLUMN x int",
        "TRUNCATE TABLE applications",
        "GRANT ALL ON applications TO PUBLIC",
    ],
    ids=["delete", "update", "insert", "drop", "create", "alter", "truncate", "grant"],
)
def test_write_operations_are_rejected(validator: SQLValidator, sql: str) -> None:
    result = validator.validate(sql)
    assert not result.is_valid
    assert result.reason


def test_delete_hidden_in_a_cte_is_rejected(validator: SQLValidator) -> None:
    """The statement is structurally a SELECT; the write hides inside the CTE."""
    result = validator.validate(
        "WITH removed AS (DELETE FROM applications RETURNING *) SELECT * FROM removed"
    )
    assert not result.is_valid
    assert "DELETE" in result.reason


# ------------------------------------------------------------- injection --
def test_stacked_statements_are_rejected(validator: SQLValidator) -> None:
    result = validator.validate("SELECT 1 FROM applications; DROP TABLE applications")
    assert not result.is_valid
    assert "one statement" in result.reason.lower()


def test_comment_hidden_stacked_statement_is_rejected(validator: SQLValidator) -> None:
    result = validator.validate(
        "SELECT target FROM applications -- harmless\nUNION SELECT 1; DELETE FROM applications"
    )
    assert not result.is_valid


def test_comments_cannot_survive_into_executed_sql(validator: SQLValidator) -> None:
    """Re-serialising from the parse tree drops anything the parser ignored."""
    result = validator.validate(
        "SELECT sk_id_curr /* injected */ FROM applications -- trailing comment"
    )
    assert result.is_valid
    assert "injected" not in result.sql
    assert "--" not in result.sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_sleep(60) FROM applications",
        "SELECT lo_import('/etc/shadow')",
        "SELECT dblink('host=evil', 'SELECT 1')",
    ],
    ids=["file-read", "sleep-dos", "lo-import", "dblink"],
)
def test_dangerous_functions_are_rejected(validator: SQLValidator, sql: str) -> None:
    assert not validator.validate(sql).is_valid


# ------------------------------------------------------- hallucination ----
def test_unknown_table_is_rejected_with_available_tables(validator: SQLValidator) -> None:
    result = validator.validate("SELECT * FROM customers")
    assert not result.is_valid
    assert "customers" in result.reason
    # The message must help the model repair itself on the retry.
    assert "applications" in result.reason


def test_unknown_column_is_rejected(validator: SQLValidator) -> None:
    result = validator.validate("SELECT credit_score FROM applications")
    assert not result.is_valid
    assert "credit_score" in result.reason


def test_unknown_qualified_column_is_rejected(validator: SQLValidator) -> None:
    result = validator.validate("SELECT a.fico_score FROM applications a")
    assert not result.is_valid
    assert "fico_score" in result.reason


def test_column_from_the_wrong_table_is_rejected(validator: SQLValidator) -> None:
    """risk_band lives on predictions, not applications."""
    result = validator.validate("SELECT a.risk_band FROM applications a")
    assert not result.is_valid


# ------------------------------------------------------------ acceptance --
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT AVG(amt_income_total) FROM applications",
        "SELECT name_education_type, COUNT(*) AS n FROM applications GROUP BY name_education_type",
        "SELECT p.risk_band, COUNT(*) FROM predictions p "
        "JOIN applications a ON a.sk_id_curr = p.sk_id_curr GROUP BY p.risk_band",
        "WITH b AS (SELECT risk_band FROM predictions) SELECT risk_band, COUNT(*) FROM b GROUP BY risk_band",
        "SELECT CASE WHEN target = 1 THEN 'Bad' ELSE 'Good' END AS outcome, COUNT(*) "
        "FROM applications GROUP BY target",
    ],
    ids=["aggregate", "group-by", "join", "cte", "case-expression"],
)
def test_legitimate_queries_are_accepted(validator: SQLValidator, sql: str) -> None:
    result = validator.validate(sql)
    assert result.is_valid, result.reason


def test_alias_in_order_by_is_accepted(validator: SQLValidator) -> None:
    """A computed alias is a legitimate reference and must not be flagged."""
    result = validator.validate(
        "SELECT name_education_type, AVG(target) AS default_rate FROM applications "
        "GROUP BY name_education_type ORDER BY default_rate DESC"
    )
    assert result.is_valid, result.reason


def test_markdown_fences_are_stripped(validator: SQLValidator) -> None:
    result = validator.validate("```sql\nSELECT COUNT(*) FROM applications\n```")
    assert result.is_valid
    assert "```" not in result.sql


# ----------------------------------------------------------- row capping --
def test_limit_is_injected_when_absent(validator: SQLValidator) -> None:
    result = validator.validate("SELECT sk_id_curr FROM applications")
    assert result.is_valid
    assert result.limit_applied == 200
    assert "LIMIT" in result.sql.upper()


def test_oversized_limit_is_tightened(validator: SQLValidator) -> None:
    result = validator.validate("SELECT sk_id_curr FROM applications LIMIT 999999")
    assert result.limit_applied == 200


def test_small_limit_is_preserved(validator: SQLValidator) -> None:
    result = validator.validate("SELECT sk_id_curr FROM applications LIMIT 5")
    assert result.limit_applied == 5


# ------------------------------------------------------------ edge cases --
@pytest.mark.parametrize(
    "sql", ["", "   ", "show me all the defaulters", "SELECT"],
    ids=["empty", "whitespace", "prose", "truncated"],
)
def test_malformed_input_is_rejected(validator: SQLValidator, sql: str) -> None:
    assert not validator.validate(sql).is_valid


def test_result_is_falsy_when_invalid(validator: SQLValidator) -> None:
    assert not validator.validate("DROP TABLE applications")
    assert validator.validate("SELECT COUNT(*) FROM applications")


def test_star_select_warns(validator: SQLValidator) -> None:
    result = validator.validate("SELECT * FROM applications")
    assert result.is_valid
    assert any("SELECT *" in warning for warning in result.warnings)


def test_validator_uses_live_schema(live_schema) -> None:
    """The whitelist comes from the real database, not a hardcoded list."""
    validator = SQLValidator(live_schema, max_rows=50)
    assert validator.validate("SELECT COUNT(*) FROM applications").is_valid
    assert not validator.validate("SELECT COUNT(*) FROM nonexistent_table").is_valid


# ------------------------------------------------- advanced adversarial ----
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT sk_id_curr FROM applications UNION SELECT usename FROM pg_user",
        "SELECT table_name FROM information_schema.tables",
        "SELECT sk_id_curr FROM applications WHERE sk_id_curr IN (SELECT id FROM secrets)",
        "SELECT pg_catalog.pg_read_file('/etc/passwd')",
        "DrOp TaBlE applications",
        "dElEtE FrOm applications",
        "COPY applications TO '/tmp/out.csv'",
        "SET ROLE postgres",
        "WITH u AS (UPDATE applications SET target = 0 RETURNING *) SELECT * FROM u",
        "SELECT 1 FROM applications\n;\nDROP TABLE applications",
        "SELECT (SELECT ssn FROM applications) FROM applications",
    ],
    ids=[
        "catalog-table", "union-to-catalog", "information-schema", "unknown-subquery-table",
        "schema-qualified-function", "case-mangled-drop", "mixed-case-delete", "copy-to-file",
        "set-role", "update-in-cte", "newline-stacked", "nested-unknown-column",
    ],
)
def test_advanced_attacks_are_rejected(validator: SQLValidator, sql: str) -> None:
    result = validator.validate(sql)
    assert not result.is_valid, f"leaked: {result.sql}"


def test_cte_body_is_validated_like_any_other_query(validator: SQLValidator) -> None:
    """A CTE is not a blind spot -- its body goes through every check."""
    for sql in (
        "WITH x AS (SELECT * FROM pg_user) SELECT * FROM x",
        "WITH x AS (SELECT * FROM secrets) SELECT * FROM x",
        "WITH x AS (SELECT ssn FROM applications) SELECT * FROM x",
    ):
        assert not validator.validate(sql).is_valid


def test_cte_may_not_shadow_a_real_table(validator: SQLValidator) -> None:
    """Rejected on fail-closed grounds, not because it is exploitable.

    Probing showed shadowing is not an escape: the CTE body is still validated,
    so a shadowed name cannot smuggle in a catalog table, an unknown column or a
    write. It is refused because it makes a statement mean something other than
    what it appears to say, and no legitimate generated query needs it.
    """
    result = validator.validate("WITH applications AS (SELECT 1 AS x) SELECT x FROM applications")
    assert not result.is_valid
    assert "applications" in result.reason

    # A CTE with its own name remains perfectly acceptable.
    assert validator.validate(
        "WITH banded AS (SELECT risk_band FROM predictions) "
        "SELECT risk_band, COUNT(*) FROM banded GROUP BY risk_band"
    ).is_valid


def test_round_arguments_are_cast_for_postgres(validator: SQLValidator) -> None:
    """PostgreSQL has no ROUND(double precision, int); the validator repairs it.

    Fixed deterministically rather than by prompt instruction. The rule was in
    the system prompt and demonstrated in a worked example, and the model still
    reverted to the uncast form once the conversation carried a few turns of
    history -- so the rewrite happens where it cannot be ignored.
    """
    result = validator.validate(
        "SELECT risk_band, ROUND(AVG(probability_of_default) * 100, 2) AS pct "
        "FROM predictions GROUP BY risk_band"
    )
    assert result.is_valid
    assert "CAST(" in result.sql.upper()
    assert any("NUMERIC cast" in warning for warning in result.warnings)


def test_existing_casts_are_left_alone(validator: SQLValidator) -> None:
    result = validator.validate(
        "SELECT ROUND(AVG(probability_of_default)::numeric, 2) AS pct FROM predictions"
    )
    assert result.is_valid
    # Exactly one cast: the one the author wrote.
    assert result.sql.upper().count("CAST(") <= 1
    assert not any("NUMERIC cast" in warning for warning in result.warnings)


def test_single_argument_round_is_untouched(validator: SQLValidator) -> None:
    """ROUND(x) works on any numeric type and needs no repair."""
    result = validator.validate("SELECT ROUND(amt_income_total) AS r FROM applications")
    assert result.is_valid
    assert not any("NUMERIC cast" in warning for warning in result.warnings)
