"""Natural-language question to SQL, end to end.

This is the orchestrator the UI talks to. One call takes a question and returns
an answer, the SQL that produced it, and the rows behind it -- with every safety
layer applied in between.

The flow, and what each step defends against:

1. **Memory** supplies recent turns so follow-ups resolve.
2. **Prompt assembly** grounds the model in a compact curated schema plus
   worked examples.
3. **Generation** produces candidate SQL, or an explicit refusal.
4. **Refusal handling** -- a model that says ``CANNOT_ANSWER`` is believed. This
   is the cheapest hallucination control available: an admitted gap beats an
   invented column.
5. **Validation** rejects anything unsafe or ungrounded.
6. **One repair attempt** -- the failure reason is fed back for a single retry.
   Models fix a named unknown column reliably; a second retry mostly burns
   tokens, so the loop is capped at one.
7. **Execution** on a read-only connection with a statement timeout. A database
   error also earns a repair attempt: a query can be perfectly valid to the
   parser and still fail on the server -- PostgreSQL rejecting
   ``ROUND(double precision, int)`` is the case that motivated this -- and the
   error text names the fix precisely enough for the model to apply it.
8. **Summarisation** strictly from the returned rows.
9. **Recording** the turn into memory and into the audit log, successes and
   failures alike -- a rejected query is a security event and a refusal is
   evidence the grounding controls did their job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.talk_to_data.llm_client import LLMClient, LLMUnavailableError, get_llm_client
from src.talk_to_data.memory import ConversationMemory
from src.talk_to_data.prompt_templates import PROMPT_VERSION, build_sql_prompt
from src.talk_to_data.query_runner import execute_sql, fetch_schema, summarise_result
from src.talk_to_data.sql_validator import SQLValidator
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Marker the model is instructed to emit when a question cannot be answered
# from the available columns.
REFUSAL_MARKER: str = "CANNOT_ANSWER"

# One repair attempt only. Beyond that the model tends to loop on the same
# mistake, and each retry costs a full prompt.
MAX_REPAIR_ATTEMPTS: int = 1


@dataclass
class AskResult:
    """Everything one question produced, for the UI and for audit."""

    question: str
    success: bool
    answer: str = ""
    sql: str = ""
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    row_count: int = 0
    error: str = ""
    refused: bool = False
    repair_attempted: bool = False
    validation_warnings: list[str] = field(default_factory=list)
    tables_used: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0
    summary_source: str = ""
    prompt_version: str = PROMPT_VERSION

    def __bool__(self) -> bool:
        return self.success

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, excluding the row data."""
        return {
            "question": self.question,
            "success": self.success,
            "answer": self.answer,
            "sql": self.sql,
            "row_count": self.row_count,
            "error": self.error,
            "refused": self.refused,
            "repair_attempted": self.repair_attempted,
            "tables_used": self.tables_used,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.prompt_tokens + self.completion_tokens,
            },
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "summary_source": self.summary_source,
            "prompt_version": self.prompt_version,
        }


class TalkToData:
    """Conversational query interface over the credit-risk database.

    Args:
        client: LLM client; the configured provider is used by default.
        validator: SQL validator; built from the live schema by default.
        memory: Conversation memory; a fresh one is created by default.
        schema: Schema override, mainly for tests.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        validator: SQLValidator | None = None,
        memory: ConversationMemory | None = None,
        schema: dict[str, set[str]] | None = None,
    ) -> None:
        self.client = client or get_llm_client()
        self.memory = memory or ConversationMemory()
        if validator is not None:
            self.validator = validator
        else:
            self.validator = SQLValidator(
                schema=schema if schema is not None else fetch_schema(),
                max_rows=settings.sql_max_rows,
            )

    # ------------------------------------------------------------- status --
    def is_available(self) -> tuple[bool, str]:
        """Whether the chat feature can run, and why not if it cannot."""
        return self.client.is_available()

    # ---------------------------------------------------------- generation --
    def generate_sql(self, question: str, feedback: str = "") -> tuple[str, int, int]:
        """Ask the model for SQL.

        Args:
            question: The analyst's question.
            feedback: A rejection reason from a previous attempt, appended so
                the model can repair its own output.

        Returns:
            ``(raw_text, prompt_tokens, completion_tokens)``.
        """
        history = self.memory.render_context()
        system_prompt, user_prompt = build_sql_prompt(question, history=history)

        if feedback:
            user_prompt += (
                f"\n\nYour previous attempt was REJECTED for this reason:\n{feedback}\n"
                "Write a corrected query that uses only the columns listed in the schema above."
            )

        response = self.client.complete(system_prompt, user_prompt)
        return response.text, response.prompt_tokens, response.completion_tokens

    @staticmethod
    def _extract_refusal(text: str) -> str | None:
        """Return the refusal reason if the model declined, else None."""
        if REFUSAL_MARKER not in text.upper():
            return None
        _, _, reason = text.partition(":")
        return reason.strip() or "The question cannot be answered from the available data."

    # ---------------------------------------------------------------- ask --
    def ask(self, question: str, summarise: bool = True) -> AskResult:
        """Answer one natural-language question.

        Args:
            question: The analyst's question.
            summarise: Generate a natural-language answer alongside the rows.

        Returns:
            An :class:`AskResult`. Failures are returned, never raised: an
            unanswerable question is a normal conversational outcome.
        """
        started = time.perf_counter()
        result = AskResult(question=question, success=False)

        available, reason = self.client.is_available()
        if not available:
            result.error = reason
            result.elapsed_seconds = time.perf_counter() - started
            self.memory.record(question, succeeded=False, error=reason)
            self._audit(result)
            return result

        feedback = ""
        last_reason = ""
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            try:
                raw, prompt_tokens, completion_tokens = self.generate_sql(question, feedback)
            except LLMUnavailableError as error:
                result.error = str(error)
                result.elapsed_seconds = time.perf_counter() - started
                self.memory.record(question, succeeded=False, error=result.error)
                self._audit(result)
                return result

            result.prompt_tokens += prompt_tokens
            result.completion_tokens += completion_tokens
            result.repair_attempted = attempt > 0

            # The model declining is a success for grounding, not a failure.
            refusal = self._extract_refusal(raw)
            if refusal:
                result.refused = True
                result.error = refusal
                result.answer = (
                    f"I can't answer that from the available data. {refusal}"
                )
                result.elapsed_seconds = time.perf_counter() - started
                self.memory.record(question, succeeded=False, error=refusal)
                self._audit(result)
                return result

            validation = self.validator.validate(raw)
            if not validation.is_valid:
                last_reason = feedback = validation.reason
                logger.info("Attempt %d rejected by the validator: %s", attempt + 1, validation.reason)
                continue

            execution = execute_sql(validation.sql)
            if not execution.success:
                # Valid to the parser, refused by the server. The database's own
                # message is usually specific enough to repair from.
                last_reason = execution.error
                feedback = f"The query was valid but the database rejected it: {execution.error}"
                logger.info("Attempt %d failed at execution: %s", attempt + 1, execution.error)
                continue

            result.sql = validation.sql
            result.validation_warnings = validation.warnings
            result.tables_used = validation.tables
            result.rows = execution.rows
            result.row_count = execution.row_count
            result.success = True
            break
        else:
            result.error = f"Could not produce a working query. {last_reason}"
            result.elapsed_seconds = time.perf_counter() - started
            self.memory.record(question, succeeded=False, error=result.error)
            self._audit(result)
            return result

        if summarise:
            result.answer, result.summary_source = summarise_result(
                question, result.sql, result.rows, client=self.client
            )

        result.elapsed_seconds = time.perf_counter() - started
        self.memory.record(
            question, sql=result.sql, row_count=result.row_count,
            answer=result.answer, succeeded=True,
        )
        self._audit(result)
        logger.info(
            "Answered in %.2fs using %d tokens (%d rows)",
            result.elapsed_seconds, result.prompt_tokens + result.completion_tokens,
            result.row_count,
        )
        return result

    @staticmethod
    def _audit(result: AskResult) -> None:
        """Record one turn to the audit log, whatever its outcome."""
        from src.utils.audit import record_query

        record_query(
            question=result.question,
            sql=result.sql,
            success=result.success,
            row_count=result.row_count,
            error=result.error,
            refused=result.refused,
            tables=result.tables_used,
            elapsed_seconds=result.elapsed_seconds,
            tokens=result.prompt_tokens + result.completion_tokens,
            summary_source=result.summary_source,
            prompt_version=result.prompt_version,
        )

    def reset(self) -> None:
        """Clear the conversation history."""
        self.memory.clear()


def ask(question: str, **kwargs: Any) -> AskResult:
    """One-shot convenience wrapper for a single question."""
    return TalkToData(**kwargs).ask(question)


def main() -> None:  # pragma: no cover - CLI entry point
    """Interactive command-line chat against the database."""
    import sys

    interface = TalkToData()
    available, message = interface.is_available()
    print(f"\nTalk to your credit-risk data  ({message})")
    if not available:
        print("\nChat is unavailable. Every other feature still works.")
        return

    questions = sys.argv[1:]
    if questions:
        for question in questions:
            _print_result(interface.ask(question))
        return

    print("Type a question, or 'quit' to exit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            _print_result(interface.ask(question))


def _print_result(result: AskResult) -> None:  # pragma: no cover
    """Render one answer to the terminal."""
    print(f"\nQ: {result.question}")
    if not result.success:
        print(f"  ! {result.error}")
        return
    print(f"\n  SQL:\n{result.sql}\n")
    print(f"  Answer: {result.answer}\n")
    if not result.rows.empty:
        print(result.rows.head(15).to_string(index=False))
    print(
        f"\n  [{result.row_count} rows | {result.elapsed_seconds:.2f}s | "
        f"{result.prompt_tokens + result.completion_tokens} tokens | "
        f"summary via {result.summary_source}]\n"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
