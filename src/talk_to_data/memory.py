"""Conversation memory for multi-turn talk-to-data.

Follow-ups are the point. An analyst asks "what is the default rate by
education?" and then "and for women?" -- the second question is meaningless
without the first, so some history has to reach the model.

The design constraint is token cost. History is re-sent on every turn, so an
unbounded transcript makes each turn more expensive than the last and
eventually crowds out the schema. Three measures keep it bounded:

* only the last ``MEMORY_MAX_TURNS`` turns are retained;
* each turn is stored as a **compact summary** -- the question, the SQL, and a
  one-line shape of the result -- never the returned rows, which can be large
  and are rarely what a follow-up depends on;
* failed turns are kept but clearly marked, so the model can learn from a
  rejection instead of repeating it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Hard cap on how much of a stored SQL string is replayed into the prompt.
MAX_SQL_CHARS: int = 400


@dataclass
class ConversationTurn:
    """One question and what came of it."""

    question: str
    sql: str = ""
    row_count: int = 0
    answer: str = ""
    succeeded: bool = True
    error: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_context(self) -> str:
        """Render the turn compactly for replay into a later prompt."""
        if not self.succeeded:
            return f"Q: {self.question}\n   (failed: {self.error[:120]})"
        sql = self.sql if len(self.sql) <= MAX_SQL_CHARS else f"{self.sql[:MAX_SQL_CHARS]}..."
        single_line = " ".join(sql.split())
        return f"Q: {self.question}\nSQL: {single_line}\n   -> {self.row_count} row(s)"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, for transcript export."""
        return {
            "question": self.question,
            "sql": self.sql,
            "row_count": self.row_count,
            "answer": self.answer,
            "succeeded": self.succeeded,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationMemory:
    """A bounded, token-aware conversation history.

    Args:
        max_turns: Turns retained. Defaults to ``MEMORY_MAX_TURNS``.
    """

    def __init__(self, max_turns: int | None = None) -> None:
        self.max_turns = max_turns if max_turns is not None else settings.memory_max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max(self.max_turns, 1))
        self._total_seen = 0

    # ---------------------------------------------------------------- state --
    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> list[ConversationTurn]:
        """Retained turns, oldest first."""
        return list(self._turns)

    @property
    def total_questions_asked(self) -> int:
        """Every question this session, including turns already evicted."""
        return self._total_seen

    def add(self, turn: ConversationTurn) -> None:
        """Record a turn, evicting the oldest once the cap is reached."""
        self._total_seen += 1
        if self.max_turns <= 0:
            return  # memory explicitly disabled
        self._turns.append(turn)

    def record(
        self,
        question: str,
        sql: str = "",
        row_count: int = 0,
        answer: str = "",
        succeeded: bool = True,
        error: str = "",
    ) -> ConversationTurn:
        """Convenience wrapper that builds and stores a turn."""
        turn = ConversationTurn(
            question=question, sql=sql, row_count=row_count,
            answer=answer, succeeded=succeeded, error=error,
        )
        self.add(turn)
        return turn

    def clear(self) -> None:
        """Forget everything, including the counter."""
        self._turns.clear()
        self._total_seen = 0

    # -------------------------------------------------------------- context --
    def render_context(self, max_turns: int | None = None) -> str:
        """Render recent turns as prompt text.

        Args:
            max_turns: Include only the most recent ``max_turns``; defaults to
                everything retained.

        Returns:
            The formatted history, or an empty string when there is none. An
            empty string is important: it lets the prompt builder omit the
            history section entirely on the first turn rather than sending a
            header with nothing under it.
        """
        turns = self.turns
        if max_turns is not None:
            turns = turns[-max_turns:]
        if not turns:
            return ""
        return "\n\n".join(turn.as_context() for turn in turns)

    def last_successful(self) -> ConversationTurn | None:
        """The most recent turn that produced a result, if any."""
        for turn in reversed(self._turns):
            if turn.succeeded:
                return turn
        return None

    def export(self) -> list[dict[str, Any]]:
        """Full transcript, for download or audit."""
        return [turn.to_dict() for turn in self._turns]

    def stats(self) -> dict[str, Any]:
        """Summary counters for the UI."""
        successes = sum(1 for turn in self._turns if turn.succeeded)
        return {
            "retained_turns": len(self._turns),
            "max_turns": self.max_turns,
            "total_questions_asked": self._total_seen,
            "successful": successes,
            "failed": len(self._turns) - successes,
            "approx_context_tokens": len(self.render_context()) // 4,
        }
