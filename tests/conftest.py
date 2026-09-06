"""Shared pytest fixtures.

Every test runs against the committed synthetic sample data, so the suite is
fully self-contained: no Kaggle download, no database, no API key required.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force sample mode before any project module reads configuration.
os.environ.setdefault("DATA_MODE", "sample")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    """Directory holding the committed synthetic fixtures."""
    return PROJECT_ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def application_train(sample_dir: Path):
    """The synthetic ``application_train`` table."""
    import pandas as pd

    return pd.read_csv(sample_dir / "application_train.csv")


@pytest.fixture(scope="session")
def bureau(sample_dir: Path):
    """The synthetic ``bureau`` table."""
    import pandas as pd

    return pd.read_csv(sample_dir / "bureau.csv")


@pytest.fixture(scope="session")
def joined_dataset():
    """The application table LEFT JOINed with aggregated bureau features."""
    from src.data.loader import build_dataset

    return build_dataset("train")


@pytest.fixture(scope="session")
def fitted_preprocessor(joined_dataset):
    """A :class:`CreditPreprocessor` fitted on the sample training split."""
    from src.data.preprocessor import CreditPreprocessor

    return CreditPreprocessor().fit(joined_dataset)


@pytest.fixture(scope="session")
def feature_matrix(fitted_preprocessor, joined_dataset):
    """The model-ready feature frame produced from the sample data."""
    return fitted_preprocessor.transform(joined_dataset)


@pytest.fixture(scope="session")
def trained_artifacts(joined_dataset):
    """Train the full bake-off once and persist artifacts for the session.

    Saving to disk is required because :func:`src.ml.predict.load_bundle` is the
    real inference path and reads from ``models/``. Artifacts are gitignored and
    fully regenerable, so overwriting them is harmless.
    """
    from src.ml import train as train_module
    from src.ml.predict import load_bundle

    outcome = train_module.train(save=True)
    load_bundle.cache_clear()  # pick up the artifacts just written
    return outcome


@pytest.fixture(scope="session")
def analytics_db(tmp_path_factory):
    """A populated SQLite analytics database.

    The talk-to-data suite must run with no Postgres server and no model
    runtime, so the whole stack is exercised against SQLite built from the same
    loader that populates Postgres in production.
    """
    from sqlalchemy import create_engine

    from src.data.db_loader import load_database

    path = tmp_path_factory.mktemp("db") / "analytics.db"
    engine = create_engine(f"sqlite:///{path}", future=True)
    load_database(include_predictions=True, engine=engine)
    return engine


@pytest.fixture(scope="session")
def live_schema(analytics_db):
    """The schema whitelist read back from the populated test database."""
    from src.talk_to_data.query_runner import fetch_schema

    return fetch_schema(analytics_db)


class FakeLLMClient:
    """Scripted LLM stand-in.

    Talk-to-data logic must be testable without a model: real generation is
    non-deterministic and slow, and neither property belongs in a unit test.
    """

    provider_name = "fake"

    def __init__(self, responses: list[str] | None = None, available: bool = True) -> None:
        self.responses = list(responses or [])
        self.available = available
        self.calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return "fake-model"

    def is_available(self) -> tuple[bool, str]:
        return self.available, "fake client ready" if self.available else "fake client disabled"

    @property
    def generation_calls(self) -> list[tuple[str, str]]:
        """Only the SQL-generation calls.

        The same client also serves summarisation, so tests that count
        generation attempts must filter -- otherwise a summary call looks like
        an extra retry.
        """
        return [call for call in self.calls if "SQL analyst" in call[0]]

    @property
    def summary_calls(self) -> list[tuple[str, str]]:
        """Only the summarisation calls."""
        return [call for call in self.calls if "summarising" in call[0]]

    def complete(self, system_prompt: str, user_prompt: str):
        from src.talk_to_data.llm_client import LLMResponse

        self.calls.append((system_prompt, user_prompt))
        is_summary = "summarising" in system_prompt
        if is_summary:
            # A canned, grounded summary: these tests exercise orchestration,
            # not summary quality, and an ungrounded string would trip the
            # grounding guard and change what is being tested.
            text = "The query returned results."
        else:
            text = self.responses.pop(0) if self.responses else "SELECT 1 AS x FROM applications"
        return LLMResponse(
            text=text, provider=self.provider_name, model=self.model_name,
            prompt_tokens=len(user_prompt) // 4, completion_tokens=len(text) // 4,
        )


@pytest.fixture
def fake_llm():
    """Factory for scripted LLM clients."""
    return FakeLLMClient
