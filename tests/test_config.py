"""Configuration layer smoke tests."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import DataMode, LLMProvider, Settings, settings


def test_settings_import() -> None:
    """The documented Phase 1 acceptance check: settings import cleanly."""
    assert settings.app_name
    assert settings.cv_folds >= 2


def test_paths_are_absolute() -> None:
    """Relative .env paths resolve against the repo root, not the CWD."""
    for path in (settings.data_dir, settings.sample_data_dir,
                 settings.models_dir, settings.reports_dir):
        assert isinstance(path, Path) and path.is_absolute()


def test_active_data_dir_follows_mode() -> None:
    """DATA_MODE switches the read location without any other code change."""
    sample = Settings(data_mode=DataMode.SAMPLE)
    real = Settings(data_mode=DataMode.REAL)
    assert sample.active_data_dir == sample.sample_data_dir
    assert real.active_data_dir == real.data_dir


def test_database_urls_are_distinct_roles() -> None:
    """The NL->SQL path must connect as a different, restricted role."""
    config = Settings(use_sqlite_fallback=False)
    assert config.postgres_user in config.database_url
    assert config.postgres_readonly_user in config.readonly_database_url
    assert config.database_url != config.readonly_database_url


def test_sqlite_fallback_url() -> None:
    """The fallback path produces a valid SQLite URL."""
    config = Settings(use_sqlite_fallback=True)
    assert config.database_url.startswith("sqlite:///")


def test_llm_disabled_without_key() -> None:
    """A selected provider with no key must report itself as disabled."""
    assert Settings(llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key="").llm_enabled is False
    assert Settings(llm_provider=LLMProvider.NONE).llm_enabled is False
    assert Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="sk-x").llm_enabled is True


def test_llm_api_key_follows_provider() -> None:
    """Each provider reads its own key, never another provider's."""
    config = Settings(
        llm_provider=LLMProvider.GEMINI, google_api_key="g-key", openai_api_key="o-key"
    )
    assert config.llm_api_key == "g-key"
