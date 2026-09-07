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


def test_default_provider_is_local_and_keyless() -> None:
    """The shipped default must work with no credential at all.

    This is the contract behind "docker-compose up and nothing else": the
    bundled local model runs on-box, so applicant data never leaves the host
    and an evaluator needs no API key to exercise the chat feature.
    """
    config = Settings(_env_file=None)
    assert config.llm_provider is LLMProvider.OLLAMA
    assert config.llm_enabled is True
    assert config.llm_api_key == ""
    assert config.llm_disabled_reason == ""


def test_only_hosted_providers_require_a_key() -> None:
    assert LLMProvider.OLLAMA.requires_api_key is False
    assert LLMProvider.NONE.requires_api_key is False
    for provider in (LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GEMINI):
        assert provider.requires_api_key is True


def test_llm_disabled_without_key() -> None:
    """A hosted provider with no key reports as disabled, and never crashes."""
    assert Settings(llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key="").llm_enabled is False
    assert Settings(llm_provider=LLMProvider.NONE).llm_enabled is False
    assert Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="sk-x").llm_enabled is True


def test_disabled_reason_is_actionable() -> None:
    """The degraded-mode message must name the fix, not just state failure."""
    reason = Settings(llm_provider=LLMProvider.GEMINI, google_api_key="").llm_disabled_reason
    assert "GOOGLE_API_KEY" in reason
    assert "ollama" in reason.lower()

    # The 'none' message must serve both deployment shapes: the full stack has
    # a local model, a lightweight cloud deployment does not.
    none_reason = Settings(llm_provider=LLMProvider.NONE).llm_disabled_reason
    assert "docker compose up" in none_reason.lower()
    assert "openai" in none_reason.lower()
    assert "every other section" in none_reason.lower()


def test_llm_api_key_follows_provider() -> None:
    """Each provider reads its own key, never another provider's."""
    config = Settings(
        llm_provider=LLMProvider.GEMINI, google_api_key="g-key", openai_api_key="o-key"
    )
    assert config.llm_api_key == "g-key"


def test_env_example_documents_every_setting() -> None:
    """A setting that exists in code but not in .env.example is undiscoverable.

    The brief asks for ".env.example with all required environment variables",
    so this is checked rather than maintained by hand.
    """
    import re

    from src.utils.config import PROJECT_ROOT

    fields = {name.upper() for name in Settings.model_fields}
    documented = set(
        re.findall(r"^([A-Z][A-Z0-9_]*)=", (PROJECT_ROOT / ".env.example").read_text(), re.M)
    )
    assert not (fields - documented), f"undocumented settings: {sorted(fields - documented)}"
    assert not (documented - fields), f"stale entries: {sorted(documented - fields)}"


def test_tests_do_not_write_to_the_real_artifact_directories() -> None:
    """The suite must not overwrite a trained model with a fixture-trained one.

    Regression test: running pytest replaced the model trained on the full
    307k-row dataset with one fitted to the 4,000-row synthetic fixtures, and
    the only symptom was the reported metrics quietly changing.
    """
    from src.utils.config import PROJECT_ROOT, settings

    assert PROJECT_ROOT / "models" != settings.models_dir
    assert PROJECT_ROOT / "reports" != settings.reports_dir
