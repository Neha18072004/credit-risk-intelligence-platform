"""Tests for the provider-agnostic LLM client and its degradation behaviour."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest

from src.talk_to_data.llm_client import (
    AnthropicClient,
    DisabledClient,
    GeminiClient,
    LLMUnavailableError,
    OllamaClient,
    OpenAIClient,
    get_llm_client,
)
from src.utils.config import LLMProvider, Settings


def test_default_provider_is_local(monkeypatch) -> None:
    """The shipped default must need no credential."""
    client = get_llm_client(LLMProvider.OLLAMA)
    assert isinstance(client, OllamaClient)


@pytest.mark.parametrize(
    "provider", [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GEMINI]
)
def test_hosted_provider_without_key_degrades(monkeypatch, provider) -> None:
    """Selecting a hosted provider with no key must not crash the app."""
    monkeypatch.setattr(
        "src.talk_to_data.llm_client.settings",
        Settings(llm_provider=provider, openai_api_key="", anthropic_api_key="", google_api_key=""),
    )
    client = get_llm_client(provider)
    assert isinstance(client, DisabledClient)
    available, message = client.is_available()
    assert not available
    assert message


def test_provider_none_degrades() -> None:
    client = get_llm_client(LLMProvider.NONE)
    assert isinstance(client, DisabledClient)
    assert not client.is_available()[0]


def test_disabled_client_raises_only_on_use() -> None:
    """Construction must never raise; only an actual completion attempt does."""
    client = DisabledClient("no provider configured")
    assert not client.is_available()[0]
    with pytest.raises(LLMUnavailableError, match="no provider"):
        client.complete("system", "user")


# ------------------------------------------------------------- ollama ----
def test_ollama_reports_unavailable_when_unreachable() -> None:
    client = OllamaClient(base_url="http://127.0.0.1:1")
    available, message = client.is_available()
    assert not available
    assert "ollama serve" in message or "runtime" in message.lower()


def test_ollama_falls_back_to_secondary_model() -> None:
    """A missing preferred model must not break the feature."""
    client = OllamaClient(model="qwen2.5-coder:7b", fallback_model="llama3.1:8b")
    with patch.object(client, "available_models", return_value=["llama3.1:8b"]):
        assert client.resolve_model() == "llama3.1:8b"


def test_ollama_prefers_the_configured_model() -> None:
    client = OllamaClient(model="qwen2.5-coder:7b", fallback_model="llama3.1:8b")
    with patch.object(
        client, "available_models", return_value=["qwen2.5-coder:7b", "llama3.1:8b"]
    ):
        assert client.resolve_model() == "qwen2.5-coder:7b"


def test_ollama_tolerates_latest_suffix() -> None:
    """Ollama normalises tags, so matching must ignore an absent ':latest'."""
    client = OllamaClient(model="qwen2.5-coder:7b", fallback_model="llama3.1:8b")
    with patch.object(client, "available_models", return_value=["qwen2.5-coder:7b:latest"]):
        assert client.resolve_model() == "qwen2.5-coder:7b"


def test_ollama_uses_any_installed_model_as_last_resort() -> None:
    client = OllamaClient(model="absent-a", fallback_model="absent-b")
    with patch.object(client, "available_models", return_value=["mistral:7b"]):
        assert client.resolve_model() == "mistral:7b"


def test_ollama_completion_parses_usage() -> None:
    client = OllamaClient()
    payload = {
        "message": {"content": "SELECT 1"},
        "prompt_eval_count": 120,
        "eval_count": 8,
    }
    with patch.object(client, "_request", return_value=payload), patch.object(
        client, "resolve_model", return_value="test-model"
    ):
        response = client.complete("system", "user")
    assert response.text == "SELECT 1"
    assert response.prompt_tokens == 120
    assert response.completion_tokens == 8
    assert response.total_tokens == 128
    assert response.provider == "ollama"


def test_ollama_connection_error_becomes_actionable(monkeypatch) -> None:
    client = OllamaClient()
    with patch.object(client, "resolve_model", return_value="m"), patch.object(
        client, "_request", side_effect=urllib.error.URLError("connection refused")
    ):
        with pytest.raises(LLMUnavailableError, match="Could not reach"):
            client.complete("system", "user")


def test_ollama_lists_no_models_when_server_is_down() -> None:
    assert OllamaClient(base_url="http://127.0.0.1:1").available_models() == []


# ------------------------------------------------------------- hosted ----
def test_hosted_clients_report_missing_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.talk_to_data.llm_client.settings",
        Settings(openai_api_key="", anthropic_api_key="", google_api_key=""),
    )
    for client in (OpenAIClient(), AnthropicClient(), GeminiClient()):
        available, message = client.is_available()
        assert not available
        assert "API_KEY" in message
        with pytest.raises(LLMUnavailableError):
            client.complete("system", "user")


def test_ollama_distinguishes_downloading_from_unreachable() -> None:
    """The first-run state is 'still downloading', not 'broken'.

    A server that is up with no model yet is the normal state for several
    minutes after `docker-compose up`, and reporting it as unreachable would
    send a new user debugging a problem that does not exist.
    """
    client = OllamaClient()

    with patch.object(client, "available_models", return_value=[]), patch.object(
        client, "server_reachable", return_value=True
    ):
        available, message = client.is_available()
        assert not available
        assert "downloading" in message.lower()
        assert "ollama-pull" in message
        assert "Every other feature works" in message

    with patch.object(client, "available_models", return_value=[]), patch.object(
        client, "server_reachable", return_value=False
    ):
        available, message = client.is_available()
        assert not available
        assert "Cannot reach" in message
