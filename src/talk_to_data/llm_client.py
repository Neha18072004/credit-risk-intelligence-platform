"""Provider-agnostic LLM client.

One interface, four back-ends, selected by ``LLM_PROVIDER``. The default is
**Ollama running locally**, which matters for two reasons:

* **Data residency.** Talk-to-data sends schema context and query results --
  derived from real applicant records -- to the model. On the local runtime that
  never leaves the host, which is the correct posture for a credit system.
* **Zero friction.** ``docker-compose up`` yields a working chatbot with no API
  key, no billing account and no signup.

The hosted providers are optional overrides. Selecting one without its key does
not crash anything: :func:`get_llm_client` returns a client that reports itself
unavailable with an actionable message, and the rest of the app carries on.

Ollama is reached over its plain HTTP API using only the standard library, so
adding the local path costs no new dependency.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Final

from src.utils.config import LLMProvider, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ollama pulls a multi-GB model on first use, so the availability probe is quick
# but generation is allowed to be slow.
PROBE_TIMEOUT_SECONDS: Final[float] = 5.0


@dataclass
class LLMResponse:
    """A single completion and what it cost."""

    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMUnavailableError(RuntimeError):
    """Raised when a completion is attempted against an unusable provider."""


class LLMClient(ABC):
    """Common surface every provider implements."""

    provider_name: str = "unknown"

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a single completion for the given prompts."""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Return ``(available, message)`` without raising.

        Called by the UI on load so the Chat tab can explain itself rather than
        failing on the user's first question.
        """

    @property
    def model_name(self) -> str:
        return settings.llm_model


# --------------------------------------------------------------------------- #
# Local runtime (default)
# --------------------------------------------------------------------------- #
class OllamaClient(LLMClient):
    """Local Ollama runtime. No API key, no network egress.

    Args:
        base_url: Ollama server address.
        model: Model tag to generate with.
        fallback_model: Used when ``model`` is not present on the server.
    """

    provider_name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.fallback_model = fallback_model or settings.ollama_fallback_model
        self._resolved_model: str | None = None

    # ------------------------------------------------------------ internal --
    def _request(self, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        """Issue a JSON request against the Ollama API."""
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST" if data else "GET",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def available_models(self) -> list[str]:
        """List model tags present on the server ( empty if unreachable )."""
        try:
            payload = self._request("/api/tags", None, PROBE_TIMEOUT_SECONDS)
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            return []
        return [entry.get("name", "") for entry in payload.get("models", [])]

    def resolve_model(self) -> str:
        """Pick the configured model, or the fallback if it is not installed.

        Matching ignores an absent ``:latest`` suffix, because Ollama reports
        tags in a normalised form that does not always match what was requested.
        """
        if self._resolved_model:
            return self._resolved_model

        installed = self.available_models()
        if not installed:
            self._resolved_model = self.model
            return self._resolved_model

        def present(tag: str) -> bool:
            candidates = {tag, f"{tag}:latest", tag.removesuffix(":latest")}
            return any(name in candidates for name in installed)

        if present(self.model):
            self._resolved_model = self.model
        elif present(self.fallback_model):
            logger.warning(
                "Model %s not installed; falling back to %s", self.model, self.fallback_model
            )
            self._resolved_model = self.fallback_model
        else:
            logger.warning(
                "Neither %s nor %s is installed; using the first available model",
                self.model, self.fallback_model,
            )
            self._resolved_model = installed[0] if installed else self.model
        return self._resolved_model

    # -------------------------------------------------------------- public --
    def is_available(self) -> tuple[bool, str]:
        installed = self.available_models()
        if not installed:
            return False, (
                f"Cannot reach the local model runtime at {self.base_url}. "
                "If you are running outside Docker, start it with 'ollama serve' and set "
                "OLLAMA_BASE_URL=http://localhost:11434. Under docker-compose the 'ollama' "
                "service provides this automatically."
            )
        resolved = self.resolve_model()
        return True, f"Local runtime ready at {self.base_url} using '{resolved}'."

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        model = self.resolve_model()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        }
        started = time.perf_counter()
        try:
            body = self._request("/api/chat", payload, settings.llm_timeout_seconds)
        except urllib.error.HTTPError as error:
            raise LLMUnavailableError(
                f"Local model '{model}' returned HTTP {error.code}. "
                f"It may still be downloading -- try again shortly."
            ) from error
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise LLMUnavailableError(
                f"Could not reach the local model runtime at {self.base_url}: {error}"
            ) from error

        return LLMResponse(
            text=body.get("message", {}).get("content", "").strip(),
            provider=self.provider_name,
            model=model,
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
            latency_seconds=time.perf_counter() - started,
        )


# --------------------------------------------------------------------------- #
# Hosted providers (optional overrides)
# --------------------------------------------------------------------------- #
class OpenAIClient(LLMClient):
    """OpenAI chat completions."""

    provider_name = "openai"

    def is_available(self) -> tuple[bool, str]:
        if not settings.openai_api_key:
            return False, "OPENAI_API_KEY is not set."
        return True, f"OpenAI ready with '{self.model_name}'."

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not settings.openai_api_key:
            raise LLMUnavailableError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds)
        started = time.perf_counter()
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        usage = completion.usage
        return LLMResponse(
            text=(completion.choices[0].message.content or "").strip(),
            provider=self.provider_name,
            model=self.model_name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_seconds=time.perf_counter() - started,
        )


class AnthropicClient(LLMClient):
    """Anthropic messages API."""

    provider_name = "anthropic"

    def is_available(self) -> tuple[bool, str]:
        if not settings.anthropic_api_key:
            return False, "ANTHROPIC_API_KEY is not set."
        return True, f"Anthropic ready with '{self.model_name}'."

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not settings.anthropic_api_key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set.")
        import anthropic

        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=float(settings.llm_timeout_seconds)
        )
        started = time.perf_counter()
        message = client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        text = "".join(block.text for block in message.content if hasattr(block, "text"))
        return LLMResponse(
            text=text.strip(),
            provider=self.provider_name,
            model=self.model_name,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            latency_seconds=time.perf_counter() - started,
        )


class GeminiClient(LLMClient):
    """Google Gemini."""

    provider_name = "gemini"

    def is_available(self) -> tuple[bool, str]:
        if not settings.google_api_key:
            return False, "GOOGLE_API_KEY is not set."
        return True, f"Gemini ready with '{self.model_name}'."

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not settings.google_api_key:
            raise LLMUnavailableError("GOOGLE_API_KEY is not set.")
        import google.generativeai as genai

        genai.configure(api_key=settings.google_api_key)
        model = genai.GenerativeModel(
            model_name=self.model_name, system_instruction=system_prompt
        )
        started = time.perf_counter()
        response = model.generate_content(
            user_prompt,
            generation_config={
                "temperature": settings.llm_temperature,
                "max_output_tokens": settings.llm_max_tokens,
            },
        )
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=(response.text or "").strip(),
            provider=self.provider_name,
            model=self.model_name,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_seconds=time.perf_counter() - started,
        )


class DisabledClient(LLMClient):
    """Stand-in used when no provider is usable.

    Exists so that the absence of an LLM is a *feature state* rather than an
    error path: the UI asks :meth:`is_available`, shows the reason, and every
    non-chat feature keeps working.
    """

    provider_name = "disabled"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def is_available(self) -> tuple[bool, str]:
        return False, self.reason

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        raise LLMUnavailableError(self.reason)


_CLIENTS: Final[dict[LLMProvider, type[LLMClient]]] = {
    LLMProvider.OLLAMA: OllamaClient,
    LLMProvider.OPENAI: OpenAIClient,
    LLMProvider.ANTHROPIC: AnthropicClient,
    LLMProvider.GEMINI: GeminiClient,
}


def get_llm_client(provider: LLMProvider | None = None) -> LLMClient:
    """Return the client for the configured provider.

    Never raises. A missing key or an unknown provider yields a
    :class:`DisabledClient` carrying an actionable explanation, because the app
    must start and remain usable without an LLM.

    Args:
        provider: Override the configured provider.

    Returns:
        A ready :class:`LLMClient`.
    """
    selected = provider or settings.llm_provider

    if selected is LLMProvider.NONE:
        return DisabledClient(settings.llm_disabled_reason)
    if selected.requires_api_key and not settings.llm_api_key:
        return DisabledClient(settings.llm_disabled_reason)

    client_type = _CLIENTS.get(selected)
    if client_type is None:  # pragma: no cover - guarded by the settings enum
        return DisabledClient(f"Unknown LLM provider '{selected}'.")

    logger.debug("Using %s provider with model %s", selected.value, settings.llm_model)
    return client_type()
