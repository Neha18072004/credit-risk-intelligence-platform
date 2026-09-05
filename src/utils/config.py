"""Central application configuration.

All runtime configuration is read from environment variables (loaded from a
local ``.env`` file when present).  Nothing in this project hardcodes a secret,
host, model name or filesystem path -- every such value flows through the
:class:`Settings` object exported here as the module-level ``settings``
singleton.

Usage::

    from src.utils.config import settings

    print(settings.data_mode)
    engine = create_engine(settings.database_url)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: this file lives at <root>/src/utils/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class DataMode(str, Enum):
    """Which dataset the pipeline reads."""

    SAMPLE = "sample"
    REAL = "real"


class LLMProvider(str, Enum):
    """Supported LLM back-ends.

    ``OLLAMA`` is the default and needs no API key: the model runs locally as a
    Docker Compose service alongside the app. Applicant data therefore never
    leaves the host, which is the correct posture for a credit-risk system, and
    the whole platform runs with ``docker-compose up`` and no credentials.

    The hosted providers are optional overrides for anyone who prefers them.
    ``NONE`` disables the chat feature cleanly.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    NONE = "none"

    @property
    def requires_api_key(self) -> bool:
        """True for hosted providers; the local runtime needs no credential."""
        return self in {LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GEMINI}


class Settings(BaseSettings):
    """Typed, validated application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- app ---
    app_name: str = "Credit Risk Intelligence Platform"
    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"

    # --------------------------------------------------------------- data ---
    data_mode: DataMode = DataMode.SAMPLE
    data_dir: Path = PROJECT_ROOT / "data"
    sample_data_dir: Path = PROJECT_ROOT / "data" / "sample"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    max_rows: int = Field(default=0, ge=0, description="0 = read every row")

    # ----------------------------------------------------------- postgres ---
    postgres_user: str = "credit"
    postgres_password: str = "credit_pass_change_me"
    postgres_db: str = "credit_risk"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_readonly_user: str = "credit_readonly"
    postgres_readonly_password: str = "readonly_pass_change_me"

    use_sqlite_fallback: bool = False
    sqlite_path: Path = PROJECT_ROOT / "data" / "credit_risk.db"

    # ---------------------------------------------------------------- llm ---
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    # qwen2.5-coder is chosen for its text-to-SQL accuracy at 7B; llama3.1:8b is
    # the general-purpose fallback if the preferred model cannot be pulled.
    llm_model: str = "qwen2.5-coder:7b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_timeout_seconds: int = Field(default=120, gt=0)

    # --- local runtime (default; no credential required) ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_fallback_model: str = "llama3.1:8b"
    ollama_keep_alive: str = "10m"

    # --- optional hosted overrides ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # ---------------------------------------------------- sql guardrails ---
    sql_max_rows: int = Field(default=200, gt=0)
    sql_timeout_seconds: int = Field(default=15, gt=0)
    sql_allow_only_select: bool = True
    memory_max_turns: int = Field(default=6, ge=0)

    # ----------------------------------------------------------- training ---
    random_seed: int = 42
    cv_folds: int = Field(default=5, ge=2)
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"
    risk_score_scale: int = Field(default=1000, gt=0)
    risk_band_low_max: float = Field(default=0.05, gt=0.0, lt=1.0)
    risk_band_medium_max: float = Field(default=0.15, gt=0.0, lt=1.0)
    surrogate_tree_max_depth: int = Field(default=4, ge=2, le=6)

    # ---------------------------------------------------------- streamlit ---
    streamlit_server_port: int = 8501
    streamlit_server_address: str = "0.0.0.0"

    # --------------------------------------------------------- validators ---
    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper().strip()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}, got {value!r}")
        return upper

    @field_validator("data_dir", "sample_data_dir", "models_dir", "reports_dir", "sqlite_path")
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        """Resolve relative paths against the project root, not the CWD.

        This keeps behaviour identical whether the app is launched from the repo
        root, from ``src/``, or from inside the Docker image.
        """
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    # -------------------------------------------------- derived properties --
    @property
    def active_data_dir(self) -> Path:
        """Directory the loader should read from, per :attr:`data_mode`."""
        return self.sample_data_dir if self.data_mode is DataMode.SAMPLE else self.data_dir

    @property
    def figures_dir(self) -> Path:
        """Where EDA charts are written (and read back by the UI)."""
        return self.reports_dir / "figures"

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL for the read/write application role."""
        if self.use_sqlite_fallback:
            return f"sqlite:///{self.sqlite_path}"
        return (
            f"postgresql+psycopg2://{quote_plus(self.postgres_user)}:"
            f"{quote_plus(self.postgres_password)}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def readonly_database_url(self) -> str:
        """SQLAlchemy URL for the restricted role used by NL->SQL execution.

        SQLite has no role system, so the fallback path returns the same URL;
        safety there rests entirely on :mod:`src.talk_to_data.sql_validator`.
        """
        if self.use_sqlite_fallback:
            return f"sqlite:///{self.sqlite_path}"
        return (
            f"postgresql+psycopg2://{quote_plus(self.postgres_readonly_user)}:"
            f"{quote_plus(self.postgres_readonly_password)}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def llm_api_key(self) -> str:
        """API key for the selected provider ("" for local or unset providers)."""
        return {
            LLMProvider.OLLAMA: "",
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.GEMINI: self.google_api_key,
            LLMProvider.NONE: "",
        }[self.llm_provider].strip()

    @property
    def llm_enabled(self) -> bool:
        """Whether the talk-to-data feature can run as configured.

        The local provider is always enabled -- it needs no credential, so the
        default install works out of the box. A hosted provider is enabled only
        when its key is actually present; otherwise the chat degrades to a clear
        message instead of failing at request time.
        """
        if self.llm_provider is LLMProvider.NONE:
            return False
        if not self.llm_provider.requires_api_key:
            return True
        return bool(self.llm_api_key)

    @property
    def llm_disabled_reason(self) -> str:
        """Operator-facing explanation of why chat is unavailable ("" if it is)."""
        if self.llm_enabled:
            return ""
        if self.llm_provider is LLMProvider.NONE:
            return (
                "LLM_PROVIDER is set to 'none'. Set LLM_PROVIDER=ollama to use the "
                "bundled local model (no API key required)."
            )
        key_variable = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.GEMINI: "GOOGLE_API_KEY",
        }[self.llm_provider]
        return (
            f"LLM_PROVIDER is '{self.llm_provider.value}' but {key_variable} is not set. "
            "Either set that key, or switch to LLM_PROVIDER=ollama to run the bundled "
            "local model with no credential."
        )

    def ensure_directories(self) -> None:
        """Create the writable output directories if they do not yet exist."""
        for directory in (self.models_dir, self.reports_dir, self.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton."""
    return Settings()


settings: Settings = get_settings()
