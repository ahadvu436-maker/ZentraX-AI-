"""Root settings: env vars (``APP_`` prefix, ``__`` for nesting) > .env > defaults.

Example: ``APP_DATABASE__HOST=db`` ``APP_SECURITY__SECRET_KEY=...``
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    APP_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_REJECT_RATIO,
    DEFAULT_MAX_WORKERS,
    ENV_NESTED_DELIMITER,
    ENV_PREFIX,
)
from .database_config import DatabaseSettings
from .environment import Environment
from .logging_config import LoggingSettings
from .model_config import ModelSettings
from .security_config import SecuritySettings


class PipelineSettings(BaseModel):
    data_dir: Path = Path("data")
    dataset_root: Path = Path("data/versions")
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1)
    max_workers: int = Field(DEFAULT_MAX_WORKERS, ge=1, le=64)
    max_reject_ratio: float = Field(DEFAULT_MAX_REJECT_RATIO, ge=0.0, le=1.0)


class ObservabilitySettings(BaseModel):
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    otlp_endpoint: str | None = None
    sentry_dsn: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    environment: Environment = Environment.DEVELOPMENT
    service_name: str = APP_NAME
    debug: bool = False

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @model_validator(mode="after")
    def _enforce_deployed_safety(self) -> "Settings":
        if self.environment.is_deployed:
            problems: list[str] = []
            if self.debug:
                problems.append("debug must be false")
            if self.security.has_default_secret:
                problems.append("security.secret_key must be set (>=32 chars)")
            if self.environment.is_production and self.database.sslmode == "disable":
                problems.append("database.sslmode must not be 'disable'")
            if problems:
                raise ValueError(f"Unsafe {self.environment.value} configuration: " + "; ".join(problems))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached process-wide settings. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
