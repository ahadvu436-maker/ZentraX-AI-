"""
ZentraX AI — Application Configuration
========================================
Centralized, type-safe environment configuration using Pydantic v2 Settings.

All values are loaded from environment variables / a `.env` file and validated
at startup. This module should be imported as a singleton via `get_settings()`
so configuration is parsed once and cached for the lifetime of the process.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import (
    AnyHttpUrl,
    Field,
    PostgresDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings.

    Values are resolved in this order of precedence (highest first):
      1. Actual environment variables
      2. Variables defined in `.env`
      3. Defaults declared below

    NOTE: Secrets (JWT_SECRET_KEY, DB password, etc.) have NO defaults in
    production-sensitive fields — the app will fail fast at startup if they
    are missing, rather than silently running insecurely.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Core / Environment
    # ------------------------------------------------------------------
    PROJECT_NAME: str = "ZentraX AI"
    ENVIRONMENT: Environment = Environment.LOCAL
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: LogLevel = LogLevel.INFO

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)

    # ------------------------------------------------------------------
    # Security / JWT
    # ------------------------------------------------------------------
    JWT_SECRET_KEY: SecretStr = Field(
        ...,
        description="Secret key used to sign JWTs. MUST be set via env var — no default.",
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, gt=0)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, gt=0)

    # Field-level encryption key for PII at rest (privacy-first requirement)
    DATA_ENCRYPTION_KEY: SecretStr = Field(
        ...,
        description="Symmetric key (e.g. Fernet) for encrypting sensitive fields at rest.",
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    BACKEND_CORS_ORIGINS: list[AnyHttpUrl] = Field(default_factory=list)

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> list[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # ------------------------------------------------------------------
    # Database — PostgreSQL + pgvector (async)
    # ------------------------------------------------------------------
    POSTGRES_SCHEME: str = "postgresql+asyncpg"
    POSTGRES_USER: str = Field(..., description="Database username")
    POSTGRES_PASSWORD: SecretStr = Field(..., description="Database password")
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = Field(default=5432, ge=1, le=65535)
    POSTGRES_DB: str = Field(..., description="Database name")

    # Optional: fully-formed DSN overrides the individual fields above if provided
    DATABASE_URL: PostgresDsn | None = None

    # Connection pool tuning
    DB_POOL_SIZE: int = Field(default=10, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0)
    DB_POOL_TIMEOUT_SECONDS: int = Field(default=30, ge=1)
    DB_POOL_RECYCLE_SECONDS: int = Field(default=1800, ge=1)
    DB_ECHO_SQL: bool = False

    # pgvector
    PGVECTOR_EXTENSION: str = "vector"
    EMBEDDING_DIMENSIONS: int = Field(
        default=1536,
        gt=0,
        description="Vector dimension size; must match the embedding model in use.",
    )

    @model_validator(mode="after")
    def assemble_database_dsn(self) -> "Settings":
        if self.DATABASE_URL is None:
            self.DATABASE_URL = PostgresDsn.build(
                scheme=self.POSTGRES_SCHEME,
                username=self.POSTGRES_USER,
                password=self.POSTGRES_PASSWORD.get_secret_value(),
                host=self.POSTGRES_HOST,
                port=self.POSTGRES_PORT,
                path=self.POSTGRES_DB,
            )
        return self

    # ------------------------------------------------------------------
    # Redis (cache / rate-limiting / sessions)
    # ------------------------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, gt=0)

    # ------------------------------------------------------------------
    # Privacy / Compliance
    # ------------------------------------------------------------------
    DATA_RETENTION_DAYS: int = Field(default=90, ge=0)
    ENABLE_AUDIT_LOGGING: bool = True
    ANONYMIZE_LOGS: bool = True

    # ------------------------------------------------------------------
    # Derived / convenience properties
    # ------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION

    @property
    def sqlalchemy_database_uri(self) -> str:
        assert self.DATABASE_URL is not None
        return str(self.DATABASE_URL)

    @model_validator(mode="after")
    def _validate_production_safety(self) -> "Settings":
        if self.is_production and self.DEBUG:
            raise ValueError("DEBUG must be False when ENVIRONMENT=production")
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor.

    Use this as a FastAPI dependency (`Depends(get_settings)`) or import
    directly. `lru_cache` ensures the .env file is parsed only once.
    """
    return Settings()


settings = get_settings()