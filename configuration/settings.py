"""Central settings orchestrator.

``Settings.load()`` resolves the environment, merges ``.env`` files and the
process environment, builds every section (logging, database, model, security)
and then applies cross-section rules for staging / production.

Typical use::

    from Configuration import get_settings

    settings = get_settings()          # cached, immutable
    settings.configure_logging()
    engine = create_engine(settings.database.url.get_secret_value(),
                           **settings.database.engine_kwargs())
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .constants import (
    APP_NAME_DEFAULT,
    APP_VERSION_DEFAULT,
    BASE_DIR,
    DEFAULT_DATA_DIR,
    ENV_PREFIX_APP,
    ENV_VAR_ENVIRONMENT,
    ENV_VAR_SECRET_KEY,
    SEMVER_PATTERN,
    DatabaseSSLMode,
    LogLevel,
)
from .database_config import DatabaseSettings
from .environment import (
    ConfigurationError,
    EnvBaseModel,
    Environment,
    load_environment,
    resolve_path,
)
from .logging_config import LoggingSettings, configure_logging
from .model_config import ModelSettings
from .security_config import SecuritySettings, generate_secret_key

_log = logging.getLogger(__name__)


class AppSettings(EnvBaseModel):
    """General application options (env prefix ``APP_``)."""

    env_prefix: ClassVar[str] = ENV_PREFIX_APP
    env_reserved: ClassVar[frozenset[str]] = frozenset({ENV_VAR_ENVIRONMENT})

    name: str = Field(default=APP_NAME_DEFAULT, min_length=1, max_length=100)
    version: str = Field(default=APP_VERSION_DEFAULT, pattern=SEMVER_PATTERN)
    debug: bool = False
    host: str = Field(default="127.0.0.1", min_length=1, max_length=255)
    port: int = Field(default=8000, ge=1, le=65_535)
    workers: int = Field(default=1, ge=1, le=64)
    data_dir: Path = DEFAULT_DATA_DIR

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        return resolve_path(value)


class Settings(BaseModel):
    """Immutable aggregate of every configuration section."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    environment: Environment
    app: AppSettings
    logging: LoggingSettings
    database: DatabaseSettings
    model: ModelSettings
    security: SecuritySettings

    # ------------------------------------------------------------------ #
    # Cross-section rules
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _enforce_environment_rules(self) -> Self:
        problems: list[str] = []

        if self.environment.is_strict:
            if self.app.debug:
                problems.append("APP_DEBUG must be false")
            if self.database.is_sqlite:
                problems.append("DATABASE_URL must not use SQLite")
            if self.security.allows_any_origin:
                problems.append("SECURITY_CORS_ORIGINS must not contain '*'")
            if self.security.allows_any_host:
                problems.append("SECURITY_ALLOWED_HOSTS must not contain '*'")
            if not self.security.cookie_secure:
                problems.append("SECURITY_COOKIE_SECURE must be true")
            if not self.security.tls_required:
                problems.append("SECURITY_TLS_REQUIRED must be true")

        if self.environment.is_production:
            if (
                not self.database.is_sqlite
                and self.database.ssl_mode is DatabaseSSLMode.DISABLE
            ):
                problems.append("DATABASE_SSL_MODE must not be 'disable'")
            if self.security.encryption_key is None:
                problems.append("SECURITY_ENCRYPTION_KEY is required")
            if self.logging.level is LogLevel.DEBUG:
                problems.append("LOG_LEVEL must not be DEBUG")

        if problems:
            listing = "\n".join(f"  - {p}" for p in problems)
            raise ValueError(
                f"Unsafe configuration for {self.environment.value!r}:\n{listing}"
            )
        return self

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        environment: str | Environment | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        env_dir: Path | str | None = None,
    ) -> Self:
        """Load and validate all settings.

        Args:
            environment: Force a mode; otherwise ``APP_ENV`` decides.
            environ: Variables to use instead of ``os.environ`` (tests).
            env_dir: Directory containing ``.env`` files (default: project root).

        Raises:
            ConfigurationError: Missing, malformed or unsafe configuration.
        """
        env, variables = load_environment(environment, environ=environ, env_dir=env_dir)

        if ENV_VAR_SECRET_KEY not in variables or not variables[ENV_VAR_SECRET_KEY].strip():
            if env.is_strict:
                raise ConfigurationError(
                    f"{ENV_VAR_SECRET_KEY} is required in {env.value}. "
                    "Generate one with: python -c "
                    '"from Configuration.security_config import generate_secret_key; '
                    'print(generate_secret_key())"'
                )
            variables[ENV_VAR_SECRET_KEY] = generate_secret_key()
            _log.warning(
                "%s not set; using an ephemeral key for %s (tokens will not "
                "survive restarts).",
                ENV_VAR_SECRET_KEY,
                env.value,
            )

        sections: tuple[type[EnvBaseModel], ...] = (
            AppSettings, LoggingSettings, DatabaseSettings, ModelSettings, SecuritySettings,
        )
        for section in sections:
            for key in section.unknown_env_keys(variables):
                _log.warning("Ignoring unknown configuration variable %s", key)

        try:
            return cls(
                environment=env,
                app=AppSettings.from_env(variables, defaults=env.profile("app")),
                logging=LoggingSettings.from_env(variables, defaults=env.profile("logging")),
                database=DatabaseSettings.from_env(variables, defaults=env.profile("database")),
                model=ModelSettings.from_env(variables, defaults=env.profile("model")),
                security=SecuritySettings.from_env(variables, defaults=env.profile("security")),
            )
        except ValidationError as exc:  # cross-section rule violations
            messages = [
                str(err["msg"]).removeprefix("Value error, ")
                for err in exc.errors(
                    include_url=False, include_context=False, include_input=False
                )
            ]
            raise ConfigurationError("\n".join(messages)) from exc

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def is_production(self) -> bool:
        return self.environment.is_production

    @property
    def is_development(self) -> bool:
        return self.environment.is_development

    def configure_logging(self) -> None:
        configure_logging(
            self.logging,
            environment=self.environment.value,
            service_name=self.app.name,
            version=self.app.version,
        )

    def ensure_directories(self) -> None:
        """Create data, model and (if enabled) log directories."""
        self.app.data_dir.mkdir(parents=True, exist_ok=True)
        self.model.ensure_directories()
        if self.logging.file_enabled:
            self.logging.directory.mkdir(parents=True, exist_ok=True)

    def to_safe_dict(self) -> dict[str, Any]:
        """JSON-friendly dump in which every secret is masked (safe to log)."""
        data = self.model_dump(mode="json")
        data["database"]["url"] = self.database.masked_url
        return data

    def __repr__(self) -> str:
        return (
            f"Settings(environment={self.environment.value!r}, "
            f"app={self.app.name!r}, base_dir={str(BASE_DIR)!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings (loaded from ``os.environ`` on first use)."""
    return Settings.load()


def reload_settings() -> Settings:
    """Drop the cache and load again (after changing the environment)."""
    get_settings.cache_clear()
    return get_settings()
