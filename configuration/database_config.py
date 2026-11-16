"""Database connection settings and connection-pool parameters."""

from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr, field_validator

from .constants import (
    ALLOWED_DATABASE_SCHEMES,
    DEFAULT_DATABASE_URL,
    ENV_PREFIX_DATABASE,
    DatabaseSSLMode,
)
from .environment import EnvBaseModel


class DatabaseSettings(EnvBaseModel):
    """Database options (env prefix ``DATABASE_``; e.g. ``DATABASE_URL``)."""

    env_prefix: ClassVar[str] = ENV_PREFIX_DATABASE

    url: SecretStr = SecretStr(DEFAULT_DATABASE_URL)

    # Pool parameters (SQLAlchemy semantics; ignored for SQLite).
    pool_size: int = Field(default=10, ge=1, le=200)
    max_overflow: int = Field(default=20, ge=0, le=200)
    pool_timeout: int = Field(default=30, ge=1, le=300, description="seconds")
    pool_recycle: int = Field(default=1_800, ge=-1, description="seconds, -1 disables")
    pool_pre_ping: bool = True

    connect_timeout: int = Field(default=10, ge=1, le=120, description="seconds")
    statement_timeout_ms: int = Field(default=30_000, ge=0, le=3_600_000)
    ssl_mode: DatabaseSSLMode = DatabaseSSLMode.PREFER
    echo: bool = False

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()
        if not raw:
            raise ValueError("database URL must not be empty")
        parts = urlsplit(raw)
        scheme = parts.scheme.lower()
        if scheme not in ALLOWED_DATABASE_SCHEMES:
            allowed = ", ".join(sorted(ALLOWED_DATABASE_SCHEMES))
            raise ValueError(f"unsupported database scheme {scheme!r}; allowed: {allowed}")
        if not scheme.startswith("sqlite"):
            if not parts.hostname:
                raise ValueError("database URL must include a host")
            if not parts.path.strip("/"):
                raise ValueError("database URL must include a database name")
            _ = parts.port  # raises ValueError for an invalid / out-of-range port
        return SecretStr(raw)

    @field_validator("ssl_mode", mode="before")
    @classmethod
    def _normalise_ssl_mode(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    # ------------------------------------------------------------------ #
    # Derived values
    # ------------------------------------------------------------------ #
    @property
    def scheme(self) -> str:
        return urlsplit(self.url.get_secret_value()).scheme.lower()

    @property
    def dialect(self) -> str:
        return self.scheme.split("+", 1)[0]

    @property
    def driver(self) -> str | None:
        _, _, driver = self.scheme.partition("+")
        return driver or None

    @property
    def is_sqlite(self) -> bool:
        return self.dialect == "sqlite"

    @property
    def is_async(self) -> bool:
        return self.driver in {"asyncpg", "aiosqlite", "aiomysql"}

    @property
    def masked_url(self) -> str:
        """URL safe for logs: the password is replaced with ``***``."""
        raw = self.url.get_secret_value()
        parts = urlsplit(raw)
        if parts.username is None and parts.password is None:
            return raw
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
        user = parts.username or ""
        credentials = f"{user}:***@" if parts.password is not None else f"{user}@"
        return urlunsplit(parts._replace(netloc=credentials + host))

    def engine_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``sqlalchemy.create_engine`` / ``create_async_engine``."""
        if self.is_sqlite:
            return {
                "echo": self.echo,
                "connect_args": {"check_same_thread": False},
            }

        connect_args: dict[str, Any] = {}
        if self.dialect == "postgresql":
            if self.driver == "asyncpg":
                connect_args["timeout"] = self.connect_timeout
                connect_args["ssl"] = self.ssl_mode.value
                connect_args["command_timeout"] = (
                    self.statement_timeout_ms / 1_000 or None
                )
            else:
                connect_args["connect_timeout"] = self.connect_timeout
                connect_args["sslmode"] = self.ssl_mode.value
                if self.statement_timeout_ms:
                    connect_args["options"] = (
                        f"-c statement_timeout={self.statement_timeout_ms}"
                    )
        else:  # mysql / mariadb
            connect_args["connect_timeout"] = self.connect_timeout

        return {
            "echo": self.echo,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": self.pool_pre_ping,
            "connect_args": connect_args,
        }
