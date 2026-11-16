"""Structured logging configuration (JSON or console) built on the stdlib.

Features: JSON formatter with ISO-8601 UTC timestamps, per-task context
(correlation ids) via ``contextvars``, secret redaction, optional rotating
file output and quieting of noisy third-party loggers.
"""

from __future__ import annotations

import json
import logging
import logging.config
import re
from collections.abc import Mapping
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from pydantic import Field, field_validator

from .constants import (
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILE_NAME,
    DEFAULT_QUIET_LOGGERS,
    ENV_PREFIX_LOGGING,
    LOG_CONTEXT_CORRELATION_KEY,
    MIB,
    LogFormat,
    LogLevel,
)
from .environment import EnvBaseModel, resolve_path


class LoggingSettings(EnvBaseModel):
    """Logging options (env prefix ``LOG_``)."""

    env_prefix: ClassVar[str] = ENV_PREFIX_LOGGING

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.JSON
    console_enabled: bool = True
    file_enabled: bool = False
    directory: Path = DEFAULT_LOG_DIR
    file_name: str = Field(default=DEFAULT_LOG_FILE_NAME, min_length=1, max_length=255)
    max_bytes: int = Field(default=10 * MIB, ge=1_024, le=1_024 * MIB)
    backup_count: int = Field(default=5, ge=0, le=100)
    redact_sensitive: bool = True
    quiet_loggers: tuple[str, ...] = DEFAULT_QUIET_LOGGERS

    @field_validator("level", mode="before")
    @classmethod
    def _upper_level(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @field_validator("format", mode="before")
    @classmethod
    def _lower_format(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @field_validator("directory")
    @classmethod
    def _resolve_log_dir(cls, value: Path) -> Path:
        return resolve_path(value)

    @field_validator("file_name")
    @classmethod
    def _plain_file_name(cls, value: str) -> str:
        if Path(value).name != value:
            raise ValueError("file_name must be a bare file name, not a path")
        return value

    @property
    def file_path(self) -> Path:
        return self.directory / self.file_name


# --------------------------------------------------------------------------- #
# Context propagation
# --------------------------------------------------------------------------- #
_EMPTY_CONTEXT: Final[Mapping[str, Any]] = MappingProxyType({})
_log_context: ContextVar[Mapping[str, Any]] = ContextVar(
    "configuration_log_context", default=_EMPTY_CONTEXT
)


def bind_log_context(**values: Any) -> Token[Mapping[str, Any]]:
    """Attach key/values to every log record of the current context.

    Returns a token for :func:`reset_log_context`.
    """
    merged = {**_log_context.get(), **values}
    return _log_context.set(MappingProxyType(merged))


def reset_log_context(token: Token[Mapping[str, Any]]) -> None:
    _log_context.reset(token)


def clear_log_context() -> None:
    _log_context.set(_EMPTY_CONTEXT)


def get_log_context() -> dict[str, Any]:
    return dict(_log_context.get())


# --------------------------------------------------------------------------- #
# Filters & formatters
# --------------------------------------------------------------------------- #
class ContextFilter(logging.Filter):
    """Copies the bound context onto each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _log_context.get()
        record.correlation_id = ctx.get(LOG_CONTEXT_CORRELATION_KEY, "-")
        record.context = dict(ctx)
        return True


_KEY_VALUE_SECRET = re.compile(
    r"""(?ix)
    \b(password|passwd|secret|token|api[_-]?key|authorization)\b
    (["']?\s*[=:]\s*["']?)
    (?:(?:bearer|basic)\s+)?
    ([^\s,;&"']+)
    """
)
_BEARER_SECRET = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+")
REDACTED: Final[str] = "***REDACTED***"


def redact(text: str) -> str:
    """Mask ``password=...``, ``token: ...`` and ``Bearer ...`` style secrets."""
    text = _KEY_VALUE_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return _BEARER_SECRET.sub(lambda m: f"{m.group(1)} {REDACTED}", text)


class RedactionFilter(logging.Filter):
    """Masks secrets in the formatted message before any handler sees it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


_STANDARD_ATTRS: Final[frozenset[str]] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName", "correlation_id", "context"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra ``logger.info(..., extra={})`` are kept."""

    def __init__(self, *, service: str, environment: str, version: str) -> None:
        super().__init__()
        self._static = {"service": service, "environment": environment, "version": version}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **self._static,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }
        context = getattr(record, "context", None) or get_log_context()
        if context:
            payload["context"] = context
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


CONSOLE_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | [%(correlation_id)s] %(message)s"
)


# --------------------------------------------------------------------------- #
# dictConfig builder
# --------------------------------------------------------------------------- #
def build_logging_config(
    settings: LoggingSettings,
    *,
    environment: str,
    service_name: str,
    version: str,
) -> dict[str, Any]:
    """Return a ``logging.config.dictConfig`` compatible mapping."""
    level = settings.level.value

    filters: dict[str, Any] = {"context": {"()": ContextFilter}}
    handler_filters = ["context"]
    if settings.redact_sensitive:
        filters["redact"] = {"()": RedactionFilter}
        handler_filters.insert(0, "redact")

    formatters: dict[str, Any] = {
        "json": {
            "()": JsonFormatter,
            "service": service_name,
            "environment": environment,
            "version": version,
        },
        "console": {"format": CONSOLE_FORMAT},
    }
    formatter = settings.format.value

    handlers: dict[str, Any] = {}
    if settings.console_enabled:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": formatter,
            "filters": handler_filters,
            "stream": "ext://sys.stdout",
        }
    if settings.file_enabled:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": formatter,
            "filters": handler_filters,
            "filename": str(settings.file_path),
            "maxBytes": settings.max_bytes,
            "backupCount": settings.backup_count,
            "encoding": "utf-8",
        }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": filters,
        "formatters": formatters,
        "handlers": handlers,
        "root": {"level": level, "handlers": list(handlers)},
        "loggers": {
            name: {"level": "WARNING", "propagate": True}
            for name in settings.quiet_loggers
        },
    }


def configure_logging(
    settings: LoggingSettings,
    *,
    environment: str,
    service_name: str,
    version: str,
) -> None:
    """Apply the logging configuration to the running process."""
    if settings.file_enabled:
        settings.directory.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(
        build_logging_config(
            settings,
            environment=environment,
            service_name=service_name,
            version=version,
        )
    )
    logging.captureWarnings(True)


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)
