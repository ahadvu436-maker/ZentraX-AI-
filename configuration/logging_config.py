"""Logging settings and a ``dictConfig`` builder with a JSON formatter."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel

_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields passed via ``extra=`` are included."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        return json.dumps(payload, default=str)


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_output: bool = True
    log_file: str | None = None


def build_logging_config(cfg: LoggingSettings) -> dict[str, Any]:
    handlers: dict[str, Any] = {
        "console": {"class": "logging.StreamHandler", "formatter": "main", "stream": "ext://sys.stdout"}
    }
    if cfg.log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "main",
            "filename": cfg.log_file,
            "maxBytes": 10_000_000,
            "backupCount": 5,
        }
    formatter: dict[str, Any] = (
        {"()": f"{__name__}.JsonFormatter"}
        if cfg.json_output
        else {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}
    )
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"main": formatter},
        "handlers": handlers,
        "root": {"level": cfg.level, "handlers": list(handlers)},
    }


def configure_logging(cfg: LoggingSettings) -> None:
    import logging.config

    logging.config.dictConfig(build_logging_config(cfg))
