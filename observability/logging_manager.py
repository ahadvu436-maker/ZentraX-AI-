"""Structured JSON logging with trace correlation and runtime level control.

Highlights
----------
* One JSON object per line (stdout by default) -> ready for Loki / ELK / Datadog.
* Automatic ``trace_id`` / ``span_id`` correlation and request-scoped context
  fields via :func:`log_context` (``contextvars`` -> safe across ``await``).
* Non-blocking: records are enqueued and written by a background thread; when the
  queue is full records are dropped (and counted) instead of stalling the app.
* Sensitive keys (password, token, authorization, ...) are redacted.
* Runtime level changes, globally or per logger.
"""

from __future__ import annotations

import atexit
import contextvars
import json
import logging
import queue
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, TextIO, Union

from .tracing_manager import current_trace_ids

__all__ = [
    "LoggingManager",
    "JsonFormatter",
    "StructuredLogger",
    "ContextFilter",
    "log_context",
    "get_log_context",
    "redact",
    "coerce_level",
    "DEFAULT_REDACT_KEYS",
]

LevelLike = Union[int, str]

_LEVELS = {
    "CRITICAL": logging.CRITICAL, "FATAL": logging.CRITICAL, "ERROR": logging.ERROR,
    "WARNING": logging.WARNING, "WARN": logging.WARNING, "INFO": logging.INFO,
    "DEBUG": logging.DEBUG, "NOTSET": logging.NOTSET,
}

DEFAULT_REDACT_KEYS: FrozenSet[str] = frozenset(
    {
        "password", "passwd", "secret", "token", "api_key", "apikey", "authorization",
        "cookie", "set-cookie", "access_token", "refresh_token", "private_key",
        "client_secret", "credit_card", "ssn",
    }
)

_RESERVED = (
    frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
    | {"message", "asctime", "obs_context", "trace_id", "span_id", "taskName"}
)

_log_context: "contextvars.ContextVar[Mapping[str, Any]]" = contextvars.ContextVar(
    "observability_log_context", default={}
)


def coerce_level(level: LevelLike) -> int:
    if isinstance(level, int):
        return level
    try:
        return _LEVELS[str(level).strip().upper()]
    except KeyError:
        raise ValueError(f"Unknown log level: {level!r}") from None


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach fields (e.g. ``request_id``, ``user_id``) to every log line in scope."""
    token = _log_context.set({**_log_context.get(), **fields})
    try:
        yield
    finally:
        _log_context.reset(token)


def get_log_context() -> Dict[str, Any]:
    return dict(_log_context.get())


def redact(obj: Any, keys: FrozenSet[str] = DEFAULT_REDACT_KEYS, _depth: int = 0) -> Any:
    """Return a copy of ``obj`` with values of sensitive keys replaced by ``***``."""
    if _depth > 8:
        return "<max-depth>"
    if isinstance(obj, Mapping):
        return {
            str(k): ("***" if str(k).lower() in keys else redact(v, keys, _depth + 1))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [redact(v, keys, _depth + 1) for v in obj]
    return obj


class ContextFilter(logging.Filter):
    """Snapshots trace ids and bound context at *emit* time (caller's thread/task)."""

    def filter(self, record: logging.LogRecord) -> bool:
        ids = current_trace_ids()
        record.obs_context = {"fields": dict(_log_context.get()), "trace": ids}
        record.trace_id = ids.get("trace_id", "-")
        record.span_id = ids.get("span_id", "-")
        return True


class JsonFormatter(logging.Formatter):
    """Render ``LogRecord`` as a single-line JSON document."""

    def __init__(
        self,
        service: str = "service",
        environment: str = "production",
        version: str = "0.0.0",
        redact_keys: FrozenSet[str] = DEFAULT_REDACT_KEYS,
        static_fields: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._base = {"service": service, "environment": environment, "version": version}
        self._base.update(static_fields or {})
        self._redact_keys = redact_keys

    def format(self, record: logging.LogRecord) -> str:
        ctx = getattr(record, "obs_context", None) or {
            "fields": dict(_log_context.get()),
            "trace": current_trace_ids(),
        }
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(self._base)
        payload.update(ctx["trace"])
        payload["source"] = {"module": record.module, "function": record.funcName, "line": record.lineno}
        payload["process"] = record.process
        payload["thread"] = record.threadName

        def add(key: str, value: Any) -> None:
            payload[f"extra_{key}" if key in payload else key] = value

        for key, value in ctx["fields"].items():
            add(key, redact({key: value}, self._redact_keys)[key])
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                add(key, redact({key: value}, self._redact_keys)[key])

        if record.exc_info and record.exc_info[0] is not None:
            etype, evalue, etb = record.exc_info
            payload["exception"] = {
                "type": f"{etype.__module__}.{etype.__qualname__}",
                "message": str(evalue),
                "stacktrace": "".join(traceback.format_exception(etype, evalue, etb)),
            }
        elif record.exc_text:
            payload["exception"] = {"stacktrace": record.exc_text}
        if record.stack_info:
            payload["stack_info"] = record.stack_info

        try:
            return json.dumps(payload, default=str, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):  # circular refs etc.
            payload = {k: payload[k] for k in ("timestamp", "level", "logger", "message", "service")}
            payload["formatter_error"] = "unserializable log payload"
            return json.dumps(payload, default=str, ensure_ascii=False, separators=(",", ":"))


class _DroppingQueueHandler(QueueHandler):
    """Queue handler that never blocks and never mangles the record."""

    def __init__(self, q: "queue.Queue[Any]") -> None:
        super().__init__(q)
        self.dropped = 0

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record  # keep exc_info / extras for the JSON formatter in the listener thread

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1


_ADAPTER_KWARGS = frozenset({"exc_info", "stack_info", "stacklevel", "extra"})


class StructuredLogger(logging.LoggerAdapter):
    """``logger.info("order placed", order_id=42, amount=9.9)`` -> JSON fields."""

    def process(self, msg: Any, kwargs: Any) -> Any:
        extra = dict(self.extra or {})
        extra.update(kwargs.pop("extra", None) or {})
        for key in [k for k in kwargs if k not in _ADAPTER_KWARGS]:
            extra[key] = kwargs.pop(key)
        for key in [k for k in extra if k in _RESERVED]:
            extra[f"ctx_{key}"] = extra.pop(key)
        kwargs["extra"] = extra
        return msg, kwargs


class LoggingManager:
    """Owns the process-wide logging configuration."""

    def __init__(
        self,
        service_name: str = "service",
        *,
        environment: str = "production",
        version: str = "0.0.0",
        level: LevelLike = "INFO",
        json_output: bool = True,
        stream: Optional[TextIO] = None,
        async_logging: bool = True,
        queue_size: int = 10_000,
        redact_keys: Optional[FrozenSet[str]] = None,
        static_fields: Optional[Mapping[str, Any]] = None,
        replace_handlers: bool = True,
    ) -> None:
        self.service_name = service_name
        self.environment = environment
        self.version = version
        self._level = coerce_level(level)
        self.json_output = json_output
        self._stream = stream
        self.async_logging = async_logging
        self._queue_size = queue_size
        self._redact_keys = redact_keys or DEFAULT_REDACT_KEYS
        self._static_fields = dict(static_fields or {})
        self._replace_handlers = replace_handlers
        self._lock = threading.RLock()
        self._configured = False
        self._listener: Optional[QueueListener] = None
        self._root_handler: Optional[logging.Handler] = None
        self._queue_handler: Optional[_DroppingQueueHandler] = None
        self._removed: List[logging.Handler] = []
        self._prev_level = logging.WARNING

    # -- lifecycle -----------------------------------------------------------
    def configure(self) -> "LoggingManager":
        """Install handlers on the root logger. Idempotent."""
        with self._lock:
            if self._configured:
                return self
            out = logging.StreamHandler(self._stream or sys.stdout)
            if self.json_output:
                out.setFormatter(
                    JsonFormatter(self.service_name, self.environment, self.version,
                                  self._redact_keys, self._static_fields)
                )
            else:
                out.setFormatter(
                    logging.Formatter(
                        "%(asctime)s %(levelname)-8s [%(trace_id)s] %(name)s: %(message)s"
                    )
                )
            if self.async_logging:
                q: "queue.Queue[Any]" = queue.Queue(maxsize=self._queue_size)
                self._listener = QueueListener(q, out, respect_handler_level=True)
                handler: logging.Handler = _DroppingQueueHandler(q)
                self._queue_handler = handler  # type: ignore[assignment]
                handler.addFilter(ContextFilter())
                self._listener.start()
            else:
                out.addFilter(ContextFilter())
                handler = out
            root = logging.getLogger()
            self._prev_level = root.level
            if self._replace_handlers:
                self._removed = list(root.handlers)
                for h in self._removed:
                    root.removeHandler(h)
            root.addHandler(handler)
            root.setLevel(self._level)
            self._root_handler = handler
            self._configured = True
            atexit.register(self.shutdown)
        return self

    def shutdown(self) -> None:
        """Flush queued records, restore previous handlers. Idempotent."""
        with self._lock:
            if not self._configured:
                return
            root = logging.getLogger()
            if self._root_handler is not None:
                root.removeHandler(self._root_handler)
            if self._listener is not None:
                self._listener.stop()  # drains the queue
                self._listener = None
            for h in self._removed:
                root.addHandler(h)
            self._removed = []
            root.setLevel(self._prev_level)
            self._root_handler = None
            self._configured = False
            try:
                atexit.unregister(self.shutdown)
            except Exception:  # pragma: no cover
                pass

    # -- api -----------------------------------------------------------------
    def get_logger(self, name: Optional[str] = None, **bound: Any) -> StructuredLogger:
        return StructuredLogger(logging.getLogger(name), bound)

    def set_level(self, level: LevelLike, logger_name: Optional[str] = None) -> None:
        """Change level at runtime; ``logger_name=None`` targets the root logger."""
        value = coerce_level(level)
        logging.getLogger(logger_name).setLevel(value)
        if logger_name is None:
            self._level = value

    def get_level(self, logger_name: Optional[str] = None) -> str:
        return logging.getLevelName(logging.getLogger(logger_name).getEffectiveLevel())

    @staticmethod
    def bind(**fields: Any):  # type: ignore[no-untyped-def]
        """Alias of :func:`log_context`."""
        return log_context(**fields)

    @property
    def dropped_records(self) -> int:
        return self._queue_handler.dropped if self._queue_handler is not None else 0

    @property
    def is_configured(self) -> bool:
        return self._configured
