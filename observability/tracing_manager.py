"""Distributed tracing for sync and asyncio workflows.

Features
--------
* ``contextvars``-based span propagation: works across ``await`` points, across
  ``asyncio.create_task`` / ``asyncio.gather`` and (via :func:`propagate_context`)
  across thread pools.
* W3C ``traceparent`` inject / extract for cross-service propagation.
* Deterministic head-based sampling (parent decision always wins).
* Non-blocking export: finished spans are handed to a background worker thread.
* Pluggable exporters (in-memory ring buffer and logging exporters included).
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import logging
import queue
import random
import re
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TypeVar,
)

__all__ = [
    "SpanKind",
    "StatusCode",
    "SpanContext",
    "Span",
    "SpanExporter",
    "InMemorySpanExporter",
    "LogSpanExporter",
    "TracingManager",
    "current_span",
    "current_trace_ids",
    "propagate_context",
]

_log = logging.getLogger("observability.tracing")
F = TypeVar("F", bound=Callable[..., Any])

_TRACEPARENT_RE = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_ZERO_TRACE = "0" * 32
_ZERO_SPAN = "0" * 16


class SpanKind(str, Enum):
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class StatusCode(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


def _new_trace_id() -> str:
    value = random.getrandbits(128)
    return f"{value or 1:032x}"


def _new_span_id() -> str:
    value = random.getrandbits(64)
    return f"{value or 1:016x}"


@dataclass(frozen=True)
class SpanContext:
    """Immutable identity of a span; this is what crosses process boundaries."""

    trace_id: str
    span_id: str
    sampled: bool = True
    remote: bool = False

    def to_traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"

    @classmethod
    def from_traceparent(cls, value: str) -> Optional["SpanContext"]:
        match = _TRACEPARENT_RE.match((value or "").strip().lower())
        if not match:
            return None
        version, trace_id, span_id, flags = match.groups()
        if version == "ff" or trace_id == _ZERO_TRACE or span_id == _ZERO_SPAN:
            return None
        return cls(trace_id, span_id, bool(int(flags, 16) & 0x01), remote=True)


class Span:
    """A single timed operation. Created through :class:`TracingManager`."""

    __slots__ = (
        "name", "context", "parent_span_id", "kind", "attributes", "events",
        "status", "status_message", "start_time_ns", "service",
        "_start_perf", "_end_perf", "_on_end", "_max_attributes", "_max_events",
    )

    def __init__(
        self,
        name: str,
        context: SpanContext,
        parent_span_id: Optional[str],
        kind: SpanKind,
        service: str,
        attributes: Optional[Mapping[str, Any]],
        on_end: Optional[Callable[["Span"], None]],
        max_attributes: int = 64,
        max_events: int = 128,
    ) -> None:
        self.name = name
        self.context = context
        self.parent_span_id = parent_span_id
        self.kind = kind
        self.service = service
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.status = StatusCode.UNSET
        self.status_message = ""
        self.start_time_ns = time.time_ns()
        self._start_perf = time.perf_counter_ns()
        self._end_perf: Optional[int] = None
        self._on_end = on_end
        self._max_attributes = max_attributes
        self._max_events = max_events
        if attributes:
            self.set_attributes(attributes)

    # -- state -------------------------------------------------------------
    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        return self.context.span_id

    @property
    def is_recording(self) -> bool:
        return self.context.sampled and self._end_perf is None

    @property
    def duration_ms(self) -> float:
        end = self._end_perf if self._end_perf is not None else time.perf_counter_ns()
        return (end - self._start_perf) / 1e6

    # -- mutation ----------------------------------------------------------
    def set_attribute(self, key: str, value: Any) -> None:
        if not self.is_recording:
            return
        if key not in self.attributes and len(self.attributes) >= self._max_attributes:
            return
        self.attributes[key] = value

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def add_event(self, name: str, attributes: Optional[Mapping[str, Any]] = None) -> None:
        if not self.is_recording or len(self.events) >= self._max_events:
            return
        self.events.append(
            {"name": name, "time_unix_nano": time.time_ns(), "attributes": dict(attributes or {})}
        )

    def set_status(self, status: StatusCode, message: str = "") -> None:
        if not self.is_recording:
            return
        # ERROR is sticky: a later OK must not mask a failure.
        if self.status is StatusCode.ERROR and status is not StatusCode.ERROR:
            return
        self.status = status
        self.status_message = message

    def record_exception(self, exc: BaseException, *, escaped: bool = True) -> None:
        if not self.is_recording:
            return
        self.add_event(
            "exception",
            {
                "exception.type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "exception.message": str(exc),
                "exception.escaped": escaped,
                "exception.stacktrace": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        )
        self.set_status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}")

    def end(self) -> None:
        if self._end_perf is not None:
            return
        self._end_perf = time.perf_counter_ns()
        if self._on_end is not None:
            try:
                self._on_end(self)
            except Exception:  # pragma: no cover - never break the caller
                _log.exception("span end callback failed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time_unix_nano": self.start_time_ns,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": dict(self.attributes),
            "events": list(self.events),
        }


_current_span: "contextvars.ContextVar[Optional[Span]]" = contextvars.ContextVar(
    "observability_current_span", default=None
)


def current_span() -> Optional[Span]:
    """Return the active span for this task/thread, if any."""
    return _current_span.get()


def current_trace_ids() -> Dict[str, str]:
    """Return ``{"trace_id", "span_id"}`` of the active span or an empty dict."""
    span = _current_span.get()
    if span is None:
        return {}
    return {"trace_id": span.context.trace_id, "span_id": span.context.span_id}


def propagate_context(func: F) -> F:
    """Wrap a sync callable so it runs inside a *copy* of the caller's context.

    Use for ``ThreadPoolExecutor.submit`` / ``loop.run_in_executor`` (``asyncio.to_thread``
    and ``asyncio.create_task`` already copy the context automatically).
    """
    ctx = contextvars.copy_context()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return ctx.copy().run(func, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Exporters
# --------------------------------------------------------------------------
class SpanExporter:
    """Base class for span exporters. Called from the background worker thread."""

    def export(self, spans: Sequence[Span]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def shutdown(self) -> None:
        return None


class InMemorySpanExporter(SpanExporter):
    """Ring buffer of recent spans; powers dashboards and tests."""

    def __init__(self, max_spans: int = 5000) -> None:
        self._spans: Deque[Dict[str, Any]] = deque(maxlen=max_spans)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[Span]) -> None:
        dicts = [s.to_dict() for s in spans]
        with self._lock:
            self._spans.extend(dicts)

    def spans(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._spans)

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        return [s for s in self.spans() if s["trace_id"] == trace_id]

    def recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Summaries of the most recent traces, newest first."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for span in self.spans():
            grouped.setdefault(span["trace_id"], []).append(span)
        summaries = []
        for trace_id, spans in grouped.items():
            root = next((s for s in spans if s["parent_span_id"] is None), spans[0])
            summaries.append(
                {
                    "trace_id": trace_id,
                    "root": root["name"],
                    "duration_ms": root["duration_ms"],
                    "span_count": len(spans),
                    "error": any(s["status"] == "error" for s in spans),
                    "start": min(s["start_time_unix_nano"] for s in spans),
                }
            )
        summaries.sort(key=lambda s: s["start"], reverse=True)
        return summaries[:limit]

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


class LogSpanExporter(SpanExporter):
    """Emit finished spans as structured log records (default level DEBUG)."""

    def __init__(self, level: int = logging.DEBUG, logger: Optional[logging.Logger] = None) -> None:
        self._level = level
        self._logger = logger or logging.getLogger("observability.spans")

    def export(self, spans: Sequence[Span]) -> None:
        for span in spans:
            if self._logger.isEnabledFor(self._level):
                self._logger.log(
                    self._level,
                    "span finished: %s",
                    span.name,
                    extra={"span": span.to_dict()},
                )


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------
class _Flush:
    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()


_STOP = object()


class _SpanScope:
    """Context manager usable with both ``with`` and ``async with``."""

    __slots__ = ("_tracer", "_name", "_kind", "_attributes", "_parent", "_record", "_span", "_token")

    def __init__(
        self,
        tracer: "TracingManager",
        name: str,
        kind: SpanKind,
        attributes: Optional[Mapping[str, Any]],
        parent: Optional[SpanContext],
        record_exceptions: bool,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._kind = kind
        self._attributes = attributes
        self._parent = parent
        self._record = record_exceptions
        self._span: Optional[Span] = None
        self._token: Optional[contextvars.Token] = None  # type: ignore[type-arg]

    def __enter__(self) -> Span:
        self._span = self._tracer._create_span(self._name, self._kind, self._attributes, self._parent)
        self._token = _current_span.set(self._span)
        return self._span

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        span = self._span
        assert span is not None
        try:
            if exc is not None:
                if isinstance(exc, Exception):
                    if self._record:
                        span.record_exception(exc)
                    else:
                        span.set_status(StatusCode.ERROR, type(exc).__name__)
                elif isinstance(exc, asyncio.CancelledError):
                    span.set_attribute("span.cancelled", True)
        finally:
            span.end()
            if self._token is not None:
                try:
                    _current_span.reset(self._token)
                except ValueError:  # exited in a different context (e.g. async generator)
                    pass
        return False

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)


class TracingManager:
    """Creates spans, applies sampling, propagates context and exports spans."""

    def __init__(
        self,
        service_name: str = "service",
        *,
        sampling_ratio: float = 1.0,
        exporters: Optional[Sequence[SpanExporter]] = None,
        queue_size: int = 10_000,
        max_batch: int = 256,
        flush_interval: float = 2.0,
        max_attributes: int = 64,
        max_events: int = 128,
    ) -> None:
        if not 0.0 <= sampling_ratio <= 1.0:
            raise ValueError("sampling_ratio must be within [0, 1]")
        self.service_name = service_name
        self.sampling_ratio = sampling_ratio
        self._exporters: List[SpanExporter] = list(exporters or [])
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=queue_size)
        self._max_batch = max_batch
        self._flush_interval = flush_interval
        self._max_attributes = max_attributes
        self._max_events = max_events
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = False
        self.dropped_spans = 0

    # -- configuration -------------------------------------------------------
    def add_exporter(self, exporter: SpanExporter) -> None:
        self._exporters.append(exporter)

    @property
    def exporters(self) -> Sequence[SpanExporter]:
        return tuple(self._exporters)

    def get_exporter(self, kind: type) -> Optional[Any]:
        return next((e for e in self._exporters if isinstance(e, kind)), None)

    # -- span creation -------------------------------------------------------
    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Mapping[str, Any]] = None,
        parent: Optional[SpanContext] = None,
        record_exceptions: bool = True,
    ) -> _SpanScope:
        """Open a span; use as ``with`` or ``async with``; yields the :class:`Span`."""
        return _SpanScope(self, name, kind, attributes, parent, record_exceptions)

    def trace(
        self,
        name: Optional[str] = None,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> Callable[[F], F]:
        """Decorator that wraps a sync or async function in a span."""

        def decorator(func: F) -> F:
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.start_span(span_name, kind=kind, attributes=attributes):
                        return await func(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.start_span(span_name, kind=kind, attributes=attributes):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

        return decorator

    def _should_sample(self, trace_id: str) -> bool:
        if self.sampling_ratio >= 1.0:
            return True
        if self.sampling_ratio <= 0.0:
            return False
        return int(trace_id[-16:], 16) / float(1 << 64) < self.sampling_ratio

    def _create_span(
        self,
        name: str,
        kind: SpanKind,
        attributes: Optional[Mapping[str, Any]],
        parent: Optional[SpanContext],
    ) -> Span:
        parent_ctx = parent
        if parent_ctx is None:
            active = _current_span.get()
            parent_ctx = active.context if active is not None else None
        if parent_ctx is not None:
            trace_id, parent_id, sampled = parent_ctx.trace_id, parent_ctx.span_id, parent_ctx.sampled
        else:
            trace_id = _new_trace_id()
            parent_id = None
            sampled = self._should_sample(trace_id)
        ctx = SpanContext(trace_id, _new_span_id(), sampled)
        return Span(
            name, ctx, parent_id, kind, self.service_name, attributes, self._on_span_end,
            self._max_attributes, self._max_events,
        )

    # -- propagation ---------------------------------------------------------
    def inject(self, carrier: MutableMapping[str, str], span: Optional[Span] = None) -> MutableMapping[str, str]:
        """Write ``traceparent`` for the given/active span into ``carrier`` (headers)."""
        span = span or _current_span.get()
        if span is not None:
            carrier["traceparent"] = span.context.to_traceparent()
        return carrier

    @staticmethod
    def extract(carrier: Mapping[str, str]) -> Optional[SpanContext]:
        """Read a remote parent from headers (case-insensitive)."""
        for key, value in carrier.items():
            if key.lower() == "traceparent":
                return SpanContext.from_traceparent(value)
        return None

    # -- export pipeline -----------------------------------------------------
    def _on_span_end(self, span: Span) -> None:
        if not span.context.sampled or self._closed or not self._exporters:
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait(span)
        except queue.Full:
            self.dropped_spans += 1

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run, name="obs-span-exporter", daemon=True)
            self._worker.start()

    def _export(self, batch: List[Span]) -> None:
        for exporter in self._exporters:
            try:
                exporter.export(batch)
            except Exception:
                _log.exception("span exporter %s failed", type(exporter).__name__)

    def _run(self) -> None:
        batch: List[Span] = []
        while True:
            try:
                item = self._queue.get(timeout=self._flush_interval)
            except queue.Empty:
                continue
            stop = False
            items = [item]
            while len(items) < self._max_batch:
                try:
                    items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            flushes: List[_Flush] = []
            for it in items:
                if it is _STOP:
                    stop = True
                elif isinstance(it, _Flush):
                    flushes.append(it)
                else:
                    batch.append(it)
            if batch:
                self._export(batch)
                batch = []
            for flush in flushes:
                flush.event.set()
            if stop:
                return

    def force_flush(self, timeout: float = 5.0) -> bool:
        """Block until queued spans were handed to exporters."""
        if self._worker is None or not self._worker.is_alive():
            return True
        marker = _Flush()
        try:
            self._queue.put(marker, timeout=timeout)
        except queue.Full:
            return False
        return marker.event.wait(timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self.force_flush(timeout)
        self._closed = True
        if self._worker is not None and self._worker.is_alive():
            try:
                self._queue.put(_STOP, timeout=timeout)
            except queue.Full:
                pass
            self._worker.join(timeout)
        for exporter in self._exporters:
            try:
                exporter.shutdown()
            except Exception:
                _log.exception("span exporter shutdown failed")
