"""Exception tracking: fingerprinting, grouping, rate limiting, breadcrumbs, global hooks.

Errors with the same type and call-site share a *fingerprint* (message text and line
numbers are ignored) and are grouped, so a failure storm shows up as one group with a
high count instead of thousands of events. Per-fingerprint rate limiting protects
sinks (Sentry-style forwarders, alerting) from floods.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Tuple, Type, TypeVar

from .logging_manager import get_log_context, redact
from .metrics_collector import MetricsCollector
from .tracing_manager import current_trace_ids

__all__ = ["ErrorTracker", "ErrorEvent", "ErrorGroup"]

_log = logging.getLogger("observability.errors")
F = TypeVar("F", bound=Callable[..., Any])

_DEFAULT_IGNORED: Tuple[Type[BaseException], ...] = (
    KeyboardInterrupt, SystemExit, GeneratorExit, asyncio.CancelledError,
)


@dataclass
class ErrorEvent:
    event_id: str
    fingerprint: str
    exc_type: str
    message: str
    stacktrace: str
    frames: List[Dict[str, Any]]
    timestamp: float
    level: str = "error"
    handled: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, str] = field(default_factory=dict)
    breadcrumbs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ErrorGroup:
    fingerprint: str
    exc_type: str
    message: str
    count: int = 0
    suppressed: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    unhandled: int = 0
    last_event: Optional[ErrorEvent] = None
    _window_start: float = 0.0
    _window_count: int = 0

    def to_dict(self, include_event: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "fingerprint": self.fingerprint, "type": self.exc_type, "message": self.message,
            "count": self.count, "suppressed": self.suppressed, "unhandled": self.unhandled,
            "first_seen": self.first_seen, "last_seen": self.last_seen,
        }
        if include_event and self.last_event is not None:
            data["last_event"] = self.last_event.to_dict()
        return data


Sink = Callable[[ErrorEvent], None]


class _CaptureScope:
    """``with`` / ``async with`` scope that captures exceptions raised inside it."""

    __slots__ = ("_tracker", "_reraise", "_context")

    def __init__(self, tracker: "ErrorTracker", reraise: bool, context: Optional[Mapping[str, Any]]) -> None:
        self._tracker = tracker
        self._reraise = reraise
        self._context = context

    def __enter__(self) -> "_CaptureScope":
        return self

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        if exc is None or not isinstance(exc, Exception):
            return False  # never swallow CancelledError / KeyboardInterrupt / SystemExit
        self._tracker.capture_exception(exc, context=self._context, handled=not self._reraise)
        return not self._reraise

    async def __aenter__(self) -> "_CaptureScope":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)


class ErrorTracker:
    """Collects, groups and forwards exceptions."""

    def __init__(
        self,
        *,
        metrics: Optional[MetricsCollector] = None,
        max_groups: int = 500,
        max_breadcrumbs: int = 50,
        max_events_per_fingerprint: int = 20,
        rate_window_seconds: float = 60.0,
        ignore: Tuple[Type[BaseException], ...] = _DEFAULT_IGNORED,
        log_events: bool = True,
        sinks: Optional[List[Sink]] = None,
    ) -> None:
        self.max_groups = max_groups
        self.max_events_per_fingerprint = max_events_per_fingerprint
        self.rate_window_seconds = rate_window_seconds
        self._ignore = ignore
        self._log_events = log_events
        self._sinks: List[Sink] = list(sinks or [])
        self._groups: "OrderedDict[str, ErrorGroup]" = OrderedDict()
        self._recent: Deque[ErrorEvent] = deque(maxlen=200)
        self._breadcrumbs: Deque[Dict[str, Any]] = deque(maxlen=max_breadcrumbs)
        self._lock = threading.RLock()
        self.total_errors = 0
        self._counter = (
            metrics.registry.counter("errors_total", "Captured exceptions", ("exception", "handled"))
            if metrics else None
        )
        self._prev_excepthook: Optional[Callable[..., Any]] = None
        self._prev_thread_hook: Optional[Callable[..., Any]] = None
        self._hooked_loops: List[Tuple[asyncio.AbstractEventLoop, Any]] = []

    # -- sinks / breadcrumbs -------------------------------------------------
    def add_sink(self, sink: Sink) -> None:
        """Register a callable invoked with each (non rate-limited) :class:`ErrorEvent`."""
        self._sinks.append(sink)

    def add_breadcrumb(self, category: str, message: str, **data: Any) -> None:
        """Record a lightweight trail of what happened before an error."""
        self._breadcrumbs.append(
            {"ts": time.time(), "category": category, "message": message, "data": redact(data)}
        )

    # -- fingerprinting ------------------------------------------------------
    @staticmethod
    def fingerprint_for(exc: BaseException, frames: Optional[List[traceback.FrameSummary]] = None) -> str:
        frames = frames if frames is not None else traceback.extract_tb(exc.__traceback__)
        parts = [f"{type(exc).__module__}.{type(exc).__qualname__}"]
        parts += [f"{os.path.basename(f.filename)}:{f.name}" for f in frames[-3:]]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]

    # -- capture -------------------------------------------------------------
    def capture_exception(
        self,
        exc: Optional[BaseException] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        handled: bool = True,
        level: str = "error",
    ) -> Optional[ErrorEvent]:
        """Record an exception. With no argument, uses the exception being handled."""
        if exc is None:
            exc = sys.exc_info()[1]
        if exc is None or isinstance(exc, self._ignore):
            return None
        try:
            return self._capture(exc, context, handled, level)
        except Exception:  # the tracker must never become the source of an incident
            _log.exception("error tracker failed to capture an exception")
            return None

    def capture_message(
        self, message: str, *, level: str = "warning", context: Optional[Mapping[str, Any]] = None
    ) -> Optional[ErrorEvent]:
        """Record a non-exception problem as an event in its own group."""
        try:
            return self._capture(_MessageError(message), context, True, level)
        except Exception:
            _log.exception("error tracker failed to capture a message")
            return None

    def _capture(
        self, exc: BaseException, context: Optional[Mapping[str, Any]], handled: bool, level: str
    ) -> Optional[ErrorEvent]:
        now = time.time()
        tb_frames = traceback.extract_tb(exc.__traceback__)
        fingerprint = self.fingerprint_for(exc, tb_frames)
        if isinstance(exc, _MessageError):
            fingerprint = hashlib.sha1(f"msg|{exc.args[0]}".encode("utf-8")).hexdigest()[:16]
        type_name = f"{type(exc).__module__}.{type(exc).__qualname__}"
        message = str(exc)

        with self._lock:
            self.total_errors += 1
            group = self._groups.get(fingerprint)
            if group is None:
                group = ErrorGroup(fingerprint, type_name, message[:500], first_seen=now)
                self._groups[fingerprint] = group
                while len(self._groups) > self.max_groups:
                    self._groups.popitem(last=False)  # evict least recently seen
            else:
                self._groups.move_to_end(fingerprint)
            group.count += 1
            group.last_seen = now
            if not handled:
                group.unhandled += 1
            if now - group._window_start > self.rate_window_seconds:
                group._window_start, group._window_count = now, 0
            group._window_count += 1
            limited = group._window_count > self.max_events_per_fingerprint
            if limited:
                group.suppressed += 1

        if self._counter is not None:
            self._counter.inc(exception=type(exc).__name__, handled=str(handled).lower())
        if limited:
            return None

        merged: Dict[str, Any] = {**get_log_context(), **(context or {})}
        event = ErrorEvent(
            event_id=uuid.uuid4().hex,
            fingerprint=fingerprint,
            exc_type=type_name,
            message=message,
            stacktrace="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            frames=[
                {"file": f.filename, "line": f.lineno, "function": f.name, "code": f.line}
                for f in tb_frames[-25:]
            ],
            timestamp=now,
            level=level,
            handled=handled,
            context=redact(merged),
            trace=current_trace_ids(),
            breadcrumbs=list(self._breadcrumbs),
        )
        with self._lock:
            group.last_event = event
            self._recent.append(event)

        if self._log_events:
            _log.log(
                logging.CRITICAL if level == "fatal" else logging.ERROR,
                "%s: %s", type_name, message,
                extra={"fingerprint": fingerprint, "event_id": event.event_id, "handled": handled,
                       "error_count": group.count},
                exc_info=(type(exc), exc, exc.__traceback__) if exc.__traceback__ else None,
            )
        for sink in list(self._sinks):
            try:
                sink(event)
            except Exception:
                _log.exception("error sink %r failed", sink)
        return event

    # -- ergonomics ----------------------------------------------------------
    def capture(self, *, reraise: bool = True, context: Optional[Mapping[str, Any]] = None) -> _CaptureScope:
        """``with tracker.capture(): ...`` -> records, and re-raises unless ``reraise=False``."""
        return _CaptureScope(self, reraise, context)

    def track(self, *, reraise: bool = True, context: Optional[Mapping[str, Any]] = None) -> Callable[[F], F]:
        """Decorator version of :meth:`capture` (sync and async)."""

        def decorator(func: F) -> F:
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    async with self.capture(reraise=reraise, context=context):
                        return await func(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.capture(reraise=reraise, context=context):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    # -- queries -------------------------------------------------------------
    def top_groups(self, limit: int = 10) -> List[ErrorGroup]:
        with self._lock:
            groups = list(self._groups.values())
        return sorted(groups, key=lambda g: g.count, reverse=True)[:limit]

    def recent(self, limit: int = 20) -> List[ErrorEvent]:
        with self._lock:
            return list(self._recent)[-limit:][::-1]

    def get_group(self, fingerprint: str) -> Optional[ErrorGroup]:
        with self._lock:
            return self._groups.get(fingerprint)

    def error_rate(self, window_seconds: float = 300.0) -> float:
        """Errors per minute observed within the window (based on recent events)."""
        cutoff = time.time() - window_seconds
        with self._lock:
            n = sum(1 for e in self._recent if e.timestamp >= cutoff)
        return n / (window_seconds / 60.0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_errors": self.total_errors,
            "groups": [g.to_dict() for g in self.top_groups(20)],
        }

    def reset(self) -> None:
        with self._lock:
            self._groups.clear()
            self._recent.clear()
            self._breadcrumbs.clear()
            self.total_errors = 0

    # -- global hooks --------------------------------------------------------
    def install_global_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Capture uncaught exceptions in the main thread, other threads and asyncio tasks."""
        if self._prev_excepthook is None:
            self._prev_excepthook = sys.excepthook

            def excepthook(exc_type: Any, exc: BaseException, tb: Any) -> None:
                self.capture_exception(exc, handled=False, level="fatal")
                assert self._prev_excepthook is not None
                self._prev_excepthook(exc_type, exc, tb)

            sys.excepthook = excepthook

        if self._prev_thread_hook is None:
            self._prev_thread_hook = threading.excepthook

            def thread_hook(args: Any) -> None:
                if args.exc_value is not None:
                    self.capture_exception(
                        args.exc_value, handled=False, level="fatal",
                        context={"thread": getattr(args.thread, "name", None)},
                    )
                assert self._prev_thread_hook is not None
                self._prev_thread_hook(args)

            threading.excepthook = thread_hook

        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        if loop is not None and not any(l is loop for l, _ in self._hooked_loops):
            previous = loop.get_exception_handler()

            def loop_handler(lp: asyncio.AbstractEventLoop, ctx: Dict[str, Any]) -> None:
                exc = ctx.get("exception")
                if isinstance(exc, BaseException):
                    self.capture_exception(exc, handled=False, context={"asyncio": ctx.get("message")})
                else:
                    self.capture_message(str(ctx.get("message", "asyncio error")), level="error")
                if previous is not None:
                    previous(lp, ctx)
                else:
                    lp.default_exception_handler(ctx)

            loop.set_exception_handler(loop_handler)
            self._hooked_loops.append((loop, previous))

    def uninstall_global_handlers(self) -> None:
        if self._prev_excepthook is not None:
            sys.excepthook = self._prev_excepthook
            self._prev_excepthook = None
        if self._prev_thread_hook is not None:
            threading.excepthook = self._prev_thread_hook
            self._prev_thread_hook = None
        for loop, previous in self._hooked_loops:
            try:
                loop.set_exception_handler(previous)
            except Exception:  # pragma: no cover
                pass
        self._hooked_loops.clear()


class _MessageError(Exception):
    """Internal carrier for :meth:`ErrorTracker.capture_message`."""
