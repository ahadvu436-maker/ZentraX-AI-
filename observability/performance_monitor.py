"""Real-time performance monitoring and on-demand profiling.

* ``measure()`` / ``@timed()``: per-operation latency windows (p50/p95/p99), error and
  slow-call counts, wall *and* CPU time, exported to the metrics registry.
* Event-loop lag monitor: the best single indicator of an overloaded asyncio service.
* ``profile()``: cProfile for a code block (guarded: one profiler at a time).
* Optional ``tracemalloc`` helpers to find allocation hot spots.
"""

from __future__ import annotations

import asyncio
import cProfile
import functools
import gc
import inspect
import io
import logging
import pstats
import threading
import time
import tracemalloc
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, TypeVar

from .metrics_collector import MetricsCollector

__all__ = ["PerformanceMonitor", "OperationStats", "ProfileResult"]

_log = logging.getLogger("observability.performance")
F = TypeVar("F", bound=Callable[..., Any])
_MAX_OPERATIONS = 1000


@dataclass
class OperationStats:
    operation: str
    count: int
    errors: int
    slow: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class ProfileResult:
    report: str = ""


class _OpWindow:
    __slots__ = ("durations", "count", "errors", "slow", "total", "min", "max")

    def __init__(self, size: int) -> None:
        self.durations: Deque[float] = deque(maxlen=size)
        self.count = self.errors = self.slow = 0
        self.total = 0.0
        self.min = float("inf")
        self.max = 0.0


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


class _MeasureScope:
    """Context manager for ``with`` and ``async with``; exposes ``duration_s`` / ``cpu_s``."""

    __slots__ = ("_mon", "_op", "_threshold", "_t0", "_c0", "duration_s", "cpu_s")

    def __init__(self, monitor: "PerformanceMonitor", operation: str, threshold: Optional[float]) -> None:
        self._mon = monitor
        self._op = operation
        self._threshold = threshold
        self.duration_s = 0.0
        self.cpu_s = 0.0

    def __enter__(self) -> "_MeasureScope":
        self._c0 = time.thread_time()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        self.duration_s = time.perf_counter() - self._t0
        self.cpu_s = time.thread_time() - self._c0
        self._mon.record(
            self._op, self.duration_s, error=exc is not None and isinstance(exc, Exception),
            threshold=self._threshold, cpu_s=self.cpu_s,
        )
        return False

    async def __aenter__(self) -> "_MeasureScope":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)


class PerformanceMonitor:
    """Tracks latency of named operations and the health of the event loop."""

    _profile_lock = threading.Lock()

    def __init__(
        self,
        metrics: Optional[MetricsCollector] = None,
        *,
        slow_threshold_s: float = 1.0,
        window_size: int = 1024,
        loop_lag_interval: float = 0.5,
        loop_lag_warn_s: float = 0.1,
        on_slow: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.slow_threshold_s = slow_threshold_s
        self.window_size = window_size
        self.loop_lag_interval = loop_lag_interval
        self.loop_lag_warn_s = loop_lag_warn_s
        self._on_slow = on_slow
        self._ops: Dict[str, _OpWindow] = {}
        self._thresholds: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._lag: Deque[float] = deque(maxlen=512)
        self._task: Optional["asyncio.Task[None]"] = None
        self._metrics = metrics
        if metrics is not None:
            r = metrics.registry
            self._h_op = r.histogram("operation_duration_seconds", "Duration of instrumented operations", ("operation",))
            self._c_err = r.counter("operation_errors_total", "Failed instrumented operations", ("operation",))
            self._c_slow = r.counter("slow_operations_total", "Operations exceeding their slow threshold", ("operation",))
            self._g_lag = r.gauge("event_loop_lag_seconds", "Most recent asyncio event loop lag")
            self._h_lag = r.histogram(
                "event_loop_lag_distribution_seconds", "Event loop lag distribution", (),
                (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
            )

    # -- operation timing ----------------------------------------------------
    def set_threshold(self, operation: str, seconds: float) -> None:
        """Override the slow threshold for one operation."""
        self._thresholds[operation] = seconds

    def measure(self, operation: str, *, threshold: Optional[float] = None) -> _MeasureScope:
        return _MeasureScope(self, operation, threshold)

    def timed(self, operation: Optional[str] = None, *, threshold: Optional[float] = None) -> Callable[[F], F]:
        """Decorator for sync or async callables."""

        def decorator(func: F) -> F:
            name = operation or f"{func.__module__}.{func.__qualname__}"
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    async with self.measure(name, threshold=threshold):
                        return await func(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.measure(name, threshold=threshold):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def record(
        self,
        operation: str,
        duration_s: float,
        *,
        error: bool = False,
        threshold: Optional[float] = None,
        cpu_s: Optional[float] = None,
    ) -> None:
        limit = threshold or self._thresholds.get(operation) or self.slow_threshold_s
        is_slow = duration_s > limit
        with self._lock:
            win = self._ops.get(operation)
            if win is None:
                if len(self._ops) >= _MAX_OPERATIONS:
                    return
                win = self._ops[operation] = _OpWindow(self.window_size)
            win.durations.append(duration_s)
            win.count += 1
            win.total += duration_s
            win.min = min(win.min, duration_s)
            win.max = max(win.max, duration_s)
            if error:
                win.errors += 1
            if is_slow:
                win.slow += 1
        if self._metrics is not None:
            self._h_op.observe(duration_s, operation=operation)
            if error:
                self._c_err.inc(operation=operation)
            if is_slow:
                self._c_slow.inc(operation=operation)
        if is_slow:
            _log.warning(
                "slow operation: %s", operation,
                extra={
                    "operation": operation, "duration_ms": round(duration_s * 1000, 2),
                    "cpu_ms": None if cpu_s is None else round(cpu_s * 1000, 2),
                    "threshold_ms": round(limit * 1000, 2),
                },
            )
            if self._on_slow is not None:
                try:
                    self._on_slow(operation, duration_s)
                except Exception:
                    _log.exception("on_slow callback failed")

    def stats(self, operation: str) -> Optional[OperationStats]:
        with self._lock:
            win = self._ops.get(operation)
            if win is None or win.count == 0:
                return None
            data = sorted(win.durations)
            count, errors, slow, total, lo, hi = win.count, win.errors, win.slow, win.total, win.min, win.max
        return OperationStats(
            operation, count, errors, slow, total / count * 1000, lo * 1000, hi * 1000,
            _percentile(data, 0.50) * 1000, _percentile(data, 0.95) * 1000,
            _percentile(data, 0.99) * 1000, total * 1000,
        )

    def all_stats(self, sort_by: str = "p95_ms", limit: Optional[int] = None) -> List[OperationStats]:
        with self._lock:
            names = list(self._ops)
        result = [s for s in (self.stats(n) for n in names) if s is not None]
        result.sort(key=lambda s: getattr(s, sort_by), reverse=True)
        return result[:limit] if limit else result

    # -- event loop lag ------------------------------------------------------
    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self._lag_loop(), name="obs-loop-lag")

    async def _lag_loop(self) -> None:
        interval = self.loop_lag_interval
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(interval)
            lag = max(0.0, time.perf_counter() - t0 - interval)
            self._lag.append(lag)
            if self._metrics is not None:
                self._g_lag.set(lag)
                self._h_lag.observe(lag)
            if lag > self.loop_lag_warn_s:
                _log.warning(
                    "event loop lag detected",
                    extra={"lag_ms": round(lag * 1000, 2), "warn_ms": round(self.loop_lag_warn_s * 1000, 2)},
                )

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @property
    def loop_lag_ms(self) -> Dict[str, float]:
        data = sorted(self._lag)
        return {
            "last": (self._lag[-1] * 1000) if self._lag else 0.0,
            "p99": _percentile(data, 0.99) * 1000,
            "max": (data[-1] * 1000) if data else 0.0,
        }

    # -- profiling -----------------------------------------------------------
    @contextmanager
    def profile(self, sort: str = "cumulative", limit: int = 30) -> Iterator[ProfileResult]:
        """cProfile a block. Only one profile can run at a time (raises ``RuntimeError``)."""
        if not self._profile_lock.acquire(blocking=False):
            raise RuntimeError("another profiling session is already running")
        result = ProfileResult()
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            try:
                yield result
            finally:
                profiler.disable()
                buf = io.StringIO()
                pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats(sort).print_stats(limit)
                result.report = buf.getvalue()
        finally:
            self._profile_lock.release()

    # -- memory --------------------------------------------------------------
    @staticmethod
    def start_memory_tracking(frames: int = 10) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start(frames)

    @staticmethod
    def stop_memory_tracking() -> None:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    @staticmethod
    def memory_top(limit: int = 10) -> List[Dict[str, Any]]:
        """Top allocation sites (requires :meth:`start_memory_tracking`)."""
        if not tracemalloc.is_tracing():
            return []
        snapshot = tracemalloc.take_snapshot().filter_traces(
            (tracemalloc.Filter(False, tracemalloc.__file__), tracemalloc.Filter(False, "<frozen importlib._bootstrap>"))
        )
        return [
            {"location": str(s.traceback[0]), "size_kb": round(s.size / 1024, 1), "count": s.count}
            for s in snapshot.statistics("lineno")[:limit]
        ]

    @staticmethod
    def gc_stats() -> Dict[str, Any]:
        return {"counts": gc.get_count(), "stats": gc.get_stats(), "enabled": gc.isenabled()}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "operations": [s.to_dict() for s in self.all_stats(limit=25)],
            "event_loop_lag_ms": self.loop_lag_ms,
        }
