"""Self-monitoring: rolling metrics, threshold alerts, and operation tracking."""
from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


@dataclass(frozen=True)
class MetricSample:
    name: str
    value: float
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Threshold:
    metric: str
    max_value: Optional[float] = None
    min_value: Optional[float] = None
    severity: Severity = Severity.WARNING

    def violated_by(self, v: float) -> bool:
        return (self.max_value is not None and v > self.max_value) or \
               (self.min_value is not None and v < self.min_value)


@dataclass(frozen=True)
class Alert:
    metric: str
    value: float
    severity: Severity
    message: str
    ts: float = field(default_factory=time.time)


AlertHandler = Callable[[Alert], None]


class SelfMonitor:
    """Thread-safe rolling-window metric store with threshold-based alerting."""

    def __init__(self, window: int = 500, max_alerts: int = 1000) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._series: Dict[str, Deque[MetricSample]] = defaultdict(lambda: deque(maxlen=window))
        self._thresholds: Dict[str, List[Threshold]] = defaultdict(list)
        self._handlers: List[AlertHandler] = []
        self._alerts: Deque[Alert] = deque(maxlen=max_alerts)
        self._lock = threading.RLock()

    def add_threshold(self, t: Threshold) -> None:
        with self._lock:
            self._thresholds[t.metric].append(t)

    def on_alert(self, handler: AlertHandler) -> None:
        with self._lock:
            self._handlers.append(handler)

    def record(self, name: str, value: float) -> None:
        fired: List[Alert] = []
        with self._lock:
            self._series[name].append(MetricSample(name, float(value)))
            for t in self._thresholds.get(name, ()):
                if t.violated_by(value):
                    a = Alert(name, float(value), t.severity,
                              f"{name}={value:.4g} violates threshold "
                              f"(min={t.min_value}, max={t.max_value})")
                    self._alerts.append(a)
                    fired.append(a)
            handlers = list(self._handlers)
        for a in fired:  # outside lock: handlers may call back into the monitor
            for h in handlers:
                try:
                    h(a)
                except Exception:  # a faulty handler must never break monitoring
                    pass

    def stats(self, name: str) -> Dict[str, float]:
        with self._lock:
            vals = [s.value for s in self._series.get(name, ())]
        if not vals:
            return {}
        vals_sorted = sorted(vals)
        p95 = vals_sorted[min(len(vals_sorted) - 1, int(0.95 * len(vals_sorted)))]
        return {"count": len(vals), "mean": statistics.fmean(vals), "min": vals_sorted[0],
                "max": vals_sorted[-1], "p95": p95, "last": vals[-1]}

    def alerts(self, min_severity: Severity = Severity.INFO) -> List[Alert]:
        with self._lock:
            return [a for a in self._alerts if a.severity >= min_severity]

    @contextmanager
    def track(self, operation: str) -> Iterator[None]:
        """Times an operation; records latency and success/failure (re-raises errors)."""
        t0 = time.perf_counter()
        try:
            yield
            self.record(f"{operation}.failure", 0.0)
        except BaseException:
            self.record(f"{operation}.failure", 1.0)
            raise
        finally:
            self.record(f"{operation}.latency_s", time.perf_counter() - t0)

    def failure_rate(self, operation: str) -> float:
        s = self.stats(f"{operation}.failure")
        return s.get("mean", 0.0)

    def introspect(self) -> Dict[str, Any]:
        with self._lock:
            names = list(self._series)
        return {"metrics": {n: self.stats(n) for n in names},
                "alert_count": len(self._alerts)}
