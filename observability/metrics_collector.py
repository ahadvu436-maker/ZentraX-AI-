"""Application + system metrics with Prometheus exposition.

* ``Counter`` / ``Gauge`` / ``Histogram`` with labels, thread-safe, with a
  per-metric series cap (cardinality protection).
* ``MetricsCollector`` adds RED metrics (rate, errors, duration), throughput and
  latency percentiles, and periodic CPU/memory/disk/process sampling.
* ``psutil`` is used when installed; a stdlib/``/proc`` fallback is used otherwise.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
from bisect import bisect_left
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "MetricsCollector",
    "SystemMetricsReader",
    "DEFAULT_BUCKETS",
]

_log = logging.getLogger("observability.metrics")

DEFAULT_BUCKETS: Tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)
DEFAULT_MAX_SERIES = 5000
_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")

Sample = Tuple[str, Dict[str, str], float]


def _fmt_value(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt_labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{_escape(str(v))}"' for k, v in labels.items()) + "}"


class _MetricBase:
    kind = "untyped"

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Sequence[str] = (),
        max_series: int = DEFAULT_MAX_SERIES,
    ) -> None:
        if not _NAME_RE.match(name):
            raise ValueError(f"Invalid metric name: {name!r}")
        for label in label_names:
            if not _NAME_RE.match(label) or label.startswith("__"):
                raise ValueError(f"Invalid label name: {label!r}")
        self.name = name
        self.description = description
        self.label_names: Tuple[str, ...] = tuple(label_names)
        self.max_series = max_series
        self.dropped_series = 0
        self._lock = threading.Lock()

    def _key(self, labels: Mapping[str, Any]) -> Tuple[str, ...]:
        if len(labels) != len(self.label_names):
            raise ValueError(
                f"{self.name}: expected labels {self.label_names}, got {tuple(labels)}"
            )
        try:
            return tuple(str(labels[n]) for n in self.label_names)
        except KeyError as exc:
            raise ValueError(f"{self.name}: missing label {exc}") from None

    def _labels_dict(self, key: Tuple[str, ...]) -> Dict[str, str]:
        return dict(zip(self.label_names, key))

    def collect(self) -> List[Sample]:  # pragma: no cover - interface
        raise NotImplementedError

    def snapshot(self) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError


class Counter(_MetricBase):
    """Monotonically increasing value."""

    kind = "counter"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._values: Dict[Tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        if amount < 0:
            raise ValueError("Counter can only increase")
        key = self._key(labels)
        with self._lock:
            if key not in self._values and len(self._values) >= self.max_series:
                self.dropped_series += 1
                return
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: Any) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def total(self) -> float:
        with self._lock:
            return sum(self._values.values())

    def collect(self) -> List[Sample]:
        with self._lock:
            items = sorted(self._values.items())
        return [(self.name, self._labels_dict(k), v) for k, v in items]

    def snapshot(self) -> List[Dict[str, Any]]:
        return [{"labels": lbl, "value": v} for _, lbl, v in self.collect()]


class Gauge(_MetricBase):
    """Value that can go up and down."""

    kind = "gauge"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._values: Dict[Tuple[str, ...], float] = {}

    def set(self, value: float, **labels: Any) -> None:
        key = self._key(labels)
        with self._lock:
            if key not in self._values and len(self._values) >= self.max_series:
                self.dropped_series += 1
                return
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        key = self._key(labels)
        with self._lock:
            if key not in self._values and len(self._values) >= self.max_series:
                self.dropped_series += 1
                return
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: Any) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: Any) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def collect(self) -> List[Sample]:
        with self._lock:
            items = sorted(self._values.items())
        return [(self.name, self._labels_dict(k), v) for k, v in items]

    def snapshot(self) -> List[Dict[str, Any]]:
        return [{"labels": lbl, "value": v} for _, lbl, v in self.collect()]


class _HistSeries:
    __slots__ = ("counts", "sum", "count")

    def __init__(self, n: int) -> None:
        self.counts = [0] * (n + 1)  # last slot = +Inf
        self.sum = 0.0
        self.count = 0


class Histogram(_MetricBase):
    """Bucketed distribution with quantile estimation."""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
        max_series: int = DEFAULT_MAX_SERIES,
    ) -> None:
        super().__init__(name, description, label_names, max_series)
        bounds = sorted(float(b) for b in buckets if not math.isinf(b))
        if not bounds:
            raise ValueError("Histogram needs at least one finite bucket")
        self.buckets: Tuple[float, ...] = tuple(bounds)
        self._series: Dict[Tuple[str, ...], _HistSeries] = {}

    def observe(self, value: float, **labels: Any) -> None:
        key = self._key(labels)
        idx = bisect_left(self.buckets, value)
        with self._lock:
            series = self._series.get(key)
            if series is None:
                if len(self._series) >= self.max_series:
                    self.dropped_series += 1
                    return
                series = self._series[key] = _HistSeries(len(self.buckets))
            series.counts[idx] += 1
            series.sum += value
            series.count += 1

    def _aggregate(self, filters: Mapping[str, Any]) -> Tuple[List[int], int, float]:
        positions = []
        for name, value in filters.items():
            if name not in self.label_names:
                raise ValueError(f"{self.name}: unknown label {name!r}")
            positions.append((self.label_names.index(name), str(value)))
        counts = [0] * (len(self.buckets) + 1)
        total, ssum = 0, 0.0
        with self._lock:
            for key, series in self._series.items():
                if all(key[i] == v for i, v in positions):
                    for i, c in enumerate(series.counts):
                        counts[i] += c
                    total += series.count
                    ssum += series.sum
        return counts, total, ssum

    def _quantile(self, counts: Sequence[int], total: int, q: float) -> Optional[float]:
        if total == 0:
            return None
        rank = min(max(q, 0.0), 1.0) * total
        cumulative = 0
        for i, c in enumerate(counts):
            prev = cumulative
            cumulative += c
            if c and cumulative >= rank:
                if i >= len(self.buckets):
                    return self.buckets[-1]
                lower = 0.0 if i == 0 else self.buckets[i - 1]
                return lower + (self.buckets[i] - lower) * ((rank - prev) / c)
        return self.buckets[-1]

    def quantile(self, q: float, **filters: Any) -> Optional[float]:
        """Estimate quantile ``q`` (0..1) across all series matching ``filters``."""
        counts, total, _ = self._aggregate(filters)
        return self._quantile(counts, total, q)

    def stats(self, **filters: Any) -> Dict[str, Optional[float]]:
        counts, total, ssum = self._aggregate(filters)
        return {
            "count": float(total),
            "sum": ssum,
            "mean": (ssum / total) if total else None,
            "p50": self._quantile(counts, total, 0.50),
            "p95": self._quantile(counts, total, 0.95),
            "p99": self._quantile(counts, total, 0.99),
        }

    def collect(self) -> List[Sample]:
        out: List[Sample] = []
        with self._lock:
            items = sorted(self._series.items())
            copies = [(k, list(s.counts), s.sum, s.count) for k, s in items]
        for key, counts, ssum, count in copies:
            base = self._labels_dict(key)
            cumulative = 0
            for bound, c in zip(self.buckets, counts):
                cumulative += c
                out.append((f"{self.name}_bucket", {**base, "le": repr(bound)}, float(cumulative)))
            out.append((f"{self.name}_bucket", {**base, "le": "+Inf"}, float(count)))
            out.append((f"{self.name}_sum", dict(base), ssum))
            out.append((f"{self.name}_count", dict(base), float(count)))
        return out

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            keys = sorted(self._series)
        result = []
        for key in keys:
            lbl = self._labels_dict(key)
            result.append({"labels": lbl, **self.stats(**lbl)})
        return result


class MetricsRegistry:
    """Holds metrics and renders them (Prometheus text format / JSON snapshot)."""

    def __init__(self) -> None:
        self._metrics: Dict[str, _MetricBase] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, cls: type, name: str, description: str, labels: Sequence[str], **kw: Any) -> Any:
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if type(existing) is not cls or existing.label_names != tuple(labels):
                    raise ValueError(f"Metric {name!r} already registered with a different definition")
                return existing
            metric = cls(name, description, labels, **kw)
            self._metrics[name] = metric
            return metric

    def counter(self, name: str, description: str = "", labels: Sequence[str] = ()) -> Counter:
        return self._get_or_create(Counter, name, description, labels)

    def gauge(self, name: str, description: str = "", labels: Sequence[str] = ()) -> Gauge:
        return self._get_or_create(Gauge, name, description, labels)

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> Histogram:
        return self._get_or_create(Histogram, name, description, labels, buckets=buckets)

    def get(self, name: str) -> Optional[_MetricBase]:
        return self._metrics.get(name)

    def metrics(self) -> List[_MetricBase]:
        with self._lock:
            return list(self._metrics.values())

    def render_prometheus(self) -> str:
        lines: List[str] = []
        for metric in sorted(self.metrics(), key=lambda m: m.name):
            samples = metric.collect()
            if not samples:
                continue
            if metric.description:
                lines.append(f"# HELP {metric.name} {_escape(metric.description)}")
            lines.append(f"# TYPE {metric.name} {metric.kind}")
            for name, labels, value in samples:
                lines.append(f"{name}{_fmt_labels(labels)} {_fmt_value(value)}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> Dict[str, Any]:
        return {
            m.name: {"type": m.kind, "description": m.description, "series": m.snapshot()}
            for m in sorted(self.metrics(), key=lambda m: m.name)
        }


# --------------------------------------------------------------------------
# System metrics
# --------------------------------------------------------------------------
class SystemMetricsReader:
    """Reads CPU / memory / disk / process stats. Uses psutil when available."""

    def __init__(self, disk_path: Optional[str] = None) -> None:
        self._psutil: Any = None
        self._proc: Any = None
        self._disk_path = disk_path or os.path.abspath(os.sep)
        try:
            import psutil  # type: ignore[import-not-found]

            self._psutil = psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)
            psutil.cpu_percent(None)
        except Exception:
            self._psutil = None
        self._last_wall = time.monotonic()
        self._last_cpu = time.process_time()
        self._last_stat = self._read_proc_stat()

    @property
    def backend(self) -> str:
        return "psutil" if self._psutil else "stdlib"

    @staticmethod
    def _read_proc_stat() -> Optional[Tuple[float, float]]:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as fh:
                parts = fh.readline().split()[1:]
            values = [float(p) for p in parts]
            idle = values[3] + (values[4] if len(values) > 4 else 0.0)
            return sum(values), idle
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _read_meminfo() -> Optional[Tuple[float, float]]:
        try:
            info: Dict[str, float] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    if key in ("MemTotal", "MemAvailable"):
                        info[key] = float(rest.split()[0]) * 1024
            return info["MemTotal"], info["MemAvailable"]
        except (OSError, KeyError, ValueError, IndexError):
            return None

    @staticmethod
    def _rss_bytes() -> Optional[float]:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as fh:
                return float(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError):
            pass
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return float(rss if sys.platform == "darwin" else rss * 1024)
        except Exception:
            return None

    def read(self) -> Dict[str, Optional[float]]:
        data: Dict[str, Optional[float]] = {
            "process_cpu_percent": None, "system_cpu_percent": None,
            "process_rss_bytes": None, "system_memory_percent": None,
            "system_memory_available_bytes": None, "process_threads": None,
            "process_open_fds": None, "load1": None, "disk_used_percent": None,
        }
        now_wall, now_cpu = time.monotonic(), time.process_time()
        if self._psutil is not None:
            ps = self._psutil
            try:
                data["process_cpu_percent"] = float(self._proc.cpu_percent(None))
                data["system_cpu_percent"] = float(ps.cpu_percent(None))
                data["process_rss_bytes"] = float(self._proc.memory_info().rss)
                vm = ps.virtual_memory()
                data["system_memory_percent"] = float(vm.percent)
                data["system_memory_available_bytes"] = float(vm.available)
                data["process_threads"] = float(self._proc.num_threads())
                if hasattr(self._proc, "num_fds"):
                    data["process_open_fds"] = float(self._proc.num_fds())
            except Exception:
                _log.debug("psutil read failed", exc_info=True)
        else:
            dw = now_wall - self._last_wall
            if dw > 0:
                data["process_cpu_percent"] = max(0.0, (now_cpu - self._last_cpu) / dw * 100.0)
            stat = self._read_proc_stat()
            if stat and self._last_stat and stat[0] > self._last_stat[0]:
                dt, di = stat[0] - self._last_stat[0], stat[1] - self._last_stat[1]
                data["system_cpu_percent"] = max(0.0, min(100.0, (1 - di / dt) * 100.0))
            self._last_stat = stat
            data["process_rss_bytes"] = self._rss_bytes()
            mem = self._read_meminfo()
            if mem:
                data["system_memory_percent"] = (1 - mem[1] / mem[0]) * 100.0
                data["system_memory_available_bytes"] = mem[1]
            data["process_threads"] = float(threading.active_count())
            try:
                data["process_open_fds"] = float(len(os.listdir("/proc/self/fd")))
            except OSError:
                pass
        self._last_wall, self._last_cpu = now_wall, now_cpu
        try:
            data["load1"] = os.getloadavg()[0]
        except (OSError, AttributeError):
            pass
        try:
            usage = shutil.disk_usage(self._disk_path)
            data["disk_used_percent"] = usage.used / usage.total * 100.0
        except OSError:
            pass
        return data


class _RateWindow:
    """Per-second buckets for cheap sliding-window rate calculation."""

    def __init__(self, seconds: int = 300) -> None:
        self.seconds = seconds
        self._buckets: Deque[List[int]] = deque()
        self._lock = threading.Lock()
        self._created = time.monotonic()

    def add(self, n: int = 1) -> None:
        sec = int(time.monotonic())
        with self._lock:
            if self._buckets and self._buckets[-1][0] == sec:
                self._buckets[-1][1] += n
            else:
                self._buckets.append([sec, n])
            horizon = sec - self.seconds
            while self._buckets and self._buckets[0][0] <= horizon:
                self._buckets.popleft()

    def count(self, window: Optional[float] = None) -> int:
        window = min(window or self.seconds, self.seconds)
        horizon = time.monotonic() - window
        with self._lock:
            return sum(n for sec, n in self._buckets if sec + 1 > horizon)

    def rate(self, window: Optional[float] = None) -> float:
        window = min(window or 60.0, self.seconds)
        effective = max(1.0, min(window, time.monotonic() - self._created))
        return self.count(window) / effective


class _RequestTracker:
    """Yielded by :meth:`MetricsCollector.track_request`; set ``status_code`` as needed."""

    __slots__ = ("_collector", "method", "route", "status_code", "_start")

    def __init__(self, collector: "MetricsCollector", method: str, route: str) -> None:
        self._collector = collector
        self.method = method
        self.route = route
        self.status_code = 200
        self._start = 0.0

    def __enter__(self) -> "_RequestTracker":
        self._collector.in_flight.inc()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        if exc is not None and self.status_code < 500:
            self.status_code = 500
        self._collector.in_flight.dec()
        self._collector.record_request(
            self.method, self.route, self.status_code, time.perf_counter() - self._start
        )
        return False

    async def __aenter__(self) -> "_RequestTracker":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)


class MetricsCollector:
    """High-level facade: RED metrics + system sampler + registry."""

    def __init__(
        self,
        service_name: str = "service",
        *,
        registry: Optional[MetricsRegistry] = None,
        collect_system: bool = True,
        interval: float = 15.0,
        latency_buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        self.service_name = service_name
        self.registry = registry or MetricsRegistry()
        self.interval = interval
        self.collect_system_enabled = collect_system
        r = self.registry
        self.http_requests = r.counter(
            "http_requests_total", "Total HTTP requests", ("method", "route", "status")
        )
        self.http_duration = r.histogram(
            "http_request_duration_seconds", "HTTP request latency in seconds",
            ("method", "route"), latency_buckets,
        )
        self.in_flight = r.gauge("http_requests_in_flight", "Requests currently being served")
        self._gauges = {
            "process_cpu_percent": r.gauge("process_cpu_percent", "Process CPU usage percent"),
            "system_cpu_percent": r.gauge("system_cpu_percent", "System-wide CPU usage percent"),
            "process_rss_bytes": r.gauge("process_resident_memory_bytes", "Process resident memory"),
            "system_memory_percent": r.gauge("system_memory_used_percent", "System memory used percent"),
            "system_memory_available_bytes": r.gauge("system_memory_available_bytes", "Available system memory"),
            "process_threads": r.gauge("process_threads", "Process thread count"),
            "process_open_fds": r.gauge("process_open_fds", "Open file descriptors"),
            "load1": r.gauge("system_load1", "1-minute load average"),
            "disk_used_percent": r.gauge("disk_used_percent", "Root disk used percent"),
        }
        self._gc_gauge = r.gauge("python_gc_collections", "GC collections per generation", ("generation",))
        self._reader = SystemMetricsReader()
        self._requests_window = _RateWindow(300)
        self._errors_window = _RateWindow(300)
        self.last_system: Dict[str, Optional[float]] = {}
        self._task: Optional["asyncio.Task[None]"] = None
        self._started_at = time.time()

    # -- request metrics -----------------------------------------------------
    def record_request(self, method: str, route: str, status: int, duration_s: float) -> None:
        self.http_requests.inc(method=method, route=route, status=str(status))
        self.http_duration.observe(duration_s, method=method, route=route)
        self._requests_window.add()
        if status >= 500:
            self._errors_window.add()

    def track_request(self, method: str, route: str) -> _RequestTracker:
        """``with`` / ``async with`` helper that records a request on exit."""
        return _RequestTracker(self, method, route)

    def throughput(self, window: float = 60.0) -> float:
        """Requests per second over the trailing window."""
        return self._requests_window.rate(window)

    def error_ratio(self, window: float = 60.0) -> float:
        """Share of 5xx responses over the trailing window (0..1)."""
        total = self._requests_window.count(window)
        return (self._errors_window.count(window) / total) if total else 0.0

    def latency_percentiles(self, **filters: Any) -> Dict[str, Optional[float]]:
        """Latency in milliseconds; optionally filter by ``method=`` / ``route=``."""
        stats = self.http_duration.stats(**filters)

        def ms(v: Optional[float]) -> Optional[float]:
            return None if v is None else round(v * 1000.0, 3)

        return {
            "count": stats["count"], "mean_ms": ms(stats["mean"]),
            "p50_ms": ms(stats["p50"]), "p95_ms": ms(stats["p95"]), "p99_ms": ms(stats["p99"]),
        }

    # -- system metrics ------------------------------------------------------
    def collect_system(self) -> Dict[str, Optional[float]]:
        data = self._reader.read()
        for key, value in data.items():
            if value is not None and key in self._gauges:
                self._gauges[key].set(value)
        for gen, stats in enumerate(gc.get_stats()):
            self._gc_gauge.set(stats.get("collections", 0), generation=str(gen))
        self.last_system = data
        return data

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        if self.collect_system_enabled:
            self.collect_system()
            self._task = asyncio.get_running_loop().create_task(self._run(), name="obs-system-metrics")

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                self.collect_system()
            except Exception:
                _log.exception("system metrics collection failed")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # -- export --------------------------------------------------------------
    def render_prometheus(self) -> str:
        return self.registry.render_prometheus()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "service": self.service_name,
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "throughput_rps": round(self.throughput(), 3),
            "error_ratio": round(self.error_ratio(), 5),
            "in_flight": self.in_flight.value(),
            "latency": self.latency_percentiles(),
            "system": dict(self.last_system),
            "system_backend": self._reader.backend,
        }
