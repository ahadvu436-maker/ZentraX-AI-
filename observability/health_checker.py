"""Liveness / readiness / startup probes with timeouts, caching and hysteresis.

A check function may be sync or async and may return ``None``/``True`` (healthy),
``False`` (unhealthy), a :class:`HealthStatus`, ``(ok, message)``, a ``dict`` or a
:class:`CheckResult`. Exceptions and timeouts count as unhealthy.

Aggregation: any *critical* unhealthy check -> UNHEALTHY; a non-critical failure
or any degraded check -> DEGRADED; otherwise HEALTHY. HTTP mapping: 200 for
healthy/degraded, 503 for unhealthy.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .metrics_collector import MetricsCollector, SystemMetricsReader

__all__ = [
    "HealthStatus",
    "ProbeType",
    "CheckResult",
    "HealthReport",
    "HealthChecker",
    "tcp_check",
    "memory_check",
    "disk_check",
]

_log = logging.getLogger("observability.health")


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ProbeType(str, Enum):
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


@dataclass
class CheckResult:
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    critical: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "status": self.status.value, "message": self.message,
            "duration_ms": round(self.duration_ms, 2), "timestamp": self.timestamp,
            "critical": self.critical, "details": self.details,
        }


@dataclass
class HealthReport:
    status: HealthStatus
    probe: ProbeType
    checks: List[CheckResult]
    service: str = "service"
    uptime_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def http_status(self) -> int:
        return 503 if self.status is HealthStatus.UNHEALTHY else 200

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value, "probe": self.probe.value, "service": self.service,
            "uptime_seconds": round(self.uptime_seconds, 1), "timestamp": self.timestamp,
            "checks": [c.to_dict() for c in self.checks],
        }


CheckFunc = Callable[[], Union[Awaitable[Any], Any]]


@dataclass
class _Registration:
    name: str
    func: CheckFunc
    probes: Tuple[ProbeType, ...]
    critical: bool
    timeout: float
    cache_ttl: float
    failure_threshold: int
    success_threshold: int
    cached: Optional[CheckResult] = None
    cached_at: float = 0.0
    stable: Optional[HealthStatus] = None
    failures: int = 0
    successes: int = 0


def _normalize(name: str, raw: Any, critical: bool) -> CheckResult:
    if isinstance(raw, CheckResult):
        raw.critical = critical
        return raw
    message = ""
    details: Dict[str, Any] = {}
    if isinstance(raw, tuple) and raw:
        message = str(raw[1]) if len(raw) > 1 else ""
        raw = raw[0]
    if isinstance(raw, dict):
        details = dict(raw)
        message = message or str(raw.get("message", ""))
        raw = raw.get("status", raw.get("healthy", True))
    if raw is None or raw is True:
        status = HealthStatus.HEALTHY
    elif raw is False:
        status = HealthStatus.UNHEALTHY
    elif isinstance(raw, HealthStatus):
        status = raw
    elif isinstance(raw, str):
        try:
            status = HealthStatus(raw.lower())
        except ValueError:
            status = HealthStatus.HEALTHY if raw.lower() in ("ok", "up", "pass") else HealthStatus.UNHEALTHY
    else:
        status = HealthStatus.HEALTHY if raw else HealthStatus.UNHEALTHY
    return CheckResult(name, status, message, details=details, critical=critical)


class HealthChecker:
    """Registry and runner for health probes."""

    def __init__(
        self,
        service_name: str = "service",
        *,
        metrics: Optional[MetricsCollector] = None,
        max_concurrency: int = 16,
    ) -> None:
        self.service_name = service_name
        self._regs: Dict[str, _Registration] = {}
        self._started = False
        self._shutting_down = False
        self._started_at = time.monotonic()
        self._max_concurrency = max_concurrency
        self._last: Dict[ProbeType, HealthReport] = {}
        self._task: Optional["asyncio.Task[None]"] = None
        self._gauge = (
            metrics.registry.gauge("health_check_status", "1 healthy, 0.5 degraded, 0 unhealthy", ("check",))
            if metrics else None
        )

    # -- registration --------------------------------------------------------
    def register(
        self,
        name: str,
        func: CheckFunc,
        *,
        probes: Iterable[ProbeType] = (ProbeType.READINESS,),
        critical: bool = True,
        timeout: float = 2.0,
        cache_ttl: float = 0.0,
        failure_threshold: int = 1,
        success_threshold: int = 1,
    ) -> None:
        """Register a check. ``failure_threshold`` consecutive failures are needed to flip to
        UNHEALTHY (and ``success_threshold`` successes to recover) which prevents flapping."""
        if failure_threshold < 1 or success_threshold < 1:
            raise ValueError("thresholds must be >= 1")
        self._regs[name] = _Registration(
            name, func, tuple(probes), critical, timeout, cache_ttl, failure_threshold, success_threshold
        )

    def unregister(self, name: str) -> None:
        self._regs.pop(name, None)

    @property
    def check_names(self) -> List[str]:
        return list(self._regs)

    # -- lifecycle flags -----------------------------------------------------
    def mark_started(self) -> None:
        self._started = True

    def begin_shutdown(self) -> None:
        """Fail readiness immediately so load balancers drain traffic before exit."""
        self._shutting_down = True

    @property
    def started(self) -> bool:
        return self._started

    # -- execution -----------------------------------------------------------
    async def _invoke(self, reg: _Registration) -> CheckResult:
        start = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(reg.func):
                raw = await asyncio.wait_for(reg.func(), reg.timeout)
            else:
                loop = asyncio.get_running_loop()
                raw = await asyncio.wait_for(loop.run_in_executor(None, reg.func), reg.timeout)
                if inspect.isawaitable(raw):
                    raw = await asyncio.wait_for(raw, max(0.05, reg.timeout - (time.perf_counter() - start)))
            result = _normalize(reg.name, raw, reg.critical)
        except (asyncio.TimeoutError, TimeoutError):
            result = CheckResult(reg.name, HealthStatus.UNHEALTHY, f"timed out after {reg.timeout}s", critical=reg.critical)
        except Exception as exc:
            result = CheckResult(
                reg.name, HealthStatus.UNHEALTHY, f"{type(exc).__name__}: {exc}", critical=reg.critical
            )
        result.duration_ms = (time.perf_counter() - start) * 1000.0
        return self._apply_hysteresis(reg, result)

    @staticmethod
    def _apply_hysteresis(reg: _Registration, result: CheckResult) -> CheckResult:
        raw = result.status
        if raw is HealthStatus.UNHEALTHY:
            reg.failures += 1
            reg.successes = 0
            if reg.stable is None or reg.failures >= reg.failure_threshold:
                reg.stable = HealthStatus.UNHEALTHY
            else:
                result.status = reg.stable
                result.message = f"{result.message} (failure {reg.failures}/{reg.failure_threshold})"
        else:
            reg.successes += 1
            reg.failures = 0
            if reg.stable is HealthStatus.UNHEALTHY and reg.successes < reg.success_threshold:
                result.status = HealthStatus.UNHEALTHY
                result.message = f"recovering ({reg.successes}/{reg.success_threshold})"
            else:
                reg.stable = raw
        return result

    async def _run_one(self, reg: _Registration, use_cache: bool, sem: asyncio.Semaphore) -> CheckResult:
        now = time.monotonic()
        if use_cache and reg.cache_ttl > 0 and reg.cached and now - reg.cached_at < reg.cache_ttl:
            return reg.cached
        async with sem:
            result = await self._invoke(reg)
        reg.cached, reg.cached_at = result, time.monotonic()
        if self._gauge is not None:
            value = {HealthStatus.HEALTHY: 1.0, HealthStatus.DEGRADED: 0.5}.get(result.status, 0.0)
            self._gauge.set(value, check=reg.name)
        return result

    @staticmethod
    def _aggregate(results: List[CheckResult]) -> HealthStatus:
        status = HealthStatus.HEALTHY
        for r in results:
            if r.status is HealthStatus.UNHEALTHY:
                if r.critical:
                    return HealthStatus.UNHEALTHY
                status = HealthStatus.DEGRADED
            elif r.status is HealthStatus.DEGRADED:
                status = HealthStatus.DEGRADED
        return status

    async def check(self, probe: ProbeType = ProbeType.READINESS, *, use_cache: bool = True) -> HealthReport:
        """Run all checks registered for ``probe`` concurrently and aggregate."""
        regs = [r for r in self._regs.values() if probe in r.probes]
        sem = asyncio.Semaphore(self._max_concurrency)
        results = list(await asyncio.gather(*(self._run_one(r, use_cache, sem) for r in regs)))
        status = self._aggregate(results)

        if probe is ProbeType.STARTUP and not self._started:
            results.append(CheckResult("startup", HealthStatus.UNHEALTHY, "application still starting"))
            status = HealthStatus.UNHEALTHY
        elif probe is ProbeType.READINESS:
            if not self._started:
                results.append(CheckResult("startup", HealthStatus.UNHEALTHY, "application still starting"))
                status = HealthStatus.UNHEALTHY
            elif self._shutting_down:
                results.append(CheckResult("shutdown", HealthStatus.UNHEALTHY, "draining before shutdown"))
                status = HealthStatus.UNHEALTHY

        report = HealthReport(
            status, probe, results, self.service_name, time.monotonic() - self._started_at
        )
        self._last[probe] = report
        return report

    async def is_alive(self) -> bool:
        return (await self.check(ProbeType.LIVENESS)).status is not HealthStatus.UNHEALTHY

    async def is_ready(self) -> bool:
        return (await self.check(ProbeType.READINESS)).status is not HealthStatus.UNHEALTHY

    def last_report(self, probe: ProbeType = ProbeType.READINESS) -> Optional[HealthReport]:
        return self._last.get(probe)

    # -- background refresh --------------------------------------------------
    async def start(self, interval: float = 15.0) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self._run(interval), name="obs-health")

    async def _run(self, interval: float) -> None:
        while True:
            for probe in (ProbeType.LIVENESS, ProbeType.READINESS):
                try:
                    await self.check(probe, use_cache=False)
                except Exception:
                    _log.exception("background health check failed")
            await asyncio.sleep(interval)

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# --------------------------------------------------------------------------
# Ready-made checks
# --------------------------------------------------------------------------
def tcp_check(host: str, port: int, timeout: float = 1.0) -> Callable[[], Awaitable[Tuple[bool, str]]]:
    """Check that a TCP endpoint (database, cache, broker) accepts connections."""

    async def _check() -> Tuple[bool, str]:
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        except (OSError, asyncio.TimeoutError) as exc:
            return False, f"{host}:{port} unreachable ({type(exc).__name__})"
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True, f"{host}:{port} reachable"

    return _check


def memory_check(max_used_percent: float = 90.0) -> Callable[[], Any]:
    """DEGRADED above ``max_used_percent`` system memory used, UNHEALTHY above 98%."""
    reader = SystemMetricsReader()

    def _check() -> Any:
        used = reader.read().get("system_memory_percent")
        if used is None:
            return True, "memory usage unavailable"
        status = HealthStatus.HEALTHY
        if used > 98.0:
            status = HealthStatus.UNHEALTHY
        elif used > max_used_percent:
            status = HealthStatus.DEGRADED
        return status, f"{used:.1f}% memory used"

    return _check


def disk_check(path: str = "/", max_used_percent: float = 90.0) -> Callable[[], Any]:
    """DEGRADED above ``max_used_percent`` disk usage, UNHEALTHY above 98%."""

    def _check() -> Any:
        usage = shutil.disk_usage(path)
        used = usage.used / usage.total * 100.0
        status = HealthStatus.HEALTHY
        if used > 98.0:
            status = HealthStatus.UNHEALTHY
        elif used > max_used_percent:
            status = HealthStatus.DEGRADED
        return status, f"{used:.1f}% of {path} used"

    return _check
