"""Central orchestrator: wires logging, metrics, tracing, health, performance and errors.

Typical usage::

    obs = Observability(ObservabilityConfig(service_name="orders", environment="prod"))
    async with obs:                      # configures logging, starts background tasks
        async with obs.observe("charge_card", attributes={"order_id": 42}):
            ...
        app = obs.asgi_middleware(app)   # RED metrics + traces + /healthz /readyz /metrics

``observe()`` is the one-liner that gives an operation a span, a latency measurement
and exception capture at once, all correlated through the same trace id.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, MutableMapping, Optional, TextIO, TypeVar

from .dashboards.metrics_dashboard import MetricsDashboard
from .error_tracker import ErrorTracker
from .health_checker import HealthChecker, HealthStatus, ProbeType
from .logging_manager import LoggingManager, StructuredLogger, log_context
from .metrics_collector import MetricsCollector
from .performance_monitor import PerformanceMonitor
from .tracing_manager import (
    InMemorySpanExporter,
    Span,
    SpanKind,
    StatusCode,
    TracingManager,
)

__all__ = [
    "ObservabilityConfig",
    "Observability",
    "ObservabilityMiddleware",
    "init_observability",
    "get_observability",
]

F = TypeVar("F", bound=Callable[..., Any])
_log = logging.getLogger("observability.core")


@dataclass
class ObservabilityConfig:
    service_name: str = "service"
    environment: str = "production"
    version: str = "0.0.0"
    log_level: str = "INFO"
    json_logs: bool = True
    async_logging: bool = True
    log_stream: Optional[TextIO] = None
    metrics_enabled: bool = True
    system_metrics_interval: float = 15.0
    tracing_enabled: bool = True
    trace_sampling_ratio: float = 1.0
    health_check_interval: float = 15.0
    slow_operation_threshold: float = 1.0
    event_loop_monitor: bool = True
    install_error_hooks: bool = True

    @classmethod
    def from_env(cls, prefix: str = "OBS_", **overrides: Any) -> "ObservabilityConfig":
        """Build from environment variables, e.g. ``OBS_SERVICE_NAME``, ``OBS_LOG_LEVEL``."""
        values: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            raw = os.environ.get(prefix + f.name.upper())
            if raw is None or f.name == "log_stream":
                continue
            default = f.default
            if isinstance(default, bool):
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, float):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        values.update(overrides)
        return cls(**values)


# --------------------------------------------------------------------------
# Scopes
# --------------------------------------------------------------------------
class _ObserveScope:
    """span + latency measurement + exception capture, as one ``with`` / ``async with``."""

    def __init__(
        self,
        obs: "Observability",
        operation: str,
        attributes: Optional[Mapping[str, Any]],
        kind: SpanKind,
        reraise: bool,
    ) -> None:
        self._span_scope = obs.tracing.start_span(operation, kind=kind, attributes=attributes)
        self._perf = obs.performance.measure(operation)
        self._err = obs.errors.capture(reraise=reraise, context={"operation": operation})

    def __enter__(self) -> Span:
        span = self._span_scope.__enter__()
        self._perf.__enter__()
        self._err.__enter__()
        return span

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        # Reverse order; capture runs while the span is still current (trace ids attached).
        suppress = self._err.__exit__(exc_type, exc, tb)
        self._perf.__exit__(exc_type, exc, tb)
        self._span_scope.__exit__(exc_type, exc, tb)
        return bool(suppress)

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)


class _RequestInfo:
    """Handle yielded by :meth:`Observability.request`."""

    def __init__(self, span: Span, tracker: Any) -> None:
        self.span = span
        self._tracker = tracker
        self.response_headers: Dict[str, str] = {}

    @property
    def status_code(self) -> int:
        return int(self._tracker.status_code)

    @status_code.setter
    def status_code(self, value: int) -> None:
        self._tracker.status_code = int(value)
        self.span.set_attribute("http.status_code", int(value))
        if value >= 500:
            self.span.set_status(StatusCode.ERROR, f"HTTP {value}")


class _RequestScope:
    """Server-side request scope for any framework (extract -> span -> RED metrics -> errors)."""

    def __init__(self, obs: "Observability", method: str, route: str, headers: Optional[Mapping[str, str]]) -> None:
        parent = obs.tracing.extract(headers) if headers else None
        self._obs = obs
        self._span_scope = obs.tracing.start_span(
            f"{method} {route}", kind=SpanKind.SERVER, parent=parent,
            attributes={"http.method": method, "http.route": route},
        )
        self._tracker = obs.metrics.track_request(method, route)
        self._err = obs.errors.capture(reraise=True, context={"http.method": method, "http.route": route})
        self._info: Optional[_RequestInfo] = None

    def __enter__(self) -> _RequestInfo:
        span = self._span_scope.__enter__()
        self._tracker.__enter__()
        self._err.__enter__()
        self._info = _RequestInfo(span, self._tracker)
        self._obs.tracing.inject(self._info.response_headers, span)
        return self._info

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        self._err.__exit__(exc_type, exc, tb)
        if self._info is not None:
            self._info.span.set_attribute("http.status_code", self._tracker.status_code if exc is None else 500)
        self._tracker.__exit__(exc_type, exc, tb)
        self._span_scope.__exit__(exc_type, exc, tb)
        return False

    async def __aenter__(self) -> _RequestInfo:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
class Observability:
    """Owns and coordinates every observability component."""

    def __init__(self, config: Optional[ObservabilityConfig] = None, **overrides: Any) -> None:
        cfg = config or ObservabilityConfig()
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        self.config = cfg
        name = cfg.service_name

        self.logging = LoggingManager(
            name, environment=cfg.environment, version=cfg.version, level=cfg.log_level,
            json_output=cfg.json_logs, stream=cfg.log_stream, async_logging=cfg.async_logging,
        )
        self.metrics = MetricsCollector(
            name, collect_system=cfg.metrics_enabled, interval=cfg.system_metrics_interval
        )
        self.span_store = InMemorySpanExporter()
        self.tracing = TracingManager(
            name, sampling_ratio=cfg.trace_sampling_ratio if cfg.tracing_enabled else 0.0,
            exporters=[self.span_store],
        )
        self.health = HealthChecker(name, metrics=self.metrics)
        self.performance = PerformanceMonitor(self.metrics, slow_threshold_s=cfg.slow_operation_threshold)
        self.errors = ErrorTracker(metrics=self.metrics)
        self.dashboard = MetricsDashboard(
            service_name=name, metrics=self.metrics, health=self.health, errors=self.errors,
            performance=self.performance, tracing=self.tracing,
        )
        self._set_up = False
        self._running = False

    # -- lifecycle -----------------------------------------------------------
    def setup(self) -> "Observability":
        """Synchronous part of start-up: logging + process-wide error hooks. Idempotent."""
        if self._set_up:
            return self
        self.logging.configure()
        if self.config.install_error_hooks:
            self.errors.install_global_handlers()
        self._set_up = True
        return self

    async def start(self) -> "Observability":
        """Configure everything and start background tasks. Idempotent."""
        if self._running:
            return self
        self.setup()
        if self.config.install_error_hooks:
            self.errors.install_global_handlers()  # attaches to the running loop
        await self.metrics.start()
        if self.config.event_loop_monitor:
            await self.performance.start()
        if self.config.health_check_interval > 0:
            await self.health.start(self.config.health_check_interval)
        await self.dashboard.start()
        self.health.mark_started()
        self._running = True
        _log.info(
            "observability started",
            extra={"environment": self.config.environment, "version": self.config.version,
                   "sampling_ratio": self.tracing.sampling_ratio, "metrics_backend": self.metrics.snapshot()["system_backend"]},
        )
        return self

    async def shutdown(self) -> None:
        """Drain traffic, stop tasks, flush spans and logs. Safe to call twice."""
        if not self._set_up:
            return
        self.health.begin_shutdown()
        _log.info("observability shutting down")
        await self.dashboard.stop()
        await self.health.stop()
        await self.performance.stop()
        await self.metrics.stop()
        self.errors.uninstall_global_handlers()
        self.tracing.shutdown()
        self.logging.shutdown()
        self._set_up = False
        self._running = False

    async def __aenter__(self) -> "Observability":
        return await self.start()

    async def __aexit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> None:
        await self.shutdown()

    # -- instrumentation -----------------------------------------------------
    def get_logger(self, name: Optional[str] = None, **bound: Any) -> StructuredLogger:
        return self.logging.get_logger(name, **bound)

    def observe(
        self,
        operation: str,
        *,
        attributes: Optional[Mapping[str, Any]] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        reraise: bool = True,
    ) -> _ObserveScope:
        """Span + latency + exception capture for a block (``with`` or ``async with``)."""
        return _ObserveScope(self, operation, attributes, kind, reraise)

    def instrument(self, operation: Optional[str] = None, *, reraise: bool = True) -> Callable[[F], F]:
        """Decorator form of :meth:`observe` for sync and async functions."""
        import functools
        import inspect

        def decorator(func: F) -> F:
            name = operation or f"{func.__module__}.{func.__qualname__}"
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    async with self.observe(name, reraise=reraise):
                        return await func(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.observe(name, reraise=reraise):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def request(self, method: str, route: str, headers: Optional[Mapping[str, str]] = None) -> _RequestScope:
        """Server request scope for non-ASGI frameworks; continues a remote ``traceparent``."""
        return _RequestScope(self, method, route, headers)

    def bind(self, **fields: Any):  # type: ignore[no-untyped-def]
        """Add fields to every log line in scope (``with obs.bind(request_id=...)``)."""
        return log_context(**fields)

    def asgi_middleware(self, app: Any, **kwargs: Any) -> "ObservabilityMiddleware":
        return ObservabilityMiddleware(app, self, **kwargs)

    # -- endpoints / snapshot ------------------------------------------------
    async def health_report(self, probe: ProbeType = ProbeType.READINESS) -> Dict[str, Any]:
        report = await self.health.check(probe)
        return {"http_status": report.http_status, **report.to_dict()}

    def metrics_text(self) -> str:
        return self.metrics.render_prometheus()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "service": self.config.service_name,
            "environment": self.config.environment,
            "version": self.config.version,
            "metrics": self.metrics.snapshot(),
            "performance": self.performance.snapshot(),
            "errors": self.errors.snapshot(),
            "recent_traces": self.span_store.recent_traces(10),
            "dropped": {"logs": self.logging.dropped_records, "spans": self.tracing.dropped_spans},
        }


# --------------------------------------------------------------------------
# ASGI middleware
# --------------------------------------------------------------------------
_ID_SEGMENT = re.compile(
    r"/(?:\d+|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-fA-F]{24,})(?=/|$)"
)


def _default_route(scope: Mapping[str, Any]) -> str:
    """Low-cardinality route label: framework route template if known, else ids collapsed."""
    route = scope.get("route")
    template = getattr(route, "path", None) or getattr(route, "path_format", None)
    if isinstance(template, str):
        return template
    return _ID_SEGMENT.sub("/{id}", scope.get("path", "/")) or "/"


class ObservabilityMiddleware:
    """Pure-ASGI middleware (works with FastAPI/Starlette/Quart/...).

    Per request it continues the incoming ``traceparent``, creates a SERVER span, records
    RED metrics, captures unhandled exceptions and returns ``traceparent`` to the caller.
    It can also serve ``/healthz`` (liveness), ``/readyz`` (readiness) and ``/metrics``.
    """

    def __init__(
        self,
        app: Any,
        obs: Observability,
        *,
        expose_endpoints: bool = True,
        health_path: str = "/healthz",
        ready_path: str = "/readyz",
        metrics_path: str = "/metrics",
        exclude_paths: tuple = (),
        route_resolver: Callable[[Mapping[str, Any]], str] = _default_route,
    ) -> None:
        self.app = app
        self.obs = obs
        self.expose = expose_endpoints
        self.health_path, self.ready_path, self.metrics_path = health_path, ready_path, metrics_path
        self.exclude = set(exclude_paths)
        self.route_resolver = route_resolver

    async def __call__(self, scope: MutableMapping[str, Any], receive: Callable[..., Awaitable[Any]],
                       send: Callable[..., Awaitable[None]]) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "/")
        if self.expose and path in (self.health_path, self.ready_path, self.metrics_path):
            await self._serve_endpoint(path, send)
            return
        if path in self.exclude:
            await self.app(scope, receive, send)
            return

        obs = self.obs
        method = scope.get("method", "GET")
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        parent = obs.tracing.extract(headers)
        status = 500
        start = time.perf_counter()
        obs.metrics.in_flight.inc()
        with obs.tracing.start_span(
            f"{method} {path}", kind=SpanKind.SERVER, parent=parent,
            attributes={"http.method": method, "http.target": path,
                        "http.user_agent": headers.get("user-agent", "")[:200]},
        ) as span:
            tp = {}
            obs.tracing.inject(tp, span)

            async def send_wrapper(message: MutableMapping[str, Any]) -> None:
                nonlocal status
                if message["type"] == "http.response.start":
                    status = int(message["status"])
                    out = list(message.get("headers", []))
                    out.append((b"traceparent", tp.get("traceparent", "").encode("latin-1")))
                    message = {**message, "headers": out}
                await send(message)

            try:
                with obs.bind(http_method=method, http_path=path):
                    await self.app(scope, receive, send_wrapper)
            except Exception as exc:
                status = 500
                obs.errors.capture_exception(
                    exc, handled=False, context={"http.method": method, "http.path": path}
                )
                raise
            finally:
                obs.metrics.in_flight.dec()
                route = self.route_resolver(scope)
                span.name = f"{method} {route}"
                span.set_attribute("http.route", route)
                span.set_attribute("http.status_code", status)
                if status >= 500:
                    span.set_status(StatusCode.ERROR, f"HTTP {status}")
                obs.metrics.record_request(method, route, status, time.perf_counter() - start)

    async def _serve_endpoint(self, path: str, send: Callable[..., Awaitable[None]]) -> None:
        import json

        if path == self.metrics_path:
            body = self.obs.metrics_text().encode("utf-8")
            status, ctype = 200, b"text/plain; version=0.0.4; charset=utf-8"
        else:
            probe = ProbeType.LIVENESS if path == self.health_path else ProbeType.READINESS
            report = await self.obs.health.check(probe)
            body = json.dumps(report.to_dict()).encode("utf-8")
            status, ctype = report.http_status, b"application/json"
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", ctype), (b"cache-control", b"no-store"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


# --------------------------------------------------------------------------
# Process-wide singleton helpers
# --------------------------------------------------------------------------
_default: Optional[Observability] = None


def init_observability(config: Optional[ObservabilityConfig] = None, **overrides: Any) -> Observability:
    """Create (once) and return the process-wide :class:`Observability` instance."""
    global _default
    if _default is None:
        _default = Observability(config, **overrides)
        _default.setup()
    return _default


def get_observability() -> Observability:
    if _default is None:
        raise RuntimeError("Observability is not initialised; call init_observability() first")
    return _default
