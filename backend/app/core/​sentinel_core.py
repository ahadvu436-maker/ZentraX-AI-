"""
sentinel_core.py — Request-level security sentinel.

Responsibilities:
    1. Track per-client request activity in sliding windows (in-memory).
    2. Score incoming requests for anomalies: burst rate, error-response
       bursts, repeated auth failures, and suspicious path/query patterns.
    3. "Self-heal": clients that cross a threshold are quarantined
       (blocked) for a cooldown period, then automatically un-quarantined
       and their anomaly score decays back to normal — no manual reset
       required.
    4. Emit structured, privacy-safe firewall log events (client identity
       is a salted hash, never a raw IP, consistent with this project's
       no-PII-in-logs stance).

This is detection/mitigation logic only — sliding-window counters and
threshold-based blocking. It does not fingerprint devices, inspect
request bodies, or make outbound calls of any kind.

Notes on scope / what this is NOT:
    - Not a replacement for a real WAF or DDoS mitigation at the network
      edge — this operates at the ASGI layer and only sees what reaches
      the app process.
    - The "suspicious pattern" checks are coarse heuristics (e.g. path
      traversal sequences, null bytes) meant to flag-and-log, not a
      signature database. Tune `SUSPICIOUS_PATTERNS` for your app.
    - State is in-process and non-persistent by design (privacy-first —
      nothing about client behavior is written to disk). For multi-worker
      deployments, back this with a shared store (e.g. Redis) behind the
      same `SentinelStore` interface if you need cross-process state.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("app.sentinel")

RequestResponseEndpoint = Callable[[Request], Awaitable[Response]]


# --------------------------------------------------------------------------
# Config & enums
# --------------------------------------------------------------------------

class ThreatLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    RATE_BURST = "rate_burst"
    ERROR_BURST = "error_burst"
    AUTH_FAILURE_BURST = "auth_failure_burst"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


class SentinelConfig(BaseModel):
    """Tunable thresholds. Defaults are conservative starting points —
    tune against your own traffic before relying on them in production."""

    # Sliding window for rate tracking.
    window_seconds: float = Field(default=60.0, gt=0)
    max_requests_per_window: int = Field(default=120, gt=0)

    # Error-response tracking (5xx/4xx from this app, within the window).
    max_errors_per_window: int = Field(default=20, gt=0)

    # Auth-failure tracking (401/403 responses specifically).
    max_auth_failures_per_window: int = Field(default=8, gt=0)

    # Quarantine ("self-healing") behavior.
    quarantine_seconds: float = Field(default=300.0, gt=0)
    max_quarantine_seconds: float = Field(default=3600.0, gt=0)
    quarantine_backoff_multiplier: float = Field(default=2.0, ge=1.0)

    # Suspicious-pattern detection on path + query string.
    suspicious_patterns: tuple[str, ...] = (
        r"\.\./",           # path traversal
        r"%2e%2e%2f",        # encoded path traversal
        r"\x00",             # null byte injection
        r"<script[\s>]",     # naive XSS probe
        r"union\s+select",   # naive SQLi probe
    )

    # Privacy: identify clients by salted hash, never raw IP, in logs.
    client_hash_salt: str = Field(default="change-me-in-deployment")


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------

class AnomalyEvent(BaseModel):
    """A single logged security-relevant event. Safe to serialize to
    structured logs — contains no raw client identifiers or request
    bodies."""

    client_ref: str
    anomaly_type: AnomalyType
    threat_level: ThreatLevel
    path: str
    method: str
    detail: str
    timestamp: float


class ThreatAssessment(BaseModel):
    """Result of evaluating a single incoming request, before it's
    allowed to proceed to the route handler."""

    allowed: bool
    threat_level: ThreatLevel
    anomaly: AnomalyType | None = None
    reason: str | None = None
    retry_after_seconds: float | None = None


@dataclass
class ClientActivity:
    """Mutable, in-memory sliding-window state for one client."""

    request_times: deque[float] = field(default_factory=deque)
    error_times: deque[float] = field(default_factory=deque)
    auth_failure_times: deque[float] = field(default_factory=deque)
    quarantined_until: float = 0.0
    quarantine_strikes: int = 0


# --------------------------------------------------------------------------
# Store (swap-in point for a shared backend, e.g. Redis, if needed)
# --------------------------------------------------------------------------

class SentinelStore:
    """In-memory client-state store guarded by an asyncio.Lock.

    Kept as its own class so a Redis-backed (or similar) implementation
    can be substituted behind the same interface for multi-worker
    deployments, without touching SentinelCore's logic.
    """

    def __init__(self) -> None:
        self._clients: dict[str, ClientActivity] = {}
        self._lock = asyncio.Lock()

    async def get(self, client_ref: str) -> ClientActivity:
        async with self._lock:
            activity = self._clients.get(client_ref)
            if activity is None:
                activity = ClientActivity()
                self._clients[client_ref] = activity
            return activity

    async def prune_idle(self, idle_seconds: float = 3600.0) -> int:
        """Drop clients with no recent activity and no active quarantine,
        so memory doesn't grow unbounded over a long-running process.
        Call this periodically from a background task."""
        now = time.monotonic()
        removed = 0
        async with self._lock:
            stale_refs = [
                ref
                for ref, activity in self._clients.items()
                if activity.quarantined_until < now
                and (not activity.request_times or activity.request_times[-1] < now - idle_seconds)
            ]
            for ref in stale_refs:
                del self._clients[ref]
                removed += 1
        return removed


# --------------------------------------------------------------------------
# Core sentinel logic
# --------------------------------------------------------------------------

class SentinelCore:
    """Stateful anomaly detector + adaptive quarantine manager.

    One instance should be shared across the app's lifetime (e.g. created
    once at startup and reused by the middleware), so its sliding-window
    state persists across requests.
    """

    def __init__(self, config: SentinelConfig | None = None, store: SentinelStore | None = None):
        self.config = config or SentinelConfig()
        self.store = store or SentinelStore()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.suspicious_patterns
        ]

    # -- client identity -------------------------------------------------

    def _client_ref(self, request: Request) -> str:
        """Salted hash of the client IP. Never stores or logs the raw IP —
        the hash is stable enough to correlate repeat behavior without
        being reversible to an identity."""
        raw_ip = request.client.host if request.client else "unknown"
        digest = hashlib.sha256(f"{self.config.client_hash_salt}:{raw_ip}".encode()).hexdigest()
        return digest[:16]

    # -- window maintenance ------------------------------------------------

    @staticmethod
    def _trim_window(times: deque[float], now: float, window_seconds: float) -> None:
        while times and now - times[0] > window_seconds:
            times.popleft()

    # -- pattern detection -------------------------------------------------

    def _matches_suspicious_pattern(self, request: Request) -> str | None:
        target = f"{request.url.path}?{request.url.query}"
        for pattern in self._compiled_patterns:
            if pattern.search(target):
                return pattern.pattern
        return None

    # -- main entry point: called before the route handler runs -----------

    async def evaluate_request(self, request: Request) -> ThreatAssessment:
        now = time.monotonic()
        client_ref = self._client_ref(request)
        activity = await self.store.get(client_ref)

        # 1. Already quarantined?
        if activity.quarantined_until > now:
            retry_after = activity.quarantined_until - now
            return ThreatAssessment(
                allowed=False,
                threat_level=ThreatLevel.HIGH,
                anomaly=AnomalyType.RATE_BURST,
                reason="Client is temporarily quarantined",
                retry_after_seconds=round(retry_after, 1),
            )

        # 2. Suspicious pattern in path/query — log and quarantine immediately.
        matched = self._matches_suspicious_pattern(request)
        if matched:
            await self._quarantine(client_ref, activity, now)
            await self._log_event(
                client_ref, AnomalyType.SUSPICIOUS_PATTERN, ThreatLevel.CRITICAL,
                request, detail=f"matched pattern: {matched}",
            )
            return ThreatAssessment(
                allowed=False,
                threat_level=ThreatLevel.CRITICAL,
                anomaly=AnomalyType.SUSPICIOUS_PATTERN,
                reason="Request matched a known-suspicious pattern",
            )

        # 3. Rate burst check.
        activity.request_times.append(now)
        self._trim_window(activity.request_times, now, self.config.window_seconds)
        if len(activity.request_times) > self.config.max_requests_per_window:
            await self._quarantine(client_ref, activity, now)
            await self._log_event(
                client_ref, AnomalyType.RATE_BURST, ThreatLevel.HIGH, request,
                detail=f"{len(activity.request_times)} requests in "
                       f"{self.config.window_seconds:.0f}s window",
            )
            return ThreatAssessment(
                allowed=False,
                threat_level=ThreatLevel.HIGH,
                anomaly=AnomalyType.RATE_BURST,
                reason="Request rate limit exceeded",
                retry_after_seconds=self.config.quarantine_seconds,
            )

        return ThreatAssessment(allowed=True, threat_level=ThreatLevel.NONE)

    # -- called after the route handler runs, with the response ----------

    async def record_response(self, request: Request, status_code: int) -> None:
        """Feed the response status back into the client's window so
        error-burst and auth-failure-burst detection can act on the
        *next* request. This is what makes quarantine adaptive rather
        than a one-shot check."""
        now = time.monotonic()
        client_ref = self._client_ref(request)
        activity = await self.store.get(client_ref)

        if status_code in (401, 403):
            activity.auth_failure_times.append(now)
            self._trim_window(activity.auth_failure_times, now, self.config.window_seconds)
            if len(activity.auth_failure_times) > self.config.max_auth_failures_per_window:
                await self._quarantine(client_ref, activity, now)
                await self._log_event(
                    client_ref, AnomalyType.AUTH_FAILURE_BURST, ThreatLevel.HIGH, request,
                    detail=f"{len(activity.auth_failure_times)} auth failures in window",
                )
        elif status_code >= 400:
            activity.error_times.append(now)
            self._trim_window(activity.error_times, now, self.config.window_seconds)
            if len(activity.error_times) > self.config.max_errors_per_window:
                await self._quarantine(client_ref, activity, now)
                await self._log_event(
                    client_ref, AnomalyType.ERROR_BURST, ThreatLevel.MEDIUM, request,
                    detail=f"{len(activity.error_times)} error responses in window",
                )

    # -- self-healing: quarantine with escalating-then-decaying backoff --

    async def _quarantine(self, client_ref: str, activity: ClientActivity, now: float) -> None:
        activity.quarantine_strikes += 1
        # Escalating backoff for repeat offenders, capped at max_quarantine_seconds —
        # this is the "self-healing" behavior: no manual unblock is needed,
        # good behavior during a full cooldown lets strikes decay back down.
        duration = min(
            self.config.quarantine_seconds
            * (self.config.quarantine_backoff_multiplier ** (activity.quarantine_strikes - 1)),
            self.config.max_quarantine_seconds,
        )
        activity.quarantined_until = now + duration

    async def decay_strikes(self, idle_seconds: float = 1800.0) -> None:
        """Background maintenance: clients who've been quiet for a while
        have their strike count reduced, so a single old incident doesn't
        permanently escalate future quarantine durations. Call this
        periodically (e.g. every few minutes) from an app lifespan task."""
        now = time.monotonic()
        async with self.store._lock:  # noqa: SLF001 - maintenance-only internal access
            for activity in self.store._clients.values():  # noqa: SLF001
                last_seen = activity.request_times[-1] if activity.request_times else 0.0
                if activity.quarantine_strikes > 0 and now - last_seen > idle_seconds:
                    activity.quarantine_strikes = max(0, activity.quarantine_strikes - 1)

    # -- logging -----------------------------------------------------------

    async def _log_event(
        self,
        client_ref: str,
        anomaly_type: AnomalyType,
        threat_level: ThreatLevel,
        request: Request,
        detail: str,
    ) -> None:
        event = AnomalyEvent(
            client_ref=client_ref,
            anomaly_type=anomaly_type,
            threat_level=threat_level,
            path=request.url.path,
            method=request.method,
            detail=detail,
            timestamp=time.time(),
        )
        logger.warning("sentinel_event", extra=event.model_dump(mode="json"))


# --------------------------------------------------------------------------
# ASGI middleware
# --------------------------------------------------------------------------

class SentinelMiddleware(BaseHTTPMiddleware):
    """Wire this into your FastAPI app:

        sentinel = SentinelCore(SentinelConfig(client_hash_salt=settings.jwt_secret_key.get_secret_value()))
        app.add_middleware(SentinelMiddleware, sentinel=sentinel)

    Reuse the same `SentinelCore` instance for a background task that
    periodically calls `sentinel.decay_strikes()` and
    `sentinel.store.prune_idle()`.
    """

    def __init__(self, app, sentinel: SentinelCore):
        super().__init__(app)
        self.sentinel = sentinel

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        assessment = await self.sentinel.evaluate_request(request)

        if not assessment.allowed:
            headers = {}
            if assessment.retry_after_seconds is not None:
                headers["Retry-After"] = str(int(assessment.retry_after_seconds))
            return JSONResponse(
                status_code=429 if assessment.anomaly == AnomalyType.RATE_BURST else 403,
                content={"detail": assessment.reason or "Request blocked"},
                headers=headers,
            )

        response = await call_next(request)
        await self.sentinel.record_response(request, response.status_code)
        return response
