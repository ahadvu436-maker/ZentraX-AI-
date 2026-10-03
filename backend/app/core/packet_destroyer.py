"""
packet_destroyer.py

Asynchronous rate-limiting / circuit-breaking middleware for FastAPI.

This module detects abusive traffic patterns (brute-force login
attempts, request floods, DDoS-style bursts) using a sliding-window
counter per client key (IP, API key, etc.) and responds by:

  * Throttling (HTTP 429) once a soft threshold is exceeded.
  * Temporarily banning (HTTP 403) a key once a hard threshold is
    exceeded, for a configurable cooldown period.
  * Tripping a global circuit breaker if overall request volume
    spikes far beyond baseline, shedding load to protect the service
    (fail-fast instead of falling over).

It does NOT send any payload, packet, or response back to the
originating connection beyond a standard HTTP status code -- there is
no "reflection," spoofing, or counter-attack traffic generated
against the client's source address. That would itself be an
offensive action against a network address that cannot be reliably
attributed, and is out of scope for a defensive module.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response


@dataclass
class RateLimiterConfig:
    window_seconds: float = 10.0
    soft_limit: int = 50          # requests per window -> 429
    hard_limit: int = 150         # requests per window -> temp ban
    ban_seconds: float = 300.0
    global_window_seconds: float = 5.0
    global_soft_limit: int = 5000     # total RPS-ish across all clients
    global_breaker_cooldown: float = 15.0
    max_tracked_keys: int = 200_000   # memory bound


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


@dataclass(slots=True)
class _ClientWindow:
    hits: deque[float] = field(default_factory=deque)
    banned_until: float = 0.0


class GlobalCircuitBreaker:
    """Trips when aggregate traffic across all clients spikes."""

    def __init__(self, config: RateLimiterConfig) -> None:
        self._config = config
        self._hits: deque[float] = deque()
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._lock = asyncio.Lock()

    async def record_and_check(self) -> CircuitState:
        now = time.monotonic()
        async with self._lock:
            if self._state is CircuitState.OPEN:
                if now - self._opened_at >= self._config.global_breaker_cooldown:
                    self._state = CircuitState.CLOSED
                    self._hits.clear()
                else:
                    return CircuitState.OPEN

            self._hits.append(now)
            cutoff = now - self._config.global_window_seconds
            while self._hits and self._hits[0] < cutoff:
                self._hits.popleft()

            if len(self._hits) > self._config.global_soft_limit:
                self._state = CircuitState.OPEN
                self._opened_at = now
                return CircuitState.OPEN

            return CircuitState.CLOSED


class AdaptiveRateLimiter:
    """
    Sliding-window, per-key rate limiter with temporary bans.
    Pure accounting/decision logic, no transport side effects.
    """

    def __init__(self, config: RateLimiterConfig) -> None:
        self._config = config
        self._clients: dict[str, _ClientWindow] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int]:
        """
        Returns (allowed, status_code_if_blocked).
        status_code_if_blocked is 0 when allowed.
        """
        now = time.monotonic()
        async with self._lock:
            self._evict_if_full()
            window = self._clients.setdefault(key, _ClientWindow())

            if window.banned_until > now:
                return False, 403

            cutoff = now - self._config.window_seconds
            while window.hits and window.hits[0] < cutoff:
                window.hits.popleft()

            window.hits.append(now)
            count = len(window.hits)

            if count > self._config.hard_limit:
                window.banned_until = now + self._config.ban_seconds
                window.hits.clear()
                return False, 403

            if count > self._config.soft_limit:
                return False, 429

            return True, 0

    def _evict_if_full(self) -> None:
        if len(self._clients) <= self._config.max_tracked_keys:
            return
        # Drop the oldest-looking 10% by last-activity heuristic.
        drop_count = max(1, len(self._clients) // 10)
        for key in list(self._clients.keys())[:drop_count]:
            self._clients.pop(key, None)


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key}"
    if request.client:
        return request.client.host
    return "unknown"


class PacketDestroyerMiddleware(BaseHTTPMiddleware):
    """
    Drop-in ASGI middleware: rejects abusive requests early, before
    they reach application route handlers, minimizing wasted work
    under load.
    """

    def __init__(
        self,
        app: FastAPI,
        config: RateLimiterConfig | None = None,
        exempt_paths: frozenset[str] = frozenset({"/health", "/metrics"}),
    ) -> None:
        super().__init__(app)
        self._config = config or RateLimiterConfig()
        self._limiter = AdaptiveRateLimiter(self._config)
        self._breaker = GlobalCircuitBreaker(self._config)
        self._exempt_paths = exempt_paths

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in self._exempt_paths:
            return await call_next(request)

        breaker_state = await self._breaker.record_and_check()
        if breaker_state is CircuitState.OPEN:
            return JSONResponse(
                status_code=503,
                content={"detail": "Service under load shedding, retry later."},
                headers={"Retry-After": str(int(self._config.global_breaker_cooldown))},
            )

        key = _client_key(request)
        allowed, status_code = await self._limiter.check(key)
        if not allowed:
            detail = "Too many requests." if status_code == 429 else "Temporarily blocked."
            headers = {"Retry-After": str(int(self._config.window_seconds))}
            return JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)

        return await call_next(request)


def install_rate_limiter(
    app: FastAPI, config: RateLimiterConfig | None = None
) -> PacketDestroyerMiddleware:
    middleware = PacketDestroyerMiddleware(app, config=config)
    app.add_middleware(PacketDestroyerMiddleware, config=config)
    return middleware
