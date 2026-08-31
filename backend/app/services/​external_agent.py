"""
external_agent.py — Outbound async client for third-party / cloud-provider APIs.

Responsibilities:
    1. A single, reusable async HTTP client (httpx) with sane timeouts,
       connection pooling, and lifecycle management via an async context
       manager — one `ExternalAgent` per target service, shared across
       requests rather than opening a client per call.
    2. Retry with exponential backoff + jitter on transient failures
       (timeouts, connection errors, 429, 5xx), bounded by a circuit
       breaker so a persistently-down dependency stops getting hammered.
    3. Pluggable auth strategies — static API key or OAuth2
       client-credentials with cached/auto-refreshed bearer tokens —
       so cloud-provider integrations (AWS-style, GCP-style, generic
       OAuth2 SaaS APIs) share the same request path.
    4. "Auto-negotiation": when a service exposes multiple API versions,
       the agent tries the preferred version first and transparently
       falls back to the next one on 404/406, caching whichever version
       worked so subsequent calls skip straight to it.
    5. Privacy-consistent logging — request/response bodies and auth
       headers are never logged; only method, URL path, status, attempt
       count, and latency.

This module intentionally does not know about any specific cloud
provider's API shape — it's a resilient transport layer. Build a
thin provider-specific client on top of `ExternalAgent.request()`.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("app.external_agent")

_REDACTED_HEADERS = {"authorization", "x-api-key", "cookie"}


# --------------------------------------------------------------------------
# Config & models
# --------------------------------------------------------------------------

class RetryConfig(BaseModel):
    max_attempts: int = Field(default=4, ge=1)
    base_delay_seconds: float = Field(default=0.5, gt=0)
    max_delay_seconds: float = Field(default=20.0, gt=0)
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    recovery_seconds: float = Field(default=30.0, gt=0)
    half_open_max_calls: int = Field(default=1, ge=1)


class ExternalAgentConfig(BaseModel):
    base_url: str
    request_timeout_seconds: float = Field(default=10.0, gt=0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    # Ordered most-preferred first. The agent tries [0], then falls back
    # down the list on 404/406, and remembers what worked.
    api_versions: tuple[str, ...] = ("v2", "v1")


class RequestOutcome(BaseModel):
    status_code: int
    attempts: int
    api_version_used: str | None = None
    duration_ms: float = 0.0


# --------------------------------------------------------------------------
# Auth strategies
# --------------------------------------------------------------------------

class AuthStrategy(Protocol):
    async def apply(self, headers: dict[str, str]) -> None:
        """Mutate `headers` in place to add whatever auth is needed."""
        ...


class StaticApiKeyAuth:
    """Simple bearer/API-key auth with a fixed, pre-provisioned secret."""

    def __init__(self, api_key: str, header_name: str = "Authorization", prefix: str = "Bearer "):
        self._api_key = api_key
        self._header_name = header_name
        self._prefix = prefix

    async def apply(self, headers: dict[str, str]) -> None:
        headers[self._header_name] = f"{self._prefix}{self._api_key}"


@dataclass
class _CachedToken:
    value: str
    expires_at: float


class OAuth2ClientCredentialsAuth:
    """OAuth2 client-credentials flow with a cached, auto-refreshed
    bearer token. Suited to most cloud-provider and SaaS OAuth2 APIs.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str | None = None,
        refresh_leeway_seconds: float = 30.0,
    ):
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._leeway = refresh_leeway_seconds
        self._cached: _CachedToken | None = None
        self._lock = asyncio.Lock()

    async def apply(self, headers: dict[str, str]) -> None:
        token = await self._get_token()
        headers["Authorization"] = f"Bearer {token}"

    async def _get_token(self) -> str:
        async with self._lock:
            now = time.monotonic()
            if self._cached and self._cached.expires_at - self._leeway > now:
                return self._cached.value

            data = {
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            }
            if self._scope:
                data["scope"] = self._scope

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self._token_url, data=data)
                response.raise_for_status()
                payload = response.json()

            token = payload["access_token"]
            expires_in = float(payload.get("expires_in", 300))
            self._cached = _CachedToken(value=token, expires_at=now + expires_in)
            return token


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    async def before_call(self) -> None:
        async with self._lock:
            if self._state is CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self._config.recovery_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                else:
                    raise CircuitOpenError("Circuit breaker is open for this service")

            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._config.half_open_max_calls:
                    raise CircuitOpenError("Circuit breaker is half-open and at trial capacity")
                self._half_open_calls += 1

    async def on_success(self) -> None:
        async with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    async def on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            if self._state is CircuitState.HALF_OPEN or self._failure_count >= self._config.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    @property
    def state(self) -> CircuitState:
        return self._state


# --------------------------------------------------------------------------
# External agent
# --------------------------------------------------------------------------

class ExternalAgentError(Exception):
    """Raised when a request ultimately fails after retries/circuit checks."""


class ExternalAgent:
    """Use as an async context manager, one instance per target service:

        agent = ExternalAgent(
            ExternalAgentConfig(base_url="https://api.example-cloud.com"),
            auth=OAuth2ClientCredentialsAuth(...),
        )
        async with agent:
            data = await agent.request("GET", "/widgets/123")
    """

    def __init__(self, config: ExternalAgentConfig, auth: AuthStrategy | None = None):
        self.config = config
        self._auth = auth
        self._client: httpx.AsyncClient | None = None
        self._breaker = CircuitBreaker(config.circuit_breaker)
        self._negotiated_version: str | None = None
        self._version_lock = asyncio.Lock()

    async def __aenter__(self) -> ExternalAgent:
        timeout = httpx.Timeout(
            timeout=self.config.request_timeout_seconds,
            connect=self.config.connect_timeout_seconds,
        )
        self._client = httpx.AsyncClient(base_url=self.config.base_url, timeout=timeout)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ExternalAgent must be used as an async context manager")
        return self._client

    # -- public API ----------------------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        versioned: bool = True,
    ) -> tuple[httpx.Response, RequestOutcome]:
        """Issue a request with retry, circuit breaking, and (if
        `versioned`) API-version negotiation. Returns the final response
        plus metadata about how it got there.
        """
        if versioned and self.config.api_versions:
            return await self._request_with_version_negotiation(
                method, path, json_body=json_body, params=params, headers=headers
            )
        response = await self._request_with_resilience(
            method, path, json_body=json_body, params=params, headers=headers
        )
        return response

    # -- version negotiation --------------------------------------------------

    async def _request_with_version_negotiation(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> tuple[httpx.Response, RequestOutcome]:
        async with self._version_lock:
            known_good = self._negotiated_version
        versions_to_try = [known_good] if known_good else list(self.config.api_versions)
        if known_good:
            versions_to_try += [v for v in self.config.api_versions if v != known_good]

        last_response: httpx.Response | None = None
        last_outcome: RequestOutcome | None = None

        for version in versions_to_try:
            versioned_path = f"/{version}{path if path.startswith('/') else f'/{path}'}"
            response, outcome = await self._request_with_resilience(
                method, versioned_path, json_body=json_body, params=params, headers=headers
            )
            outcome.api_version_used = version
            last_response, last_outcome = response, outcome

            if response.status_code not in (404, 406):
                async with self._version_lock:
                    self._negotiated_version = version
                return response, outcome

            logger.info(
                "external_agent_version_fallback",
                extra={"tried_version": version, "path": path, "status_code": response.status_code},
            )

        assert last_response is not None and last_outcome is not None
        return last_response, last_outcome

    # -- retry + circuit breaker ------------------------------------------------

    async def _request_with_resilience(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> tuple[httpx.Response, RequestOutcome]:
        request_headers = dict(headers or {})
        if self._auth is not None:
            await self._auth.apply(request_headers)

        start = time.perf_counter()
        attempt = 0
        last_exc: Exception | None = None

        while attempt < self.config.retry.max_attempts:
            attempt += 1
            try:
                await self._breaker.before_call()
            except CircuitOpenError:
                raise ExternalAgentError(f"Circuit open for {self.config.base_url}") from None

            try:
                response = await self.client.request(
                    method, path, json=json_body, params=params, headers=request_headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                await self._breaker.on_failure()
                await self._log_attempt(method, path, attempt, status_code=None, error=str(exc))
                await self._sleep_backoff(attempt)
                continue

            if response.status_code in self.config.retry.retryable_status_codes:
                await self._breaker.on_failure()
                await self._log_attempt(method, path, attempt, status_code=response.status_code, error=None)
                if attempt < self.config.retry.max_attempts:
                    await self._sleep_backoff(attempt, retry_after=response.headers.get("Retry-After"))
                    continue
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                return response, RequestOutcome(
                    status_code=response.status_code, attempts=attempt, duration_ms=duration_ms
                )

            await self._breaker.on_success()
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            await self._log_attempt(method, path, attempt, status_code=response.status_code, error=None)
            return response, RequestOutcome(
                status_code=response.status_code, attempts=attempt, duration_ms=duration_ms
            )

        raise ExternalAgentError(
            f"Request to {path} failed after {attempt} attempts"
        ) from last_exc

    async def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after is not None:
            try:
                delay = float(retry_after)
                await asyncio.sleep(min(delay, self.config.retry.max_delay_seconds))
                return
            except ValueError:
                pass  # fall through to computed backoff

        base = self.config.retry.base_delay_seconds * (2 ** (attempt - 1))
        capped = min(base, self.config.retry.max_delay_seconds)
        jitter = random.uniform(0, capped * 0.25)
        await asyncio.sleep(capped + jitter)

    async def _log_attempt(
        self,
        method: str,
        path: str,
        attempt: int,
        status_code: int | None,
        error: str | None,
    ) -> None:
        fields = {
            "method": method,
            "path": path,
            "attempt": attempt,
            "status_code": status_code,
            "circuit_state": self._breaker.state.value,
        }
        if error:
            fields["error"] = error
            logger.warning("external_agent_attempt", extra=fields)
        else:
            logger.info("external_agent_attempt", extra=fields)


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Shared helper for any code that needs to log outbound headers for
    debugging — strips auth/cookie/api-key values first."""
    return {
        k: ("<redacted>" if k.lower() in _REDACTED_HEADERS else v)
        for k, v in headers.items()
    }
