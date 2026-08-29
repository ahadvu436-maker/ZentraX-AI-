"""
ZentraX AI — AI Gateway
==========================
Central, provider-agnostic entry point for all outbound calls to AI model
providers (Anthropic, OpenAI, ...). Routers (`chat.py`, `toolkit.py`)
should depend on `get_ai_gateway()` and call `AIGateway.generate_completion()`
— they should never call a provider's HTTP API directly.

Responsibilities concentrated here, so they exist exactly once:
- A single shared `httpx.AsyncClient` (connection pooling) instead of a
  new client per request.
- A uniform request/response shape across providers.
- Per-call timeout enforcement independent of any single provider's own
  timeout behavior.
- Retries with exponential backoff for transient failures only.
- Ordered fallback across providers (primary -> configured fallbacks).
- A lightweight in-process circuit breaker so a provider that's clearly
  down stops eating request latency on every call.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel

from app.config.settings import settings

logger = logging.getLogger(__name__)


# ========================================================================
# Shared request/response types
# ========================================================================
class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class CompletionRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    model: str | None = None  # override the provider's configured default


class CompletionResponse(BaseModel):
    content: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float


# ========================================================================
# Exceptions
# ========================================================================
class AIGatewayError(Exception):
    """Base class for all AI Gateway errors."""


class ProviderAuthError(AIGatewayError):
    """Invalid/missing API key — never retried."""


class ProviderTimeoutError(AIGatewayError):
    """Provider did not respond within the configured timeout."""


class ProviderRateLimitedError(AIGatewayError):
    """Provider returned a 429 — retryable with backoff."""


class ProviderResponseError(AIGatewayError):
    """Provider returned an unexpected status/payload shape."""


class ProviderUnavailableError(AIGatewayError):
    """Circuit breaker is open for this provider; skipped without a call."""


class AllProvidersFailedError(AIGatewayError):
    """Every provider in the chain (primary + fallbacks) failed."""

    def __init__(self, attempts: dict[str, str]):
        self.attempts = attempts  # provider name -> error message
        summary = "; ".join(f"{name}: {msg}" for name, msg in attempts.items())
        super().__init__(f"All AI providers failed. {summary}")


# Errors worth retrying (same provider, backoff) vs. failing straight to
# the next provider in the chain.
_RETRYABLE_EXCEPTIONS = (ProviderTimeoutError, ProviderRateLimitedError, ProviderResponseError)


# ========================================================================
# Circuit breaker (per-provider, in-process)
# ========================================================================
@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: datetime | None = None

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = None

    def record_failure(self, threshold: int) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= threshold and self.opened_at is None:
            self.opened_at = datetime.now(timezone.utc)

    def is_open(self, cooldown_seconds: float) -> bool:
        if self.opened_at is None:
            return False
        if datetime.now(timezone.utc) - self.opened_at >= timedelta(seconds=cooldown_seconds):
            # Cooldown elapsed — allow a trial request through (half-open).
            self.opened_at = None
            self.consecutive_failures = 0
            return False
        return True


# NOTE: in-process only — in a multi-instance deployment each process has
# its own breaker state. Fine for reducing latency on a single instance;
# move to a Redis-backed counter if you need cluster-wide breaker state.
_circuit_states: dict[str, _CircuitState] = {}


def _get_circuit(provider_name: str) -> _CircuitState:
    return _circuit_states.setdefault(provider_name, _CircuitState())


# ========================================================================
# Provider abstraction
# ========================================================================
class BaseAIProvider(ABC):
    name: str

    def __init__(self, http_client: httpx.AsyncClient):
        self._http = http_client

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Perform the provider-specific API call and normalize the response."""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """False if required credentials are missing — used to skip silently."""


class AnthropicProvider(BaseAIProvider):
    name = "anthropic"
    _API_URL = "https://api.anthropic.com/v1/messages"
    _API_VERSION = "2023-06-01"

    @property
    def is_configured(self) -> bool:
        return settings.ANTHROPIC_API_KEY is not None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        if not self.is_configured:
            raise ProviderAuthError("ANTHROPIC_API_KEY is not configured.")

        system_prompt = "\n".join(
            m.content for m in request.messages if m.role == ChatRole.SYSTEM
        ) or None
        conversation = [
            {"role": m.role.value, "content": m.content}
            for m in request.messages
            if m.role != ChatRole.SYSTEM
        ]

        payload: dict[str, Any] = {
            "model": request.model or settings.ANTHROPIC_DEFAULT_MODEL,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": conversation,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": settings.ANTHROPIC_API_KEY.get_secret_value(),
            "anthropic-version": self._API_VERSION,
            "content-type": "application/json",
        }

        start = time.perf_counter()
        try:
            response = await self._http.post(self._API_URL, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Anthropic request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderResponseError(f"Anthropic request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 401:
            raise ProviderAuthError("Anthropic rejected the configured API key.")
        if response.status_code == 429:
            raise ProviderRateLimitedError("Anthropic rate limit exceeded.")
        if response.status_code >= 400:
            raise ProviderResponseError(
                f"Anthropic returned {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        try:
            content = "".join(
                block.get("text", "") for block in data["content"] if block.get("type") == "text"
            )
            usage = data.get("usage", {})
        except (KeyError, TypeError) as exc:
            raise ProviderResponseError(f"Unexpected Anthropic response shape: {exc}") from exc

        return CompletionResponse(
            content=content,
            provider=self.name,
            model=data.get("model", payload["model"]),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            latency_ms=latency_ms,
        )


class OpenAIProvider(BaseAIProvider):
    name = "openai"
    _API_URL = "https://api.openai.com/v1/chat/completions"

    @property
    def is_configured(self) -> bool:
        return settings.OPENAI_API_KEY is not None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        if not self.is_configured:
            raise ProviderAuthError("OPENAI_API_KEY is not configured.")

        payload: dict[str, Any] = {
            "model": request.model or settings.OPENAI_DEFAULT_MODEL,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
        }
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY.get_secret_value()}",
            "Content-Type": "application/json",
        }

        start = time.perf_counter()
        try:
            response = await self._http.post(self._API_URL, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"OpenAI request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderResponseError(f"OpenAI request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 401:
            raise ProviderAuthError("OpenAI rejected the configured API key.")
        if response.status_code == 429:
            raise ProviderRateLimitedError("OpenAI rate limit exceeded.")
        if response.status_code >= 400:
            raise ProviderResponseError(
                f"OpenAI returned {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError(f"Unexpected OpenAI response shape: {exc}") from exc

        return CompletionResponse(
            content=content,
            provider=self.name,
            model=data.get("model", payload["model"]),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
        )


# ========================================================================
# Gateway — orchestrates retries, fallback, and the circuit breaker
# ========================================================================
class AIGateway:
    """
    Provider-agnostic facade used by routers/services.

    Usage:
        gateway = get_ai_gateway()
        response = await gateway.generate_completion(
            CompletionRequest(messages=[ChatMessage(role=ChatRole.USER, content="Hi")])
        )
    """

    def __init__(self, http_client: httpx.AsyncClient):
        self._providers: dict[str, BaseAIProvider] = {
            "anthropic": AnthropicProvider(http_client),
            "openai": OpenAIProvider(http_client),
        }

    def _provider_chain(self) -> list[BaseAIProvider]:
        ordered_names = [settings.AI_PRIMARY_PROVIDER, *settings.AI_FALLBACK_PROVIDERS]
        chain = []
        for name in ordered_names:
            provider = self._providers.get(name)
            if provider is None:
                logger.warning("Configured provider '%s' is not implemented; skipping.", name)
                continue
            chain.append(provider)
        return chain

    async def _call_with_retries(self, provider: BaseAIProvider, request: CompletionRequest) -> CompletionResponse:
        circuit = _get_circuit(provider.name)

        if circuit.is_open(settings.AI_CIRCUIT_BREAKER_COOLDOWN_SECONDS):
            raise ProviderUnavailableError(
                f"Circuit breaker open for '{provider.name}'; skipping without a call."
            )

        last_error: Exception | None = None
        for attempt in range(settings.AI_MAX_RETRIES_PER_PROVIDER + 1):
            try:
                result = await asyncio.wait_for(
                    provider.complete(request),
                    timeout=settings.AI_REQUEST_TIMEOUT_SECONDS,
                )
                circuit.record_success()
                return result
            except asyncio.TimeoutError as exc:
                last_error = ProviderTimeoutError(
                    f"'{provider.name}' exceeded the {settings.AI_REQUEST_TIMEOUT_SECONDS}s gateway timeout."
                )
            except ProviderAuthError:
                # Never retry bad credentials — retrying won't help and just
                # burns time before falling through to the next provider.
                circuit.record_failure(settings.AI_CIRCUIT_BREAKER_THRESHOLD)
                raise
            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = exc

            circuit.record_failure(settings.AI_CIRCUIT_BREAKER_THRESHOLD)

            if attempt < settings.AI_MAX_RETRIES_PER_PROVIDER:
                backoff = (2 ** attempt) + random.uniform(0, 0.5)
                logger.info(
                    "Retrying provider='%s' attempt=%d/%d after %.2fs (%s)",
                    provider.name,
                    attempt + 1,
                    settings.AI_MAX_RETRIES_PER_PROVIDER,
                    backoff,
                    last_error,
                )
                await asyncio.sleep(backoff)

        assert last_error is not None
        raise last_error

    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Try the primary provider, then each configured fallback in order.

        Raises `AllProvidersFailedError` only if every provider in the
        chain fails (or is unconfigured/circuit-open) — callers get one
        exception type to handle regardless of how many providers exist.
        """
        chain = self._provider_chain()
        if not chain:
            raise AIGatewayError(
                "No AI providers configured. Set AI_PRIMARY_PROVIDER and ensure "
                "its corresponding API key is set."
            )

        attempts: dict[str, str] = {}
        for provider in chain:
            if not provider.is_configured:
                attempts[provider.name] = "not configured (missing API key)"
                continue
            try:
                return await self._call_with_retries(provider, request)
            except AIGatewayError as exc:
                logger.warning("Provider '%s' failed, trying next: %s", provider.name, exc)
                attempts[provider.name] = str(exc)
                continue

        raise AllProvidersFailedError(attempts)


# ========================================================================
# Lifecycle — shared HTTP client + FastAPI dependency
# ========================================================================
_http_client: httpx.AsyncClient | None = None
_gateway: AIGateway | None = None


async def init_ai_gateway() -> None:
    """Call once at application startup (see lifespan handler in main.py)."""
    global _http_client, _gateway
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.AI_REQUEST_TIMEOUT_SECONDS),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    _gateway = AIGateway(_http_client)
    logger.info(
        "AI Gateway initialized. primary=%s fallbacks=%s",
        settings.AI_PRIMARY_PROVIDER,
        settings.AI_FALLBACK_PROVIDERS,
    )


async def close_ai_gateway() -> None:
    """Call once at application shutdown."""
    global _http_client, _gateway
    if _http_client is not None:
        await _http_client.aclose()
    _http_client = None
    _gateway = None


def get_ai_gateway() -> AIGateway:
    """
    FastAPI dependency: `gateway: AIGateway = Depends(get_ai_gateway)`.

    Raises RuntimeError if called before `init_ai_gateway()` has run —
    a signal that the app's lifespan handler isn't wired up correctly,
    not something a route should try to recover from.
    """
    if _gateway is None:
        raise RuntimeError(
            "AI Gateway is not initialized. Ensure init_ai_gateway() runs in "
            "the application's lifespan startup handler."
        )
    return _gateway