"""Retry policies and the :class:`RetryManager` (exponential back-off with jitter)."""
from __future__ import annotations

import asyncio
import logging
import random
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Tuple, Type

from .execution_core import (
    ExecutionError, ResourceLimitError, RetryExhaustedError, TaskCancelledError,
)

logger = logging.getLogger(__name__)

__all__ = ["RetryPolicy", "RetryManager"]

OnRetry = Callable[[int, BaseException, float], None]


@dataclass(frozen=True)
class RetryPolicy:
    """How and when a failing task is retried.

    Attributes:
        max_attempts: Total attempts including the first (1 = no retries).
        base_delay: Delay before the first retry, in seconds.
        backoff_factor: Multiplier applied per further retry.
        max_delay: Upper bound for any delay.
        jitter: Fractional random spread (0.1 = +/-10%) to avoid thundering herds.
        retry_on: Exception types that may be retried.
        no_retry_on: Exception types that are never retried (wins over ``retry_on``).
        retry_if: Optional extra predicate evaluated on the exception.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    backoff_factor: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.1
    retry_on: Tuple[Type[BaseException], ...] = (Exception,)
    no_retry_on: Tuple[Type[BaseException], ...] = (TaskCancelledError, ResourceLimitError)
    retry_if: Optional[Callable[[BaseException], bool]] = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be >= 0")
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be >= 1")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be within [0, 1]")

    @classmethod
    def none(cls) -> "RetryPolicy":
        """A policy that never retries."""
        return cls(max_attempts=1)

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        """Whether another attempt should follow failed attempt number ``attempt`` (1-based)."""
        if attempt >= self.max_attempts:
            return False
        if isinstance(exc, self.no_retry_on) or not isinstance(exc, self.retry_on):
            return False
        if self.retry_if is not None:
            try:
                return bool(self.retry_if(exc))
            except Exception:  # noqa: BLE001
                logger.exception("retry_if predicate raised; not retrying")
                return False
        return True

    def delay_for(self, attempt: int) -> float:
        """Back-off delay after failed attempt number ``attempt`` (1-based)."""
        delay = min(self.max_delay, self.base_delay * (self.backoff_factor ** (attempt - 1)))
        if self.jitter:
            delay *= 1 + random.uniform(-self.jitter, self.jitter)
        return max(0.0, min(delay, self.max_delay * (1 + self.jitter)))


class RetryManager:
    """Runs callables under a :class:`RetryPolicy`.

    Both entry points return ``(value, attempts)`` on success and raise
    :class:`RetryExhaustedError` (with ``last_exception``) otherwise, including for
    non-retryable errors and cancellation, so callers need one ``except`` clause.
    ``shutdown()`` interrupts back-off sleeps and cancels pending retries.
    """

    def __init__(self, default_policy: Optional[RetryPolicy] = None) -> None:
        self.default_policy = default_policy or RetryPolicy.none()
        self._stop = threading.Event()

    def shutdown(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _cancelled(self, attempt: int, label: str) -> RetryExhaustedError:
        return RetryExhaustedError(attempt, TaskCancelledError("retry manager shut down"), label)

    def _notify(self, on_retry: Optional[OnRetry], attempt: int, exc: BaseException, delay: float) -> None:
        logger.info("retrying %s after attempt %d (%s: %s) in %.2fs", "task", attempt,
                    type(exc).__name__, exc, delay)
        if on_retry is not None:
            try:
                on_retry(attempt, exc, delay)
            except Exception:  # noqa: BLE001
                logger.exception("on_retry callback failed")

    def run(
        self, fn: Callable[[], Any], policy: Optional[RetryPolicy] = None, *,
        label: str = "", on_retry: Optional[OnRetry] = None,
    ) -> Tuple[Any, int]:
        """Synchronously call ``fn`` until it succeeds or the policy gives up."""
        policy = policy or self.default_policy
        attempt = 0
        while True:
            attempt += 1
            if self._stop.is_set():
                raise self._cancelled(attempt - 1, label)
            try:
                return fn(), attempt
            except Exception as exc:  # noqa: BLE001 - KeyboardInterrupt etc. propagate
                if not policy.should_retry(exc, attempt):
                    raise RetryExhaustedError(attempt, exc, label) from exc
                delay = policy.delay_for(attempt)
                self._notify(on_retry, attempt, exc, delay)
                if self._stop.wait(delay):
                    raise self._cancelled(attempt, label) from exc

    async def arun(
        self, fn: Callable[[], Awaitable[Any]], policy: Optional[RetryPolicy] = None, *,
        label: str = "", on_retry: Optional[OnRetry] = None,
    ) -> Tuple[Any, int]:
        """Async variant of :meth:`run`; ``fn`` returns an awaitable."""
        policy = policy or self.default_policy
        attempt = 0
        while True:
            attempt += 1
            if self._stop.is_set():
                raise self._cancelled(attempt - 1, label)
            try:
                return await fn(), attempt
            except Exception as exc:  # noqa: BLE001 - CancelledError is BaseException
                if not policy.should_retry(exc, attempt):
                    raise RetryExhaustedError(attempt, exc, label) from exc
                delay = policy.delay_for(attempt)
                self._notify(on_retry, attempt, exc, delay)
                await asyncio.sleep(delay)
