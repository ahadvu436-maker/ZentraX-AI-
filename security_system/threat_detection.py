"""Behavioural and signature-based threat detection.

Detectors (each yields a score in ``[0, 1]``):

* blocklisted source                         -> 1.0
* request flooding per source                -> rate limit breach
* brute-force login failures per IP/account  -> online password guessing
* password spraying                          -> one IP failing across many accounts
* privilege probing                          -> repeated authorisation denials
* injection signatures in payloads           -> SQLi / XSS / command / traversal

Scores are combined with a noisy-OR (``1 - prod(1 - s_i)``) so several weak
signals add up to a strong one. The resulting level maps to an action:
``ALLOW`` / ``MONITOR`` / ``CHALLENGE`` (step-up auth) / ``BLOCK``.

State is in-memory and per-process. For a multi-node deployment back the
counters with a shared store (for example Redis) behind the same interface.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, Mapping

from .input_validation import InputValidator
from .policies.security_policy import SecurityPolicy

__all__ = [
    "EventType",
    "ThreatLevel",
    "ThreatAction",
    "SecurityEvent",
    "ThreatAssessment",
    "ThreatDetector",
]

_SIGNATURE_SCORES: Mapping[str, float] = {
    "sql_injection": 0.85,
    "xss": 0.8,
    "command_injection": 0.9,
    "path_traversal": 0.85,
    "template_injection": 0.6,
    "ldap_injection": 0.6,
    "oversized_payload": 0.3,
}


class EventType(str, Enum):
    REQUEST = "request"
    LOGIN_SUCCESS = "auth.login_success"
    LOGIN_FAILURE = "auth.login_failure"
    TOKEN_INVALID = "auth.token_invalid"
    AUTHZ_DENIED = "authz.denied"


class ThreatLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ThreatAction(str, Enum):
    ALLOW = "allow"
    MONITOR = "monitor"
    CHALLENGE = "challenge"
    BLOCK = "block"


@dataclass(frozen=True)
class SecurityEvent:
    event_type: EventType
    principal_id: str | None = None
    ip: str | None = None
    payload: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ThreatAssessment:
    score: float
    level: ThreatLevel
    action: ThreatAction
    indicators: tuple[str, ...]


class _SlidingWindow:
    """Per-key event counter over a fixed time window."""

    def __init__(self, window_seconds: float) -> None:
        self._window = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._adds = 0

    def add(self, key: str, now: float) -> int:
        bucket = self._events.setdefault(key, deque())
        bucket.append(now)
        self._prune(bucket, now)
        self._adds += 1
        if self._adds % 1000 == 0:
            self._gc(now)
        return len(bucket)

    def count(self, key: str, now: float) -> int:
        bucket = self._events.get(key)
        if bucket is None:
            return 0
        self._prune(bucket, now)
        return len(bucket)

    def _prune(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self._window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def _gc(self, now: float) -> None:
        for key in [k for k, b in self._events.items() if not b or b[-1] <= now - self._window]:
            del self._events[key]


class ThreatDetector:
    """Thread-safe scoring engine. Call :meth:`assess` for every security-relevant event."""

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        *,
        validator: InputValidator | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._policy = policy or SecurityPolicy()
        self._rl = self._policy.rate_limit
        self._validator = validator or InputValidator()
        self._clock = clock
        window = float(self._rl.window_seconds)
        self._requests = _SlidingWindow(window)
        self._ip_failures = _SlidingWindow(window)
        self._principal_failures = _SlidingWindow(window)
        self._denials = _SlidingWindow(window)
        self._spray: dict[str, deque[tuple[float, str]]] = {}
        self._blocked: dict[str, float] = {}
        self._lock = threading.Lock()

    # --------------------------------------------------------------- blocklist
    def block(self, ip: str, *, seconds: float | None = None) -> None:
        """Block a source; ``seconds=None`` uses the policy's auto-block duration."""
        duration = float(self._rl.auto_block_seconds) if seconds is None else seconds
        with self._lock:
            self._blocked[ip] = self._clock() + duration

    def unblock(self, ip: str) -> None:
        with self._lock:
            self._blocked.pop(ip, None)

    def is_blocked(self, ip: str | None) -> bool:
        if not ip:
            return False
        with self._lock:
            return self._is_blocked(ip, self._clock())

    def _is_blocked(self, ip: str, now: float) -> bool:
        until = self._blocked.get(ip)
        if until is None:
            return False
        if until <= now:
            del self._blocked[ip]
            return False
        return True

    # ---------------------------------------------------------------- assessing
    def assess(self, event: SecurityEvent) -> ThreatAssessment:
        """Record ``event`` and return the combined threat assessment."""
        now = event.timestamp
        indicators: list[str] = []
        scores: list[float] = []

        with self._lock:
            if event.ip and self._is_blocked(event.ip, now):
                return ThreatAssessment(1.0, ThreatLevel.CRITICAL, ThreatAction.BLOCK, ("source_blocked",))

            if event.ip:
                total = self._requests.add(event.ip, now)
                limit = self._rl.max_requests
                if total > limit:
                    overshoot = min(0.3, (total - limit) / limit * 0.3)
                    scores.append(0.6 + overshoot)
                    indicators.append("rate_limit_exceeded")

            failure_types = (EventType.LOGIN_FAILURE, EventType.TOKEN_INVALID)
            if event.event_type in failure_types:
                if event.ip:
                    fails = self._ip_failures.add(event.ip, now)
                    threshold = self._rl.brute_force_threshold
                    if fails >= threshold:
                        scores.append(0.85)
                        indicators.append("brute_force_ip")
                    elif fails * 2 >= threshold:
                        scores.append(0.4)
                        indicators.append("elevated_failures_ip")
                if event.principal_id and event.event_type is EventType.LOGIN_FAILURE:
                    p_fails = self._principal_failures.add(event.principal_id, now)
                    if p_fails >= self._policy.lockout.max_failed_attempts:
                        scores.append(0.7)
                        indicators.append("brute_force_account")
                    self._track_spray(event, now, scores, indicators)

            if event.event_type is EventType.AUTHZ_DENIED and event.principal_id:
                denied = self._denials.add(event.principal_id, now)
                if denied >= self._rl.authz_denial_threshold:
                    scores.append(0.7)
                    indicators.append("privilege_probing")

        if event.payload:
            for name in self._validator.detect_injection(event.payload):
                scores.append(_SIGNATURE_SCORES.get(name, 0.7))
                indicators.append(name)

        combined = 1.0
        for score in scores:
            combined *= 1.0 - min(max(score, 0.0), 1.0)
        score = round(1.0 - combined, 4)
        level = self._level(score)
        action = {
            ThreatLevel.NONE: ThreatAction.ALLOW,
            ThreatLevel.LOW: ThreatAction.MONITOR,
            ThreatLevel.MEDIUM: ThreatAction.CHALLENGE,
            ThreatLevel.HIGH: ThreatAction.BLOCK,
            ThreatLevel.CRITICAL: ThreatAction.BLOCK,
        }[level]

        if level is ThreatLevel.CRITICAL and event.ip:
            with self._lock:
                self._blocked[event.ip] = now + self._rl.auto_block_seconds
            indicators.append("auto_blocked")

        return ThreatAssessment(score, level, action, tuple(indicators))

    # --------------------------------------------------------------- internals
    def _track_spray(
        self,
        event: SecurityEvent,
        now: float,
        scores: list[float],
        indicators: list[str],
    ) -> None:
        if not event.ip or not event.principal_id:
            return
        bucket = self._spray.setdefault(event.ip, deque())
        bucket.append((now, event.principal_id))
        cutoff = now - self._rl.window_seconds
        while bucket and bucket[0][0] <= cutoff:
            bucket.popleft()
        if len({user for _, user in bucket}) >= self._rl.password_spray_distinct_users:
            scores.append(0.9)
            indicators.append("password_spraying")
        if len(self._spray) > 10_000:  # bound memory under attack
            for key in [k for k, b in self._spray.items() if not b or b[-1][0] <= cutoff]:
                del self._spray[key]

    @staticmethod
    def _level(score: float) -> ThreatLevel:
        if score < 0.2:
            return ThreatLevel.NONE
        if score < 0.4:
            return ThreatLevel.LOW
        if score < 0.6:
            return ThreatLevel.MEDIUM
        if score < 0.8:
            return ThreatLevel.HIGH
        return ThreatLevel.CRITICAL
