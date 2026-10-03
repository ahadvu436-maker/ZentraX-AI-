"""
quantum_deception.py

Live intrusion tracking and alerting module.

Consumes security events (e.g. from trap_decoy_engine.py hits or
packet_destroyer.py bans) and:

  * Scores and classifies threat severity.
  * Maintains a rolling, in-memory timeline of recent incidents per
    source key, for live dashboards.
  * Fans out alerts asynchronously to configurable sinks (webhook,
    email, Slack, SIEM, log file, etc.) without blocking request
    handling.
  * Optionally triggers an account/IP "lockdown" callback (supplied
    by the host application) once a severity threshold is crossed --
    e.g. force-disable an account or push an IP onto a block-list
    consumed by packet_destroyer.py.

This module only observes and reports. It never attempts to probe,
trace back to, deanonymize, or take any action against the
originating client beyond what the host application's own
lockdown_callback chooses to do within its own systems (e.g. banning
a key in its own rate limiter). No outbound network traffic is sent
to the attacker.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger("quantum_deception")

AlertSink = Callable[["SecurityAlert"], Awaitable[None]]
LockdownCallback = Callable[[str, "ThreatLevel"], Awaitable[None]]


class ThreatLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(slots=True)
class SecurityEvent:
    source_key: str          # e.g. client IP or API key
    kind: str                # "decoy_hit", "rate_limit_ban", "auth_failure", ...
    timestamp: float = field(default_factory=time.time)
    detail: dict[str, Any] = field(default_factory=dict)
    base_score: int = 1


@dataclass(slots=True)
class SecurityAlert:
    source_key: str
    level: ThreatLevel
    score: int
    event_count: int
    window_seconds: float
    recent_kinds: tuple[str, ...]
    timestamp: float = field(default_factory=time.time)


@dataclass
class DeceptionConfig:
    window_seconds: float = 300.0        # rolling scoring window
    medium_threshold: int = 5
    high_threshold: int = 15
    critical_threshold: int = 30
    max_tracked_sources: int = 100_000
    history_per_source: int = 200


class _SourceTimeline:
    __slots__ = ("events", "score")

    def __init__(self, maxlen: int) -> None:
        self.events: deque[SecurityEvent] = deque(maxlen=maxlen)
        self.score: int = 0


class IntrusionTracker:
    """
    Aggregates security events per source key, computes a rolling
    threat score, raises alerts across configured sinks, and invokes
    a host-supplied lockdown callback at high severity.
    """

    def __init__(
        self,
        config: DeceptionConfig | None = None,
        sinks: list[AlertSink] | None = None,
        lockdown_callback: LockdownCallback | None = None,
    ) -> None:
        self._config = config or DeceptionConfig()
        self._sinks = sinks or []
        self._lockdown_callback = lockdown_callback
        self._timelines: dict[str, _SourceTimeline] = {}
        self._lock = asyncio.Lock()
        self._locked_down: set[str] = set()

    def add_sink(self, sink: AlertSink) -> None:
        self._sinks.append(sink)

    async def record(self, event: SecurityEvent) -> SecurityAlert | None:
        now = time.time()
        async with self._lock:
            self._evict_if_full()
            timeline = self._timelines.setdefault(
                event.source_key,
                _SourceTimeline(maxlen=self._config.history_per_source),
            )
            timeline.events.append(event)

            cutoff = now - self._config.window_seconds
            while timeline.events and timeline.events[0].timestamp < cutoff:
                timeline.events.popleft()

            score = sum(e.base_score for e in timeline.events)
            timeline.score = score
            level = self._classify(score)
            event_count = len(timeline.events)
            recent_kinds = tuple(e.kind for e in timeline.events)[-10:]

        if level is None:
            return None

        alert = SecurityAlert(
            source_key=event.source_key,
            level=level,
            score=score,
            event_count=event_count,
            window_seconds=self._config.window_seconds,
            recent_kinds=recent_kinds,
        )
        await self._dispatch(alert)
        return alert

    def _classify(self, score: int) -> ThreatLevel | None:
        cfg = self._config
        if score >= cfg.critical_threshold:
            return ThreatLevel.CRITICAL
        if score >= cfg.high_threshold:
            return ThreatLevel.HIGH
        if score >= cfg.medium_threshold:
            return ThreatLevel.MEDIUM
        return None

    async def _dispatch(self, alert: SecurityAlert) -> None:
        sink_tasks = [self._safe_sink(sink, alert) for sink in self._sinks]
        if sink_tasks:
            await asyncio.gather(*sink_tasks, return_exceptions=True)

        if alert.level >= ThreatLevel.HIGH and self._lockdown_callback is not None:
            if alert.source_key not in self._locked_down:
                self._locked_down.add(alert.source_key)
                try:
                    await self._lockdown_callback(alert.source_key, alert.level)
                except Exception:
                    logger.exception("Lockdown callback failed for %s", alert.source_key)

    async def _safe_sink(self, sink: AlertSink, alert: SecurityAlert) -> None:
        try:
            await sink(alert)
        except Exception:
            logger.exception("Alert sink failed")

    def _evict_if_full(self) -> None:
        if len(self._timelines) <= self._config.max_tracked_sources:
            return
        drop_count = max(1, len(self._timelines) // 10)
        for key in list(self._timelines.keys())[:drop_count]:
            self._timelines.pop(key, None)
            self._locked_down.discard(key)

    def snapshot(self, source_key: str) -> dict[str, Any] | None:
        timeline = self._timelines.get(source_key)
        if timeline is None:
            return None
        return {
            "source_key": source_key,
            "score": timeline.score,
            "event_count": len(timeline.events),
            "recent_kinds": [e.kind for e in timeline.events][-10:],
        }


async def log_sink(alert: SecurityAlert) -> None:
    logger.warning(
        "SECURITY ALERT level=%s source=%s score=%d events=%d recent=%s",
        alert.level.name, alert.source_key, alert.score,
        alert.event_count, alert.recent_kinds,
    )


def webhook_sink(post_fn: Callable[[dict[str, Any]], Awaitable[None]]) -> AlertSink:
    """
    Wrap an async HTTP POST function (e.g. an httpx.AsyncClient.post
    partial) into an AlertSink that ships alerts to Slack/SIEM/etc.
    Kept generic so this module has no hard dependency on a specific
    HTTP client library.
    """

    async def _sink(alert: SecurityAlert) -> None:
        payload = {
            "level": alert.level.name,
            "source_key": alert.source_key,
            "score": alert.score,
            "event_count": alert.event_count,
            "window_seconds": alert.window_seconds,
            "recent_kinds": list(alert.recent_kinds),
            "timestamp": alert.timestamp,
        }
        await post_fn(payload)

    return _sink


class SupportsBan(Protocol):
    async def ban(self, key: str, duration_seconds: float) -> None: ...


def make_lockdown_callback(
    limiter_with_ban: SupportsBan, duration_seconds: float = 3600.0
) -> LockdownCallback:
    """
    Convenience adapter: wires a HIGH/CRITICAL alert straight into
    banning the source key on something implementing `.ban()` (for
    example an extended AdaptiveRateLimiter from packet_destroyer.py).
    """

    async def _callback(source_key: str, level: ThreatLevel) -> None:
        multiplier = 4 if level is ThreatLevel.CRITICAL else 1
        await limiter_with_ban.ban(source_key, duration_seconds * multiplier)

    return _callback
