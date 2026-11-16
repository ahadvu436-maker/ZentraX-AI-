"""Topic-based publish/subscribe bus with per-subscriber isolation and backpressure policies."""
from __future__ import annotations

import asyncio
import enum
import inspect
import itertools
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple, Union

from ..communication_core import Component, ValidationError

logger = logging.getLogger(__name__)

_SEGMENT = re.compile(r"^[A-Za-z0-9_\-:]+$")
_MAX_TOPIC_LEN, _MAX_SEGMENTS = 256, 16


def validate_topic(topic: str, *, allow_wildcards: bool = False) -> str:
    """Validate a dotted topic (``orders.created``). Patterns may use ``*`` (one
    segment) and ``#`` (zero or more segments) as whole segments."""
    if not isinstance(topic, str) or not topic or len(topic) > _MAX_TOPIC_LEN:
        raise ValidationError("invalid topic")
    parts = topic.split(".")
    if len(parts) > _MAX_SEGMENTS:
        raise ValidationError("topic has too many segments")
    for seg in parts:
        if _SEGMENT.match(seg) or (allow_wildcards and seg in ("*", "#")):
            continue
        raise ValidationError(f"invalid topic segment {seg!r}")
    return topic


def _match(p: Tuple[str, ...], t: Tuple[str, ...]) -> bool:
    if not p:
        return not t
    head = p[0]
    if head == "#":
        return len(p) == 1 or any(_match(p[1:], t[i:]) for i in range(len(t) + 1))
    if not t:
        return False
    return (head == "*" or head == t[0]) and _match(p[1:], t[1:])


@lru_cache(maxsize=8192)
def topic_matches(pattern: str, topic: str) -> bool:
    """True if ``topic`` matches ``pattern`` (supports ``*`` and ``#``)."""
    return _match(tuple(pattern.split(".")), tuple(topic.split(".")))


@dataclass(frozen=True, eq=False)
class Event:
    topic: str
    payload: Any = None
    source: str = ""
    correlation_id: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))


class OverflowPolicy(str, enum.Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"  # publisher waits; use only for trusted, fast subscribers


EventHandler = Callable[[Event], Union[None, Awaitable[None]]]
ErrorHook = Callable[["Subscription", Event, BaseException], Any]


@dataclass(eq=False)
class Subscription:
    pattern: str
    handler: EventHandler
    name: str
    queue: "asyncio.Queue[Event]"
    overflow: OverflowPolicy
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    active: bool = True
    task: Optional["asyncio.Task[None]"] = None
    delivered: int = 0
    dropped: int = 0
    errors: int = 0


class EventBus(Component):
    """Each subscriber has its own bounded queue and worker task, so a slow or
    failing subscriber never blocks publishers or other subscribers; events to one
    subscriber are delivered in publish order."""

    def __init__(self, *, on_error: Optional[ErrorHook] = None) -> None:
        super().__init__()
        self._subs: List[Subscription] = []
        self._on_error = on_error
        self._names = itertools.count(1)
        self._published = 0

    def subscribe(self, pattern: str, handler: EventHandler, *, name: Optional[str] = None,
                  queue_size: int = 256, overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST) -> Subscription:
        validate_topic(pattern, allow_wildcards=True)
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        sub = Subscription(pattern, handler, name or f"sub-{next(self._names)}",
                           asyncio.Queue(maxsize=queue_size), overflow)
        self._subs.append(sub)
        if self.running:
            self._spawn(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        sub.active = False
        if sub in self._subs:
            self._subs.remove(sub)
        if sub.task is not None:
            sub.task.cancel()

    async def publish(self, event: Event) -> int:
        """Fan out to matching subscribers; returns how many queues accepted the event."""
        validate_topic(event.topic)
        self._published += 1
        accepted = 0
        for sub in tuple(self._subs):
            if not sub.active or not topic_matches(sub.pattern, event.topic):
                continue
            if await self._offer(sub, event):
                accepted += 1
        return accepted

    async def emit(self, topic: str, payload: Any = None, **kwargs: Any) -> Event:
        event = Event(topic=topic, payload=payload, **kwargs)
        await self.publish(event)
        return event

    def stats(self) -> Dict[str, Any]:
        return {
            "published": self._published,
            "subscribers": [{"name": s.name, "pattern": s.pattern, "queued": s.queue.qsize(),
                             "delivered": s.delivered, "dropped": s.dropped, "errors": s.errors}
                            for s in self._subs],
        }

    # -- lifecycle ---------------------------------------------------------- #
    async def _on_start(self) -> None:
        for sub in self._subs:
            self._spawn(sub)

    async def _on_stop(self, drain_timeout_s: float = 5.0) -> None:
        pending = [s.queue.join() for s in self._subs if s.task is not None]
        if pending:
            try:
                await asyncio.wait_for(asyncio.gather(*pending), drain_timeout_s)
            except asyncio.TimeoutError:
                logger.warning("EventBus drain timed out; dropping queued events")
        tasks = [s.task for s in self._subs if s.task is not None]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for s in self._subs:
            s.task = None

    # -- internals ---------------------------------------------------------- #
    def _spawn(self, sub: Subscription) -> None:
        if sub.task is None or sub.task.done():
            sub.task = asyncio.create_task(self._worker(sub), name=f"eventbus-{sub.name}")

    async def _offer(self, sub: Subscription, event: Event) -> bool:
        q = sub.queue
        if sub.overflow is OverflowPolicy.BLOCK:
            await q.put(event)
            return True
        if q.full():
            if sub.overflow is OverflowPolicy.DROP_NEWEST:
                sub.dropped += 1
                return False
            try:
                q.get_nowait()
                q.task_done()
                sub.dropped += 1
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(event)
            return True
        except asyncio.QueueFull:
            sub.dropped += 1
            return False

    async def _worker(self, sub: Subscription) -> None:
        while True:
            event = await sub.queue.get()
            try:
                res = sub.handler(event)
                if inspect.isawaitable(res):
                    await res
                sub.delivered += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                sub.errors += 1
                logger.exception("subscriber %s failed on %s", sub.name, event.topic)
                if self._on_error:
                    try:
                        r = self._on_error(sub, event, exc)
                        if inspect.isawaitable(r):
                            await r
                    except Exception:
                        logger.exception("bus error hook failed")
            finally:
                sub.queue.task_done()
