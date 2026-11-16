"""High-level event API over the bus: validation, decorators, history/replay, wait_for."""
from __future__ import annotations

import asyncio
import collections
import logging
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Tuple

from .communication_core import CommunicationTimeoutError, Component, ValidationError
from .messaging.event_bus import (
    Event, EventBus, EventHandler, OverflowPolicy, Subscription, topic_matches, validate_topic,
)

logger = logging.getLogger(__name__)
Validator = Callable[[Any], None]  # raise ValueError/ValidationError for bad payloads


class EventManager(Component):
    """Application-facing event facade.

    * topic registration with optional payload validators (``strict_topics`` rejects
      unregistered topics),
    * ``@manager.on("pattern")`` handler decorator,
    * bounded history with replay,
    * ``wait_for`` to await a specific event.
    """

    def __init__(self, bus: EventBus, *, history_size: int = 1000, strict_topics: bool = False) -> None:
        super().__init__()
        self._bus = bus
        self._strict = strict_topics
        self._topics: List[Tuple[str, Optional[Validator], str]] = []
        self._history: Deque[Event] = collections.deque(maxlen=history_size)
        self._handlers: Dict[EventHandler, Subscription] = {}

    # -- registration ------------------------------------------------------- #
    def register_topic(self, pattern: str, validator: Optional[Validator] = None, description: str = "") -> None:
        validate_topic(pattern, allow_wildcards=True)
        self._topics.append((pattern, validator, description))

    def topics(self) -> List[Tuple[str, str]]:
        return [(p, d) for p, _, d in self._topics]

    # -- publishing --------------------------------------------------------- #
    async def publish(self, topic: str, payload: Any = None, *, source: str = "",
                      correlation_id: Optional[str] = None,
                      headers: Optional[Mapping[str, str]] = None) -> Event:
        validate_topic(topic)
        matching = [(p, v) for p, v, _ in self._topics if topic_matches(p, topic)]
        if self._strict and not matching:
            raise ValidationError(f"topic '{topic}' is not registered")
        for _, validator in matching:
            if validator is None:
                continue
            try:
                validator(payload)
            except ValidationError:
                raise
            except (ValueError, TypeError) as exc:
                raise ValidationError(f"invalid payload for '{topic}': {exc}") from exc
        event = Event(topic=topic, payload=payload, source=source, correlation_id=correlation_id,
                      headers=headers or {})
        self._history.append(event)
        await self._bus.publish(event)
        return event

    # -- subscribing -------------------------------------------------------- #
    def subscribe(self, pattern: str, handler: EventHandler, *, name: Optional[str] = None,
                  queue_size: int = 256, overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST) -> Subscription:
        sub = self._bus.subscribe(pattern, handler, name=name, queue_size=queue_size, overflow=overflow)
        self._handlers[handler] = sub
        return sub

    def on(self, pattern: str, **kwargs: Any) -> Callable[[EventHandler], EventHandler]:
        """Decorator: ``@events.on("orders.*")``."""
        def deco(fn: EventHandler) -> EventHandler:
            self.subscribe(pattern, fn, **kwargs)
            return fn
        return deco

    def off(self, handler: EventHandler) -> bool:
        sub = self._handlers.pop(handler, None)
        if sub is None:
            return False
        self._bus.unsubscribe(sub)
        return True

    # -- history ------------------------------------------------------------ #
    def history(self, pattern: str = "#", limit: int = 100) -> List[Event]:
        events = [e for e in self._history if topic_matches(pattern, e.topic)]
        return events[-limit:] if limit > 0 else []

    async def replay(self, pattern: str, handler: EventHandler, *, since: Optional[float] = None) -> int:
        """Invoke ``handler`` for matching historic events (oldest first). Returns count."""
        import inspect
        count = 0
        for e in [e for e in self._history if topic_matches(pattern, e.topic)
                  and (since is None or e.timestamp >= since)]:
            res = handler(e)
            if inspect.isawaitable(res):
                await res
            count += 1
        return count

    async def wait_for(self, pattern: str, *, predicate: Optional[Callable[[Event], bool]] = None,
                       timeout: float = 10.0) -> Event:
        """Await the next event matching ``pattern`` (and ``predicate``)."""
        fut: "asyncio.Future[Event]" = asyncio.get_running_loop().create_future()

        def handler(event: Event) -> None:
            try:
                ok = predicate is None or predicate(event)
            except Exception:
                ok = False
            if ok and not fut.done():
                fut.set_result(event)

        sub = self._bus.subscribe(pattern, handler, name="wait_for", queue_size=64)
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise CommunicationTimeoutError(f"timed out waiting for '{pattern}'") from None
        finally:
            self._bus.unsubscribe(sub)
