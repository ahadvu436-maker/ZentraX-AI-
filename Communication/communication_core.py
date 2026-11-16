"""Shared foundations and the ``CommunicationCore`` orchestrator.

This module defines the error hierarchy, the ``Component`` lifecycle base class,
a token-bucket rate limiter, and ``CommunicationCore`` which coordinates the
router, event manager/bus, WebSocket manager, protocol handler, notification
manager, queues and API gateway.

Sibling modules are imported lazily (``create_default``) or only for typing, so
every other module can import from here without circular imports.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .api_gateway import ApiGateway, Principal
    from .event_manager import EventManager
    from .messaging.event_bus import Event, EventBus
    from .messaging.message_queue import AsyncMessageQueue, QueueManager, QueueMessage
    from .message_router import MessageRouter
    from .notification_manager import Notification, NotificationManager
    from .protocol_handler import Envelope, ProtocolHandler
    from .websocket_manager import Connection, WebSocketManager

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class CommunicationError(Exception):
    """Base error. ``code``/``status`` are safe to expose to clients."""

    code: str = "communication_error"
    status: int = 500

    def __init__(self, message: str = "", *, code: Optional[str] = None, status: Optional[int] = None) -> None:
        super().__init__(message or (code or self.code))
        if code:
            self.code = code
        if status:
            self.status = status


class ProtocolError(CommunicationError):
    code, status = "protocol_error", 400


class ValidationError(CommunicationError):
    code, status = "validation_error", 422


class RoutingError(CommunicationError):
    code, status = "no_route", 404


class AuthenticationError(CommunicationError):
    code, status = "unauthenticated", 401


class AuthorizationError(CommunicationError):
    code, status = "forbidden", 403


class CommunicationTimeoutError(CommunicationError):
    code, status = "timeout", 504


class QueueError(CommunicationError):
    code, status = "queue_error", 500


class QueueFullError(QueueError):
    code, status = "queue_full", 503


class QueueClosedError(QueueError):
    code, status = "queue_closed", 503


class ConnectionClosedError(CommunicationError):
    code, status = "connection_closed", 410


class ConnectionLimitError(CommunicationError):
    code, status = "connection_limit", 503


class NotificationError(CommunicationError):
    code, status = "notification_error", 502


class RateLimitError(CommunicationError):
    code, status = "rate_limited", 429

    def __init__(self, message: str = "rate limit exceeded", *, retry_after: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class TokenBucket:
    """Simple token bucket. Not thread-safe; intended for single-event-loop use."""

    def __init__(self, rate: float, capacity: float) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be > 0")
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._stamp = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._stamp) * self._rate)
        self._stamp = now

    def allow(self, cost: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False

    def retry_after(self, cost: float = 1.0) -> float:
        self._refill()
        return max(0.0, (cost - self._tokens) / self._rate)


class Component:
    """Idempotent async start/stop lifecycle. Subclasses override ``_on_start``/``_on_stop``."""

    def __init__(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        await self._on_start()
        self._running = True

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        await self._on_stop()

    async def _on_start(self) -> None:  # pragma: no cover - default no-op
        return None

    async def _on_stop(self) -> None:  # pragma: no cover - default no-op
        return None

    async def __aenter__(self) -> "Component":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class CommunicationCore(Component):
    """Central manager wiring all communication components together.

    Data flows:
      * WebSocket inbound -> protocol decode -> :meth:`handle_incoming` ->
        router / event manager / subscription management -> reply.
      * Events published anywhere -> event bus -> forwarded to subscribed sockets.
      * HTTP -> API gateway -> router / events / notifications.
    """

    def __init__(
        self, *, protocol: "ProtocolHandler", router: "MessageRouter", bus: "EventBus",
        events: "EventManager", ws: "WebSocketManager", notifications: "NotificationManager",
        queues: "QueueManager", gateway: "ApiGateway",
    ) -> None:
        super().__init__()
        self.protocol, self.router, self.bus, self.events = protocol, router, bus, events
        self.ws, self.notifications, self.queues, self.gateway = ws, notifications, queues, gateway
        self._forwarder: Any = None

    @classmethod
    def create_default(
        cls, *, secret: Optional[bytes] = None, api_keys: Optional[Mapping[str, "Principal"]] = None,
        max_connections: int = 10_000, max_message_bytes: int = 1_048_576,
        allowed_origins: Optional[List[str]] = None,
    ) -> "CommunicationCore":
        """Build a fully wired core. ``api_keys`` maps raw API keys to principals;
        with no keys configured, authenticated HTTP routes reject everything."""
        from .api_gateway import ApiGateway, ApiKeyAuthenticator
        from .event_manager import EventManager
        from .message_router import MessageRouter
        from .messaging.event_bus import EventBus
        from .messaging.message_queue import QueueManager
        from .notification_manager import NotificationManager, WebSocketChannel
        from .protocol_handler import ProtocolHandler
        from .websocket_manager import WebSocketManager

        protocol = ProtocolHandler(secret=secret, max_message_bytes=max_message_bytes)
        bus = EventBus()
        queues = QueueManager()
        auth = ApiKeyAuthenticator()
        for key, principal in (api_keys or {}).items():
            auth.add_key(key, principal)
        core = cls(
            protocol=protocol, router=MessageRouter(), bus=bus, events=EventManager(bus),
            ws=WebSocketManager(protocol, max_connections=max_connections),
            notifications=NotificationManager(queues.get_or_create("notifications", max_attempts=4)),
            queues=queues, gateway=ApiGateway(auth, max_body_bytes=max_message_bytes),
        )
        core.allowed_origins = list(allowed_origins or [])
        core.ws.on_message = core.handle_incoming
        core.ws.on_connect = core._on_connect
        core.notifications.add_channel(WebSocketChannel(core.ws, protocol))
        core.gateway.mount_core_routes(core)
        return core

    allowed_origins: List[str] = []

    # -- lifecycle ---------------------------------------------------------- #
    async def _on_start(self) -> None:
        started: List[Component] = []
        try:
            for comp in (self.bus, self.events, self.queues, self.ws, self.notifications):
                await comp.start()
                started.append(comp)
            self._forwarder = self.bus.subscribe("#", self._forward_event, name="ws-forwarder", queue_size=1024)
        except Exception:
            for comp in reversed(started):
                await comp.stop()
            raise

    async def _on_stop(self) -> None:
        for comp in (self.notifications, self.ws, self.queues, self.events, self.bus):
            try:
                await comp.stop()
            except Exception:
                logger.exception("Error stopping %s", type(comp).__name__)

    # -- public API --------------------------------------------------------- #
    async def publish_event(self, topic: str, payload: Any = None, **kwargs: Any) -> "Event":
        return await self.events.publish(topic, payload, **kwargs)

    async def send(self, envelope: "Envelope", context: Optional[Any] = None) -> Any:
        """Route an envelope locally and return the handler result."""
        return await self.router.route(envelope, context)

    async def notify(self, notification: "Notification") -> str:
        return await self.notifications.notify(notification)

    async def enqueue(self, queue: str, payload: Any, **kwargs: Any) -> "QueueMessage":
        return await self.queues.get_or_create(queue).put(payload, **kwargs)

    def consume(self, queue: str, handler: Callable[["QueueMessage"], Awaitable[None]], *,
                concurrency: int = 1, handler_timeout_s: Optional[float] = None) -> None:
        self.queues.get_or_create(queue).start_workers(
            handler, concurrency=concurrency, handler_timeout_s=handler_timeout_s)

    def health(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "bus": self.bus.stats(), "router": self.router.stats(), "websockets": self.ws.stats(),
            "queues": self.queues.stats(), "notifications": self.notifications.stats(),
        }

    # -- inbound WebSocket handling ---------------------------------------- #
    async def handle_incoming(self, conn: "Connection", env: "Envelope") -> None:
        """Dispatch one decoded client envelope. Raising is safe: the WebSocket
        manager converts exceptions into ERROR envelopes."""
        from .protocol_handler import MessageType as MT

        if env.type is MT.PING:
            await self.ws.send(conn.id, self.protocol.make(MT.PONG, correlation_id=env.id))
        elif env.type in (MT.SUBSCRIBE, MT.UNSUBSCRIBE):
            topics = env.payload.get("topics") if isinstance(env.payload, dict) else None
            if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
                raise ValidationError("payload.topics must be a list of strings")
            for t in topics:
                (self.ws.subscribe if env.type is MT.SUBSCRIBE else self.ws.unsubscribe)(conn.id, t)
            await self.ws.send(conn.id, self.protocol.make(MT.ACK, correlation_id=env.id, payload={"topics": topics}))
        elif env.type is MT.EVENT:
            await self.events.publish(env.topic, env.payload, source=f"ws:{conn.id}", correlation_id=env.correlation_id)
        elif env.type is MT.REQUEST:
            from .message_router import RouteContext

            ctx = RouteContext(connection=conn, principal=conn.principal)
            try:
                result = await self.router.route(env, ctx)
                reply = self.protocol.make(MT.RESPONSE, topic=env.topic, payload=result, correlation_id=env.id)
            except Exception as exc:
                reply = self.protocol.make_error(env, exc)
            await self.ws.send(conn.id, reply)
        else:
            logger.debug("Ignoring client message of type %s", env.type.value)

    async def _forward_event(self, event: "Event") -> None:
        from .protocol_handler import MessageType as MT

        env = self.protocol.make(
            MT.EVENT, topic=event.topic, payload=event.payload, correlation_id=event.correlation_id,
            headers={"event_id": event.id, "source": event.source} if event.source else {"event_id": event.id},
        )
        await self.ws.publish_to_subscribers(event.topic, env)

    async def _on_connect(self, conn: "Connection") -> None:
        principal = conn.principal
        if principal is not None:
            await self.notifications.deliver_pending(principal.id)
