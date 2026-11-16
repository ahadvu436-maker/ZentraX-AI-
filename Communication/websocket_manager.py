"""WebSocket connection management: registry, subscriptions, backpressure, heartbeats.

The manager is framework-agnostic: any object with ``send_text``/``receive_text``/
``close`` (Starlette/FastAPI ``WebSocket``) works, and ``WebsocketsAdapter``
adapts the ``websockets`` library. The caller must *accept* the handshake
before calling :meth:`WebSocketManager.serve`.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, Set

from .communication_core import (
    Component, CommunicationError, ConnectionClosedError, ConnectionLimitError, RateLimitError,
    TokenBucket, ProtocolError, AuthorizationError,
)
from .messaging.event_bus import topic_matches, validate_topic
from .protocol_handler import Envelope, MessageType, ProtocolHandler

logger = logging.getLogger(__name__)


class WebSocketLike(Protocol):
    async def send_text(self, data: str) -> None: ...
    async def receive_text(self) -> str: ...
    async def close(self, code: int = 1000) -> None: ...


class WebsocketsAdapter:
    """Adapt a ``websockets`` connection to :class:`WebSocketLike`."""

    def __init__(self, conn: Any) -> None:
        self._c = conn

    async def send_text(self, data: str) -> None:
        await self._c.send(data)

    async def receive_text(self) -> str:
        msg = await self._c.recv()
        if isinstance(msg, bytes):
            raise ProtocolError("binary frames are not supported")
        return msg

    async def close(self, code: int = 1000) -> None:
        await self._c.close(code)


@dataclass(eq=False)
class Connection:
    ws: WebSocketLike
    principal: Any
    remote: str
    bucket: TokenBucket
    outbox: "asyncio.Queue[str]"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: Dict[str, Any] = field(default_factory=dict)
    subscriptions: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    strikes: int = 0
    open: bool = True
    writer: Optional["asyncio.Task[None]"] = None


ConnHook = Callable[[Connection], Awaitable[None]]
MessageHook = Callable[[Connection, Envelope], Awaitable[None]]
SubscribePolicy = Callable[[Connection, str], bool]


class WebSocketManager(Component):
    """Tracks connections and delivers messages with bounded per-connection outboxes.

    * slow consumers (full outbox) are disconnected rather than blocking senders,
    * per-connection token-bucket rate limiting; repeated violations disconnect,
    * heartbeat pings and idle timeout,
    * ``subscribe_policy`` authorises topic subscriptions.
    """

    def __init__(self, protocol: ProtocolHandler, *, max_connections: int = 10_000, outbox_size: int = 256,
                 send_timeout_s: float = 10.0, heartbeat_interval_s: float = 20.0, idle_timeout_s: float = 60.0,
                 messages_per_second: float = 50.0, burst: int = 100, max_subscriptions: int = 100,
                 max_strikes: int = 10, subscribe_policy: Optional[SubscribePolicy] = None) -> None:
        super().__init__()
        self._protocol = protocol
        self._max_conn, self._outbox_size, self._send_timeout = max_connections, outbox_size, send_timeout_s
        self._hb, self._idle = heartbeat_interval_s, idle_timeout_s
        self._rate, self._burst = messages_per_second, burst
        self._max_subs, self._max_strikes = max_subscriptions, max_strikes
        self._policy = subscribe_policy
        self.on_message: Optional[MessageHook] = None
        self.on_connect: Optional[ConnHook] = None
        self.on_disconnect: Optional[ConnHook] = None
        self._conns: Dict[str, Connection] = {}
        self._by_user: Dict[str, Set[str]] = {}
        self._bg: Set["asyncio.Task[Any]"] = set()
        self._hb_task: Optional["asyncio.Task[None]"] = None
        self._stats = {"connected_total": 0, "messages_in": 0, "messages_out": 0, "slow_consumers": 0}

    # -- lifecycle ---------------------------------------------------------- #
    async def _on_start(self) -> None:
        self._hb_task = asyncio.create_task(self._heartbeat_loop(), name="ws-heartbeat")

    async def _on_stop(self) -> None:
        if self._hb_task:
            self._hb_task.cancel()
            await asyncio.gather(self._hb_task, return_exceptions=True)
        await asyncio.gather(*(self.disconnect(c, 1001, "server shutdown") for c in list(self._conns)),
                             return_exceptions=True)
        for t in list(self._bg):
            t.cancel()
        await asyncio.gather(*self._bg, return_exceptions=True)

    # -- connection handling ------------------------------------------------ #
    async def serve(self, ws: WebSocketLike, *, principal: Any = None, remote: str = "",
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register ``ws`` and run its receive loop until it closes."""
        conn = await self._register(ws, principal, remote, metadata)
        try:
            while conn.open:
                try:
                    raw = await ws.receive_text()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    break  # client disconnect or transport error
                conn.last_seen = time.monotonic()
                self._stats["messages_in"] += 1
                if not conn.bucket.allow():
                    if await self._strike(conn, RateLimitError(retry_after=conn.bucket.retry_after())):
                        break
                    continue
                try:
                    env = self._protocol.decode(raw)
                except ProtocolError as exc:
                    if await self._strike(conn, exc):
                        break
                    continue
                if env.type is MessageType.PONG:
                    continue
                await self._dispatch(conn, env)
        finally:
            await self.disconnect(conn.id)

    async def disconnect(self, conn_or_id: Any, code: int = 1000, reason: str = "") -> None:
        """Idempotently close and unregister a connection."""
        cid = conn_or_id.id if isinstance(conn_or_id, Connection) else conn_or_id
        conn = self._conns.pop(cid, None)
        if conn is None:
            return
        conn.open = False
        if conn.principal is not None:
            ids = self._by_user.get(conn.principal.id)
            if ids:
                ids.discard(cid)
                if not ids:
                    self._by_user.pop(conn.principal.id, None)
        if conn.writer and conn.writer is not asyncio.current_task():
            conn.writer.cancel()
        try:
            await asyncio.wait_for(conn.ws.close(code), 2.0)
        except Exception:
            pass
        if self.on_disconnect:
            try:
                await self.on_disconnect(conn)
            except Exception:
                logger.exception("on_disconnect hook failed")

    # -- sending ------------------------------------------------------------ #
    async def send(self, conn_id: str, env: Envelope) -> None:
        conn = self._conns.get(conn_id)
        if conn is None or not conn.open:
            raise ConnectionClosedError(f"connection {conn_id} is closed")
        self._enqueue(conn, self._protocol.encode(env))

    async def send_to_user(self, user_id: str, env: Envelope) -> int:
        """Send to every connection of ``user_id``; returns the number reached."""
        text = self._protocol.encode(env)
        return sum(self._enqueue(self._conns[c], text) for c in tuple(self._by_user.get(user_id, ())) if c in self._conns)

    async def broadcast(self, env: Envelope, *, exclude: Set[str] = frozenset()) -> int:  # type: ignore[assignment]
        text = self._protocol.encode(env)
        return sum(self._enqueue(c, text) for c in tuple(self._conns.values()) if c.id not in exclude)

    async def publish_to_subscribers(self, topic: str, env: Envelope) -> int:
        """Send ``env`` to connections whose subscription patterns match ``topic``."""
        targets = [c for c in tuple(self._conns.values())
                   if any(topic_matches(p, topic) for p in c.subscriptions)]
        if not targets:
            return 0
        text = self._protocol.encode(env)
        return sum(self._enqueue(c, text) for c in targets)

    # -- subscriptions ------------------------------------------------------ #
    def subscribe(self, conn_id: str, pattern: str) -> None:
        conn = self._get(conn_id)
        validate_topic(pattern, allow_wildcards=True)
        if len(conn.subscriptions) >= self._max_subs and pattern not in conn.subscriptions:
            raise AuthorizationError("subscription limit reached", code="subscription_limit")
        if self._policy is not None and not self._policy(conn, pattern):
            raise AuthorizationError(f"not allowed to subscribe to '{pattern}'")
        conn.subscriptions.add(pattern)

    def unsubscribe(self, conn_id: str, pattern: str) -> None:
        self._get(conn_id).subscriptions.discard(pattern)

    # -- introspection ------------------------------------------------------ #
    def get(self, conn_id: str) -> Optional[Connection]:
        return self._conns.get(conn_id)

    def connections_for(self, user_id: str) -> int:
        return len(self._by_user.get(user_id, ()))

    def stats(self) -> Dict[str, Any]:
        return {"connections": len(self._conns), **self._stats}

    # -- internals ---------------------------------------------------------- #
    def _get(self, conn_id: str) -> Connection:
        conn = self._conns.get(conn_id)
        if conn is None:
            raise ConnectionClosedError(f"connection {conn_id} is closed")
        return conn

    async def _register(self, ws: WebSocketLike, principal: Any, remote: str,
                        metadata: Optional[Dict[str, Any]]) -> Connection:
        if len(self._conns) >= self._max_conn:
            try:
                await ws.close(1013)
            except Exception:
                pass
            raise ConnectionLimitError("too many connections")
        conn = Connection(ws=ws, principal=principal, remote=remote,
                          bucket=TokenBucket(self._rate, self._burst),
                          outbox=asyncio.Queue(maxsize=self._outbox_size), metadata=dict(metadata or {}))
        conn.writer = asyncio.create_task(self._writer(conn), name=f"ws-writer-{conn.id[:8]}")
        self._conns[conn.id] = conn
        if principal is not None:
            self._by_user.setdefault(principal.id, set()).add(conn.id)
        self._stats["connected_total"] += 1
        if self.on_connect:
            try:
                await self.on_connect(conn)
            except Exception:
                logger.exception("on_connect hook failed")
        return conn

    def _enqueue(self, conn: Connection, text: str) -> bool:
        if not conn.open:
            return False
        try:
            conn.outbox.put_nowait(text)
            return True
        except asyncio.QueueFull:
            self._stats["slow_consumers"] += 1
            logger.warning("disconnecting slow consumer %s", conn.id)
            self._spawn(self.disconnect(conn, 1013, "slow consumer"))
            return False

    async def _writer(self, conn: Connection) -> None:
        try:
            while True:
                text = await conn.outbox.get()
                await asyncio.wait_for(conn.ws.send_text(text), self._send_timeout)
                self._stats["messages_out"] += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("writer for %s failed", conn.id, exc_info=True)
            self._spawn(self.disconnect(conn, 1011, "send failure"))

    async def _dispatch(self, conn: Connection, env: Envelope) -> None:
        if self.on_message is None:
            return
        try:
            await self.on_message(conn, env)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not isinstance(exc, CommunicationError):
                logger.exception("message handler failed")
            self._enqueue(conn, self._protocol.encode(self._protocol.make_error(env, exc)))

    async def _strike(self, conn: Connection, exc: CommunicationError) -> bool:
        """Record a violation, notify the client. Returns True if the connection should close."""
        conn.strikes += 1
        try:
            self._enqueue(conn, self._protocol.encode(self._protocol.make_error(None, exc)))
        except CommunicationError:
            pass
        return conn.strikes >= self._max_strikes

    def _spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._hb)
            now = time.monotonic()
            ping = self._protocol.encode(self._protocol.make(MessageType.PING))
            for conn in tuple(self._conns.values()):
                if now - conn.last_seen > self._idle:
                    self._spawn(self.disconnect(conn, 1001, "idle timeout"))
                else:
                    self._enqueue(conn, ping)
