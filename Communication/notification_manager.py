"""Multi-channel notifications with queued delivery, retries, dedupe and offline inbox."""
from __future__ import annotations

import asyncio
import enum
import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Collection, Deque, Dict, FrozenSet, List, Mapping, Optional, Protocol, Tuple
from urllib.parse import urlparse

from .communication_core import Component, NotificationError, ValidationError
from .messaging.message_queue import AsyncMessageQueue, QueueMessage
from .protocol_handler import MessageType, ProtocolHandler
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class NotificationLevel(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


_LEVEL_ORDER = {NotificationLevel.INFO: 0, NotificationLevel.WARNING: 1,
                NotificationLevel.ERROR: 2, NotificationLevel.CRITICAL: 3}


@dataclass(frozen=True, eq=False)
class Notification:
    """``channels`` empty means "all registered channels"."""

    recipient: str
    title: str
    body: str = ""
    level: NotificationLevel = NotificationLevel.INFO
    channels: Tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)
    dedupe_key: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "recipient": self.recipient, "title": self.title, "body": self.body,
                "level": self.level.value, "data": dict(self.data), "created_at": self.created_at}


class RecipientOfflineError(NotificationError):
    """Raised by a channel when the recipient cannot be reached right now (not retried)."""
    code, status = "recipient_offline", 202


class NotificationChannel(Protocol):
    name: str

    async def send(self, notification: Notification) -> None:
        """Deliver or raise. Raise ``RecipientOfflineError`` for offline users; any
        other exception triggers a queued retry."""


class WebSocketChannel:
    name = "websocket"

    def __init__(self, ws: WebSocketManager, protocol: ProtocolHandler) -> None:
        self._ws, self._protocol = ws, protocol

    async def send(self, n: Notification) -> None:
        env = self._protocol.make(MessageType.EVENT, topic="notification", payload=n.to_dict())
        if await self._ws.send_to_user(n.recipient, env) == 0:
            raise RecipientOfflineError(f"{n.recipient} is offline")


class LogChannel:
    name = "log"

    async def send(self, n: Notification) -> None:
        logger.log(logging.WARNING if _LEVEL_ORDER[n.level] >= 2 else logging.INFO,
                   "notify[%s] %s: %s", n.recipient, n.title, n.body)


class CallbackChannel:
    def __init__(self, name: str, callback: Callable[[Notification], Awaitable[None]]) -> None:
        self.name, self._cb = name, callback

    async def send(self, n: Notification) -> None:
        await self._cb(n)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a: Any, **k: Any) -> None:  # type: ignore[override]
        return None


class WebhookChannel:
    """HTTPS webhook with HMAC-SHA256 signature in ``X-Signature``; redirects refused."""

    def __init__(self, name: str, url: str, *, secret: Optional[bytes] = None,
                 allowed_hosts: Optional[Collection[str]] = None, timeout_s: float = 10.0) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValidationError("webhook URL must use https")
        if allowed_hosts is not None and parsed.hostname.lower() not in {h.lower() for h in allowed_hosts}:
            raise ValidationError(f"webhook host not allowed: {parsed.hostname}")
        self.name, self._url, self._secret, self._timeout = name, url, secret, timeout_s
        self._opener = urllib.request.build_opener(_NoRedirect)

    async def send(self, n: Notification) -> None:
        body = json.dumps(n.to_dict(), separators=(",", ":")).encode()
        await asyncio.to_thread(self._post, body)

    def _post(self, body: bytes) -> None:
        headers = {"Content-Type": "application/json"}
        if self._secret:
            headers["X-Signature"] = hmac.new(self._secret, body, hashlib.sha256).hexdigest()
        req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                resp.read(1024)
        except (urllib.error.URLError, OSError) as exc:
            raise NotificationError(f"webhook delivery failed: {exc}") from exc


@dataclass
class Preferences:
    muted_channels: FrozenSet[str] = frozenset()
    min_level: NotificationLevel = NotificationLevel.INFO


class NotificationManager(Component):
    """Fan-out to channels through a retrying queue.

    Each (notification, channel) pair is an independent queued job, so a failing
    channel is retried with backoff (then dead-lettered) without affecting others.
    Undeliverable notifications for offline users are kept in a bounded inbox and
    flushed by :meth:`deliver_pending` when the user reconnects.
    """

    def __init__(self, queue: Optional[AsyncMessageQueue] = None, *, concurrency: int = 4,
                 dedupe_window_s: float = 60.0, inbox_size: int = 100) -> None:
        super().__init__()
        self._queue = queue or AsyncMessageQueue("notifications", max_attempts=4)
        self._concurrency = concurrency
        self._dedupe_window = dedupe_window_s
        self._inbox_size = inbox_size
        self._channels: Dict[str, NotificationChannel] = {}
        self._prefs: Dict[str, Preferences] = {}
        self._inbox: Dict[str, Deque[Notification]] = {}
        self._seen: Dict[str, float] = {}
        self._stats = {"accepted": 0, "delivered": 0, "deduped": 0, "filtered": 0, "stored_offline": 0}

    def add_channel(self, channel: NotificationChannel) -> None:
        self._channels[channel.name] = channel

    def set_preferences(self, user_id: str, prefs: Preferences) -> None:
        self._prefs[user_id] = prefs

    async def _on_start(self) -> None:
        self._queue.start_workers(self._deliver, concurrency=self._concurrency, handler_timeout_s=30.0)

    async def _on_stop(self) -> None:
        await self._queue.stop()

    async def notify(self, n: Notification) -> str:
        """Validate, dedupe, filter by preferences, and enqueue delivery jobs. Returns the id."""
        if not n.recipient or not n.title:
            raise ValidationError("recipient and title are required")
        unknown = [c for c in n.channels if c not in self._channels]
        if unknown:
            raise ValidationError(f"unknown channels: {unknown}")
        if self._is_duplicate(n):
            self._stats["deduped"] += 1
            return n.id
        prefs = self._prefs.get(n.recipient)
        if prefs and _LEVEL_ORDER[n.level] < _LEVEL_ORDER[prefs.min_level]:
            self._stats["filtered"] += 1
            return n.id
        names = [c for c in (n.channels or tuple(self._channels)) if not (prefs and c in prefs.muted_channels)]
        if not names:
            self._stats["filtered"] += 1
            return n.id
        for name in names:
            await self._queue.put(QueueMessage(payload=(n, name), topic=name,
                                               priority=_LEVEL_ORDER[n.level]))
        self._stats["accepted"] += 1
        return n.id

    async def deliver_pending(self, user_id: str) -> int:
        """Flush stored notifications (oldest first) for a user who just connected."""
        inbox = self._inbox.pop(user_id, None)
        if not inbox:
            return 0
        channel = self._channels.get("websocket")
        sent = 0
        for n in list(inbox):
            try:
                if channel is None:
                    raise RecipientOfflineError("no websocket channel")
                await channel.send(n)
                sent += 1
            except NotificationError:
                self._store(n)
        return sent

    def pending_count(self, user_id: str) -> int:
        return len(self._inbox.get(user_id, ()))

    def stats(self) -> Dict[str, Any]:
        return {**self._stats, "queue": self._queue.stats()}

    # -- internals ---------------------------------------------------------- #
    async def _deliver(self, msg: QueueMessage) -> None:
        n, channel_name = msg.payload
        channel = self._channels.get(channel_name)
        if channel is None:
            return  # channel removed; drop job
        try:
            await channel.send(n)
            self._stats["delivered"] += 1
        except RecipientOfflineError:
            self._store(n)

    def _store(self, n: Notification) -> None:
        box = self._inbox.setdefault(n.recipient, deque(maxlen=self._inbox_size))
        if all(x.id != n.id for x in box):
            box.append(n)
            self._stats["stored_offline"] += 1

    def _is_duplicate(self, n: Notification) -> bool:
        if not n.dedupe_key:
            return False
        now = time.monotonic()
        if len(self._seen) > 10_000:
            self._seen = {k: t for k, t in self._seen.items() if t > now}
        key = f"{n.recipient}:{n.dedupe_key}"
        if self._seen.get(key, 0.0) > now:
            return True
        self._seen[key] = now + self._dedupe_window
        return False
