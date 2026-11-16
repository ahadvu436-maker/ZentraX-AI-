"""In-process async message queue with priorities, delays, acks, retries and dead letters."""
from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple

from ..communication_core import Component, QueueClosedError, QueueError, QueueFullError

logger = logging.getLogger(__name__)


@dataclass
class QueueMessage:
    """A unit of work. ``priority``: higher is delivered first."""

    payload: Any
    topic: str = ""
    priority: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attempts: int = 0
    enqueued_at: float = field(default_factory=time.time)
    ttl_s: Optional[float] = None
    last_error: Optional[str] = None

    @property
    def expired(self) -> bool:
        return self.ttl_s is not None and time.time() - self.enqueued_at > self.ttl_s


class AsyncMessageQueue(Component):
    """At-least-once queue.

    ``get`` leases a message for ``visibility_timeout_s``; the consumer must
    ``ack`` it, or ``nack`` to retry (exponential backoff) until ``max_attempts``
    is reached, after which it is moved to the dead-letter list. Unacknowledged
    leases expire and are retried. Capacity is bounded (backpressure).
    """

    def __init__(self, name: str = "default", *, maxsize: int = 10_000, max_attempts: int = 5,
                 visibility_timeout_s: float = 30.0, backoff_base_s: float = 0.5,
                 backoff_max_s: float = 30.0, dead_letter_size: int = 1000,
                 on_dead_letter: Optional[Callable[[QueueMessage], None]] = None) -> None:
        super().__init__()
        if maxsize < 1 or max_attempts < 1 or visibility_timeout_s <= 0:
            raise ValueError("invalid queue configuration")
        self.name = name
        self._maxsize, self._max_attempts = maxsize, max_attempts
        self._vis, self._base, self._cap = visibility_timeout_s, backoff_base_s, backoff_max_s
        self._on_dead = on_dead_letter
        self._ready: List[Tuple[int, int, QueueMessage]] = []
        self._delayed: List[Tuple[float, int, QueueMessage]] = []
        self._inflight: Dict[str, Tuple[QueueMessage, float]] = {}
        self._dead: Deque[QueueMessage] = deque(maxlen=dead_letter_size)
        self._seq = itertools.count()
        self._cond = asyncio.Condition()
        self._closed = False
        self._workers: List["asyncio.Task[None]"] = []
        self._stats = {"enqueued": 0, "acked": 0, "retried": 0, "dead": 0, "expired": 0}

    # -- producer ----------------------------------------------------------- #
    async def put(self, item: Any, *, delay_s: float = 0.0, priority: Optional[int] = None,
                  block: bool = False, timeout: Optional[float] = None) -> QueueMessage:
        """Enqueue ``item`` (payload or ``QueueMessage``). Raises ``QueueFullError`` when
        full unless ``block=True`` (then waits up to ``timeout``)."""
        msg = item if isinstance(item, QueueMessage) else QueueMessage(payload=item)
        if priority is not None:
            msg.priority = priority
        async with self._cond:
            if self._closed:
                raise QueueClosedError(f"queue '{self.name}' is closed")
            if self._size() >= self._maxsize:
                if not block:
                    raise QueueFullError(f"queue '{self.name}' is full")
                try:
                    await asyncio.wait_for(
                        self._cond.wait_for(lambda: self._closed or self._size() < self._maxsize), timeout)
                except asyncio.TimeoutError:
                    raise QueueFullError(f"queue '{self.name}' is full") from None
                if self._closed:
                    raise QueueClosedError(f"queue '{self.name}' is closed")
            self._push(msg, delay_s)
            self._stats["enqueued"] += 1
            self._cond.notify_all()
        return msg

    # -- consumer ----------------------------------------------------------- #
    async def get(self, timeout: Optional[float] = None) -> Optional[QueueMessage]:
        """Lease the next message; ``None`` on timeout. Raises ``QueueClosedError`` once
        the queue is closed and fully drained."""
        deadline = None if timeout is None else time.monotonic() + timeout
        async with self._cond:
            while True:
                self._maintain()
                while self._ready:
                    _, _, msg = heapq.heappop(self._ready)
                    if msg.expired:
                        self._stats["expired"] += 1
                        continue
                    msg.attempts += 1
                    self._inflight[msg.id] = (msg, time.monotonic() + self._vis)
                    return msg
                if self._closed and not self._delayed and not self._inflight:
                    raise QueueClosedError(f"queue '{self.name}' is closed")
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    return None
                wakeups = [w - now for w in (
                    self._delayed[0][0] if self._delayed else None,
                    min((d for _, d in self._inflight.values()), default=None),
                    deadline) if w is not None]
                try:
                    if wakeups:
                        await asyncio.wait_for(self._cond.wait(), max(0.0, min(wakeups)))
                    else:
                        await self._cond.wait()
                except asyncio.TimeoutError:
                    pass

    async def ack(self, msg_id: str) -> None:
        async with self._cond:
            if self._inflight.pop(msg_id, None) is None:
                raise QueueError(f"unknown or expired message {msg_id}")
            self._stats["acked"] += 1
            self._cond.notify_all()

    async def nack(self, msg_id: str, *, requeue: bool = True, error: Optional[str] = None,
                   delay_s: Optional[float] = None) -> None:
        async with self._cond:
            item = self._inflight.pop(msg_id, None)
            if item is None:
                raise QueueError(f"unknown or expired message {msg_id}")
            msg = item[0]
            if error:
                msg.last_error = error
            if not requeue:
                self._dead_letter(msg)
            else:
                self._retry(msg, delay_s)
            self._cond.notify_all()

    async def extend(self, msg_id: str, seconds: float) -> None:
        """Extend the lease of a long-running message."""
        async with self._cond:
            item = self._inflight.get(msg_id)
            if item is None:
                raise QueueError(f"unknown or expired message {msg_id}")
            self._inflight[msg_id] = (item[0], time.monotonic() + seconds)
            self._cond.notify_all()

    # -- workers ------------------------------------------------------------ #
    def start_workers(self, handler: Callable[[QueueMessage], Awaitable[None]], *, concurrency: int = 1,
                      handler_timeout_s: Optional[float] = None, poll_s: float = 1.0) -> None:
        """Spawn ``concurrency`` consumers; success acks, an exception nacks (retry)."""
        self._running = True  # so stop() cancels the workers
        for i in range(max(1, concurrency)):
            self._workers.append(asyncio.create_task(
                self._worker(handler, handler_timeout_s, poll_s), name=f"queue-{self.name}-{i}"))

    async def _worker(self, handler: Callable[[QueueMessage], Awaitable[None]],
                      timeout: Optional[float], poll_s: float) -> None:
        while True:
            try:
                msg = await self.get(timeout=poll_s)
            except QueueClosedError:
                return
            if msg is None:
                continue
            try:
                await (asyncio.wait_for(handler(msg), timeout) if timeout else handler(msg))
                await self.ack(msg.id)
            except asyncio.CancelledError:
                try:
                    await self.nack(msg.id, delay_s=0.0, error="worker cancelled")
                except Exception:
                    pass
                raise
            except QueueError:
                pass  # lease expired while handling; message already rescheduled
            except Exception as exc:
                logger.warning("queue %s handler failed for %s: %r", self.name, msg.id, exc)
                try:
                    await self.nack(msg.id, error=repr(exc))
                except QueueError:
                    pass

    # -- dead letters / admin ----------------------------------------------- #
    def dead_letters(self) -> List[QueueMessage]:
        return list(self._dead)

    async def requeue_dead(self, msg_id: str) -> None:
        async with self._cond:
            for m in list(self._dead):
                if m.id == msg_id:
                    self._dead.remove(m)
                    m.attempts, m.last_error = 0, None
                    self._push(m, 0.0)
                    self._cond.notify_all()
                    return
        raise QueueError(f"no dead letter {msg_id}")

    async def close(self, *, drain: bool = True) -> None:
        """Reject new messages. With ``drain=False`` pending delayed retries are dropped."""
        async with self._cond:
            self._closed = True
            if not drain:
                self._delayed.clear()
            self._cond.notify_all()

    def qsize(self) -> int:
        return len(self._ready) + len(self._delayed)

    def stats(self) -> Dict[str, Any]:
        return {**self._stats, "ready": len(self._ready), "delayed": len(self._delayed),
                "inflight": len(self._inflight), "dead_letters": len(self._dead), "closed": self._closed}

    async def _on_stop(self) -> None:
        await self.close(drain=False)
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # -- internals (call with lock held) ------------------------------------ #
    def _size(self) -> int:
        return len(self._ready) + len(self._delayed) + len(self._inflight)

    def _push(self, msg: QueueMessage, delay_s: float) -> None:
        seq = next(self._seq)
        if delay_s > 0:
            heapq.heappush(self._delayed, (time.monotonic() + delay_s, seq, msg))
        else:
            heapq.heappush(self._ready, (-msg.priority, seq, msg))

    def _maintain(self) -> None:
        now = time.monotonic()
        while self._delayed and self._delayed[0][0] <= now:
            _, _, msg = heapq.heappop(self._delayed)
            self._push(msg, 0.0)
        for mid in [m for m, (_, dl) in self._inflight.items() if dl <= now]:
            msg, _ = self._inflight.pop(mid)
            msg.last_error = "visibility timeout"
            self._retry(msg, None)

    def _retry(self, msg: QueueMessage, delay_s: Optional[float]) -> None:
        if msg.attempts >= self._max_attempts:
            self._dead_letter(msg)
            return
        delay = delay_s if delay_s is not None else min(self._cap, self._base * (2 ** max(0, msg.attempts - 1)))
        self._stats["retried"] += 1
        self._push(msg, delay)

    def _dead_letter(self, msg: QueueMessage) -> None:
        self._dead.append(msg)
        self._stats["dead"] += 1
        logger.warning("queue %s: dead-lettered %s (%s)", self.name, msg.id, msg.last_error)
        if self._on_dead:
            try:
                self._on_dead(msg)
            except Exception:
                logger.exception("on_dead_letter hook failed")


class QueueManager(Component):
    """Registry of named queues sharing one lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self._queues: Dict[str, AsyncMessageQueue] = {}

    def get_or_create(self, name: str, **kwargs: Any) -> AsyncMessageQueue:
        q = self._queues.get(name)
        if q is None:
            q = self._queues[name] = AsyncMessageQueue(name, **kwargs)
        return q

    def get(self, name: str) -> AsyncMessageQueue:
        try:
            return self._queues[name]
        except KeyError:
            raise QueueError(f"unknown queue '{name}'") from None

    async def _on_stop(self) -> None:
        for q in self._queues.values():
            await q.close(drain=False)
        await asyncio.gather(*(q.stop() for q in self._queues.values()), return_exceptions=True)

    def stats(self) -> Dict[str, Any]:
        return {n: q.stats() for n, q in self._queues.items()}
