"""Failover & sync agent.

Behaviour
---------
* Online and backlog empty  -> send straight to the cloud.
* Offline, send fails, or backlog exists -> persist to the local outbox.
  (Queuing while a backlog exists preserves ordering: new requests never
  overtake older queued ones.)
* When connectivity returns, the outbox is drained in FIFO order. A transient
  failure stops the drain (and backs off) to keep order; a permanent failure
  (e.g. HTTP 4xx) parks that one item as ``dead`` and continues.

Delivery is at-least-once: the idempotency key is sent with every request and
the cloud API must de-duplicate on it.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from local_store import LocalStore, LocalStoreError, OutboxItem
from network_monitor import ConnectionStatus, NetworkMonitor

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- errors
class SyncError(Exception):
    pass


class TransientSyncError(SyncError):
    """Worth retrying later: network error, timeout, 5xx, 408, 429."""


class PermanentSyncError(SyncError):
    """Retrying will not help: validation error, 4xx, etc."""


# --------------------------------------------------------------- cloud client
class CloudClient(Protocol):
    async def send(self, item: OutboxItem) -> None:
        """Deliver one item. Raise TransientSyncError / PermanentSyncError on failure."""
        ...


class HttpCloudClient:
    """Default client: POST {base_url}/{operation} with an Idempotency-Key header.

    Requires ``httpx`` (pip install httpx). Swap in your own CloudClient for
    other transports (gRPC, a DB driver, etc.).
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 10.0) -> None:
        import httpx  # local import keeps the module importable without httpx

        self._httpx = httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=headers)

    async def send(self, item: OutboxItem) -> None:
        httpx = self._httpx
        try:
            resp = await self._client.post(
                f"/{item.operation.lstrip('/')}",
                json=item.payload,
                headers={"Idempotency-Key": item.idempotency_key},
            )
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise TransientSyncError(f"{type(exc).__name__}: {exc}") from exc

        if resp.status_code < 300:
            return
        message = f"HTTP {resp.status_code}: {resp.text[:200]}"
        if resp.status_code >= 500 or resp.status_code in (408, 425, 429):
            raise TransientSyncError(message)
        raise PermanentSyncError(message)

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------- config
@dataclass(frozen=True)
class SyncConfig:
    batch_size: int = 50
    poll_interval: float = 5.0      # worker wake-up when no event arrives
    backoff_base: float = 2.0       # seconds; doubles per failed attempt
    backoff_max: float = 300.0


@dataclass(frozen=True)
class SubmitResult:
    status: Literal["sent", "queued"]
    idempotency_key: str


# -------------------------------------------------------------------- manager
class SyncManager:
    def __init__(
        self,
        monitor: NetworkMonitor,
        store: LocalStore,
        client: CloudClient,
        config: SyncConfig | None = None,
    ) -> None:
        self._monitor = monitor
        self._store = store
        self._client = client
        self._cfg = config or SyncConfig()
        self._drain_lock = asyncio.Lock()
        self._trigger = asyncio.Event()
        self._stopping = False
        self._worker: asyncio.Task[None] | None = None

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Start after ``store.open()`` and ``monitor.start()``."""
        if self._worker is not None:
            return
        self._stopping = False
        self._monitor.add_listener(self._on_status_change)
        self._worker = asyncio.create_task(self._run(), name="sync-worker")
        self._trigger.set()  # drain anything left over from a previous run

    async def stop(self) -> None:
        self._stopping = True
        self._monitor.remove_listener(self._on_status_change)
        self._trigger.set()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    # ------------------------------------------------------------------ API
    @property
    def mode(self) -> str:
        return "online" if self._monitor.is_online else "offline"

    async def submit(
        self, operation: str, payload: dict[str, Any], idempotency_key: str | None = None
    ) -> SubmitResult:
        """Send now if possible, otherwise queue durably. Never loses a request
        unless the local store itself fails (LocalStoreError is propagated)."""
        key = idempotency_key or uuid.uuid4().hex

        if self._monitor.is_online and await self._store.backlog_count() == 0:
            item = OutboxItem(id=0, idempotency_key=key, operation=operation, payload=payload)
            try:
                await self._client.send(item)
                return SubmitResult("sent", key)
            except TransientSyncError as exc:
                logger.warning("Direct send failed (%s); queueing locally", exc)
                self._monitor.report_failure()
            # PermanentSyncError propagates: the request itself is bad.

        await self._store.enqueue(operation, payload, key)
        self._trigger.set()
        return SubmitResult("queued", key)

    async def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "connection": self._monitor.status.value,
            "outbox": await self._store.stats(),
        }

    async def sync_now(self) -> None:
        """Manually trigger a drain (e.g. from an admin endpoint)."""
        if self._monitor.is_online:
            await self.drain()

    # ------------------------------------------------------------- internals
    async def _on_status_change(self, prev: ConnectionStatus, new: ConnectionStatus) -> None:
        if new is ConnectionStatus.ONLINE:
            logger.info("Back online; scheduling outbox sync")
            self._trigger.set()
        elif new is ConnectionStatus.OFFLINE:
            logger.warning("Switched to offline mode; requests will be queued locally")

    async def _run(self) -> None:
        while not self._stopping:
            try:
                if self._monitor.is_online:
                    await self.drain()
            except asyncio.CancelledError:
                raise
            except LocalStoreError:
                logger.exception("Local store error during sync")
            except Exception:
                logger.exception("Unexpected error in sync worker")

            try:
                await asyncio.wait_for(self._trigger.wait(), timeout=self._cfg.poll_interval)
            except asyncio.TimeoutError:
                pass
            self._trigger.clear()

    async def drain(self) -> int:
        """Send queued items in order. Returns the number delivered."""
        delivered = 0
        async with self._drain_lock:  # only one drain at a time
            while self._monitor.is_online and not self._stopping:
                batch = await self._store.peek(self._cfg.batch_size)
                if not batch:
                    break
                for item in batch:
                    if not self._monitor.is_online:
                        return delivered
                    if item.next_attempt_at > time.time():
                        return delivered  # head is backing off; keep order
                    outcome = await self._deliver(item)
                    if outcome == "ok":
                        delivered += 1
                    elif outcome == "stop":
                        return delivered
        if delivered:
            logger.info("Synced %d queued item(s)", delivered)
        return delivered

    async def _deliver(self, item: OutboxItem) -> Literal["ok", "skip", "stop"]:
        if not await self._store.claim(item.id):
            return "skip"
        try:
            await self._client.send(item)
        except asyncio.CancelledError:
            await asyncio.shield(self._store.release(item.id))
            raise
        except PermanentSyncError as exc:
            logger.error("Item %s rejected permanently: %s", item.idempotency_key, exc)
            await self._store.mark_dead(item.id, str(exc))
            return "skip"
        except Exception as exc:  # TransientSyncError and anything unexpected
            if not isinstance(exc, TransientSyncError):
                logger.exception("Unexpected error sending %s", item.idempotency_key)
            delay = self._backoff(item.attempts)
            dead = await self._store.retry_later(item.id, str(exc), delay)
            if dead:
                logger.error("Item %s exceeded max attempts; parked as dead", item.idempotency_key)
                return "skip"
            self._monitor.report_failure()
            return "stop"

        await self._store.complete(item.id)
        return "ok"

    def _backoff(self, attempts: int) -> float:
        delay = min(self._cfg.backoff_base * (2 ** attempts), self._cfg.backoff_max)
        return delay * random.uniform(0.8, 1.2)
