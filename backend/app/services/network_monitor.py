"""Asynchronous network connectivity monitor.

Probes a set of TCP endpoints concurrently. The link is considered up if ANY
probe succeeds. Status changes use hysteresis (N consecutive failures/successes)
so a single dropped packet does not flip the system between modes.

Tip: include your own cloud API host in ``probes`` -- "internet is up" and
"our backend is reachable" are not always the same thing.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class ConnectionStatus(str, enum.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Probe:
    host: str
    port: int = 443


@dataclass(frozen=True)
class MonitorConfig:
    probes: tuple[Probe, ...] = field(
        default_factory=lambda: (Probe("1.1.1.1", 443), Probe("8.8.8.8", 443))
    )
    probe_timeout: float = 3.0
    online_interval: float = 10.0   # seconds between checks while online
    offline_interval: float = 3.0   # check faster while offline to recover quickly
    failure_threshold: int = 2      # consecutive failed checks before OFFLINE
    recovery_threshold: int = 2     # consecutive good checks before ONLINE
    jitter: float = 0.1             # +/- fraction applied to intervals


# (previous_status, new_status)
StatusListener = Callable[[ConnectionStatus, ConnectionStatus], Awaitable[None]]


class NetworkMonitor:
    def __init__(self, config: MonitorConfig | None = None) -> None:
        self._cfg = config or MonitorConfig()
        if not self._cfg.probes:
            raise ValueError("At least one probe is required")
        self._status = ConnectionStatus.UNKNOWN
        self._ok_streak = 0
        self._fail_streak = 0
        self._listeners: list[StatusListener] = []
        self._notify_tasks: set[asyncio.Task[None]] = set()
        self._online_event = asyncio.Event()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    # ------------------------------------------------------------------ API
    @property
    def status(self) -> ConnectionStatus:
        return self._status

    @property
    def is_online(self) -> bool:
        return self._status is ConnectionStatus.ONLINE

    def add_listener(self, listener: StatusListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: StatusListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def start(self) -> None:
        """Run an initial check (so status is known) and start the loop."""
        if self._task is not None:
            return
        self._stopping = False
        self._apply_result(await self.check_once(), initial=True)
        self._task = asyncio.create_task(self._run(), name="network-monitor")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for task in list(self._notify_tasks):
            task.cancel()

    def report_failure(self) -> None:
        """Called by other components after a network-looking error.

        Triggers an immediate re-check instead of waiting for the next tick.
        """
        self._wake.set()

    async def wait_until_online(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._online_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def check_once(self) -> bool:
        """Probe all endpoints concurrently; True if at least one is reachable."""
        results = await asyncio.gather(
            *(self._probe(p) for p in self._cfg.probes), return_exceptions=True
        )
        return any(r is True for r in results)

    # ------------------------------------------------------------ internals
    async def _probe(self, probe: Probe) -> bool:
        writer: asyncio.StreamWriter | None = None
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(probe.host, probe.port),
                timeout=self._cfg.probe_timeout,
            )
            return True
        except (OSError, asyncio.TimeoutError):
            return False
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass

    async def _run(self) -> None:
        while not self._stopping:
            try:
                self._apply_result(await self.check_once())
            except asyncio.CancelledError:
                raise
            except Exception:  # never let the monitor die
                logger.exception("Unexpected error in connectivity check")

            base = (
                self._cfg.online_interval if self.is_online else self._cfg.offline_interval
            )
            delay = base * (1 + random.uniform(-self._cfg.jitter, self._cfg.jitter))
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    def _apply_result(self, ok: bool, initial: bool = False) -> None:
        if ok:
            self._ok_streak += 1
            self._fail_streak = 0
        else:
            self._fail_streak += 1
            self._ok_streak = 0

        new = self._status
        if initial:
            new = ConnectionStatus.ONLINE if ok else ConnectionStatus.OFFLINE
        elif self._status is not ConnectionStatus.ONLINE and self._ok_streak >= self._cfg.recovery_threshold:
            new = ConnectionStatus.ONLINE
        elif self._status is not ConnectionStatus.OFFLINE and self._fail_streak >= self._cfg.failure_threshold:
            new = ConnectionStatus.OFFLINE
        self._set_status(new)

    def _set_status(self, new: ConnectionStatus) -> None:
        prev = self._status
        if new is prev:
            return
        self._status = new
        if new is ConnectionStatus.ONLINE:
            self._online_event.set()
        else:
            self._online_event.clear()
        logger.warning("Connectivity changed: %s -> %s", prev.value, new.value)
        for listener in list(self._listeners):
            task = asyncio.create_task(self._safe_notify(listener, prev, new))
            self._notify_tasks.add(task)
            task.add_done_callback(self._notify_tasks.discard)

    @staticmethod
    async def _safe_notify(
        listener: StatusListener, prev: ConnectionStatus, new: ConnectionStatus
    ) -> None:
        try:
            await listener(prev, new)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Connectivity listener failed")
