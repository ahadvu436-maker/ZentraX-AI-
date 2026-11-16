"""Cross-device memory management.

Adds policy on top of the drivers' raw allocators: per-device quotas, a
headroom reserve, per-owner accounting, reclaim callbacks that are invoked
before an allocation is refused, and memory-pressure classification.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

from .drivers.cpu_driver import read_system_memory
from .hardware_core import (
    AllocationError,
    BaseManager,
    DeviceInfo,
    DriverError,
    EventBus,
    HardwareConfig,
    HardwareError,
    MemoryHandle,
    ResourceExhaustedError,
    Severity,
)

logger = logging.getLogger("hardware_integration.memory")

Reclaimer = Callable[[int, str], int]  # (bytes_needed, device_id) -> bytes_freed


class PressureLevel(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


@dataclass(frozen=True)
class MemoryStats:
    device_id: str
    total_bytes: int
    free_bytes: int
    managed_bytes: int  # bytes allocated through this manager
    quota_bytes: Optional[int]
    pressure: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryManager(BaseManager):
    """Policy-enforcing allocator front-end for all devices."""

    name = "memory"

    def __init__(self, devices: "Any", bus: Optional[EventBus] = None,
                 config: Optional[HardwareConfig] = None) -> None:
        super().__init__(bus)
        self.devices = devices
        self.config = config or HardwareConfig()
        self._lock = threading.RLock()
        self._owner: Dict[str, str] = {}  # handle_id -> owner
        self._handles: Dict[str, MemoryHandle] = {}
        self._quotas: Dict[str, int] = {}
        self._reclaimers: List[Reclaimer] = []

    def _on_start(self) -> None:
        return None

    def _on_stop(self) -> None:
        self.free_all()

    # -- configuration ----------------------------------------------------- #

    def set_quota(self, device_id: str, limit_bytes: Optional[int]) -> None:
        """Cap bytes this manager may hold on ``device_id`` (``None`` removes the cap)."""
        self.devices.get(device_id)
        with self._lock:
            if limit_bytes is None:
                self._quotas.pop(device_id, None)
            elif limit_bytes < 0:
                raise ValueError("quota must be >= 0")
            else:
                self._quotas[device_id] = limit_bytes

    def register_reclaimer(self, reclaimer: Reclaimer) -> Callable[[], None]:
        """Register ``reclaimer(bytes_needed, device_id) -> bytes_freed``; returns an unregister callable."""
        with self._lock:
            self._reclaimers.append(reclaimer)

        def unregister() -> None:
            with self._lock:
                if reclaimer in self._reclaimers:
                    self._reclaimers.remove(reclaimer)

        return unregister

    # -- allocation -------------------------------------------------------- #

    def _managed(self, device_id: str) -> int:
        with self._lock:
            return sum(h.size_bytes for h in self._handles.values() if h.device_id == device_id)

    def _reclaim(self, needed: int, device_id: str) -> int:
        with self._lock:
            reclaimers = list(self._reclaimers)
        freed = 0
        for reclaimer in reclaimers:  # called without the lock: reclaimers may call free()
            try:
                freed += max(0, int(reclaimer(needed - freed, device_id)))
            except Exception:  # noqa: BLE001
                logger.exception("Reclaimer failed")
            if freed >= needed:
                break
        if freed:
            logger.info("Reclaimed %d bytes on %s", freed, device_id)
        return freed

    def _headroom(self, device_id: str) -> int:
        free, total = self.devices.memory_info(device_id)
        return max(0, free - int(total * self.config.memory_reserve_fraction))

    def allocate(self, device_id: str, size_bytes: int, *, owner: str = "default", tag: str = "") -> MemoryHandle:
        """Allocate on ``device_id`` honouring quota, reserve and reclaimers."""
        driver = self.devices.ensure_usable(device_id)
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise AllocationError(f"Invalid allocation size: {size_bytes!r}", device_id=device_id)

        with self._lock:
            quota = self._quotas.get(device_id)
        if quota is not None and self._managed(device_id) + size_bytes > quota:
            raise ResourceExhaustedError(
                f"Quota exceeded on {device_id}: {self._managed(device_id)} + {size_bytes} > {quota}",
                device_id=device_id, requested=size_bytes, available=max(0, quota - self._managed(device_id)))

        headroom = self._headroom(device_id)
        if size_bytes > headroom:
            self._reclaim(size_bytes - headroom, device_id)
            headroom = self._headroom(device_id)
            if size_bytes > headroom:
                self._publish("memory_exhausted", Severity.WARNING,
                              f"{device_id}: cannot allocate {size_bytes} bytes ({headroom} available)",
                              device_id, requested=size_bytes, available=headroom)
                raise ResourceExhaustedError(
                    f"Insufficient memory on {device_id}: requested {size_bytes}, available {headroom}",
                    device_id=device_id, requested=size_bytes, available=headroom)
        try:
            handle = driver.allocate(device_id, size_bytes, tag)
        except ResourceExhaustedError:
            self._reclaim(size_bytes, device_id)  # one retry after reclaiming
            handle = driver.allocate(device_id, size_bytes, tag)
        with self._lock:
            self._handles[handle.handle_id] = handle
            self._owner[handle.handle_id] = owner
        return handle

    def free(self, handle: MemoryHandle) -> None:
        with self._lock:
            if handle.handle_id not in self._handles:
                raise AllocationError(f"Handle {handle.handle_id} was not allocated by this manager",
                                      device_id=handle.device_id)
        self.devices.driver_for(handle.device_id).free(handle)
        with self._lock:
            self._handles.pop(handle.handle_id, None)
            self._owner.pop(handle.handle_id, None)

    def free_owner(self, owner: str) -> int:
        """Free everything held by ``owner``; returns bytes released."""
        with self._lock:
            targets = [self._handles[h] for h, o in self._owner.items() if o == owner]
        return self._free_many(targets)

    def free_all(self) -> int:
        with self._lock:
            targets = list(self._handles.values())
        return self._free_many(targets)

    def _free_many(self, handles: List[MemoryHandle]) -> int:
        released = 0
        for handle in handles:
            try:
                self.free(handle)
                released += handle.size_bytes
            except (AllocationError, DriverError):
                logger.warning("Could not free %s", handle.handle_id, exc_info=True)
        return released

    @contextmanager
    def scoped(self, device_id: str, size_bytes: int, *, owner: str = "scoped", tag: str = "") -> Iterator[MemoryHandle]:
        """Allocate for the duration of a ``with`` block."""
        handle = self.allocate(device_id, size_bytes, owner=owner, tag=tag)
        try:
            yield handle
        finally:
            try:
                self.free(handle)
            except HardwareError:
                logger.warning("Scoped free failed for %s", handle.handle_id, exc_info=True)

    # -- introspection ----------------------------------------------------- #

    def owner_usage(self, owner: str) -> Dict[str, int]:
        usage: Dict[str, int] = {}
        with self._lock:
            for hid, o in self._owner.items():
                if o == owner:
                    h = self._handles[hid]
                    usage[h.device_id] = usage.get(h.device_id, 0) + h.size_bytes
        return usage

    def pressure(self, device_id: str) -> PressureLevel:
        free, total = self.devices.memory_info(device_id)
        used_pct = 100.0 * (total - free) / total if total else 0.0
        thresholds = self.config.thresholds
        is_gpu = device_id.split(":")[0] != "cpu"
        warn = thresholds.gpu_mem_warn if is_gpu else thresholds.mem_warn
        crit = thresholds.gpu_mem_crit if is_gpu else thresholds.mem_crit
        return PressureLevel.CRITICAL if used_pct >= crit else PressureLevel.ELEVATED if used_pct >= warn \
            else PressureLevel.NORMAL

    def system_memory(self) -> Dict[str, int]:
        available, total = read_system_memory()
        return {"available_bytes": available, "total_bytes": total}

    def stats(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for info in self.devices.usable_devices():
            try:
                free, total = self.devices.memory_info(info.device_id)
                level = self.pressure(info.device_id).value
            except (DriverError, AllocationError):
                continue
            with self._lock:
                quota = self._quotas.get(info.device_id)
            out.append(MemoryStats(info.device_id, total, free, self._managed(info.device_id), quota, level).to_dict())
        return out

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {"started": self._started, "live_allocations": len(self._handles),
                    "managed_bytes": sum(h.size_bytes for h in self._handles.values())}

