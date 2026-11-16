"""GPU management: device selection, allocation with cross-device fallback, execution."""

from __future__ import annotations

import itertools
import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from .hardware_core import (
    BaseManager,
    DeviceInfo,
    DeviceType,
    DeviceUnavailableError,
    DriverError,
    EventBus,
    HardwareConfig,
    HardwareError,
    MemoryHandle,
    ResourceExhaustedError,
)

logger = logging.getLogger("hardware_integration.gpu")
T = TypeVar("T")


class SelectionPolicy(str, Enum):
    MOST_FREE_MEMORY = "most_free_memory"
    LEAST_UTILIZED = "least_utilized"
    ROUND_ROBIN = "round_robin"


class GPUManager(BaseManager):
    """Schedules memory and work across GPU devices.

    Works with any driver exposing ``DeviceType.GPU`` devices (CUDA today).
    With no GPUs present the manager starts fine and simply reports none.
    """

    name = "gpu"

    def __init__(self, devices: Any, memory: Any, bus: Optional[EventBus] = None,
                 config: Optional[HardwareConfig] = None) -> None:
        super().__init__(bus)
        self.devices = devices
        self.memory = memory
        self.config = config or HardwareConfig()
        self._rr = itertools.count()
        self._lock = threading.Lock()

    def _on_start(self) -> None:
        gpus = self.gpus()
        logger.info("GPUManager: %d GPU(s) available", len(gpus))

    # -- discovery --------------------------------------------------------- #

    def gpus(self) -> List[DeviceInfo]:
        return self.devices.usable_devices(DeviceType.GPU)

    def has_gpu(self) -> bool:
        return bool(self.gpus())

    def _ranked(self, policy: SelectionPolicy, min_free_bytes: int) -> List[Tuple[DeviceInfo, int]]:
        """Usable GPUs with ``free >= min_free_bytes`` ordered by ``policy`` (best first)."""
        candidates: List[Tuple[DeviceInfo, int, float]] = []
        for info in self.gpus():
            try:
                free, _ = self.devices.memory_info(info.device_id)
                util = 0.0
                if policy is SelectionPolicy.LEAST_UTILIZED:
                    util = self.devices.telemetry(info.device_id).utilization_pct or 0.0
            except (HardwareError, DriverError):
                logger.warning("Skipping %s: query failed", info.device_id, exc_info=True)
                continue
            if free >= min_free_bytes:
                candidates.append((info, free, util))
        if policy is SelectionPolicy.MOST_FREE_MEMORY:
            candidates.sort(key=lambda c: -c[1])
        elif policy is SelectionPolicy.LEAST_UTILIZED:
            candidates.sort(key=lambda c: (c[2], -c[1]))
        elif candidates:
            with self._lock:
                shift = next(self._rr) % len(candidates)
            candidates = candidates[shift:] + candidates[:shift]
        return [(info, free) for info, free, _ in candidates]

    def select_device(self, min_free_bytes: int = 0,
                      policy: SelectionPolicy = SelectionPolicy.MOST_FREE_MEMORY) -> DeviceInfo:
        """Pick the best GPU for ``policy`` with at least ``min_free_bytes`` free."""
        self._require_started()
        if not self.has_gpu():
            raise DeviceUnavailableError("No usable GPU devices")
        ranked = self._ranked(policy, min_free_bytes)
        if not ranked:
            best = max((self.devices.memory_info(g.device_id)[0] for g in self.gpus()), default=0)
            raise ResourceExhaustedError(f"No GPU has {min_free_bytes} bytes free", requested=min_free_bytes,
                                         available=best)
        return ranked[0][0]

    # -- memory ------------------------------------------------------------ #

    def allocate(self, size_bytes: int, *, device_id: Optional[str] = None, owner: str = "default", tag: str = "",
                 policy: SelectionPolicy = SelectionPolicy.MOST_FREE_MEMORY,
                 allow_fallback: bool = True) -> MemoryHandle:
        """Allocate GPU memory.

        With ``device_id`` set, only that device is tried. Otherwise devices are
        tried in ``policy`` order, falling through on ``ResourceExhaustedError``
        when ``allow_fallback`` is true.
        """
        self._require_started()
        if device_id is not None:
            return self.memory.allocate(device_id, size_bytes, owner=owner, tag=tag)
        if not self.has_gpu():
            raise DeviceUnavailableError("No usable GPU devices")
        order = [info for info, _ in self._ranked(policy, 0)]
        if not order:
            raise DeviceUnavailableError("No GPU could be queried")
        errors: List[str] = []
        for info in order if allow_fallback else order[:1]:
            try:
                return self.memory.allocate(info.device_id, size_bytes, owner=owner, tag=tag)
            except ResourceExhaustedError as exc:
                errors.append(f"{info.device_id}: {exc}")
                logger.info("GPU %s exhausted for %d bytes; trying next", info.device_id, size_bytes)
        raise ResourceExhaustedError("Allocation failed on all GPUs: " + "; ".join(errors), requested=size_bytes,
                                     available=0)

    def free(self, handle: MemoryHandle) -> None:
        self.memory.free(handle)

    # -- execution --------------------------------------------------------- #

    def run(self, fn: Callable[..., T], *args: Any, device_id: Optional[str] = None, min_free_bytes: int = 0,
            policy: SelectionPolicy = SelectionPolicy.LEAST_UTILIZED, timeout: Optional[float] = None,
            retries: int = 0, **kwargs: Any) -> T:
        """Run ``fn`` on a chosen (or policy-selected) GPU; the device is bound/synchronised by the driver."""
        self._require_started()
        target = device_id or self.select_device(min_free_bytes, policy).device_id
        return self.devices.execute(target, fn, *args, timeout=timeout, retries=retries, **kwargs)

    # -- introspection ----------------------------------------------------- #

    def stats(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for info in self.gpus():
            try:
                sample = self.devices.telemetry(info.device_id)
            except (HardwareError, DriverError):
                continue
            out.append({"device_id": info.device_id, "name": info.name,
                        "state": self.devices.state(info.device_id).value, **sample.to_dict()})
        return out

    def health(self) -> Dict[str, Any]:
        return {"started": self._started, "gpu_count": len(self.gpus()) if self._started else 0}
