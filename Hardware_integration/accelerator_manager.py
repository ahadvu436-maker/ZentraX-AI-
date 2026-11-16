"""Accelerator management: pluggable backends, capability matching and fallback dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar

from .hardware_core import (
    BaseDriver,
    BaseManager,
    DeviceInfo,
    DeviceType,
    DeviceUnavailableError,
    DriverError,
    EventBus,
    HardwareConfig,
    HardwareError,
)

logger = logging.getLogger("hardware_integration.accelerators")
T = TypeVar("T")


@dataclass(frozen=True)
class Requirements:
    """What a workload needs from a device.

    ``capabilities`` entries match when the device reports an equal value (or a truthy
    value if ``True`` is required). ``device_types`` is the priority order.
    """

    device_types: Sequence[DeviceType] = (DeviceType.ACCELERATOR, DeviceType.GPU, DeviceType.CPU)
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    min_free_memory_bytes: int = 0
    backend: Optional[str] = None


@dataclass
class DispatchResult:
    device_id: str
    result: Any
    attempts: List[Tuple[str, str]] = field(default_factory=list)  # (device_id, error) for failed tries


class AcceleratorManager(BaseManager):
    """Routes work to the best matching device across accelerators, GPUs and CPU.

    Third-party accelerator backends (TPU/NPU/FPGA ...) plug in by implementing
    ``BaseDriver`` with devices of ``DeviceType.ACCELERATOR`` and registering via
    :meth:`register_backend`.
    """

    name = "accelerator"

    def __init__(self, devices: Any, bus: Optional[EventBus] = None,
                 config: Optional[HardwareConfig] = None) -> None:
        super().__init__(bus)
        self.devices = devices
        self.config = config or HardwareConfig()

    def _on_start(self) -> None:
        logger.info("AcceleratorManager: %d accelerator(s)", len(self.accelerators()))

    def accelerators(self) -> List[DeviceInfo]:
        return self.devices.usable_devices(DeviceType.ACCELERATOR)

    def register_backend(self, driver: BaseDriver) -> int:
        """Initialise ``driver`` and add its devices; returns the number of devices added."""
        self._require_started()
        return self.devices.add_driver(driver)

    # -- selection --------------------------------------------------------- #

    @staticmethod
    def _matches(info: DeviceInfo, req: Requirements) -> bool:
        if req.backend and info.backend != req.backend:
            return False
        for key, wanted in req.capabilities.items():
            have = info.capabilities.get(key)
            if wanted is True and not have:
                return False
            if wanted is not True and have != wanted:
                return False
        return True

    def candidates(self, req: Optional[Requirements] = None) -> List[DeviceInfo]:
        """Usable devices satisfying ``req``, ordered by type priority then free memory (desc)."""
        req = req or Requirements()
        ranked: List[Tuple[int, int, DeviceInfo]] = []
        for priority, dtype in enumerate(req.device_types):
            for info in self.devices.usable_devices(dtype):
                if not self._matches(info, req):
                    continue
                try:
                    free, _ = self.devices.memory_info(info.device_id)
                except (HardwareError, DriverError):
                    continue
                if free >= req.min_free_memory_bytes:
                    ranked.append((priority, -free, info))
        ranked.sort(key=lambda t: (t[0], t[1]))
        return [info for _, _, info in ranked]

    def select(self, req: Optional[Requirements] = None) -> DeviceInfo:
        found = self.candidates(req)
        if not found:
            raise DeviceUnavailableError(f"No usable device satisfies {req or Requirements()}")
        return found[0]

    # -- dispatch ---------------------------------------------------------- #

    def dispatch(self, fn: Callable[..., T], *args: Any, requirements: Optional[Requirements] = None,
                 timeout: Optional[float] = None, fallback: bool = True, pass_device: bool = False,
                 **kwargs: Any) -> DispatchResult:
        """Run ``fn`` on the best device, falling back down the candidate list on failure.

        With ``pass_device=True`` the chosen device id is passed as the first argument.
        Raises the last error if every candidate fails.
        """
        self._require_started()
        order = self.candidates(requirements)
        if not order:
            raise DeviceUnavailableError("No usable device satisfies the requirements")
        attempts: List[Tuple[str, str]] = []
        last: Optional[HardwareError] = None
        for info in order if fallback else order[:1]:
            call_args = (info.device_id, *args) if pass_device else args
            try:
                result = self.devices.execute(info.device_id, fn, *call_args, timeout=timeout, **kwargs)
                return DispatchResult(info.device_id, result, attempts)
            except HardwareError as exc:
                attempts.append((info.device_id, f"{type(exc).__name__}: {exc}"))
                last = exc
                logger.warning("Dispatch to %s failed (%s); %s", info.device_id, exc,
                               "falling back" if fallback else "giving up")
        assert last is not None
        raise last

    def health(self) -> Dict[str, Any]:
        return {"started": self._started, "accelerators": [a.device_id for a in self.accelerators()]
                if self._started else []}
