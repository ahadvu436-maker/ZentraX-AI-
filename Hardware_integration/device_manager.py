"""Device registry and routing layer.

``DeviceManager`` owns the drivers, discovers devices, tracks device state
(READY/BUSY/DEGRADED/FAULTED/OFFLINE), hands out advisory leases and routes
execution to the right driver while translating failures into state changes.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .hardware_core import (
    BaseDriver,
    BaseManager,
    DeviceFaultError,
    DeviceInfo,
    DeviceNotFoundError,
    DeviceState,
    DeviceType,
    DeviceUnavailableError,
    DriverError,
    EventBus,
    ExecutionError,
    HardwareConfig,
    HardwareError,
    HardwareTimeoutError,
    Severity,
    TelemetrySample,
)

logger = logging.getLogger("hardware_integration.devices")

_USABLE = (DeviceState.READY, DeviceState.BUSY, DeviceState.DEGRADED)
_DEGRADE_AFTER_FAILURES = 3


class DeviceLease:
    """Advisory lease on a device; use as a context manager."""

    def __init__(self, manager: "DeviceManager", device_id: str, owner: str, exclusive: bool) -> None:
        self._manager = manager
        self.device_id = device_id
        self.owner = owner
        self.exclusive = exclusive
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._manager._release(self)

    def __enter__(self) -> "DeviceLease":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


class DeviceManager(BaseManager):
    """Registry of all devices across all drivers."""

    name = "device"

    def __init__(self, drivers: Sequence[BaseDriver], bus: Optional[EventBus] = None,
                 config: Optional[HardwareConfig] = None) -> None:
        super().__init__(bus)
        self.config = config or HardwareConfig()
        self._drivers: List[BaseDriver] = list(drivers)
        self._lock = threading.RLock()
        self._devices: Dict[str, Tuple[DeviceInfo, BaseDriver]] = {}
        self._states: Dict[str, DeviceState] = {}
        self._failures: Dict[str, int] = {}
        self._leases: Dict[str, List[DeviceLease]] = {}

    # -- lifecycle --------------------------------------------------------- #

    def _on_start(self) -> None:
        for driver in list(self._drivers):
            self._register_driver(driver)
        if not self._devices:
            raise HardwareError("No usable hardware devices were discovered")

    def _on_stop(self) -> None:
        for driver in self._drivers:
            try:
                driver.shutdown()
            except HardwareError:
                logger.exception("Driver '%s' failed to shut down cleanly", driver.name)
        with self._lock:
            self._devices.clear()
            self._states.clear()
            self._leases.clear()
            self._failures.clear()

    def _register_driver(self, driver: BaseDriver) -> int:
        try:
            driver.initialize()
            infos = driver.enumerate_devices()
        except HardwareError as exc:
            logger.warning("Driver '%s' unavailable: %s", driver.name, exc)
            self._publish("driver_unavailable", Severity.WARNING, f"Driver '{driver.name}' unavailable: {exc}")
            return 0
        with self._lock:
            for info in infos:
                self._devices[info.device_id] = (info, driver)
                self._states[info.device_id] = DeviceState.READY
                self._failures[info.device_id] = 0
        logger.info("Driver '%s' registered %d device(s)", driver.name, len(infos))
        return len(infos)

    def add_driver(self, driver: BaseDriver) -> int:
        """Register a driver at runtime (e.g. a TPU/NPU backend). Returns devices added."""
        self._require_started()
        with self._lock:
            if driver not in self._drivers:
                self._drivers.append(driver)
        return self._register_driver(driver)

    # -- lookup ------------------------------------------------------------ #

    def list_devices(self, device_type: Optional[DeviceType] = None,
                     states: Optional[Iterable[DeviceState]] = None) -> List[DeviceInfo]:
        wanted: Optional[Set[DeviceState]] = set(states) if states is not None else None
        with self._lock:
            return [info for info, _ in self._devices.values()
                    if (device_type is None or info.device_type is device_type)
                    and (wanted is None or self._states[info.device_id] in wanted)]

    def usable_devices(self, device_type: Optional[DeviceType] = None) -> List[DeviceInfo]:
        return self.list_devices(device_type, _USABLE)

    def get(self, device_id: str) -> DeviceInfo:
        with self._lock:
            try:
                return self._devices[device_id][0]
            except KeyError:
                raise DeviceNotFoundError(f"Unknown device '{device_id}'", device_id=device_id) from None

    def driver_for(self, device_id: str) -> BaseDriver:
        with self._lock:
            try:
                return self._devices[device_id][1]
            except KeyError:
                raise DeviceNotFoundError(f"Unknown device '{device_id}'", device_id=device_id) from None

    def state(self, device_id: str) -> DeviceState:
        self.get(device_id)
        with self._lock:
            return self._states[device_id]

    def set_state(self, device_id: str, state: DeviceState, reason: str = "") -> None:
        self.get(device_id)
        with self._lock:
            previous = self._states[device_id]
            self._states[device_id] = state
        if previous is not state:
            severity = Severity.CRITICAL if state in (DeviceState.FAULTED, DeviceState.OFFLINE) else (
                Severity.WARNING if state is DeviceState.DEGRADED else Severity.INFO)
            self._publish("device_state", severity, f"{device_id}: {previous.value} -> {state.value} {reason}".strip(),
                          device_id, previous=previous.value, current=state.value, reason=reason)

    def ensure_usable(self, device_id: str) -> BaseDriver:
        """Return the driver for ``device_id`` or raise if the device cannot take work."""
        self._require_started()
        driver = self.driver_for(device_id)
        state = self.state(device_id)
        if state not in _USABLE:
            raise DeviceUnavailableError(f"Device {device_id} is {state.value}", device_id=device_id)
        return driver

    # -- leases ------------------------------------------------------------ #

    def acquire(self, device_id: str, owner: str, exclusive: bool = False) -> DeviceLease:
        """Take an advisory lease (non-blocking). Raises ``DeviceUnavailableError`` on conflict."""
        self.ensure_usable(device_id)
        with self._lock:
            held = self._leases.setdefault(device_id, [])
            if any(l.exclusive for l in held) or (exclusive and held):
                raise DeviceUnavailableError(f"Device {device_id} is leased by "
                                             f"{sorted({l.owner for l in held})}", device_id=device_id)
            lease = DeviceLease(self, device_id, owner, exclusive)
            held.append(lease)
            if exclusive and self._states[device_id] is DeviceState.READY:
                self._states[device_id] = DeviceState.BUSY
            return lease

    def _release(self, lease: DeviceLease) -> None:
        with self._lock:
            held = self._leases.get(lease.device_id, [])
            if lease in held:
                held.remove(lease)
            if not held and self._states.get(lease.device_id) is DeviceState.BUSY:
                self._states[lease.device_id] = DeviceState.READY

    def leases(self, device_id: str) -> List[Tuple[str, bool]]:
        with self._lock:
            return [(l.owner, l.exclusive) for l in self._leases.get(device_id, [])]

    # -- execution & queries ---------------------------------------------- #

    def submit(self, device_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> "Future[Any]":
        driver = self.ensure_usable(device_id)
        future = driver.submit(device_id, fn, *args, **kwargs)
        future.add_done_callback(lambda f: self._record_outcome(device_id, f.exception() if not f.cancelled() else None))
        return future

    def execute(self, device_id: str, fn: Callable[..., Any], *args: Any, timeout: Optional[float] = None,
                retries: int = 0, **kwargs: Any) -> Any:
        """Run ``fn`` on a device; retries only transient ``ExecutionError``/timeouts."""
        attempt = 0
        while True:
            try:
                driver = self.ensure_usable(device_id)
                result = driver.execute(device_id, fn, *args, timeout=timeout, **kwargs)
                self._record_outcome(device_id, None)
                return result
            except DeviceFaultError as exc:
                self._record_outcome(device_id, exc)
                raise
            except (ExecutionError, HardwareTimeoutError) as exc:
                self._record_outcome(device_id, exc)
                if attempt >= retries:
                    raise
                attempt += 1
                logger.warning("Retrying %s (%d/%d) after: %s", device_id, attempt, retries, exc)

    def _record_outcome(self, device_id: str, error: Optional[BaseException]) -> None:
        if error is None:
            with self._lock:
                self._failures[device_id] = 0
            if self._states.get(device_id) is DeviceState.DEGRADED:
                self.set_state(device_id, DeviceState.READY, "recovered after successful execution")
            return
        if isinstance(error, DeviceFaultError):
            self.set_state(device_id, DeviceState.FAULTED, str(error))
        elif isinstance(error, (ExecutionError, DriverError)):
            with self._lock:
                self._failures[device_id] = self._failures.get(device_id, 0) + 1
                too_many = self._failures[device_id] >= _DEGRADE_AFTER_FAILURES
            if too_many and self._states.get(device_id) is DeviceState.READY:
                self.set_state(device_id, DeviceState.DEGRADED, f"{_DEGRADE_AFTER_FAILURES} consecutive failures")

    def memory_info(self, device_id: str) -> Tuple[int, int]:
        """``(free_bytes, total_bytes)``."""
        return self.ensure_usable(device_id).memory_info(device_id)

    def telemetry(self, device_id: str) -> TelemetrySample:
        driver = self.driver_for(device_id)
        self._require_started()
        return driver.telemetry(device_id)

    # -- recovery ---------------------------------------------------------- #

    def recover(self, device_id: str) -> bool:
        """Health-check a device (resetting it if faulted and supported). True if usable afterwards."""
        self._require_started()
        driver = self.driver_for(device_id)
        state = driver.health_check(device_id) if self.state(device_id) is not DeviceState.FAULTED else DeviceState.FAULTED
        if state is DeviceState.FAULTED:
            try:
                driver.reset(device_id)
                state = driver.health_check(device_id)
            except HardwareError as exc:
                logger.error("Reset of %s failed: %s", device_id, exc)
        self.set_state(device_id, state, "after recovery attempt")
        with self._lock:
            self._failures[device_id] = 0
        return state in _USABLE

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {"started": self._started,
                    "devices": {d: s.value for d, s in self._states.items()},
                    "drivers": [d.name for d in self._drivers]}
