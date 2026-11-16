"""Core abstractions and orchestration for hardware integration.

This module is the dependency root of the package. It defines

* the exception hierarchy (``HardwareError`` and friends),
* the data model (``DeviceInfo``, ``TelemetrySample``, ``MemoryHandle`` ...),
* ``BaseDriver`` - the low-level device contract implemented by every driver,
* ``BaseManager`` / ``EventBus`` - the lifecycle and event plumbing shared by
  all managers, and
* ``HardwareCore`` - the orchestrator that builds, starts and stops the
  managers and the monitor in dependency order.

Managers and drivers import from here; ``HardwareCore`` imports them lazily,
so there are no import cycles.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:  # pragma: no cover
    from .accelerator_manager import AcceleratorManager
    from .cpu_manager import CPUManager
    from .device_manager import DeviceManager
    from .gpu_manager import GPUManager
    from .hardware_monitor import HardwareMonitor
    from .memory_manager import MemoryManager

logger = logging.getLogger("hardware_integration")

BytesLike = Union[bytes, bytearray, memoryview]

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class HardwareError(Exception):
    """Base class for all hardware-integration errors."""

    def __init__(self, message: str, *, device_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.device_id = device_id


class DriverError(HardwareError):
    """A driver call failed."""


class DriverNotAvailableError(DriverError):
    """The driver's runtime (library, hardware) is not present on this host."""


class DeviceFaultError(DriverError):
    """A sticky/unrecoverable device fault; the device needs a reset."""


class DeviceNotFoundError(HardwareError):
    """Unknown device id."""


class DeviceUnavailableError(HardwareError):
    """Device is faulted, offline or exclusively leased."""


class AllocationError(HardwareError):
    """Memory allocation / free failed."""


class ResourceExhaustedError(AllocationError):
    """A resource (memory, cores, quota) is exhausted."""

    def __init__(self, message: str, *, device_id: Optional[str] = None,
                 requested: Optional[int] = None, available: Optional[int] = None) -> None:
        super().__init__(message, device_id=device_id)
        self.requested = requested
        self.available = available


class ExecutionError(HardwareError):
    """Work submitted to a device failed."""


class HardwareTimeoutError(ExecutionError):
    """Work submitted to a device timed out."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    ACCELERATOR = "accelerator"


class DeviceState(str, Enum):
    READY = "ready"
    BUSY = "busy"
    DEGRADED = "degraded"
    FAULTED = "faulted"
    OFFLINE = "offline"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DeviceInfo:
    device_id: str
    name: str
    device_type: DeviceType
    backend: str
    total_memory_bytes: int
    compute_units: int
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["device_type"] = self.device_type.value
        return data


@dataclass(frozen=True)
class MemoryHandle:
    """Opaque reference to a device allocation."""

    handle_id: str
    device_id: str
    size_bytes: int
    address: Optional[int] = None
    tag: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TelemetrySample:
    device_id: str
    timestamp: float
    utilization_pct: Optional[float] = None
    memory_used_bytes: Optional[int] = None
    memory_total_bytes: Optional[int] = None
    temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def memory_used_pct(self) -> Optional[float]:
        if self.memory_used_bytes is None or not self.memory_total_bytes:
            return None
        return 100.0 * self.memory_used_bytes / self.memory_total_bytes

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["memory_used_pct"] = self.memory_used_pct
        return data


@dataclass(frozen=True)
class HardwareEvent:
    kind: str
    severity: Severity
    message: str
    device_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Thresholds:
    """Alert thresholds (percent / degrees Celsius)."""

    cpu_util_warn: float = 90.0
    cpu_util_crit: float = 98.0
    mem_warn: float = 85.0
    mem_crit: float = 95.0
    temp_warn: float = 80.0
    temp_crit: float = 90.0
    gpu_mem_warn: float = 90.0
    gpu_mem_crit: float = 98.0


@dataclass(frozen=True)
class HardwareConfig:
    enable_cuda: bool = True
    monitor_interval_s: float = 1.0
    monitor_history: int = 300
    alert_cooldown_s: float = 30.0
    memory_reserve_fraction: float = 0.05  # headroom kept free on every device
    max_cpu_workers: Optional[int] = None
    thresholds: Thresholds = field(default_factory=Thresholds)

    def __post_init__(self) -> None:
        if self.monitor_interval_s <= 0:
            raise ValueError("monitor_interval_s must be > 0")
        if not 0.0 <= self.memory_reserve_fraction < 1.0:
            raise ValueError("memory_reserve_fraction must be in [0, 1)")
        if self.monitor_history < 1:
            raise ValueError("monitor_history must be >= 1")


# --------------------------------------------------------------------------- #
# Driver contract
# --------------------------------------------------------------------------- #


class BaseDriver(ABC):
    """Low-level device contract.

    Subclasses implement the ``_``-prefixed hooks plus ``enumerate_devices``,
    ``write``/``read``, ``submit``, ``synchronize``, ``telemetry``,
    ``memory_info`` and ``health_check``. The base class provides lifecycle
    idempotence, allocation bookkeeping (so ``shutdown`` never leaks) and
    guarded execution.
    """

    name: str = "base"
    backend: str = "base"

    def __init__(self) -> None:
        self._lock = threading.RLock()  # guards allocation tables / small state only
        self._lifecycle_lock = threading.RLock()  # serialises initialize()/shutdown()
        self._initialized = False
        self._allocs: Dict[str, MemoryHandle] = {}
        self._raw: Dict[str, Any] = {}

    # -- availability / lifecycle ------------------------------------------ #

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Cheap check whether the runtime for this driver exists on the host."""

    @abstractmethod
    def _initialize(self) -> None: ...

    @abstractmethod
    def _shutdown(self) -> None: ...

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        with self._lifecycle_lock:
            if self._initialized:
                return
            try:
                self._initialize()
            except HardwareError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise DriverError(f"{self.name}: initialisation failed: {exc}") from exc
            self._initialized = True
            logger.info("Driver '%s' initialised", self.name)

    def shutdown(self) -> None:
        # The table lock (_lock) is deliberately NOT held here: draining executors
        # may need it from worker threads (e.g. future done-callbacks).
        with self._lifecycle_lock:
            if not self._initialized:
                return
            for handle in self.live_allocations():
                try:
                    self.free(handle)
                except HardwareError:
                    logger.warning("Could not free %s during shutdown", handle.handle_id, exc_info=True)
            try:
                self._shutdown()
            finally:
                self._initialized = False
                logger.info("Driver '%s' shut down", self.name)

    def __enter__(self) -> "BaseDriver":
        self.initialize()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise DriverError(f"Driver '{self.name}' is not initialised")

    # -- device discovery / state ------------------------------------------ #

    @abstractmethod
    def enumerate_devices(self) -> List[DeviceInfo]: ...

    @abstractmethod
    def memory_info(self, device_id: str) -> Tuple[int, int]:
        """Return ``(free_bytes, total_bytes)``."""

    @abstractmethod
    def telemetry(self, device_id: str) -> TelemetrySample: ...

    @abstractmethod
    def health_check(self, device_id: str) -> DeviceState: ...

    def reset(self, device_id: str) -> None:  # optional capability
        raise DriverError(f"Driver '{self.name}' does not support device reset", device_id=device_id)

    # -- memory ------------------------------------------------------------ #

    @abstractmethod
    def _allocate(self, device_id: str, size_bytes: int) -> Tuple[Optional[int], Any]:
        """Allocate; return ``(address_or_None, raw_object)``."""

    @abstractmethod
    def _free(self, handle: MemoryHandle, raw: Any) -> None: ...

    @abstractmethod
    def write(self, handle: MemoryHandle, data: BytesLike, offset: int = 0) -> None: ...

    @abstractmethod
    def read(self, handle: MemoryHandle, size: Optional[int] = None, offset: int = 0) -> bytes: ...

    def allocate(self, device_id: str, size_bytes: int, tag: str = "") -> MemoryHandle:
        self._require_initialized()
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise AllocationError(f"Invalid allocation size: {size_bytes!r}", device_id=device_id)
        address, raw = self._allocate(device_id, size_bytes)
        handle = MemoryHandle(uuid.uuid4().hex[:16], device_id, size_bytes, address, tag)
        with self._lock:
            self._allocs[handle.handle_id] = handle
            self._raw[handle.handle_id] = raw
        return handle

    def free(self, handle: MemoryHandle) -> None:
        with self._lock:
            if handle.handle_id not in self._allocs:
                raise AllocationError(f"Unknown or already-freed handle {handle.handle_id}",
                                      device_id=handle.device_id)
            raw = self._raw[handle.handle_id]
        self._free(handle, raw)  # may raise; bookkeeping is only dropped on success
        with self._lock:
            self._allocs.pop(handle.handle_id, None)
            self._raw.pop(handle.handle_id, None)

    def allocated_bytes(self, device_id: Optional[str] = None) -> int:
        with self._lock:
            return sum(h.size_bytes for h in self._allocs.values()
                       if device_id is None or h.device_id == device_id)

    def live_allocations(self) -> List[MemoryHandle]:
        with self._lock:
            return list(self._allocs.values())

    def _check_access(self, handle: MemoryHandle, offset: int, length: int) -> Any:
        with self._lock:
            raw = self._raw.get(handle.handle_id)
        if raw is None:
            raise AllocationError(f"Handle {handle.handle_id} is not live", device_id=handle.device_id)
        if offset < 0 or length < 0 or offset + length > handle.size_bytes:
            raise AllocationError(
                f"Access [{offset}, {offset + length}) out of bounds for {handle.size_bytes}-byte allocation",
                device_id=handle.device_id)
        return raw

    # -- execution --------------------------------------------------------- #

    @abstractmethod
    def submit(self, device_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> "Future[Any]":
        """Run ``fn`` on the device asynchronously; failures surface as ``ExecutionError``."""

    @abstractmethod
    def synchronize(self, device_id: str) -> None: ...

    def execute(self, device_id: str, fn: Callable[..., Any], *args: Any,
                timeout: Optional[float] = None, **kwargs: Any) -> Any:
        """Synchronous ``submit`` with timeout handling."""
        future = self.submit(device_id, fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            raise HardwareTimeoutError(f"Execution on {device_id} timed out after {timeout}s",
                                       device_id=device_id) from None

    @staticmethod
    def _guarded(device_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``fn``, wrapping foreign exceptions in ``ExecutionError``."""
        try:
            return fn(*args, **kwargs)
        except HardwareError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(f"{type(exc).__name__}: {exc}", device_id=device_id) from exc


# --------------------------------------------------------------------------- #
# Events and manager base
# --------------------------------------------------------------------------- #

EventCallback = Callable[[HardwareEvent], None]


class EventBus:
    """Thread-safe publish/subscribe with callback error isolation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: List[EventCallback] = []

    def subscribe(self, callback: EventCallback) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def publish(self, event: HardwareEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:  # noqa: BLE001 - a bad subscriber must not break hardware code
                logger.exception("Event subscriber failed for %s", event.kind)


class BaseManager(ABC):
    """Lifecycle + event plumbing shared by all managers."""

    name: str = "manager"

    def __init__(self, bus: Optional[EventBus] = None) -> None:
        self.bus = bus or EventBus()
        self._started = False
        self._lifecycle_lock = threading.RLock()

    @property
    def started(self) -> bool:
        return self._started

    @abstractmethod
    def _on_start(self) -> None: ...

    def _on_stop(self) -> None:
        return None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return
            self._on_start()
            self._started = True

    def stop(self) -> None:
        with self._lifecycle_lock:
            if not self._started:
                return
            try:
                self._on_stop()
            finally:
                self._started = False

    def _require_started(self) -> None:
        if not self._started:
            raise HardwareError(f"Manager '{self.name}' is not started")

    def health(self) -> Dict[str, Any]:
        return {"started": self._started}

    def _publish(self, kind: str, severity: Severity, message: str,
                 device_id: Optional[str] = None, **data: Any) -> None:
        self.bus.publish(HardwareEvent(kind, severity, message, device_id, data))


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def default_drivers(config: HardwareConfig) -> List[BaseDriver]:
    """CPU driver always; CUDA driver when enabled and the runtime is present."""
    from .drivers.cpu_driver import CPUDriver
    from .drivers.cuda_driver import CUDADriver

    drivers: List[BaseDriver] = [CPUDriver(max_workers=config.max_cpu_workers)]
    if config.enable_cuda and CUDADriver.is_available():
        drivers.append(CUDADriver())
    return drivers


class HardwareCore:
    """Builds the managers, starts them in dependency order and exposes them.

    Start order: devices -> cpu -> memory -> gpu -> accelerators -> monitor.
    If any manager fails to start, those already started are stopped (in
    reverse order) before the error is re-raised.
    """

    def __init__(self, config: Optional[HardwareConfig] = None,
                 drivers: Optional[Sequence[BaseDriver]] = None) -> None:
        from .accelerator_manager import AcceleratorManager
        from .cpu_manager import CPUManager
        from .device_manager import DeviceManager
        from .gpu_manager import GPUManager
        from .hardware_monitor import HardwareMonitor
        from .memory_manager import MemoryManager

        self.config = config or HardwareConfig()
        self.bus = EventBus()
        self.devices: "DeviceManager" = DeviceManager(
            list(drivers) if drivers is not None else default_drivers(self.config), self.bus, self.config)
        self.cpu: "CPUManager" = CPUManager(self.devices, self.bus, self.config)
        self.memory: "MemoryManager" = MemoryManager(self.devices, self.bus, self.config)
        self.gpu: "GPUManager" = GPUManager(self.devices, self.memory, self.bus, self.config)
        self.accelerators: "AcceleratorManager" = AcceleratorManager(self.devices, self.bus, self.config)
        self.monitor: "HardwareMonitor" = HardwareMonitor(self.devices, self.bus, self.config)
        self._managers: List[BaseManager] = [self.devices, self.cpu, self.memory, self.gpu,
                                             self.accelerators, self.monitor]
        self._running = False

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> "HardwareCore":
        if self._running:
            return self
        started: List[BaseManager] = []
        try:
            for manager in self._managers:
                manager.start()
                started.append(manager)
        except Exception as exc:
            logger.error("Startup failed in '%s': %s", manager.name, exc)
            self._stop_all(reversed(started))
            if isinstance(exc, HardwareError):
                raise
            raise HardwareError(f"Startup failed in manager '{manager.name}': {exc}") from exc
        self._running = True
        return self

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_all(reversed(self._managers))
        self._running = False

    @staticmethod
    def _stop_all(managers: Any) -> None:
        for manager in managers:
            try:
                manager.stop()
            except Exception:  # noqa: BLE001 - keep shutting down the rest
                logger.exception("Error stopping manager '%s'", manager.name)

    def __enter__(self) -> "HardwareCore":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- introspection ----------------------------------------------------- #

    def subscribe(self, callback: EventCallback) -> Callable[[], None]:
        """Subscribe to all hardware events (alerts, state changes, ...)."""
        return self.bus.subscribe(callback)

    def health(self) -> Dict[str, Any]:
        return {m.name: m.health() for m in self._managers}

    def snapshot(self) -> Dict[str, Any]:
        """Point-in-time view of devices, memory, latest telemetry and recent alerts."""
        return {
            "timestamp": time.time(),
            "devices": [{**d.to_dict(), "state": self.devices.state(d.device_id).value}
                        for d in self.devices.list_devices()],
            "memory": self.memory.stats(),
            "telemetry": {k: v.to_dict() for k, v in self.monitor.latest().items()},
            "alerts": [a.to_dict() for a in self.monitor.recent_alerts(10)],
            "health": self.health(),
        }
