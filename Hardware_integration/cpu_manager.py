"""CPU management: topology, core reservation, thread affinity and task execution."""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar

from .hardware_core import (
    BaseManager,
    DeviceType,
    EventBus,
    HardwareConfig,
    HardwareError,
    ResourceExhaustedError,
)

try:
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger("hardware_integration.cpu_manager")
T = TypeVar("T")


@dataclass(frozen=True)
class CPUTopology:
    model: str
    logical_cores: int
    physical_cores: int
    usable_cores: Tuple[int, ...]  # cores this process may run on (affinity mask)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CPUManager(BaseManager):
    """Manages the host CPU device (``DeviceType.CPU``)."""

    name = "cpu"

    def __init__(self, devices: Any, bus: Optional[EventBus] = None,
                 config: Optional[HardwareConfig] = None) -> None:
        super().__init__(bus)
        self.devices = devices
        self.config = config or HardwareConfig()
        self._lock = threading.RLock()
        self._device_id: Optional[str] = None
        self._reservations: Dict[str, FrozenSet[int]] = {}

    def _on_start(self) -> None:
        cpus = self.devices.list_devices(DeviceType.CPU)
        if not cpus:
            raise HardwareError("CPUManager: no CPU device registered")
        self._device_id = cpus[0].device_id

    def _on_stop(self) -> None:
        with self._lock:
            self._reservations.clear()

    @property
    def device_id(self) -> str:
        if self._device_id is None:
            raise HardwareError("CPUManager is not started")
        return self._device_id

    # -- topology ---------------------------------------------------------- #

    def usable_cores(self) -> Tuple[int, ...]:
        if hasattr(os, "sched_getaffinity"):
            return tuple(sorted(os.sched_getaffinity(0)))
        return tuple(range(os.cpu_count() or 1))

    def topology(self) -> CPUTopology:
        info = self.devices.get(self.device_id)
        logical = os.cpu_count() or 1
        physical = (psutil.cpu_count(logical=False) if psutil is not None else None) or logical
        return CPUTopology(info.name, logical, physical, self.usable_cores())

    # -- core reservation & affinity -------------------------------------- #

    def reserve_cores(self, count: int, owner: str) -> FrozenSet[int]:
        """Reserve ``count`` currently-unreserved cores for ``owner`` (bookkeeping, not enforcement)."""
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._lock:
            if owner in self._reservations:
                raise HardwareError(f"'{owner}' already holds a core reservation")
            taken: Set[int] = set().union(*self._reservations.values()) if self._reservations else set()
            free = [c for c in self.usable_cores() if c not in taken]
            if len(free) < count:
                raise ResourceExhaustedError(f"Only {len(free)} unreserved cores, {count} requested",
                                             device_id=self.device_id, requested=count, available=len(free))
            cores = frozenset(free[:count])
            self._reservations[owner] = cores
            return cores

    def release_cores(self, owner: str) -> None:
        with self._lock:
            self._reservations.pop(owner, None)

    def reservations(self) -> Dict[str, List[int]]:
        with self._lock:
            return {o: sorted(c) for o, c in self._reservations.items()}

    @staticmethod
    def set_affinity(cores: Iterable[int]) -> None:
        """Pin the *calling thread* (Linux) to ``cores``."""
        if not hasattr(os, "sched_setaffinity"):
            raise HardwareError("CPU affinity is not supported on this platform")
        try:
            os.sched_setaffinity(0, set(cores))
        except (OSError, ValueError) as exc:
            raise HardwareError(f"Could not set CPU affinity: {exc}") from exc

    @contextmanager
    def pinned(self, cores: Iterable[int]) -> Iterator[None]:
        """Temporarily pin the calling thread to ``cores``; restores the previous mask."""
        if not hasattr(os, "sched_getaffinity"):
            raise HardwareError("CPU affinity is not supported on this platform")
        previous = os.sched_getaffinity(0)
        self.set_affinity(cores)
        try:
            yield
        finally:
            os.sched_setaffinity(0, previous)

    # -- execution --------------------------------------------------------- #

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> "Future[T]":
        return self.devices.submit(self.device_id, fn, *args, **kwargs)

    def execute(self, fn: Callable[..., T], *args: Any, timeout: Optional[float] = None,
                retries: int = 0, **kwargs: Any) -> T:
        return self.devices.execute(self.device_id, fn, *args, timeout=timeout, retries=retries, **kwargs)

    def parallel_map(self, fn: Callable[[Any], T], items: Sequence[Any], timeout: Optional[float] = None) -> List[T]:
        """Apply ``fn`` to ``items`` on the CPU pool, preserving order; cancels the rest on first failure."""
        futures = [self.submit(fn, item) for item in items]
        try:
            return [f.result(timeout=timeout) for f in futures]
        except BaseException:
            for f in futures:
                f.cancel()
            raise

    # -- telemetry --------------------------------------------------------- #

    def utilization(self) -> Optional[float]:
        return self.devices.telemetry(self.device_id).utilization_pct

    @staticmethod
    def load_average() -> Optional[Tuple[float, float, float]]:
        try:
            return os.getloadavg()
        except (OSError, AttributeError):
            return None

    def health(self) -> Dict[str, Any]:
        return {"started": self._started, "reservations": self.reservations() if self._started else {},
                "load_average": self.load_average()}
