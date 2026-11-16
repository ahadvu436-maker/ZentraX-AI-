"""Host CPU driver.

* Memory: anonymous ``mmap`` regions (page-granular, lazily committed, with a
  real address obtainable for FFI). Failures map to ``ResourceExhaustedError``.
* Execution: bounded ``ThreadPoolExecutor``; ``synchronize`` waits for pending work.
* Telemetry: ``psutil`` when installed, otherwise ``/proc`` (Linux) / ``ctypes`` (Windows).
"""

from __future__ import annotations

import ctypes
import errno
import logging
import mmap
import os
import platform
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..hardware_core import (
    AllocationError,
    BaseDriver,
    BytesLike,
    DeviceInfo,
    DeviceNotFoundError,
    DeviceState,
    DeviceType,
    DriverError,
    MemoryHandle,
    ResourceExhaustedError,
    TelemetrySample,
)

try:  # optional dependency
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on environment
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger("hardware_integration.cpu")


# --------------------------------------------------------------------------- #
# Host probes
# --------------------------------------------------------------------------- #


def read_system_memory() -> Tuple[int, int]:
    """Return ``(available_bytes, total_bytes)`` of host RAM."""
    if psutil is not None:
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)
    try:  # Linux
        values: Dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = int(rest.split()[0]) * 1024
        if "MemTotal" in values and "MemAvailable" in values:
            return values["MemAvailable"], values["MemTotal"]
    except OSError:
        pass
    if platform.system() == "Windows":  # pragma: no cover
        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullAvailPhys), int(status.ullTotalPhys)
    raise DriverError("Cannot determine system memory on this platform; install 'psutil'")


class _CpuSampler:
    """Non-blocking CPU utilisation (percent) from successive samples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prev: Optional[Tuple[int, int]] = None
        if psutil is not None:
            psutil.cpu_percent(interval=None)  # prime

    @staticmethod
    def _proc_stat() -> Optional[Tuple[int, int]]:
        try:
            with open("/proc/stat", encoding="ascii") as fh:
                fields = [int(x) for x in fh.readline().split()[1:9]]
        except (OSError, ValueError):
            return None
        idle = fields[3] + fields[4]
        return idle, sum(fields)

    def sample(self) -> Optional[float]:
        if psutil is not None:
            return float(psutil.cpu_percent(interval=None))
        current = self._proc_stat()
        if current is not None:
            with self._lock:
                prev, self._prev = self._prev, current
            if prev is None:
                return None
            d_idle, d_total = current[0] - prev[0], current[1] - prev[1]
            return 100.0 * (1 - d_idle / d_total) if d_total > 0 else None
        try:  # coarse fallback: 1-minute load per core
            return min(100.0, 100.0 * os.getloadavg()[0] / (os.cpu_count() or 1))
        except (OSError, AttributeError):
            return None


def _cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "CPU"


def _cpu_flags() -> Set[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(("flags", "Features")):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _cpu_temperature() -> Optional[float]:
    if psutil is None or not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        readings = [t.current for entries in psutil.sensors_temperatures().values() for t in entries
                    if t.current is not None]
    except Exception:  # noqa: BLE001 - sensors are best-effort
        return None
    return float(max(readings)) if readings else None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


class CPUDriver(BaseDriver):
    """Driver for the host CPU and system RAM (device id ``cpu:0``)."""

    name = "cpu"
    backend = "cpu"
    DEVICE_ID = "cpu:0"

    def __init__(self, max_workers: Optional[int] = None) -> None:
        super().__init__()
        self._max_workers = max_workers or os.cpu_count() or 1
        self._executor: Optional[ThreadPoolExecutor] = None
        self._sampler: Optional[_CpuSampler] = None
        self._pending: Set["Future[Any]"] = set()
        self._known_ids: Set[str] = set()

    @classmethod
    def is_available(cls) -> bool:
        return True

    def _initialize(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="cpu-drv")
        self._sampler = _CpuSampler()
        read_system_memory()  # fail early if memory probing is unsupported

    def _shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._sampler = None
        self._pending.clear()

    def _guard(self, device_id: str) -> None:
        self._require_initialized()
        if device_id not in self._known_ids:
            raise DeviceNotFoundError(f"Unknown CPU device '{device_id}'", device_id=device_id)

    # -- discovery / state ------------------------------------------------- #

    def enumerate_devices(self) -> List[DeviceInfo]:
        self._require_initialized()
        _, total = read_system_memory()
        flags = _cpu_flags()
        caps: Dict[str, Any] = {
            "arch": platform.machine(),
            "threads": self._max_workers,
            **{f: (f in flags) for f in ("avx2", "avx512f", "fma", "neon") if flags},
        }
        self._known_ids.add(self.DEVICE_ID)
        return [DeviceInfo(self.DEVICE_ID, _cpu_model_name(), DeviceType.CPU, self.backend,
                           total, os.cpu_count() or 1, caps)]

    def memory_info(self, device_id: str) -> Tuple[int, int]:
        self._guard(device_id)
        available, total = read_system_memory()
        return available, total

    def telemetry(self, device_id: str) -> TelemetrySample:
        self._guard(device_id)
        available, total = read_system_memory()
        assert self._sampler is not None
        return TelemetrySample(
            device_id, time.time(), self._sampler.sample(), total - available, total,
            _cpu_temperature(), None,
            {"pending_tasks": len(self._pending)},
        )

    def health_check(self, device_id: str) -> DeviceState:
        self._guard(device_id)
        try:
            available, total = read_system_memory()
        except DriverError:
            return DeviceState.DEGRADED
        return DeviceState.DEGRADED if available < 0.02 * total else DeviceState.READY

    # -- memory ------------------------------------------------------------ #

    def _allocate(self, device_id: str, size_bytes: int) -> Tuple[Optional[int], Any]:
        self._guard(device_id)
        try:
            region = mmap.mmap(-1, size_bytes)
        except MemoryError as exc:
            raise ResourceExhaustedError(f"Out of host memory allocating {size_bytes} bytes",
                                         device_id=device_id, requested=size_bytes) from exc
        except OSError as exc:
            if exc.errno == errno.ENOMEM:
                raise ResourceExhaustedError(f"Out of host memory allocating {size_bytes} bytes",
                                             device_id=device_id, requested=size_bytes) from exc
            raise AllocationError(f"mmap failed: {exc}", device_id=device_id) from exc
        except (ValueError, OverflowError) as exc:
            raise AllocationError(f"Invalid allocation size {size_bytes}: {exc}", device_id=device_id) from exc
        address = ctypes.addressof(ctypes.c_char.from_buffer(region))  # temp export released immediately
        return address, region

    def _free(self, handle: MemoryHandle, raw: Any) -> None:
        try:
            raw.close()
        except BufferError as exc:
            raise AllocationError(f"Allocation {handle.handle_id} still has exported views; release them first",
                                  device_id=handle.device_id) from exc

    def write(self, handle: MemoryHandle, data: BytesLike, offset: int = 0) -> None:
        view = memoryview(data).cast("B")
        region = self._check_access(handle, offset, len(view))
        region[offset:offset + len(view)] = view

    def read(self, handle: MemoryHandle, size: Optional[int] = None, offset: int = 0) -> bytes:
        length = handle.size_bytes - offset if size is None else size
        region = self._check_access(handle, offset, length)
        return bytes(region[offset:offset + length])

    def view(self, handle: MemoryHandle) -> memoryview:
        """Zero-copy writable view (CPU-only). Release it before freeing the handle."""
        return memoryview(self._check_access(handle, 0, 0))

    # -- execution --------------------------------------------------------- #

    def submit(self, device_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> "Future[Any]":
        self._guard(device_id)
        assert self._executor is not None
        future = self._executor.submit(self._guarded, device_id, fn, *args, **kwargs)
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(self._discard)
        return future

    def _discard(self, future: "Future[Any]") -> None:
        with self._lock:
            self._pending.discard(future)

    def synchronize(self, device_id: str) -> None:
        self._guard(device_id)
        with self._lock:
            pending = list(self._pending)
        futures_wait(pending)
