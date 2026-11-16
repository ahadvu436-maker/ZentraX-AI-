"""NVIDIA CUDA driver built on the CUDA Driver API (``libcuda``) through ``ctypes``.

* Uses each device's *primary context*, so it interoperates with PyTorch, CuPy,
  Numba and friends running in the same process.
* Every operation binds the device context to the calling thread first
  (``cuCtxSetCurrent``); ``submit`` runs work on a dedicated single-thread
  executor per device and synchronises afterwards.
* Telemetry (utilisation, temperature, power) comes from NVML via the optional
  ``pynvml`` / ``nvidia-ml-py`` package; memory figures come from the driver API.
* Sticky errors (illegal address, launch failure, uncorrectable ECC ...) mark the
  device faulted until :meth:`CUDADriver.reset` succeeds.

NOTE: this module cannot be exercised without an NVIDIA GPU and driver; treat it as
needing validation on your target hardware before production use.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..hardware_core import (
    AllocationError,
    BaseDriver,
    BytesLike,
    DeviceFaultError,
    DeviceInfo,
    DeviceNotFoundError,
    DeviceState,
    DeviceType,
    DriverError,
    DriverNotAvailableError,
    MemoryHandle,
    ResourceExhaustedError,
    TelemetrySample,
)

try:  # optional dependency
    import pynvml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]

logger = logging.getLogger("hardware_integration.cuda")

_CUDA_SUCCESS = 0
_ERR_OUT_OF_MEMORY = 2
_ERR_NO_DEVICE = 100
_ERR_INVALID_DEVICE = 101
# Errors after which the context is unusable until reset.
_STICKY_ERRORS = {214, 700, 702, 710, 715, 716, 717, 718, 719, 999}

_ATTR_MULTIPROCESSOR_COUNT = 16
_ATTR_CC_MAJOR = 75
_ATTR_CC_MINOR = 76

_c = ctypes
_SIGNATURES: Dict[str, List[Any]] = {
    "cuInit": [_c.c_uint],
    "cuDeviceGetCount": [_c.POINTER(_c.c_int)],
    "cuDeviceGet": [_c.POINTER(_c.c_int), _c.c_int],
    "cuDeviceGetName": [_c.c_char_p, _c.c_int, _c.c_int],
    "cuDeviceGetPCIBusId": [_c.c_char_p, _c.c_int, _c.c_int],
    "cuDeviceTotalMem_v2": [_c.POINTER(_c.c_size_t), _c.c_int],
    "cuDeviceGetAttribute": [_c.POINTER(_c.c_int), _c.c_int, _c.c_int],
    "cuDevicePrimaryCtxRetain": [_c.POINTER(_c.c_void_p), _c.c_int],
    "cuDevicePrimaryCtxRelease_v2": [_c.c_int],
    "cuDevicePrimaryCtxReset_v2": [_c.c_int],
    "cuCtxSetCurrent": [_c.c_void_p],
    "cuCtxSynchronize": [],
    "cuMemAlloc_v2": [_c.POINTER(_c.c_uint64), _c.c_size_t],
    "cuMemFree_v2": [_c.c_uint64],
    "cuMemGetInfo_v2": [_c.POINTER(_c.c_size_t), _c.POINTER(_c.c_size_t)],
    "cuMemcpyHtoD_v2": [_c.c_uint64, _c.c_void_p, _c.c_size_t],
    "cuMemcpyDtoH_v2": [_c.c_void_p, _c.c_uint64, _c.c_size_t],
    "cuGetErrorString": [_c.c_int, _c.POINTER(_c.c_char_p)],
}


def _load_libcuda() -> ctypes.CDLL:
    candidates = {"win32": ["nvcuda.dll"], "darwin": ["libcuda.dylib"]}.get(sys.platform, ["libcuda.so.1", "libcuda.so"])
    found = ctypes.util.find_library("cuda")
    if found:
        candidates.append(found)
    last: Optional[Exception] = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            last = exc
            continue
        try:
            for fn_name, argtypes in _SIGNATURES.items():
                fn = getattr(lib, fn_name)
                fn.argtypes = argtypes
                fn.restype = ctypes.c_int
        except AttributeError as exc:
            raise DriverNotAvailableError(f"CUDA driver library is too old / incomplete: {exc}") from exc
        return lib
    raise DriverNotAvailableError(f"CUDA driver library not found ({last})")


@dataclass
class _Device:
    device_id: str
    ordinal: int
    handle: int
    ctx: int
    info: DeviceInfo
    pci_bus_id: str
    nvml: Any = None
    executor: Optional[ThreadPoolExecutor] = None


class CUDADriver(BaseDriver):
    """CUDA devices exposed as ``cuda:<ordinal>``."""

    name = "cuda"
    backend = "cuda"

    _probe_lock = threading.Lock()
    _probe_result: Optional[bool] = None

    def __init__(self, ordinals: Optional[List[int]] = None, use_nvml: bool = True) -> None:
        super().__init__()
        self._ordinals = ordinals
        self._use_nvml = use_nvml
        self._lib: Optional[ctypes.CDLL] = None
        self._devices: Dict[str, _Device] = {}
        self._faulted: Set[str] = set()
        self._nvml_ready = False

    # -- availability ------------------------------------------------------ #

    @classmethod
    def is_available(cls) -> bool:
        with cls._probe_lock:
            if cls._probe_result is None:
                try:
                    lib = _load_libcuda()
                    count = ctypes.c_int()
                    ok = lib.cuInit(0) == _CUDA_SUCCESS and lib.cuDeviceGetCount(ctypes.byref(count)) == _CUDA_SUCCESS
                    cls._probe_result = bool(ok and count.value > 0)
                except DriverError:
                    cls._probe_result = False
            return cls._probe_result

    # -- error handling ---------------------------------------------------- #

    def _error_string(self, code: int) -> str:
        assert self._lib is not None
        msg = ctypes.c_char_p()
        if self._lib.cuGetErrorString(code, ctypes.byref(msg)) == _CUDA_SUCCESS and msg.value:
            return msg.value.decode("ascii", "replace")
        return f"CUDA error {code}"

    def _check(self, code: int, op: str, device_id: Optional[str] = None,
               requested: Optional[int] = None) -> None:
        if code == _CUDA_SUCCESS:
            return
        text = f"{op} failed: {self._error_string(code)} (code {code})"
        if code == _ERR_OUT_OF_MEMORY:
            raise ResourceExhaustedError(text, device_id=device_id, requested=requested)
        if code == _ERR_INVALID_DEVICE:
            raise DeviceNotFoundError(text, device_id=device_id)
        if code == _ERR_NO_DEVICE:
            raise DriverNotAvailableError(text, device_id=device_id)
        if code in _STICKY_ERRORS and device_id:
            self._faulted.add(device_id)
            raise DeviceFaultError(text, device_id=device_id)
        raise DriverError(text, device_id=device_id)

    # -- lifecycle --------------------------------------------------------- #

    def _initialize(self) -> None:
        self._lib = _load_libcuda()
        self._check(self._lib.cuInit(0), "cuInit")
        count = ctypes.c_int()
        self._check(self._lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        if count.value == 0:
            raise DriverNotAvailableError("No CUDA devices found")
        ordinals = self._ordinals if self._ordinals is not None else list(range(count.value))
        if self._use_nvml and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_ready = True
            except Exception:  # noqa: BLE001
                logger.warning("NVML init failed; GPU utilisation/temperature unavailable", exc_info=True)
        try:
            for ordinal in ordinals:
                self._open_device(ordinal, count.value)
        except Exception:
            self._shutdown()
            raise

    def _open_device(self, ordinal: int, count: int) -> None:
        lib = self._lib
        assert lib is not None
        if not 0 <= ordinal < count:
            raise DeviceNotFoundError(f"CUDA ordinal {ordinal} out of range (0..{count - 1})")
        device_id = f"cuda:{ordinal}"
        dev = ctypes.c_int()
        self._check(lib.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet", device_id)
        name_buf = ctypes.create_string_buffer(128)
        self._check(lib.cuDeviceGetName(name_buf, 128, dev.value), "cuDeviceGetName", device_id)
        total = ctypes.c_size_t()
        self._check(lib.cuDeviceTotalMem_v2(ctypes.byref(total), dev.value), "cuDeviceTotalMem", device_id)
        attrs: Dict[int, int] = {}
        for attr in (_ATTR_MULTIPROCESSOR_COUNT, _ATTR_CC_MAJOR, _ATTR_CC_MINOR):
            value = ctypes.c_int()
            self._check(lib.cuDeviceGetAttribute(ctypes.byref(value), attr, dev.value),
                        "cuDeviceGetAttribute", device_id)
            attrs[attr] = value.value
        pci = ctypes.create_string_buffer(32)
        pci_ok = lib.cuDeviceGetPCIBusId(pci, 32, dev.value) == _CUDA_SUCCESS
        pci_id = pci.value.decode("ascii", "replace") if pci_ok else ""
        ctx = ctypes.c_void_p()
        self._check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value), "cuDevicePrimaryCtxRetain", device_id)
        info = DeviceInfo(
            device_id, name_buf.value.decode("utf-8", "replace"), DeviceType.GPU, self.backend,
            int(total.value), attrs[_ATTR_MULTIPROCESSOR_COUNT],
            {"compute_capability": f"{attrs[_ATTR_CC_MAJOR]}.{attrs[_ATTR_CC_MINOR]}", "cuda": True,
             "pci_bus_id": pci_id},
        )
        record = _Device(device_id, ordinal, dev.value, int(ctx.value or 0), info, pci_id,
                         self._nvml_handle(ordinal, pci_id),
                         ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"cuda{ordinal}"))
        self._devices[device_id] = record

    def _nvml_handle(self, ordinal: int, pci_id: str) -> Any:
        if not self._nvml_ready:
            return None
        try:
            try:
                return pynvml.nvmlDeviceGetHandleByPciBusId(pci_id)
            except Exception:  # noqa: BLE001 - some bindings want bytes
                return pynvml.nvmlDeviceGetHandleByPciBusId(pci_id.encode())
        except Exception:  # noqa: BLE001
            try:
                return pynvml.nvmlDeviceGetHandleByIndex(ordinal)
            except Exception:  # noqa: BLE001
                return None

    def _shutdown(self) -> None:
        for record in self._devices.values():
            if record.executor is not None:
                try:
                    record.executor.shutdown(wait=True, cancel_futures=True)
                except TypeError:  # Python < 3.9
                    record.executor.shutdown(wait=True)
            try:
                if self._lib is not None:
                    self._lib.cuDevicePrimaryCtxRelease_v2(record.handle)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to release primary context for %s", record.device_id, exc_info=True)
        self._devices.clear()
        self._faulted.clear()
        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            self._nvml_ready = False

    # -- helpers ----------------------------------------------------------- #

    def _device(self, device_id: str) -> _Device:
        self._require_initialized()
        record = self._devices.get(device_id)
        if record is None:
            raise DeviceNotFoundError(f"Unknown CUDA device '{device_id}'", device_id=device_id)
        if device_id in self._faulted:
            raise DeviceFaultError(f"{device_id} is faulted; call reset()", device_id=device_id)
        return record

    def _bind(self, record: _Device) -> None:
        assert self._lib is not None
        self._check(self._lib.cuCtxSetCurrent(record.ctx), "cuCtxSetCurrent", record.device_id)

    # -- discovery / state ------------------------------------------------- #

    def enumerate_devices(self) -> List[DeviceInfo]:
        self._require_initialized()
        return [r.info for r in self._devices.values()]

    def memory_info(self, device_id: str) -> Tuple[int, int]:
        record = self._device(device_id)
        assert self._lib is not None
        self._bind(record)
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self._check(self._lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo", device_id)
        return int(free.value), int(total.value)

    def telemetry(self, device_id: str) -> TelemetrySample:
        record = self._device(device_id)
        free, total = self.memory_info(device_id)
        util = temp = power = None
        extra: Dict[str, Any] = {}
        handle = record.nvml
        if handle is not None:
            try:
                util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:  # noqa: BLE001
                pass
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(handle, 0))  # NVML_TEMPERATURE_GPU
            except Exception:  # noqa: BLE001
                pass
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:  # noqa: BLE001
                pass
            try:
                extra["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:  # noqa: BLE001
                pass
        return TelemetrySample(device_id, time.time(), util, total - free, total, temp, power, extra)

    def health_check(self, device_id: str) -> DeviceState:
        if device_id in self._faulted:
            return DeviceState.FAULTED
        try:
            record = self._device(device_id)
            assert self._lib is not None
            self._bind(record)
            self.memory_info(device_id)
            self._check(self._lib.cuCtxSynchronize(), "cuCtxSynchronize", device_id)
        except DeviceFaultError:
            return DeviceState.FAULTED
        except DeviceNotFoundError:
            return DeviceState.OFFLINE
        except HardwareErrorTypes:
            return DeviceState.DEGRADED
        return DeviceState.READY

    def reset(self, device_id: str) -> None:
        """Best-effort primary-context reset. Invalidates all allocations on the device."""
        self._require_initialized()
        record = self._devices.get(device_id)
        if record is None:
            raise DeviceNotFoundError(f"Unknown CUDA device '{device_id}'", device_id=device_id)
        assert self._lib is not None
        with self._lock:
            for handle in [h for h in self._allocs.values() if h.device_id == device_id]:
                self._allocs.pop(handle.handle_id, None)  # memory is destroyed by the reset
                self._raw.pop(handle.handle_id, None)
        self._check(self._lib.cuDevicePrimaryCtxReset_v2(record.handle), "cuDevicePrimaryCtxReset", device_id)
        ctx = ctypes.c_void_p()
        self._check(self._lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), record.handle),
                    "cuDevicePrimaryCtxRetain", device_id)
        self._lib.cuDevicePrimaryCtxRelease_v2(record.handle)  # balance the extra retain
        record.ctx = int(ctx.value or 0)
        self._faulted.discard(device_id)
        logger.warning("Reset %s", device_id)

    # -- memory ------------------------------------------------------------ #

    def _allocate(self, device_id: str, size_bytes: int) -> Tuple[Optional[int], Any]:
        record = self._device(device_id)
        assert self._lib is not None
        self._bind(record)
        ptr = ctypes.c_uint64()
        code = self._lib.cuMemAlloc_v2(ctypes.byref(ptr), size_bytes)
        if code == _ERR_OUT_OF_MEMORY:
            try:
                available: Optional[int] = self.memory_info(device_id)[0]
            except DriverError:
                available = None
            raise ResourceExhaustedError(f"Out of device memory on {device_id} allocating {size_bytes} bytes",
                                         device_id=device_id, requested=size_bytes, available=available)
        self._check(code, "cuMemAlloc", device_id, size_bytes)
        return int(ptr.value), int(ptr.value)

    def _free(self, handle: MemoryHandle, raw: Any) -> None:
        record = self._device(handle.device_id)
        assert self._lib is not None
        self._bind(record)
        self._check(self._lib.cuMemFree_v2(int(raw)), "cuMemFree", handle.device_id)

    def write(self, handle: MemoryHandle, data: BytesLike, offset: int = 0) -> None:
        view = memoryview(data).cast("B")
        dptr = int(self._check_access(handle, offset, len(view)))
        record = self._device(handle.device_id)
        assert self._lib is not None
        self._bind(record)
        buffer = (ctypes.c_char * len(view)).from_buffer(view) if not view.readonly \
            else (ctypes.c_char * len(view)).from_buffer_copy(view)
        try:
            self._check(self._lib.cuMemcpyHtoD_v2(dptr + offset, ctypes.addressof(buffer), len(view)),
                        "cuMemcpyHtoD", handle.device_id)
        finally:
            del buffer  # release the buffer export on the caller's object

    def read(self, handle: MemoryHandle, size: Optional[int] = None, offset: int = 0) -> bytes:
        length = handle.size_bytes - offset if size is None else size
        dptr = int(self._check_access(handle, offset, length))
        record = self._device(handle.device_id)
        assert self._lib is not None
        self._bind(record)
        out = ctypes.create_string_buffer(length)
        self._check(self._lib.cuMemcpyDtoH_v2(ctypes.addressof(out), dptr + offset, length),
                    "cuMemcpyDtoH", handle.device_id)
        return out.raw[:length]

    # -- execution --------------------------------------------------------- #

    def submit(self, device_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> "Future[Any]":
        record = self._device(device_id)
        assert record.executor is not None

        def run() -> Any:
            self._bind(record)
            result = fn(*args, **kwargs)
            assert self._lib is not None
            self._check(self._lib.cuCtxSynchronize(), "cuCtxSynchronize", device_id)
            return result

        return record.executor.submit(self._guarded, device_id, run)

    def synchronize(self, device_id: str) -> None:
        record = self._device(device_id)
        assert self._lib is not None
        self._bind(record)
        self._check(self._lib.cuCtxSynchronize(), "cuCtxSynchronize", device_id)


# Exceptions health_check treats as "degraded" (everything except fault/not-found, handled above).
HardwareErrorTypes = (DriverError, AllocationError)
