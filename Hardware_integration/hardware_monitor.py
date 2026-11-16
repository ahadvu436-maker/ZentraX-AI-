"""Real-time hardware telemetry.

A daemon thread samples every usable device at a fixed interval into bounded
ring buffers, evaluates thresholds (CPU/GPU utilisation, memory, temperature)
with alert de-duplication and recovery events, and marks devices FAULTED after
repeated sampling failures.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from .hardware_core import (
    BaseManager,
    DeviceState,
    DeviceType,
    EventBus,
    HardwareConfig,
    HardwareError,
    Severity,
    TelemetrySample,
)

logger = logging.getLogger("hardware_integration.monitor")

_FAULT_AFTER_FAILURES = 3


@dataclass(frozen=True)
class Alert:
    timestamp: float
    device_id: str
    metric: str
    severity: Severity
    value: float
    threshold: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


class HardwareMonitor(BaseManager):
    """Background sampler and alert engine."""

    name = "monitor"

    def __init__(self, devices: Any, bus: Optional[EventBus] = None,
                 config: Optional[HardwareConfig] = None) -> None:
        super().__init__(bus)
        self.devices = devices
        self.config = config or HardwareConfig()
        self._interval = self.config.monitor_interval_s
        self._lock = threading.RLock()
        self._history: Dict[str, Deque[TelemetrySample]] = {}
        self._latest: Dict[str, TelemetrySample] = {}
        self._alerts: Deque[Alert] = deque(maxlen=200)
        self._active: Dict[Tuple[str, str], Severity] = {}
        self._last_alert_at: Dict[Tuple[str, str], float] = {}
        self._failures: Dict[str, int] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples_taken = 0
        self.sample_errors = 0

    # -- lifecycle --------------------------------------------------------- #

    def _on_start(self) -> None:
        self._stop_event.clear()
        self.sample_once()  # prime so latest() is populated immediately
        self._thread = threading.Thread(target=self._run, name="hw-monitor", daemon=True)
        self._thread.start()

    def _on_stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._interval * 2))
            if self._thread.is_alive():
                logger.warning("Monitor thread did not stop in time")
            self._thread = None

    def set_interval(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("interval must be > 0")
        self._interval = seconds

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            started = time.perf_counter()
            try:
                self.sample_once()
            except Exception:  # noqa: BLE001 - the sampler loop must survive anything
                self.sample_errors += 1
                logger.exception("Unexpected monitor failure")
            elapsed = time.perf_counter() - started
            if elapsed > self._interval:
                logger.debug("Sampling took %.3fs (> interval %.3fs)", elapsed, self._interval)

    # -- sampling ---------------------------------------------------------- #

    def sample_once(self) -> Dict[str, TelemetrySample]:
        """Sample every non-offline device once and evaluate alerts."""
        taken: Dict[str, TelemetrySample] = {}
        for info in self.devices.list_devices(states=(DeviceState.READY, DeviceState.BUSY, DeviceState.DEGRADED)):
            device_id = info.device_id
            try:
                sample = self.devices.telemetry(device_id)
            except HardwareError as exc:
                self._on_sample_failure(device_id, exc)
                continue
            self._failures[device_id] = 0
            taken[device_id] = sample
            with self._lock:
                self._history.setdefault(device_id, deque(maxlen=self.config.monitor_history)).append(sample)
                self._latest[device_id] = sample
            self._evaluate(info.device_type, sample)
        with self._lock:  # drop stale readings for devices that are no longer sampled (e.g. faulted)
            for stale in set(self._latest) - set(taken):
                del self._latest[stale]
        self.samples_taken += 1
        return taken

    def _on_sample_failure(self, device_id: str, exc: Exception) -> None:
        self.sample_errors += 1
        count = self._failures[device_id] = self._failures.get(device_id, 0) + 1
        logger.warning("Telemetry failed for %s (%d/%d): %s", device_id, count, _FAULT_AFTER_FAILURES, exc)
        if count >= _FAULT_AFTER_FAILURES:
            self.devices.set_state(device_id, DeviceState.FAULTED, f"telemetry failing: {exc}")

    # -- alerting ---------------------------------------------------------- #

    def _rules(self, device_type: DeviceType) -> List[Tuple[str, Optional[float], float, float]]:
        t = self.config.thresholds
        if device_type is DeviceType.CPU:
            return [("utilization_pct", None, t.cpu_util_warn, t.cpu_util_crit),
                    ("memory_used_pct", None, t.mem_warn, t.mem_crit),
                    ("temperature_c", None, t.temp_warn, t.temp_crit)]
        return [("memory_used_pct", None, t.gpu_mem_warn, t.gpu_mem_crit),
                ("temperature_c", None, t.temp_warn, t.temp_crit)]

    def _evaluate(self, device_type: DeviceType, sample: TelemetrySample) -> None:
        now = sample.timestamp
        for metric, _, warn, crit in self._rules(device_type):
            value = getattr(sample, metric)
            if value is None:
                continue
            key = (sample.device_id, metric)
            level = Severity.CRITICAL if value >= crit else Severity.WARNING if value >= warn else None
            previous = self._active.get(key)
            if level is None:
                if previous is not None:
                    del self._active[key]
                    self._emit(Alert(now, sample.device_id, metric, Severity.INFO, value, warn,
                                     f"{sample.device_id} {metric} recovered ({value:.1f})"), "alert_recovered")
                continue
            threshold = crit if level is Severity.CRITICAL else warn
            cooled = now - self._last_alert_at.get(key, 0.0) >= self.config.alert_cooldown_s
            if level is not previous or cooled:
                self._active[key] = level
                self._last_alert_at[key] = now
                self._emit(Alert(now, sample.device_id, metric, level, value, threshold,
                                 f"{sample.device_id} {metric}={value:.1f} >= {threshold:.1f}"), "alert")

    def _emit(self, alert: Alert, kind: str) -> None:
        with self._lock:
            self._alerts.append(alert)
        log = logger.error if alert.severity is Severity.CRITICAL else logger.warning \
            if alert.severity is Severity.WARNING else logger.info
        log(alert.message)
        self._publish(kind, alert.severity, alert.message, alert.device_id,
                      metric=alert.metric, value=alert.value, threshold=alert.threshold)

    # -- queries ----------------------------------------------------------- #

    def latest(self) -> Dict[str, TelemetrySample]:
        with self._lock:
            return dict(self._latest)

    def history(self, device_id: str, limit: Optional[int] = None) -> List[TelemetrySample]:
        with self._lock:
            items = list(self._history.get(device_id, ()))
        return items[-limit:] if limit else items

    def recent_alerts(self, count: int = 20) -> List[Alert]:
        with self._lock:
            return list(self._alerts)[-count:]

    def summary(self, device_id: str, window: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """avg/min/max per metric over the last ``window`` samples."""
        samples = self.history(device_id, window)
        out: Dict[str, Dict[str, float]] = {}
        for metric in ("utilization_pct", "memory_used_pct", "temperature_c", "power_w"):
            values = [v for s in samples if (v := getattr(s, metric)) is not None]
            if values:
                out[metric] = {"avg": sum(values) / len(values), "min": min(values), "max": max(values),
                               "samples": float(len(values))}
        return out

    def health(self) -> Dict[str, Any]:
        return {"started": self._started, "thread_alive": bool(self._thread and self._thread.is_alive()),
                "samples_taken": self.samples_taken, "sample_errors": self.sample_errors,
                "active_alerts": len(self._active)}
