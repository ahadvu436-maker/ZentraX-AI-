"""Thread-safe execution monitoring: events, metrics, listeners, stall watchdog."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from .execution_core import TaskResult, TaskSpec, TaskStatus

logger = logging.getLogger(__name__)

__all__ = ["MonitorEvent", "ExecutionMonitor"]

Listener = Callable[["MonitorEvent"], None]


@dataclass(frozen=True)
class MonitorEvent:
    """One observable occurrence (``task_started``, ``task_retry``, ``task_finished``, ...)."""

    kind: str
    timestamp: float
    task_id: str = ""
    name: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


class ExecutionMonitor:
    """Collects metrics and events for everything the engine runs.

    Listener callbacks run outside the internal lock and their exceptions are
    logged, never propagated, so monitoring can't break execution.
    """

    def __init__(self, history_size: int = 1000, stall_threshold: Optional[float] = None) -> None:
        self._lock = threading.RLock()
        self._events: Deque[MonitorEvent] = deque(maxlen=max(1, history_size))
        self._listeners: List[Listener] = []
        self._running: Dict[str, Dict[str, Any]] = {}
        self._counts: Dict[str, int] = {s.value: 0 for s in TaskStatus}
        self._counts.update(started=0, retries=0)
        self._total_duration = 0.0
        self._max_duration = 0.0
        self._finished = 0
        self.stall_threshold = stall_threshold
        self._stalled_reported: set = set()
        self._wd_thread: Optional[threading.Thread] = None
        self._wd_stop = threading.Event()

    # -- listeners / events ------------------------------------------------- #
    def add_listener(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit(self, kind: str, task_id: str = "", name: str = "", **detail: Any) -> MonitorEvent:
        event = MonitorEvent(kind, time.time(), task_id, name, detail)
        with self._lock:
            self._events.append(event)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001
                logger.exception("monitor listener failed for %s", kind)
        return event

    def recent_events(self, n: int = 50, kind: Optional[str] = None) -> List[MonitorEvent]:
        with self._lock:
            events = [e for e in self._events if kind is None or e.kind == kind]
        return events[-n:]

    # -- task lifecycle ----------------------------------------------------- #
    def task_started(self, spec: TaskSpec) -> None:
        with self._lock:
            self._counts["started"] += 1
            self._running[spec.task_id] = {"name": spec.name, "t0": time.monotonic()}
        self.emit("task_started", spec.task_id, spec.name)

    def task_retry(self, spec: TaskSpec, attempt: int, exc: BaseException, delay: float) -> None:
        with self._lock:
            self._counts["retries"] += 1
        self.emit("task_retry", spec.task_id, spec.name, attempt=attempt,
                  error=f"{type(exc).__name__}: {exc}", next_delay=round(delay, 3))

    def task_finished(self, result: TaskResult) -> None:
        with self._lock:
            self._running.pop(result.task_id, None)
            self._stalled_reported.discard(result.task_id)
            self._counts[result.status.value] += 1
            self._finished += 1
            self._total_duration += result.duration
            self._max_duration = max(self._max_duration, result.duration)
        self.emit("task_finished", result.task_id, result.name, status=result.status.value,
                  attempts=result.attempts, duration=round(result.duration, 4), error=result.error)

    # -- introspection ------------------------------------------------------ #
    def running_tasks(self) -> Dict[str, float]:
        """Mapping ``task_id -> seconds running``."""
        now = time.monotonic()
        with self._lock:
            return {tid: now - info["t0"] for tid, info in self._running.items()}

    def stalled_tasks(self, threshold: Optional[float] = None) -> Dict[str, float]:
        limit = threshold if threshold is not None else self.stall_threshold
        if not limit:
            return {}
        return {tid: age for tid, age in self.running_tasks().items() if age >= limit}

    def snapshot(self) -> Dict[str, Any]:
        """Point-in-time metrics."""
        with self._lock:
            counts = dict(self._counts)
            finished = self._finished
            done = counts["succeeded"] + counts["failed"] + counts["timeout"] + counts["cancelled"]
            return {
                **counts,
                "running": len(self._running),
                "finished": finished,
                "success_rate": (counts["succeeded"] / done) if done else None,
                "avg_duration": (self._total_duration / finished) if finished else 0.0,
                "max_duration": self._max_duration,
            }

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._counts = {s.value: 0 for s in TaskStatus}
            self._counts.update(started=0, retries=0)
            self._total_duration = self._max_duration = 0.0
            self._finished = 0

    # -- watchdog ----------------------------------------------------------- #
    def start_watchdog(self, interval: float = 5.0) -> None:
        """Start a daemon thread that emits ``task_stalled`` once per stalled task."""
        with self._lock:
            if self._wd_thread and self._wd_thread.is_alive():
                return
            self._wd_stop.clear()
            self._wd_thread = threading.Thread(
                target=self._watch, args=(interval,), daemon=True, name="exec-monitor-watchdog")
            self._wd_thread.start()

    def stop_watchdog(self) -> None:
        self._wd_stop.set()
        thread = self._wd_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _watch(self, interval: float) -> None:
        while not self._wd_stop.wait(interval):
            for tid, age in self.stalled_tasks().items():
                with self._lock:
                    if tid in self._stalled_reported:
                        continue
                    self._stalled_reported.add(tid)
                    name = self._running.get(tid, {}).get("name", "")
                logger.warning("task %s (%s) appears stalled after %.1fs", name, tid, age)
                self.emit("task_stalled", tid, name, running_for=round(age, 2))
