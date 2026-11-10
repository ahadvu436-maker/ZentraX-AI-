"""
omni_cognition_nexus.py

Central cognitive pipeline for the system:
  1. Async event loop / event bus for real-time data exchange between modules
     (core_growth_engine.py, autonomous_executor.py, ...).
  2. Thread-safe central memory with versioning and change subscriptions.
  3. Crisis fast-path: a dedicated high-priority lane with deadline-bounded
     handlers and an O(1) precomputed playbook for instant decisions.

Usage:
    nexus = OmniCognitionNexus()
    nexus.register(GrowthEngineAdapter(core_growth_engine))
    nexus.register(ExecutorAdapter(autonomous_executor))
    nexus.start()                       # runs in its own thread
    nexus.publish("sensor.cpu", {"load": 0.97}, priority=Priority.CRISIS)
    ...
    nexus.stop()
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("omni_nexus")


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
class Priority(IntEnum):
    CRISIS = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


_seq = itertools.count()


@dataclass(order=True)
class Event:
    priority: int
    seq: int = field(compare=True)
    topic: str = field(compare=False, default="")
    payload: Any = field(compare=False, default=None)
    source: str = field(compare=False, default="unknown")
    ts: float = field(compare=False, default_factory=time.monotonic)
    deadline_s: Optional[float] = field(compare=False, default=None)

    @staticmethod
    def make(topic: str, payload: Any = None, *, source: str = "unknown",
             priority: Priority = Priority.NORMAL,
             deadline_s: Optional[float] = None) -> "Event":
        return Event(int(priority), next(_seq), topic, payload, source,
                     time.monotonic(), deadline_s)


# --------------------------------------------------------------------------
# Thread-safe central memory
# --------------------------------------------------------------------------
class CentralMemory:
    """Versioned key-value store, safe for use from any thread.

    - Per-key version numbers enable optimistic concurrency (compare_and_set).
    - Change subscribers are called OUTSIDE the lock to avoid deadlocks.
    - snapshot() returns a consistent point-in-time copy.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: Dict[str, Tuple[Any, int]] = {}
        self._global_version = 0
        self._subs: Dict[str, List[Callable[[str, Any, int], None]]] = defaultdict(list)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, (default, 0))[0]

    def get_versioned(self, key: str) -> Tuple[Any, int]:
        with self._lock:
            return self._data.get(key, (None, 0))

    def set(self, key: str, value: Any) -> int:
        with self._lock:
            ver = self._data.get(key, (None, 0))[1] + 1
            self._data[key] = (value, ver)
            self._global_version += 1
            subs = list(self._subs.get(key, ())) + list(self._subs.get("*", ()))
        self._notify(subs, key, value, ver)
        return ver

    def compare_and_set(self, key: str, expected_version: int, value: Any) -> bool:
        with self._lock:
            cur_ver = self._data.get(key, (None, 0))[1]
            if cur_ver != expected_version:
                return False
            self._data[key] = (value, cur_ver + 1)
            self._global_version += 1
            ver = cur_ver + 1
            subs = list(self._subs.get(key, ())) + list(self._subs.get("*", ()))
        self._notify(subs, key, value, ver)
        return True

    def update(self, key: str, fn: Callable[[Any], Any], default: Any = None) -> Any:
        """Atomic read-modify-write."""
        with self._lock:
            old, ver = self._data.get(key, (default, 0))
            new = fn(old)
            self._data[key] = (new, ver + 1)
            self._global_version += 1
            subs = list(self._subs.get(key, ())) + list(self._subs.get("*", ()))
        self._notify(subs, key, new, ver + 1)
        return new

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {k: v for k, (v, _) in self._data.items()}

    def subscribe(self, key: str, cb: Callable[[str, Any, int], None]) -> None:
        """key='*' subscribes to every change."""
        with self._lock:
            self._subs[key].append(cb)

    @property
    def version(self) -> int:
        with self._lock:
            return self._global_version

    @staticmethod
    def _notify(subs: Iterable[Callable], key: str, value: Any, ver: int) -> None:
        for cb in subs:
            try:
                cb(key, value, ver)
            except Exception:
                log.exception("memory subscriber failed for %s", key)


# --------------------------------------------------------------------------
# Module interface + adapters
# --------------------------------------------------------------------------
class NexusModule:
    """Base class. Subclass or wrap your existing modules."""

    name: str = "module"
    topics: Tuple[str, ...] = ()          # topic prefixes, e.g. ("growth.", "sensor.")

    async def handle(self, event: Event, memory: CentralMemory) -> Optional[List[Event]]:
        raise NotImplementedError


class CallableAdapter(NexusModule):
    """Wrap any object exposing handle_event(event_dict) (sync or async)."""

    def __init__(self, name: str, target: Any, topics: Tuple[str, ...],
                 method: str = "handle_event") -> None:
        self.name, self.topics = name, topics
        self._fn = getattr(target, method)

    async def handle(self, event: Event, memory: CentralMemory) -> Optional[List[Event]]:
        data = {"topic": event.topic, "payload": event.payload, "source": event.source}
        if asyncio.iscoroutinefunction(self._fn):
            result = await self._fn(data)
        else:
            result = await asyncio.get_running_loop().run_in_executor(None, self._fn, data)
        if not result:
            return None
        return [Event.make(r["topic"], r.get("payload"), source=self.name,
                           priority=Priority(r.get("priority", Priority.NORMAL)))
                for r in result]


# --------------------------------------------------------------------------
# Crisis fast-path: O(1) playbook decision
# --------------------------------------------------------------------------
@dataclass
class PlaybookEntry:
    action_topic: str                     # event emitted immediately on match
    predicate: Callable[[Any], bool]      # cheap check on payload
    budget_s: float = 0.05                # max time for handlers


class CrisisPlaybook:
    """Precomputed topic -> response table. No search, no I/O on the hot path."""

    def __init__(self) -> None:
        self._table: Dict[str, List[PlaybookEntry]] = defaultdict(list)

    def add(self, topic: str, entry: PlaybookEntry) -> None:
        self._table[topic].append(entry)

    def decide(self, event: Event) -> List[PlaybookEntry]:
        return [e for e in self._table.get(event.topic, ()) if e.predicate(event.payload)]


# --------------------------------------------------------------------------
# The Nexus
# --------------------------------------------------------------------------
class OmniCognitionNexus:
    def __init__(self, workers: int = 4, handler_timeout_s: float = 5.0) -> None:
        self.memory = CentralMemory()
        self.playbook = CrisisPlaybook()
        self._modules: List[NexusModule] = []
        self._workers = workers
        self._handler_timeout = handler_timeout_s

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._main_q: Optional[asyncio.PriorityQueue] = None
        self._crisis_q: Optional[asyncio.Queue] = None
        self._tasks: List[asyncio.Task] = []
        self._stats = defaultdict(int)

    # ---- registration ----------------------------------------------------
    def register(self, module: NexusModule) -> None:
        self._modules.append(module)
        log.info("registered module %s topics=%s", module.name, module.topics)

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, name="nexus-loop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        if not self._loop:
            return
        fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        fut.result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._loop = self._thread = None

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._main_q = asyncio.PriorityQueue()
        self._crisis_q = asyncio.Queue()
        self._tasks = [self._loop.create_task(self._crisis_worker())]
        self._tasks += [self._loop.create_task(self._worker(i)) for i in range(self._workers)]
        self._ready.set()
        self._loop.run_forever()

    async def _shutdown(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ---- publishing (callable from ANY thread) ---------------------------
    def publish(self, topic: str, payload: Any = None, *, source: str = "external",
                priority: Priority = Priority.NORMAL,
                deadline_s: Optional[float] = None) -> None:
        self._submit(Event.make(topic, payload, source=source,
                                priority=priority, deadline_s=deadline_s))

    def _submit(self, ev: Event) -> None:
        if not self._loop:
            raise RuntimeError("Nexus not started")
        self._loop.call_soon_threadsafe(self._enqueue, ev)

    def _enqueue(self, ev: Event) -> None:
        if ev.priority == Priority.CRISIS:
            self._crisis_q.put_nowait(ev)        # dedicated lane, never behind other work
        else:
            self._main_q.put_nowait(ev)
        self._stats["published"] += 1

    # ---- workers ---------------------------------------------------------
    async def _crisis_worker(self) -> None:
        while True:
            ev = await self._crisis_q.get()
            t0 = time.monotonic()
            try:
                # Step 1: instant playbook decision (no I/O, O(1) lookup).
                for entry in self.playbook.decide(ev):
                    self._enqueue(Event.make(entry.action_topic, ev.payload,
                                             source="crisis_playbook",
                                             priority=Priority.CRISIS))
                    self.memory.set(f"crisis.last_action.{ev.topic}", entry.action_topic)
                # Step 2: fan out to modules in parallel, bounded by deadline.
                budget = ev.deadline_s or 0.25
                await self._dispatch(ev, timeout=budget)
            except Exception:
                log.exception("crisis handling failed for %s", ev.topic)
            finally:
                self._stats["crisis_handled"] += 1
                self.memory.set("crisis.last_latency_ms", (time.monotonic() - t0) * 1000)

    async def _worker(self, idx: int) -> None:
        while True:
            ev = await self._main_q.get()
            try:
                await self._dispatch(ev, timeout=ev.deadline_s or self._handler_timeout)
            except Exception:
                log.exception("worker %d failed on %s", idx, ev.topic)
            finally:
                self._stats["handled"] += 1

    async def _dispatch(self, ev: Event, timeout: float) -> None:
        targets = [m for m in self._modules
                   if m.name != ev.source and any(ev.topic.startswith(p) for p in m.topics)]
        if not targets:
            return
        results = await asyncio.gather(
            *(self._call(m, ev, timeout) for m in targets), return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                for out in r:
                    self._enqueue(out)           # pipeline: outputs become new events

    async def _call(self, module: NexusModule, ev: Event, timeout: float):
        try:
            return await asyncio.wait_for(module.handle(ev, self.memory), timeout=timeout)
        except asyncio.TimeoutError:
            self._stats["timeouts"] += 1
            log.warning("%s timed out on %s (%.0f ms)", module.name, ev.topic, timeout * 1000)
        except Exception:
            self._stats["errors"] += 1
            log.exception("%s failed on %s", module.name, ev.topic)
        return None

    # ---- introspection ---------------------------------------------------
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


# --------------------------------------------------------------------------
# Example wiring
# --------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class DemoGrowth(NexusModule):
        name, topics = "core_growth_engine", ("sensor.", "growth.")

        async def handle(self, event, memory):
            memory.update("growth.events_seen", lambda n: (n or 0) + 1, default=0)
            return None

    class DemoExecutor(NexusModule):
        name, topics = "autonomous_executor", ("action.", "crisis.")

        async def handle(self, event, memory):
            log.info("EXECUTOR running %s payload=%s", event.topic, event.payload)
            return None

    nexus = OmniCognitionNexus()
    nexus.register(DemoGrowth())
    nexus.register(DemoExecutor())
    nexus.playbook.add("sensor.cpu", PlaybookEntry(
        action_topic="action.shed_load",
        predicate=lambda p: p.get("load", 0) > 0.9))
    nexus.start()

    nexus.publish("sensor.cpu", {"load": 0.5}, source="monitor")
    nexus.publish("sensor.cpu", {"load": 0.97}, source="monitor", priority=Priority.CRISIS)
    time.sleep(0.5)
    print("stats:", nexus.stats())
    print("memory:", nexus.memory.snapshot())
    nexus.stop()
