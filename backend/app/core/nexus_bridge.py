#!/usr/bin/env python3
"""
nexus_bridge.py - central async hub that discovers, activates and
interconnects a project's defense modules through an in-memory event bus.

No external dependencies: asyncio + importlib from the standard library only,
so this runs fully offline, in local memory, with no network or disk I/O
required by the bus itself.

Architecture
------------
* EventBus     pub/sub over asyncio queues. Any module can publish an Event;
                every module subscribed to that event type (or "*") gets it
                concurrently. This is how modules "share threat signals and
                alerts with each other in real time" without being wired to
                each other directly.
* Module       the standard interface every plugin implements: start(bus),
                stop(), and optionally handle(event). NexusBridge discovers
                modules by name, imports them if present, and falls back to
                a StubModule (logs only, does nothing) if the file is
                missing or doesn't expose the interface - so a bridge with
                a partial project still runs.
* NexusBridge   owns the bus and the module registry. activate_all() starts
                every enabled module concurrently (asyncio.gather). It
                subscribes to NETWORK_DOWN and THREAT_DETECTED itself and
                calls activate_all() the moment either fires.
* Triggers      NetworkMonitor polls for a default route / DNS resolution in
                a background thread (non-blocking) and publishes
                NETWORK_DOWN / NETWORK_UP. Any code - or a module - can also
                publish THREAT_DETECTED directly to trigger activation.

Module interface
-----------------
Each real module should be a .py file in --modules-dir (default: this
file's directory) exposing a class with:

    class Module:
        name = "sentinel_core"                 # optional, defaults to file stem
        subscribes = ["THREAT_DETECTED", "*"]  # event types this module wants

        async def start(self, bus: "EventBus") -> None: ...
        async def stop(self) -> None: ...
        async def handle(self, event: "Event") -> None: ...

`start` should return quickly; long-running work goes in a task the module
creates for itself (e.g. `self._task = asyncio.create_task(self._loop())`)
and cancels in `stop`. Raising from any of these never brings the bridge
down - the bridge isolates each module's exceptions and keeps the rest
running (see ModuleWrapper).

Stubbed by design (see note at the end of this file)
------------------------------------------------------
anti_forensic_shield, packet_destroyer and polymorphic_api_shield are
registered as inert stubs regardless of whether real files by those names
exist in --modules-dir: they log activation and do nothing else. Swap in
real behavior yourself if you intend something other than what those names
describe (see the note at the bottom of this file).

Usage
-----
    bridge = NexusBridge(modules_dir="./modules")
    bridge.load_all()
    asyncio.run(bridge.run_forever())

CLI:
    python nexus_bridge.py --modules-dir ./modules
    python nexus_bridge.py --modules-dir ./modules --simulate-threat
    python nexus_bridge.py --once                 # activate once and exit
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import logging
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("nexus_bridge")

# ======================================================================
# Event bus
# ======================================================================
WILDCARD = "*"

# Well-known event types. Modules are free to publish/subscribe to others.
NETWORK_DOWN = "NETWORK_DOWN"
NETWORK_UP = "NETWORK_UP"
THREAT_DETECTED = "THREAT_DETECTED"
MODULE_STARTED = "MODULE_STARTED"
MODULE_STOPPED = "MODULE_STOPPED"
MODULE_ERROR = "MODULE_ERROR"
ALERT = "ALERT"


@dataclasses.dataclass(frozen=True)
class Event:
    type: str
    source: str
    data: dict = dataclasses.field(default_factory=dict)
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = dataclasses.field(default_factory=time.time)


class EventBus:
    """In-memory async pub/sub. No disk, no network, no external deps."""

    def __init__(self, history_size=500):
        self._subscribers: dict[str, list[Callable]] = {}
        self._history: list[Event] = []
        self._history_size = history_size
        self._lock = asyncio.Lock()
        self._seq = 0

    def subscribe(self, event_type: str, handler: Callable[[Event], "asyncio.Future"]):
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler):
        if handler in self._subscribers.get(event_type, []):
            self._subscribers[event_type].remove(handler)

    async def publish(self, event: Event):
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history.pop(0)
        log.debug("event %-16s from %-28s id=%s", event.type, event.source, event.id)
        handlers = list(self._subscribers.get(event.type, [])) + \
            list(self._subscribers.get(WILDCARD, []))
        if not handlers:
            return
        results = await asyncio.gather(
            *(self._safe_call(h, event) for h in handlers), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                log.error("subscriber raised on %s: %r", event.type, r)

    @staticmethod
    async def _safe_call(handler, event):
        res = handler(event)
        if asyncio.iscoroutine(res):
            await res

    def recent(self, event_type: Optional[str] = None, limit=50):
        items = [e for e in self._history if event_type is None or e.type == event_type]
        return items[-limit:]


# ======================================================================
# Module interface + wrapper (isolates failures, enforces the contract)
# ======================================================================
class ModuleBase:
    """Modules may subclass this, or just duck-type the same methods."""
    name = "unnamed_module"
    subscribes: list[str] = [WILDCARD]

    async def start(self, bus: EventBus) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def handle(self, event: Event) -> None:
        pass


class StubModule(ModuleBase):
    """Used when a real module file is missing, broken, or deliberately stubbed."""

    def __init__(self, name, reason="no implementation loaded"):
        self.name = name
        self.reason = reason
        self.subscribes = [WILDCARD]
        self._bus = None

    async def start(self, bus: EventBus):
        self._bus = bus
        log.warning("STUB %-28s active (%s) - no real behavior", self.name, self.reason)

    async def stop(self):
        log.info("STUB %-28s stopped", self.name)

    async def handle(self, event: Event):
        log.debug("STUB %-28s saw %s (ignored)", self.name, event.type)


class ModuleWrapper:
    """Wraps one module instance: tracks lifecycle state, isolates exceptions
    so one bad module can't take the bridge or its siblings down."""

    def __init__(self, instance: ModuleBase):
        self.instance = instance
        self.name = getattr(instance, "name", type(instance).__name__)
        self.state = "idle"        # idle | starting | running | error | stopped
        self.last_error: Optional[str] = None
        self._bus: Optional[EventBus] = None
        self._handler = None

    async def start(self, bus: EventBus):
        self._bus = bus
        self.state = "starting"
        try:
            self._handler = self._on_event
            for etype in getattr(self.instance, "subscribes", [WILDCARD]):
                bus.subscribe(etype, self._handler)
            await self.instance.start(bus)
            self.state = "running"
            await bus.publish(Event(MODULE_STARTED, self.name))
        except Exception as e:
            self.state = "error"
            self.last_error = repr(e)
            log.exception("module %s failed to start", self.name)
            await bus.publish(Event(MODULE_ERROR, self.name, {"phase": "start", "error": repr(e)}))

    async def stop(self):
        if self._handler:
            for etype in getattr(self.instance, "subscribes", [WILDCARD]):
                self._bus.unsubscribe(etype, self._handler)
        try:
            await self.instance.stop()
        except Exception as e:
            log.exception("module %s failed to stop cleanly", self.name)
            self.last_error = repr(e)
        self.state = "stopped"
        if self._bus:
            await self._bus.publish(Event(MODULE_STOPPED, self.name))

    async def _on_event(self, event: Event):
        if event.source == self.name:
            return                                   # don't echo to self
        try:
            await self.instance.handle(event)
        except Exception as e:
            log.exception("module %s raised handling %s", self.name, event.type)
            if self._bus:
                await self._bus.publish(
                    Event(MODULE_ERROR, self.name,
                         {"phase": "handle", "event": event.type, "error": repr(e)}))


# ======================================================================
# Module discovery / loading
# ======================================================================
# Registered but kept inert on purpose - see the note at the end of this file.
FORCED_STUBS = {
    "anti_forensic_shield": "defeats forensic investigation (log/evidence wiping) rather than defending; not implemented",
    "packet_destroyer": "autonomous, unsupervised network disruption; not implemented",
    "polymorphic_api_shield": "self-mutating code to evade detection is a malware technique, not a shield; not implemented",
}

DEFAULT_MODULES = [
    "anti_forensic_shield", "auto_immune_patcher", "autonomous_counter_intel",
    "device_dna_guard", "encryption", "neural_threat_predictor",
    "neuro_phantom_mesh", "packet_destroyer", "polymorphic_api_shield",
    "quantum_blackhole_sync", "quantum_deception", "synthetic_agent_swarm",
    "temporal_entropy_vault", "trap_decoy_engine", "sentinel_core",
]


def _load_module_class(py_path: Path):
    """Import a .py file and return the class it exposes as `Module`
    (or the first ModuleBase subclass found), or None."""
    try:
        spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        log.error("failed to import %s: %r", py_path, e)
        return None
    if hasattr(mod, "Module"):
        return mod.Module
    for attr in vars(mod).values():
        if isinstance(attr, type) and issubclass(attr, ModuleBase) and attr is not ModuleBase:
            return attr
    return None


def build_module(name: str, modules_dir: Path) -> ModuleBase:
    if name in FORCED_STUBS:
        return StubModule(name, FORCED_STUBS[name])
    path = modules_dir / f"{name}.py"
    if not path.exists():
        return StubModule(name, f"{path.name} not found in {modules_dir}")
    cls = _load_module_class(path)
    if cls is None:
        return StubModule(name, f"{path.name} has no Module class / ModuleBase subclass")
    try:
        instance = cls()
        if not hasattr(instance, "name") or not instance.name:
            instance.name = name
        return instance
    except Exception as e:
        return StubModule(name, f"failed to instantiate: {e!r}")


# ======================================================================
# Network / threat triggers
# ======================================================================
class NetworkMonitor:
    """Polls connectivity in a background thread (non-blocking for the
    event loop) and publishes NETWORK_DOWN / NETWORK_UP transitions."""

    def __init__(self, bus: EventBus, loop: asyncio.AbstractEventLoop,
                probe=("1.1.1.1", 53), interval=5.0):
        self.bus, self.loop, self.probe, self.interval = bus, loop, probe, interval
        self._stop = False
        self._was_up = None
        self._thread = None

    def _check(self) -> bool:
        try:
            with socket.create_connection(self.probe, timeout=2):
                return True
        except OSError:
            return False

    def _run(self):
        import threading
        while not self._stop:
            up = self._check()
            if up != self._was_up:
                etype = NETWORK_UP if up else NETWORK_DOWN
                fut = asyncio.run_coroutine_threadsafe(
                    self.bus.publish(Event(etype, "network_monitor")), self.loop)
                try:
                    fut.result(timeout=5)
                except Exception as e:
                    log.error("failed to publish %s: %r", etype, e)
                self._was_up = up
            threading.Event().wait(self.interval)

    def start(self):
        import threading
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True


# ======================================================================
# Bridge
# ======================================================================
class NexusBridge:
    def __init__(self, modules_dir="./modules", module_names=None, activation_cooldown=10.0):
        self.modules_dir = Path(modules_dir)
        self.module_names = module_names or DEFAULT_MODULES
        self.bus = EventBus()
        self.wrappers: dict[str, ModuleWrapper] = {}
        self.activation_cooldown = activation_cooldown
        self._last_activation = 0.0
        self._active = False
        self._net_monitor: Optional[NetworkMonitor] = None

    def load_all(self):
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        for name in self.module_names:
            instance = build_module(name, self.modules_dir)
            self.wrappers[name] = ModuleWrapper(instance)
        log.info("loaded %d modules (%d stubbed)", len(self.wrappers),
                 sum(1 for w in self.wrappers.values() if isinstance(w.instance, StubModule)))

    async def activate_all(self, reason: str):
        now = time.time()
        if self._active and now - self._last_activation < self.activation_cooldown:
            log.debug("activation suppressed (cooldown): %s", reason)
            return
        self._last_activation = now
        self._active = True
        log.warning("ACTIVATING all modules: %s", reason)
        await self.bus.publish(Event(ALERT, "nexus_bridge", {"reason": reason}))
        await asyncio.gather(*(w.start(self.bus) for w in self.wrappers.values()))
        running = sum(1 for w in self.wrappers.values() if w.state == "running")
        log.warning("activation complete: %d/%d modules running", running, len(self.wrappers))

    async def deactivate_all(self, reason: str):
        if not self._active:
            return
        log.info("deactivating all modules: %s", reason)
        await asyncio.gather(*(w.stop() for w in self.wrappers.values()))
        self._active = False

    def status(self):
        return {name: {"state": w.state, "stub": isinstance(w.instance, StubModule),
                       "error": w.last_error} for name, w in self.wrappers.items()}

    # ---- wiring triggers to activation ----
    async def _on_network_down(self, event: Event):
        await self.activate_all(f"network down (source={event.source})")

    async def _on_threat(self, event: Event):
        await self.activate_all(f"threat detected: {event.data.get('summary', event.data)}")

    async def run_forever(self, enable_network_monitor=True):
        self.bus.subscribe(NETWORK_DOWN, self._on_network_down)
        self.bus.subscribe(THREAT_DETECTED, self._on_threat)
        if enable_network_monitor:
            self._net_monitor = NetworkMonitor(self.bus, asyncio.get_running_loop())
            self._net_monitor.start()
        log.info("nexus_bridge running; waiting for NETWORK_DOWN / THREAT_DETECTED")
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            if self._net_monitor:
                self._net_monitor.stop()
            await self.deactivate_all("shutdown")


# ======================================================================
# CLI
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modules-dir", default="./modules")
    ap.add_argument("--once", action="store_true", help="activate once, print status, exit")
    ap.add_argument("--simulate-threat", action="store_true",
                    help="publish a THREAT_DETECTED event shortly after startup")
    ap.add_argument("--no-network-monitor", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    bridge = NexusBridge(modules_dir=args.modules_dir)
    bridge.load_all()

    async def run():
        if args.once:
            await bridge.activate_all("manual (--once)")
            for name, info in bridge.status().items():
                print(f"{name:28s} {info['state']:9s} stub={info['stub']} {info['error'] or ''}")
            await bridge.deactivate_all("--once complete")
            return
        task = asyncio.create_task(bridge.run_forever(not args.no_network_monitor))
        if args.simulate_threat:
            await asyncio.sleep(2)
            await bridge.bus.publish(Event(THREAT_DETECTED, "cli",
                                           {"summary": "simulated threat (--simulate-threat)"}))
        await task

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
# Note on anti_forensic_shield, packet_destroyer, polymorphic_api_shield
# ----------------------------------------------------------------------
# These three are registered in FORCED_STUBS above and will stay inert
# (log-only) even if real files by these names exist in --modules-dir.
# That's deliberate: as named, they describe evidence destruction,
# unsupervised network disruption, and detection-evasion - techniques
# associated with malware rather than defense. If what you actually have
# in mind is different (e.g. tamper-evident logging, automated blocking
# of traffic already confirmed malicious by sentinel_core, or key/schema
# rotation for your APIs), rename the module to describe that and it will
# load normally like the others, or tell me what it does and I'll wire it
# up properly.
