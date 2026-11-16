"""Top-level supervisor wiring all subsystems together."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .capability_manager import Capability, CapabilityManager
from .cognitive_architecture import (CognitiveArchitecture, CognitiveModule,
                                     PerceptionModule, PlanningModule, WorkingMemoryModule)
from .cross_domain_transfer import CrossDomainTransferEngine
from .general_intelligence import GeneralIntelligence, Problem, Solution
from .intelligence_orchestrator import IntelligenceOrchestrator

log = logging.getLogger(__name__)
SafetyGuard = Callable[[Problem], str | None]  # return a reason string to block
EventHandler = Callable[[str, dict[str, Any]], Awaitable[None] | None]


@dataclass
class EngineConfig:
    workers: int = 4
    task_timeout_s: float = 30.0
    max_queue: int = 1000
    load_default_modules: bool = True


class ASIEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.architecture = CognitiveArchitecture()
        self.capabilities = CapabilityManager()
        self.transfer = CrossDomainTransferEngine()
        self.intelligence = GeneralIntelligence(self.architecture, self.capabilities, self.transfer)
        self.orchestrator = IntelligenceOrchestrator(
            self.intelligence, self.config.workers,
            self.config.task_timeout_s, self.config.max_queue)
        self._guards: list[SafetyGuard] = []
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self.running = False
        if self.config.load_default_modules:
            for m in (PerceptionModule(), WorkingMemoryModule(), PlanningModule()):
                self.architecture.register(m)

    # ---- extension points
    def register_module(self, m: CognitiveModule) -> None: self.architecture.register(m)
    def register_capability(self, c: Capability) -> None: self.capabilities.register(c)
    def add_safety_guard(self, g: SafetyGuard) -> None: self._guards.append(g)
    def on(self, event: str, h: EventHandler) -> None: self._handlers[event].append(h)

    async def _emit(self, event: str, **payload: Any) -> None:
        for h in self._handlers[event]:
            try:
                r = h(event, payload)
                if r is not None:
                    await r
            except Exception:
                log.exception("event handler failed for %s", event)

    # ---- lifecycle
    async def start(self) -> None:
        if self.running:
            return
        await self.architecture.initialize_all()
        await self.orchestrator.start()
        self.running = True
        await self._emit("engine.started")

    async def stop(self) -> None:
        if not self.running:
            return
        await self.orchestrator.stop()
        await self.architecture.shutdown_all()
        self.running = False
        await self._emit("engine.stopped")

    async def __aenter__(self) -> "ASIEngine":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ---- main entry
    async def solve(self, problem: Problem, priority: int = 5) -> Solution:
        if not self.running:
            raise RuntimeError("engine not running")
        for g in self._guards:
            reason = g(problem)
            if reason:
                await self._emit("problem.blocked", id=problem.id, reason=reason)
                return Solution(problem.id, False, error=f"blocked by safety guard: {reason}")
        sol = await self.orchestrator.run(problem, priority)
        await self._emit("problem.solved", id=problem.id, success=sol.success)
        return sol

    def health(self) -> dict[str, Any]:
        return {"running": self.running, "modules": self.architecture.status(),
                "capabilities": [c.name for c in self.capabilities.list()],
                "orchestrator": self.orchestrator.metrics(),
                "knowledge_units": len(self.transfer)}
