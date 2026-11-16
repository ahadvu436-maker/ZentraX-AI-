"""Cognitive architecture: abstract module contract + dependency-ordered pipeline."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class ModuleStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class CognitiveContext:
    """Mutable state passed through the pipeline."""
    task_id: str
    data: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


class CognitiveModule(ABC):
    """Base class for every cognitive module (perception, memory, planning...)."""
    name: str = "unnamed"
    dependencies: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.status = ModuleStatus.CREATED

    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def process(self, ctx: CognitiveContext) -> CognitiveContext: ...


class PerceptionModule(CognitiveModule):
    name = "perception"

    async def process(self, ctx: CognitiveContext) -> CognitiveContext:
        ctx.data["percept"] = str(ctx.data.get("input", "")).strip()
        ctx.trace.append("perceived input")
        return ctx


class WorkingMemoryModule(CognitiveModule):
    name = "working_memory"
    dependencies = ("perception",)

    def __init__(self, capacity: int = 64) -> None:
        super().__init__()
        self.capacity = capacity
        self._items: list[Any] = []

    async def process(self, ctx: CognitiveContext) -> CognitiveContext:
        self._items.append(ctx.data.get("percept"))
        self._items = self._items[-self.capacity:]
        ctx.data["recent_memory"] = list(self._items)
        ctx.trace.append("updated working memory")
        return ctx


class PlanningModule(CognitiveModule):
    name = "planning"
    dependencies = ("working_memory",)

    async def process(self, ctx: CognitiveContext) -> CognitiveContext:
        ctx.data["plan"] = ["abstract", "transfer", "select_capability", "execute", "reflect"]
        ctx.trace.append("planned")
        return ctx


class CognitiveArchitecture:
    """Registry + executor that runs modules in dependency (topological) order."""

    def __init__(self) -> None:
        self._modules: dict[str, CognitiveModule] = {}

    def register(self, module: CognitiveModule) -> None:
        if module.name in self._modules:
            raise ValueError(f"module already registered: {module.name}")
        self._modules[module.name] = module

    def unregister(self, name: str) -> None:
        self._modules.pop(name, None)

    def get(self, name: str) -> CognitiveModule:
        return self._modules[name]

    def execution_order(self) -> list[CognitiveModule]:
        order: list[CognitiveModule] = []
        state: dict[str, int] = {}

        def visit(n: str) -> None:
            if state.get(n) == 2:
                return
            if state.get(n) == 1:
                raise ValueError(f"dependency cycle at '{n}'")
            if n not in self._modules:
                raise KeyError(f"missing dependency '{n}'")
            state[n] = 1
            for d in self._modules[n].dependencies:
                visit(d)
            state[n] = 2
            order.append(self._modules[n])

        for name in self._modules:
            visit(name)
        return order

    async def initialize_all(self) -> None:
        for m in self.execution_order():
            try:
                await m.initialize()
                m.status = ModuleStatus.READY
            except Exception:
                m.status = ModuleStatus.FAILED
                log.exception("module %s failed to initialize", m.name)
                raise

    async def run_pipeline(self, ctx: CognitiveContext) -> CognitiveContext:
        for m in self.execution_order():
            if m.status is not ModuleStatus.READY:
                raise RuntimeError(f"module '{m.name}' not ready ({m.status.value})")
            ctx = await m.process(ctx)
        return ctx

    async def shutdown_all(self) -> None:
        for m in reversed(self.execution_order()):
            try:
                await m.shutdown()
            finally:
                m.status = ModuleStatus.STOPPED

    def status(self) -> dict[str, str]:
        return {n: m.status.value for n, m in self._modules.items()}
