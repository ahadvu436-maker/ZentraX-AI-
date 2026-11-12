"""High-level facade that wires the whole agent ecosystem together."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Sequence

from .agent_collaboration import CollaborationManager
from .agent_communication import MessageBus
from .agent_core import AgentConfig, BaseAgent, Reasoner, Task, TaskResult
from .agent_manager import AgentManager
from .agent_orchestrator import AgentOrchestrator, WorkflowResult
from .agent_registry import AgentFactory, AgentRegistry
from .agents import BUILTIN_AGENT_TYPES, SearchProvider
from .task_delegation import DelegationStrategy, TaskDelegator

logger = logging.getLogger("zentrax.agents.system")

__all__ = ["MultiAgentSystem"]


class MultiAgentSystem:
    """Bus + registry + manager + delegator + orchestrator + collaboration, in one object.

    Example::

        async with MultiAgentSystem() as system:
            await system.spawn("research")
            result = await system.execute("look up", {"query": "async python"}, ["research"])
    """

    def __init__(
        self,
        *,
        strategy: DelegationStrategy = DelegationStrategy.LEAST_LOADED,
        max_concurrency: int = 10,
        queue_workers: int = 4,
        health_check_interval: float = 15.0,
        auto_restart: bool = True,
        register_builtin: bool = True,
    ) -> None:
        self.bus = MessageBus()
        self.registry = AgentRegistry()
        self.manager = AgentManager(
            self.registry, self.bus, health_check_interval=health_check_interval, auto_restart=auto_restart
        )
        self.delegator = TaskDelegator(self.registry, strategy)
        self.orchestrator = AgentOrchestrator(self.delegator, max_concurrency=max_concurrency)
        self.collaboration = CollaborationManager(self.delegator)
        self._queue_workers = queue_workers
        self._started = False
        if register_builtin:
            for name, cls in BUILTIN_AGENT_TYPES.items():
                self.registry.register_type(name, cls)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._started

    async def start(self) -> "MultiAgentSystem":
        if not self._started:
            await self.manager.start_all()
            await self.manager.start_monitoring()
            await self.orchestrator.start(self._queue_workers)
            self._started = True
        return self

    async def stop(self) -> None:
        if self._started:
            await self.orchestrator.stop()
            await self.manager.shutdown()
            self._started = False

    async def __aenter__(self) -> "MultiAgentSystem":
        return await self.start()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    @classmethod
    async def with_default_agents(
        cls,
        *,
        llm: Optional[Reasoner] = None,
        search_providers: Sequence[SearchProvider] = (),
        allow_code_execution: bool = False,
        **kwargs: Any,
    ) -> "MultiAgentSystem":
        """Create a started system with one agent of each built-in type."""
        system = cls(**kwargs)
        await system.start()
        await system.spawn("research", llm=llm, search_providers=search_providers)
        await system.spawn("coding", llm=llm, allow_execution=allow_code_execution)
        await system.spawn("analysis", llm=llm)
        await system.spawn("planning", llm=llm)
        return system

    # ------------------------------------------------------------------ #
    # Agents
    # ------------------------------------------------------------------ #

    def register_agent_type(self, name: str, factory: AgentFactory, *, overwrite: bool = False) -> None:
        self.registry.register_type(name, factory, overwrite=overwrite)

    async def spawn(self, type_name: str, config: Optional[AgentConfig] = None, **kwargs: Any) -> BaseAgent:
        return await self.manager.spawn(type_name, config, **kwargs)

    async def adopt(self, agent: BaseAgent) -> BaseAgent:
        return await self.manager.adopt(agent)

    async def terminate(self, agent_id: str) -> None:
        await self.manager.terminate(agent_id)

    # ------------------------------------------------------------------ #
    # Work
    # ------------------------------------------------------------------ #

    async def run_task(self, task: Task) -> TaskResult:
        return await self.delegator.delegate(task)

    async def execute(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        capabilities: Iterable[str] = (),
        **task_kwargs: Any,
    ) -> TaskResult:
        """Build and run a task in one call."""
        task = Task(name=name, payload=payload or {}, required_capabilities=frozenset(capabilities), **task_kwargs)
        return await self.delegator.delegate(task)

    async def run_workflow(self, tasks: Sequence[Task], **kwargs: Any) -> WorkflowResult:
        return await self.orchestrator.run_workflow(tasks, **kwargs)

    async def submit(self, task: Task) -> TaskResult:
        """Run via the background priority queue and wait for the result."""
        return await self.orchestrator.submit_and_wait(task)

    # ------------------------------------------------------------------ #

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._started,
            "agent_types": self.registry.types,
            "agents": self.registry.snapshot(),
            "capabilities": self.registry.capability_index(),
            "bus": dict(self.bus.stats),
            "delegation": dict(self.delegator.stats),
        }
