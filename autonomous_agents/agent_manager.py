"""Lifecycle management: spawn, terminate, restart and health-monitor agents."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .agent_communication import MessageBus
from .agent_core import AgentConfig, AgentState, BaseAgent
from .agent_registry import AgentRegistry

logger = logging.getLogger("zentrax.agents.manager")

__all__ = ["AgentManager"]


class AgentManager:
    """Owns agent lifecycles and keeps registry + bus consistent."""

    def __init__(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        *,
        health_check_interval: float = 15.0,
        health_check_timeout: float = 5.0,
        auto_restart: bool = True,
        max_restarts: int = 3,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._interval = health_check_interval
        self._check_timeout = health_check_timeout
        self._auto_restart = auto_restart
        self._max_restarts = max_restarts
        self._restarts: Dict[str, int] = defaultdict(int)
        self._monitor_task: Optional["asyncio.Task[None]"] = None

    # ------------------------------------------------------------------ #
    # Create / destroy
    # ------------------------------------------------------------------ #

    async def spawn(
        self,
        type_name: str,
        config: Optional[AgentConfig] = None,
        *,
        start: bool = True,
        **kwargs: Any,
    ) -> BaseAgent:
        """Create an agent of a registered type, register it and (optionally) start it."""
        agent = self._registry.create(type_name, config, **kwargs)
        return await self.adopt(agent, start=start)

    async def spawn_many(
        self, type_name: str, count: int, config: Optional[AgentConfig] = None, **kwargs: Any
    ) -> List[BaseAgent]:
        return list(await asyncio.gather(*(self.spawn(type_name, config, **kwargs) for _ in range(count))))

    async def adopt(self, agent: BaseAgent, *, start: bool = True) -> BaseAgent:
        """Register an externally constructed agent."""
        self._registry.add(agent)
        try:
            self._bus.register(agent)
        except Exception:
            self._registry.remove(agent.id)
            raise
        if start:
            try:
                await agent.start()
            except Exception:
                self._bus.unregister(agent.id)
                self._registry.remove(agent.id)
                raise
        return agent

    async def terminate(self, agent_id: str, *, timeout: float = 10.0, drain: bool = True) -> None:
        agent = self._registry.require(agent_id)
        await agent.stop(timeout=timeout, drain=drain)
        self._bus.unregister(agent_id)
        self._registry.remove(agent_id)
        self._restarts.pop(agent_id, None)

    async def restart(self, agent_id: str) -> BaseAgent:
        agent = self._registry.require(agent_id)
        self._restarts[agent_id] += 1
        await agent.stop(timeout=5.0, drain=False)
        await agent.start()
        logger.info("restarted agent %s (restart #%d)", agent_id, self._restarts[agent_id])
        return agent

    async def start_all(self) -> None:
        await asyncio.gather(*(a.start() for a in self._registry.all()))

    # ------------------------------------------------------------------ #
    # Monitoring
    # ------------------------------------------------------------------ #

    async def start_monitoring(self) -> None:
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop(), name="agent-health-monitor")

    async def stop_monitoring(self) -> None:
        task, self._monitor_task = self._monitor_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.check_health()
            except Exception:  # noqa: BLE001
                logger.exception("health check sweep failed")

    async def check_health(self) -> Dict[str, bool]:
        """Probe every running/failed agent; restart unhealthy ones if enabled."""
        report: Dict[str, bool] = {}
        for agent in self._registry.all():
            if agent.state in (
                AgentState.CREATED,
                AgentState.STARTING,
                AgentState.STOPPING,
                AgentState.STOPPED,
                AgentState.PAUSED,
            ):
                continue  # intentionally not running
            healthy = agent.state != AgentState.FAILED
            if healthy:
                try:
                    healthy = bool(await asyncio.wait_for(agent.health_check(), timeout=self._check_timeout))
                except Exception:  # noqa: BLE001
                    healthy = False
            report[agent.id] = healthy
            if healthy or not self._auto_restart:
                continue
            if self._restarts[agent.id] >= self._max_restarts:
                logger.error("agent %s unhealthy; restart budget exhausted", agent.id)
                continue
            try:
                await self.restart(agent.id)
            except Exception:  # noqa: BLE001
                logger.exception("failed to restart agent %s", agent.id)
        return report

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #

    async def shutdown(self, timeout: float = 10.0) -> None:
        await self.stop_monitoring()
        agents = self._registry.all()
        await asyncio.gather(*(a.stop(timeout=timeout) for a in agents), return_exceptions=True)
        for agent in agents:
            self._bus.unregister(agent.id)
            self._registry.remove(agent.id)
        self._restarts.clear()
