"""Registry of live agent instances and of agent *types* (factories)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Collection, Dict, Iterable, Iterator, List, Optional

from .agent_core import AgentConfig, BaseAgent, RegistryError

logger = logging.getLogger("zentrax.agents.registry")

__all__ = ["AgentRegistry", "AgentFactory"]

#: Anything that builds an agent from keyword arguments (usually the class itself).
AgentFactory = Callable[..., BaseAgent]


class AgentRegistry:
    """Tracks agent instances and knows how to build new ones by type name."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self._factories: Dict[str, AgentFactory] = {}

    # ------------------------------------------------------------------ #
    # Agent types
    # ------------------------------------------------------------------ #

    def register_type(self, type_name: str, factory: AgentFactory, *, overwrite: bool = False) -> None:
        if type_name in self._factories and not overwrite:
            raise RegistryError(f"agent type {type_name!r} is already registered")
        self._factories[type_name] = factory

    def unregister_type(self, type_name: str) -> None:
        self._factories.pop(type_name, None)

    @property
    def types(self) -> List[str]:
        return sorted(self._factories)

    def create(self, type_name: str, config: Optional[AgentConfig] = None, **kwargs: Any) -> BaseAgent:
        """Build (but do not register or start) an agent of ``type_name``."""
        factory = self._factories.get(type_name)
        if factory is None:
            raise RegistryError(f"unknown agent type {type_name!r}; known types: {self.types}")
        if config is not None:
            kwargs["config"] = config
        return factory(**kwargs)

    # ------------------------------------------------------------------ #
    # Agent instances
    # ------------------------------------------------------------------ #

    def add(self, agent: BaseAgent) -> None:
        existing = self._agents.get(agent.id)
        if existing is not None and existing is not agent:
            raise RegistryError(f"agent id {agent.id!r} is already registered")
        self._agents[agent.id] = agent

    def remove(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_id)

    def require(self, agent_id: str) -> BaseAgent:
        agent = self._agents.get(agent_id)
        if agent is None:
            raise RegistryError(f"unknown agent {agent_id!r}")
        return agent

    def all(self) -> List[BaseAgent]:
        return list(self._agents.values())

    def by_type(self, agent_type: str) -> List[BaseAgent]:
        return [a for a in self._agents.values() if a.agent_type == agent_type]

    def find(
        self,
        capabilities: Iterable[str] = (),
        *,
        available_only: bool = False,
        running_only: bool = True,
        exclude: Collection[str] = (),
    ) -> List[BaseAgent]:
        """Agents that have *all* ``capabilities`` (and optionally are free/running)."""
        required = frozenset(capabilities)
        matches: List[BaseAgent] = []
        for agent in self._agents.values():
            if agent.id in exclude or not required <= agent.capabilities:
                continue
            if running_only and not agent.is_running:
                continue
            if available_only and not agent.is_available:
                continue
            matches.append(agent)
        return matches

    def capability_index(self) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}
        for agent in self._agents.values():
            for capability in agent.capabilities:
                index.setdefault(capability, []).append(agent.id)
        return index

    def snapshot(self) -> List[Dict[str, Any]]:
        return [agent.health() for agent in self._agents.values()]

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._agents

    def __iter__(self) -> Iterator[BaseAgent]:
        return iter(list(self._agents.values()))
