"""Capability-based task delegation with load balancing, retries and failover."""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
import time
from enum import Enum
from typing import Any, Collection, Dict, List, Sequence

from .agent_core import (
    AgentError,
    BaseAgent,
    NoCapableAgentError,
    Task,
    TaskResult,
    TaskStatus,
)
from .agent_registry import AgentRegistry

logger = logging.getLogger("zentrax.agents.delegation")

__all__ = ["DelegationStrategy", "TaskDelegator"]


class DelegationStrategy(str, Enum):
    LEAST_LOADED = "least_loaded"
    ROUND_ROBIN = "round_robin"
    BEST_MATCH = "best_match"  # prefer specialists (fewest unrelated capabilities)
    RANDOM = "random"


class TaskDelegator:
    """Selects an agent for a task, runs it, and retries on other agents on failure."""

    def __init__(
        self,
        registry: AgentRegistry,
        strategy: DelegationStrategy = DelegationStrategy.LEAST_LOADED,
        *,
        retry_backoff: float = 0.25,
        max_backoff: float = 8.0,
    ) -> None:
        self._registry = registry
        self.strategy = strategy
        self._retry_backoff = retry_backoff
        self._max_backoff = max_backoff
        self._round_robin: Dict[frozenset, int] = {}
        self.stats: Dict[str, int] = {"delegated": 0, "succeeded": 0, "failed": 0, "retries": 0}

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #

    def select_agent(self, task: Task, exclude: Collection[str] = ()) -> BaseAgent:
        """Pick the best agent for ``task`` or raise ``NoCapableAgentError``."""
        if task.target_agent:
            agent = self._registry.get(task.target_agent)
            if agent is None or not agent.is_running:
                raise NoCapableAgentError(f"target agent {task.target_agent!r} is not running")
            if not task.required_capabilities <= agent.capabilities:
                raise NoCapableAgentError(f"target agent {agent.id} lacks required capabilities")
            return agent

        candidates = self._registry.find(task.required_capabilities, available_only=True, exclude=exclude)
        if not candidates:  # everyone capable is saturated: queue on the least loaded one
            candidates = self._registry.find(task.required_capabilities, exclude=exclude)
        if not candidates:
            raise NoCapableAgentError(
                f"no running agent provides capabilities {sorted(task.required_capabilities)}"
            )
        return self._choose(candidates, task)

    def _choose(self, candidates: Sequence[BaseAgent], task: Task) -> BaseAgent:
        if len(candidates) == 1:
            return candidates[0]
        if self.strategy == DelegationStrategy.RANDOM:
            return random.choice(list(candidates))
        if self.strategy == DelegationStrategy.ROUND_ROBIN:
            key = task.required_capabilities
            index = self._round_robin.get(key, 0)
            self._round_robin[key] = index + 1
            ordered = sorted(candidates, key=lambda a: a.id)
            return ordered[index % len(ordered)]
        if self.strategy == DelegationStrategy.BEST_MATCH:
            return min(
                candidates,
                key=lambda a: (len(a.capabilities - task.required_capabilities), a.load, a.metrics.avg_duration),
            )
        return min(candidates, key=lambda a: (a.load, a.metrics.avg_duration))

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    async def delegate(self, task: Task) -> TaskResult:
        """Run ``task`` on a suitable agent. Never raises (except on cancellation)."""
        self.stats["delegated"] += 1
        task.status = TaskStatus.QUEUED
        attempts_allowed = task.max_retries + 1
        tried: set = set()
        last_error = "task was not executed"
        result: TaskResult = TaskResult.failure(task.id, last_error)

        for attempt in range(attempts_allowed):
            try:
                try:
                    agent = self.select_agent(task, exclude=tried)
                except NoCapableAgentError:
                    if not tried:
                        raise
                    agent = self.select_agent(task)  # every capable agent already tried: allow reuse
            except NoCapableAgentError as exc:
                result = TaskResult.failure(task.id, str(exc))
                break

            task.assigned_to = agent.id
            task.attempts += 1
            task.status = TaskStatus.RUNNING
            task.started_at = task.started_at or time.time()
            try:
                result = await agent.execute(task)
            except AgentError as exc:
                result = TaskResult.failure(task.id, f"{type(exc).__name__}: {exc}", agent.id)

            if result.success:
                break
            tried.add(agent.id)
            logger.warning(
                "task %s failed on %s (attempt %d/%d): %s", task.id, agent.id, attempt + 1, attempts_allowed, result.error
            )
            if attempt < attempts_allowed - 1:
                self.stats["retries"] += 1
                delay = min(self._retry_backoff * (2**attempt), self._max_backoff)
                await asyncio.sleep(delay)

        task.finished_at = time.time()
        task.result = result
        task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        self.stats["succeeded" if result.success else "failed"] += 1
        return result

    async def delegate_many(self, tasks: Sequence[Task]) -> List[TaskResult]:
        """Run independent tasks concurrently; results are in input order."""
        return list(await asyncio.gather(*(self.delegate(t) for t in tasks)))
