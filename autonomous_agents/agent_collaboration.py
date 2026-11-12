"""Multi-agent collaboration patterns on top of the task delegator.

Patterns
--------
* ``pipeline``     sequential stages; each stage sees the previous output
* ``fan_out``      parallel tasks with optional aggregation
* ``vote``         the same task on several agents; majority wins
* ``review_loop``  producer/reviewer iterations until approved

A :class:`Blackboard` provides shared, versioned state that collaborating
agents (or the caller) can read, write and watch.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .agent_core import Task, TaskResult, new_id
from .task_delegation import TaskDelegator

logger = logging.getLogger("zentrax.agents.collaboration")

__all__ = ["Blackboard", "CollaborationManager", "CollaborationResult"]

Watcher = Callable[[str, Any, Optional[str]], Union[None, Awaitable[None]]]


@dataclass
class CollaborationResult:
    pattern: str
    success: bool
    output: Any = None
    results: List[TaskResult] = field(default_factory=list)
    rounds: int = 1
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Blackboard:
    """Shared versioned key/value workspace with change watchers."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or new_id("bb-")
        self._data: Dict[str, Any] = {}
        self._versions: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._watchers: Dict[str, List[Watcher]] = {}
        self.history: List[Dict[str, Any]] = []

    def watch(self, key: str, callback: Watcher) -> None:
        """Call ``callback(key, value, author)`` whenever ``key`` changes (``"*"`` = any key)."""
        self._watchers.setdefault(key, []).append(callback)

    async def set(self, key: str, value: Any, author: Optional[str] = None) -> int:
        async with self._lock:
            version = self._write(key, value, author)
        await self._notify(key, value, author)
        return version

    async def update(
        self, key: str, fn: Callable[[Any], Any], default: Any = None, author: Optional[str] = None
    ) -> Any:
        """Atomic read-modify-write."""
        async with self._lock:
            value = fn(self._data.get(key, default))
            self._write(key, value, author)
        await self._notify(key, value, author)
        return value

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._data.get(key, default)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            existed = key in self._data
            self._data.pop(key, None)
            self._versions.pop(key, None)
            return existed

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return dict(self._data)

    def _write(self, key: str, value: Any, author: Optional[str]) -> int:
        self._data[key] = value
        self._versions[key] = self._versions.get(key, 0) + 1
        self.history.append({"key": key, "version": self._versions[key], "author": author, "at": time.time()})
        return self._versions[key]

    async def _notify(self, key: str, value: Any, author: Optional[str]) -> None:
        for callback in self._watchers.get(key, []) + self._watchers.get("*", []):
            try:
                outcome = callback(key, value, author)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:  # noqa: BLE001
                logger.exception("blackboard watcher failed for key %r", key)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr(value)


def _default_approved(output: Any) -> bool:
    if isinstance(output, bool):
        return output
    if isinstance(output, Mapping):
        if "approved" in output:
            return bool(output["approved"])
        if "issues" in output:
            return output.get("valid", True) and not any(i.get("severity") == "error" for i in output["issues"])
    return False


class CollaborationManager:
    def __init__(self, delegator: TaskDelegator) -> None:
        self._delegator = delegator
        self._boards: Dict[str, Blackboard] = {}

    # ------------------------------------------------------------------ #
    # Blackboards
    # ------------------------------------------------------------------ #

    def create_blackboard(self, session_id: Optional[str] = None) -> Blackboard:
        board = Blackboard(session_id)
        self._boards[board.session_id] = board
        return board

    def get_blackboard(self, session_id: str) -> Optional[Blackboard]:
        return self._boards.get(session_id)

    def close_blackboard(self, session_id: str) -> None:
        self._boards.pop(session_id, None)

    # ------------------------------------------------------------------ #
    # Patterns
    # ------------------------------------------------------------------ #

    async def pipeline(
        self,
        stages: Sequence[Task],
        *,
        initial_input: Any = None,
        inject_key: Optional[str] = None,
        stop_on_failure: bool = True,
        blackboard: Optional[Blackboard] = None,
    ) -> CollaborationResult:
        """Run ``stages`` in order. Each stage gets ``context["previous_output"]``
        (and ``payload[inject_key]`` if ``inject_key`` is given)."""
        started = time.monotonic()
        results: List[TaskResult] = []
        errors: List[str] = []
        previous = initial_input
        for index, stage in enumerate(stages):
            stage.context["previous_output"] = previous
            stage.context["stage_index"] = index
            if inject_key and previous is not None:
                stage.payload[inject_key] = previous
            result = await self._delegator.delegate(stage)
            results.append(result)
            if blackboard is not None:
                await blackboard.set(f"stage:{index}:{stage.name}", result.output if result.success else None, result.agent_id)
            if result.success:
                previous = result.output
                continue
            errors.append(f"stage {index} ({stage.name}): {result.error}")
            if stop_on_failure:
                break
        success = len(results) == len(stages) and not errors
        return CollaborationResult("pipeline", success, previous, results, 1, time.monotonic() - started, errors)

    async def fan_out(
        self,
        tasks: Sequence[Task],
        *,
        aggregator: Optional[Callable[[List[TaskResult]], Any]] = None,
        min_success: Optional[int] = None,
        blackboard: Optional[Blackboard] = None,
    ) -> CollaborationResult:
        """Run ``tasks`` in parallel. ``aggregator`` receives the successful results."""
        started = time.monotonic()
        results = await self._delegator.delegate_many(tasks)
        ok = [r for r in results if r.success]
        errors = [f"{t.name}: {r.error}" for t, r in zip(tasks, results) if not r.success]
        if blackboard is not None:
            for task, result in zip(tasks, results):
                await blackboard.set(f"result:{task.name}", result.output if result.success else None, result.agent_id)
        required = min_success if min_success is not None else len(tasks)
        output: Any
        if aggregator is not None:
            output = aggregator(ok)
            if inspect.isawaitable(output):
                output = await output
        else:
            output = [r.output for r in ok]
        success = bool(tasks) and len(ok) >= required
        return CollaborationResult("fan_out", success, output, results, 1, time.monotonic() - started, errors)

    async def vote(
        self,
        task: Task,
        *,
        voters: int = 3,
        quorum: Optional[int] = None,
        key: Optional[Callable[[Any], Any]] = None,
    ) -> CollaborationResult:
        """Run ``task`` on up to ``voters`` distinct agents; the most common output wins."""
        started = time.monotonic()
        capable = sorted(self._delegator.registry.find(task.required_capabilities), key=lambda a: a.load)[: max(1, voters)]
        if not capable:
            return CollaborationResult(
                "vote", False, None, [], 1, time.monotonic() - started, ["no capable agents available to vote"]
            )
        clones = [task.clone(target_agent=agent.id, max_retries=0) for agent in capable]
        results = await self._delegator.delegate_many(clones)
        ballots: Counter = Counter()
        samples: Dict[str, Any] = {}
        for result in results:
            if result.success:
                ballot = _canonical(key(result.output) if key else result.output)
                ballots[ballot] += 1
                samples.setdefault(ballot, result.output)
        errors = [r.error or "failed" for r in results if not r.success]
        if not ballots:
            return CollaborationResult("vote", False, None, results, 1, time.monotonic() - started, errors)
        winner, count = ballots.most_common(1)[0]
        needed = quorum if quorum is not None else len(capable) // 2 + 1
        return CollaborationResult(
            "vote",
            count >= needed,
            samples[winner],
            results,
            1,
            time.monotonic() - started,
            errors,
            {"voters": len(capable), "winner_votes": count, "agreement": round(count / len(capable), 3), "needed": needed},
        )

    async def review_loop(
        self,
        producer: Task,
        reviewer: Task,
        *,
        max_rounds: int = 3,
        is_approved: Optional[Callable[[Any], bool]] = None,
    ) -> CollaborationResult:
        """Producer creates an artifact, reviewer critiques it; repeat until approved.

        The producer sees ``context["feedback"]`` / ``["previous_artifact"]``;
        the reviewer sees ``context["artifact"]``.
        """
        started = time.monotonic()
        approve = is_approved or _default_approved
        results: List[TaskResult] = []
        artifact: Any = None
        feedback: Any = None
        for round_no in range(1, max(1, max_rounds) + 1):
            p = producer.clone()
            p.context.update({"round": round_no, "feedback": feedback, "previous_artifact": artifact})
            produced = await self._delegator.delegate(p)
            results.append(produced)
            if not produced.success:
                return CollaborationResult(
                    "review_loop", False, artifact, results, round_no, time.monotonic() - started,
                    [f"producer failed in round {round_no}: {produced.error}"],
                )
            artifact = produced.output

            r = reviewer.clone()
            r.context.update({"round": round_no, "artifact": artifact})
            reviewed = await self._delegator.delegate(r)
            results.append(reviewed)
            if not reviewed.success:
                return CollaborationResult(
                    "review_loop", False, artifact, results, round_no, time.monotonic() - started,
                    [f"reviewer failed in round {round_no}: {reviewed.error}"],
                )
            if approve(reviewed.output):
                return CollaborationResult(
                    "review_loop", True, artifact, results, round_no, time.monotonic() - started, [],
                    {"review": reviewed.output},
                )
            feedback = reviewed.output
        return CollaborationResult(
            "review_loop", False, artifact, results, max_rounds, time.monotonic() - started,
            [f"not approved after {max_rounds} round(s)"], {"review": feedback},
        )
