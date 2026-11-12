"""Workflow orchestration: dependency graphs and a priority task queue."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agent_core import Task, TaskResult, TaskStatus, WorkflowError, new_id
from .task_delegation import TaskDelegator

logger = logging.getLogger("zentrax.agents.orchestrator")

__all__ = ["AgentOrchestrator", "WorkflowResult"]


@dataclass
class WorkflowResult:
    workflow_id: str
    results: Dict[str, TaskResult] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    duration: float = 0.0

    @property
    def success(self) -> bool:
        return not self.skipped and all(r.success for r in self.results.values())

    @property
    def outputs(self) -> Dict[str, Any]:
        return {tid: r.output for tid, r in self.results.items() if r.success}

    @property
    def errors(self) -> Dict[str, str]:
        return {tid: (r.error or "unknown error") for tid, r in self.results.items() if not r.success}


class AgentOrchestrator:
    """Runs task graphs (respecting dependencies) and a background priority queue."""

    def __init__(self, delegator: TaskDelegator, *, max_concurrency: int = 10) -> None:
        self._delegator = delegator
        self._max_concurrency = max_concurrency
        self._queue: Optional["asyncio.PriorityQueue[Tuple[int, int, Task, asyncio.Future[TaskResult]]]"] = None
        self._workers: List["asyncio.Task[None]"] = []
        self._seq = itertools.count()

    # ------------------------------------------------------------------ #
    # Workflows
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate(tasks: Sequence[Task]) -> List[str]:
        """Check ids/dependencies and return a topological order of task ids."""
        by_id: Dict[str, Task] = {}
        for task in tasks:
            if task.id in by_id:
                raise WorkflowError(f"duplicate task id {task.id!r}")
            by_id[task.id] = task
        indegree: Dict[str, int] = {}
        children: Dict[str, List[str]] = defaultdict(list)
        for task in tasks:
            deps = set(task.dependencies)
            for dep in deps:
                if dep not in by_id:
                    raise WorkflowError(f"task {task.id!r} depends on unknown task {dep!r}")
                if dep == task.id:
                    raise WorkflowError(f"task {task.id!r} depends on itself")
                children[dep].append(task.id)
            indegree[task.id] = len(deps)
        queue = deque(tid for tid, deg in indegree.items() if deg == 0)
        order: List[str] = []
        while queue:
            tid = queue.popleft()
            order.append(tid)
            for child in children[tid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(tasks):
            cyclic = sorted(tid for tid, deg in indegree.items() if deg > 0)
            raise WorkflowError(f"dependency cycle detected among tasks: {cyclic}")
        return order

    async def run_workflow(
        self,
        tasks: Sequence[Task],
        *,
        fail_fast: bool = False,
        max_concurrency: Optional[int] = None,
        inject_dependency_results: bool = True,
    ) -> WorkflowResult:
        """Execute ``tasks`` respecting ``Task.dependencies``.

        Independent tasks run concurrently (highest priority first). Dependents
        of a failed task are skipped. With ``fail_fast`` the first failure
        cancels everything still running. Upstream outputs are passed to each
        task as ``task.context["dependency_results"]`` (``{task_id: output}``).
        """
        self.validate(tasks)
        started = time.monotonic()
        workflow = WorkflowResult(workflow_id=new_id("wf-"))
        by_id = {t.id: t for t in tasks}
        children: Dict[str, List[str]] = defaultdict(list)
        indegree: Dict[str, int] = {}
        for task in tasks:
            deps = set(task.dependencies)
            indegree[task.id] = len(deps)
            for dep in deps:
                children[dep].append(task.id)

        limit = asyncio.Semaphore(max_concurrency or self._max_concurrency)
        results = workflow.results

        async def run_one(task: Task) -> TaskResult:
            async with limit:
                if inject_dependency_results:
                    task.context["dependency_results"] = {
                        dep: results[dep].output for dep in task.dependencies if dep in results
                    }
                return await self._delegator.delegate(task)

        ready: List[str] = [tid for tid, deg in indegree.items() if deg == 0]
        running: Dict["asyncio.Task[TaskResult]", str] = {}
        aborted = False

        while (ready or running) and not aborted:
            ready.sort(key=lambda tid: by_id[tid].priority, reverse=True)
            while ready:
                tid = ready.pop(0)
                running[asyncio.create_task(run_one(by_id[tid]))] = tid
            done, _ = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
            failed = False
            for future in done:
                tid = running.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = TaskResult.failure(tid, f"{type(exc).__name__}: {exc}")
                results[tid] = result
                if result.success:
                    for child in children[tid]:
                        indegree[child] -= 1
                        if indegree[child] == 0:
                            ready.append(child)
                else:
                    failed = True
            if failed and fail_fast:
                aborted = True

        if running:  # fail-fast: cancel what is still in flight
            for future in running:
                future.cancel()
            await asyncio.gather(*running.keys(), return_exceptions=True)
            for tid in running.values():
                by_id[tid].status = TaskStatus.CANCELLED
                results[tid] = TaskResult.failure(tid, "cancelled (fail_fast)")

        for task in tasks:
            if task.id not in results:
                task.status = TaskStatus.SKIPPED
                workflow.skipped.append(task.id)
        workflow.duration = time.monotonic() - started
        return workflow

    # ------------------------------------------------------------------ #
    # Priority queue
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._queue is not None

    async def start(self, workers: int = 4) -> None:
        if self._queue is not None:
            return
        self._queue = asyncio.PriorityQueue()
        self._workers = [asyncio.create_task(self._worker(), name=f"orchestrator-worker-{i}") for i in range(workers)]

    async def stop(self) -> None:
        queue, self._queue = self._queue, None
        workers, self._workers = self._workers, []
        for worker in workers:
            worker.cancel()
        for worker in workers:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker
        if queue is not None:
            while not queue.empty():
                _, _, _, future = queue.get_nowait()
                if not future.done():
                    future.cancel()

    async def submit(self, task: Task) -> "asyncio.Future[TaskResult]":
        """Queue ``task``; the returned future resolves with its ``TaskResult``."""
        if self._queue is None:
            raise WorkflowError("orchestrator queue is not running; call start() first")
        future: "asyncio.Future[TaskResult]" = asyncio.get_running_loop().create_future()
        task.status = TaskStatus.QUEUED
        await self._queue.put((-int(task.priority), next(self._seq), task, future))
        return future

    async def submit_and_wait(self, task: Task) -> TaskResult:
        return await (await self.submit(task))

    async def _worker(self) -> None:
        while True:
            queue = self._queue
            if queue is None:
                return
            _, _, task, future = await queue.get()
            try:
                if future.cancelled():
                    continue
                try:
                    result = await self._delegator.delegate(task)
                except asyncio.CancelledError:
                    if not future.done():
                        future.cancel()
                    raise
                except Exception as exc:  # noqa: BLE001
                    result = TaskResult.failure(task.id, f"{type(exc).__name__}: {exc}")
                if not future.done():
                    future.set_result(result)
            finally:
                queue.task_done()
