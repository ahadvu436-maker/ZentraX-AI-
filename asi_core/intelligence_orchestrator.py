"""Priority-based, concurrency-bounded task orchestration."""
from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

from .general_intelligence import GeneralIntelligence, Problem, Solution

log = logging.getLogger(__name__)


@dataclass(order=True)
class _Job:
    priority: int
    seq: int
    problem: Problem = field(compare=False)
    future: "asyncio.Future[Solution]" = field(compare=False)


class IntelligenceOrchestrator:
    """Lower priority number = scheduled first."""

    def __init__(self, intelligence: GeneralIntelligence, workers: int = 4,
                 task_timeout_s: float = 30.0, max_queue: int = 1000) -> None:
        self.gi, self.n_workers, self.timeout = intelligence, workers, task_timeout_s
        self._q: asyncio.PriorityQueue[_Job] = asyncio.PriorityQueue(maxsize=max_queue)
        self._seq = itertools.count()
        self._workers: list[asyncio.Task[None]] = []
        self.completed = self.failed = 0

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.n_workers)]

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        while not self._q.empty():  # fail pending jobs cleanly
            j = self._q.get_nowait()
            if not j.future.done():
                j.future.cancel()

    async def submit(self, problem: Problem, priority: int = 5) -> "asyncio.Future[Solution]":
        fut: asyncio.Future[Solution] = asyncio.get_running_loop().create_future()
        await self._q.put(_Job(priority, next(self._seq), problem, fut))
        return fut

    async def run(self, problem: Problem, priority: int = 5) -> Solution:
        return await (await self.submit(problem, priority))

    async def _worker(self, idx: int) -> None:
        while True:
            job = await self._q.get()
            if job.future.cancelled():
                continue
            try:
                sol = await asyncio.wait_for(self.gi.solve(job.problem), self.timeout)
                self.completed += 1
            except asyncio.TimeoutError:
                sol = Solution(job.problem.id, False, error="timeout")
                self.failed += 1
            except Exception as e:
                log.exception("worker %d error", idx)
                sol = Solution(job.problem.id, False, error=repr(e))
                self.failed += 1
            if not job.future.done():
                job.future.set_result(sol)

    def metrics(self) -> dict[str, int]:
        return {"queued": self._q.qsize(), "completed": self.completed,
                "failed": self.failed, "workers": len(self._workers)}
