"""Resource-constrained scheduling (serial schedule generation scheme)."""
from __future__ import annotations

import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional, Tuple

from .dependency_resolver import DependencyResolver, TaskInput
from .planning_core import (Schedule, ScheduledTask, SchedulingError, to_hours)
from .resource_planner import ResourcePlanner


class ScheduleManager:
    """Places tasks on a timeline respecting dependencies and resource capacity."""

    def __init__(
        self,
        resource_planner: Optional[ResourcePlanner] = None,
        resolver: Optional[DependencyResolver] = None,
    ) -> None:
        self.resource_planner = resource_planner or ResourcePlanner()
        self.resolver = resolver or DependencyResolver()

    def build_schedule(
        self,
        tasks: TaskInput,
        start: datetime,
        priority_keys: Optional[Mapping[str, float]] = None,
    ) -> Schedule:
        """Create a schedule without mutating the real resource ledger.

        Args:
            tasks: Tasks to schedule.
            start: Earliest possible start.
            priority_keys: ``task_id -> key``; among *ready* tasks, lower keys
                are scheduled first.  Defaults to higher task priority first.

        Raises:
            SchedulingError: if not every task could be placed.
        """
        mapping = self.resolver.as_mapping(tasks)
        self.resolver.validate(mapping)
        self.resource_planner.validate_requirements(mapping.values())
        rp = self.resource_planner.fork()
        keys = dict(priority_keys) if priority_keys else {
            tid: -float(int(t.priority)) for tid, t in mapping.items()}
        dependents = self.resolver.dependents(mapping)
        waiting = {tid: set(t.dependencies) for tid, t in mapping.items()}
        heap: List[Tuple[float, str]] = [
            (keys.get(tid, 0.0), tid) for tid, w in waiting.items() if not w]
        heapq.heapify(heap)

        schedule = Schedule(start=start)
        cost = lateness = 0.0
        late = 0
        while heap:
            _, tid = heapq.heappop(heap)
            task = mapping[tid]
            ready = max(
                [start, task.earliest_start or start]
                + [schedule.entries[d].end for d in task.dependencies])
            begin = rp.earliest_available(task.resources, task.duration, ready)
            end = begin + task.duration
            rp.allocate(tid, task.resources, begin, end)
            schedule.entries[tid] = ScheduledTask(tid, begin, end, dict(task.resources))
            cost += rp.cost(task.resources, task.duration)
            if task.deadline and end > task.deadline:
                late += 1
                lateness += to_hours(end - task.deadline)
                schedule.violations.append(
                    f"Task '{tid}' finishes {to_hours(end - task.deadline):.1f}h after its deadline")
            for nxt in dependents[tid]:
                waiting[nxt].discard(tid)
                if not waiting[nxt]:
                    heapq.heappush(heap, (keys.get(nxt, 0.0), nxt))
        if len(schedule.entries) != len(mapping):
            raise SchedulingError("Not all tasks could be scheduled")
        schedule.metrics = {
            "makespan_hours": to_hours(schedule.makespan),
            "total_cost": cost,
            "late_tasks": float(late),
            "total_lateness_hours": lateness,
        }
        return schedule

    def commit_schedule(self, schedule: Schedule) -> None:
        """Reserve a schedule's resources in the real resource planner."""
        for e in schedule.entries.values():
            self.resource_planner.allocate(e.task_id, e.resources, e.start, e.end)

    def release_task(self, task_id: str) -> None:
        """Free a committed task's reservations."""
        self.resource_planner.release(task_id)
