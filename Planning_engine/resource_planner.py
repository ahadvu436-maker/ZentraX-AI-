"""Resource registry and time-based capacity ledger."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .planning_core import Resource, ResourceError, Task, to_hours

_EPS = 1e-9
_Alloc = Tuple[datetime, datetime, float, str]  # start, end, units, task_id


class ResourcePlanner:
    """Tracks resource capacity and reserved usage over time."""

    def __init__(self, resources: Optional[Iterable[Resource]] = None) -> None:
        self._resources: Dict[str, Resource] = {}
        self._ledger: Dict[str, List[_Alloc]] = {}
        for r in resources or []:
            self.add_resource(r)

    # ---- registry --------------------------------------------------------- #
    def add_resource(self, resource: Resource) -> None:
        """Register a resource; ids must be unique."""
        if resource.id in self._resources:
            raise ResourceError(f"Resource '{resource.id}' already exists")
        self._resources[resource.id] = resource
        self._ledger[resource.id] = []

    def get_resource(self, resource_id: str) -> Resource:
        """Return a resource or raise :class:`ResourceError`."""
        try:
            return self._resources[resource_id]
        except KeyError:
            raise ResourceError(f"Unknown resource '{resource_id}'") from None

    def resources(self) -> List[Resource]:
        """All registered resources."""
        return list(self._resources.values())

    def validate_requirements(self, tasks: Iterable[Task]) -> None:
        """Ensure every task only needs known resources within capacity."""
        for t in tasks:
            self._check_requirements(t.resources, t.id)

    def _check_requirements(self, req: Mapping[str, float], owner: str = "?") -> None:
        for rid, units in req.items():
            res = self.get_resource(rid)
            if units > res.capacity + _EPS:
                raise ResourceError(
                    f"Task '{owner}' needs {units} of '{rid}' but capacity is {res.capacity}"
                )

    # ---- ledger ----------------------------------------------------------- #
    def peak_usage(self, resource_id: str, start: datetime, end: datetime) -> float:
        """Maximum simultaneous reserved units within ``[start, end)``."""
        if end <= start:
            return 0.0
        allocs = [a for a in self._ledger[resource_id] if a[0] < end and a[1] > start]
        points = {start} | {a[0] for a in allocs if a[0] > start}
        return max(
            (sum(a[2] for a in allocs if a[0] <= p < a[1]) for p in points), default=0.0
        )

    def can_allocate(self, req: Mapping[str, float], start: datetime, end: datetime) -> bool:
        """True if all requirements fit in ``[start, end)``."""
        return all(
            self.peak_usage(rid, start, end) + units <= self.get_resource(rid).capacity + _EPS
            for rid, units in req.items()
        )

    def earliest_available(
        self, req: Mapping[str, float], duration: timedelta, not_before: datetime
    ) -> datetime:
        """Earliest start >= ``not_before`` where ``req`` fits for ``duration``."""
        if not req or duration <= timedelta(0):
            return not_before
        self._check_requirements(req)
        candidates = {not_before}
        for rid in req:
            candidates.update(a[1] for a in self._ledger[rid] if a[1] > not_before)
        for c in sorted(candidates):
            if self.can_allocate(req, c, c + duration):
                return c
        raise ResourceError("No feasible slot found")  # pragma: no cover

    def allocate(self, task_id: str, req: Mapping[str, float], start: datetime, end: datetime) -> None:
        """Reserve resources; raises :class:`ResourceError` on over-allocation."""
        if not self.can_allocate(req, start, end):
            raise ResourceError(f"Insufficient capacity for task '{task_id}'")
        for rid, units in req.items():
            self._ledger[rid].append((start, end, units, task_id))

    def release(self, task_id: str) -> None:
        """Drop all reservations belonging to ``task_id``."""
        for rid in self._ledger:
            self._ledger[rid] = [a for a in self._ledger[rid] if a[3] != task_id]

    def cost(self, req: Mapping[str, float], duration: timedelta) -> float:
        """Cost of using ``req`` for ``duration``."""
        return sum(
            units * self.get_resource(rid).cost_per_hour * to_hours(duration)
            for rid, units in req.items()
        )

    def utilization(self, resource_id: str, start: datetime, end: datetime) -> float:
        """Average utilisation (0..1) of a resource over a window."""
        span = (end - start).total_seconds()
        if span <= 0:
            return 0.0
        used = sum(
            (min(a[1], end) - max(a[0], start)).total_seconds() * a[2]
            for a in self._ledger[resource_id]
            if a[0] < end and a[1] > start
        )
        return used / (span * self.get_resource(resource_id).capacity)

    def fork(self) -> "ResourcePlanner":
        """Copy of this planner including current reservations (for what-if runs)."""
        clone = ResourcePlanner(self._resources.values())
        clone._ledger = {rid: list(a) for rid, a in self._ledger.items()}
        return clone
