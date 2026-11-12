"""Critical Path Method (CPM) analysis and priority-key heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional

from ..dependency_resolver import DependencyResolver, TaskInput
from ..planning_core import to_hours

_ZERO = timedelta(0)


@dataclass(frozen=True)
class TaskTiming:
    """CPM offsets (from project start) for one task."""

    earliest_start: timedelta
    earliest_finish: timedelta
    latest_start: timedelta
    latest_finish: timedelta

    @property
    def slack(self) -> timedelta:
        """Delay the task can absorb without extending the project."""
        return self.latest_start - self.earliest_start


class PathOptimizer:
    """Computes critical paths ignoring resource limits."""

    def __init__(self, resolver: Optional[DependencyResolver] = None) -> None:
        self.resolver = resolver or DependencyResolver()

    def analyze(self, tasks: TaskInput) -> Dict[str, TaskTiming]:
        """Forward/backward pass returning timing data for every task."""
        mapping = self.resolver.as_mapping(tasks)
        order = self.resolver.topological_order(mapping)
        dependents = self.resolver.dependents(mapping)
        es: Dict[str, timedelta] = {}
        ef: Dict[str, timedelta] = {}
        for tid in order:
            es[tid] = max((ef[d] for d in mapping[tid].dependencies), default=_ZERO)
            ef[tid] = es[tid] + mapping[tid].duration
        end = max(ef.values(), default=_ZERO)
        ls: Dict[str, timedelta] = {}
        lf: Dict[str, timedelta] = {}
        for tid in reversed(order):
            lf[tid] = min((ls[n] for n in dependents[tid]), default=end)
            ls[tid] = lf[tid] - mapping[tid].duration
        return {t: TaskTiming(es[t], ef[t], ls[t], lf[t]) for t in order}

    def project_duration(self, tasks: TaskInput) -> timedelta:
        """Shortest possible duration given unlimited resources."""
        return max((t.earliest_finish for t in self.analyze(tasks).values()), default=_ZERO)

    def critical_path(self, tasks: TaskInput) -> List[str]:
        """Zero-slack tasks in execution order."""
        timings = self.analyze(tasks)
        crit = [t for t, v in timings.items() if v.slack <= _ZERO]
        return sorted(crit, key=lambda t: (timings[t].earliest_start, t))

    def priority_keys(self, tasks: TaskInput, strategy: str = "slack") -> Dict[str, float]:
        """Build scheduling keys (lower = sooner) using a named heuristic.

        Strategies: ``slack``, ``priority``, ``shortest``, ``longest``, ``successors``.
        """
        mapping = self.resolver.as_mapping(tasks)
        if strategy == "slack":
            timings = self.analyze(mapping)
            return {t: to_hours(timings[t].slack) - 1e-3 * int(mapping[t].priority)
                    for t in mapping}
        if strategy == "priority":
            return {t: -float(int(v.priority)) for t, v in mapping.items()}
        if strategy == "shortest":
            return {t: to_hours(v.duration) for t, v in mapping.items()}
        if strategy == "longest":
            return {t: -to_hours(v.duration) for t, v in mapping.items()}
        if strategy == "successors":
            return {t: -float(len(self.resolver.descendants(mapping, t))) for t in mapping}
        raise ValueError(f"Unknown strategy '{strategy}'")
