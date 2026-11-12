"""Schedule optimization by heuristic comparison plus seeded randomised search."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from ..dependency_resolver import TaskInput
from ..planning_core import (OptimizationError, PlanningError, Schedule)
from ..schedule_manager import ScheduleManager
from .path_optimizer import PathOptimizer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationObjective:
    """Weighted cost function (lower is better)."""

    makespan_weight: float = 1.0   # per hour of total duration
    cost_weight: float = 0.0       # per currency unit of resource cost
    lateness_weight: float = 10.0  # per hour past a task deadline

    def score(self, schedule: Schedule) -> float:
        """Evaluate a schedule using its metrics."""
        m = schedule.metrics
        return (self.makespan_weight * m.get("makespan_hours", 0.0)
                + self.cost_weight * m.get("total_cost", 0.0)
                + self.lateness_weight * m.get("total_lateness_hours", 0.0))


@dataclass
class OptimizationResult:
    """Best schedule found and the scores of all candidates tried."""

    schedule: Schedule
    score: float
    strategy_name: str
    candidates: Dict[str, float] = field(default_factory=dict)


class PlanOptimizer:
    """Finds a better task ordering for the resource-constrained scheduler."""

    HEURISTICS = ("priority", "slack", "shortest", "longest", "successors")

    def __init__(
        self,
        schedule_manager: ScheduleManager,
        path_optimizer: Optional[PathOptimizer] = None,
        objective: Optional[OptimizationObjective] = None,
        iterations: int = 25,
        seed: Optional[int] = 42,
    ) -> None:
        self.schedule_manager = schedule_manager
        self.path_optimizer = path_optimizer or PathOptimizer(schedule_manager.resolver)
        self.objective = objective or OptimizationObjective()
        self.iterations = max(0, iterations)
        self.seed = seed

    def optimize(
        self,
        tasks: TaskInput,
        start: datetime,
        objective: Optional[OptimizationObjective] = None,
    ) -> OptimizationResult:
        """Try each heuristic, then random perturbations of the slack heuristic.

        Raises:
            OptimizationError: if no candidate schedule could be built.
            DependencyError / ResourceError: for invalid input (propagated).
        """
        obj = objective or self.objective
        mapping = self.schedule_manager.resolver.as_mapping(tasks)
        self.schedule_manager.resolver.validate(mapping)  # fail fast on bad graphs
        rng = random.Random(self.seed)
        best: Optional[OptimizationResult] = None
        scores: Dict[str, float] = {}

        def consider(name: str, keys: Dict[str, float]) -> None:
            nonlocal best
            try:
                sched = self.schedule_manager.build_schedule(mapping, start, keys)
            except PlanningError as exc:
                logger.warning("Candidate '%s' failed: %s", name, exc)
                return
            score = obj.score(sched)
            scores[name] = score
            if best is None or score < best.score - 1e-12:
                best = OptimizationResult(sched, score, name)

        for name in self.HEURISTICS:
            consider(name, self.path_optimizer.priority_keys(mapping, name))
        base = self.path_optimizer.priority_keys(mapping, "slack")
        spread = max(1.0, max((abs(v) for v in base.values()), default=1.0))
        for i in range(self.iterations):
            consider(f"random-{i}", {t: v + rng.uniform(0, spread) for t, v in base.items()})

        if best is None:
            raise OptimizationError("No feasible schedule could be generated")
        best.candidates = scores
        logger.info("Best strategy '%s' (score %.2f)", best.strategy_name, best.score)
        return best
