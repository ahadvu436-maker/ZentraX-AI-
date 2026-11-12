"""Workflow structure: parallel stages and critical path."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

from .dependency_resolver import DependencyResolver, TaskInput
from .optimization.path_optimizer import PathOptimizer


@dataclass
class Stage:
    """Tasks that may run concurrently (ignoring resource limits)."""

    index: int
    task_ids: List[str]
    duration: timedelta


@dataclass
class Workflow:
    """Stage-wise view of a task graph."""

    stages: List[Stage] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    estimated_duration: timedelta = timedelta(0)
    max_parallelism: int = 0


class WorkflowPlanner:
    """Builds :class:`Workflow` objects from tasks."""

    def __init__(
        self,
        resolver: Optional[DependencyResolver] = None,
        path_optimizer: Optional[PathOptimizer] = None,
    ) -> None:
        self.resolver = resolver or DependencyResolver()
        self.path_optimizer = path_optimizer or PathOptimizer(self.resolver)

    def build_workflow(self, tasks: TaskInput) -> Workflow:
        """Group tasks into stages and compute the critical path."""
        mapping: Dict = self.resolver.as_mapping(tasks)
        layers: List[List[str]] = self.resolver.layers(mapping)
        stages = [
            Stage(i, ids, max((mapping[t].duration for t in ids), default=timedelta(0)))
            for i, ids in enumerate(layers)
        ]
        return Workflow(
            stages=stages,
            critical_path=self.path_optimizer.critical_path(mapping),
            estimated_duration=self.path_optimizer.project_duration(mapping),
            max_parallelism=max((len(s.task_ids) for s in stages), default=0),
        )
