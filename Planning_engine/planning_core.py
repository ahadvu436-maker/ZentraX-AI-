"""Core domain models, exceptions and the :class:`PlanningEngine` orchestrator.

This module is the dependency root of the package: every other module imports
its data models (``Goal``, ``Task``, ``Resource``, ``Schedule``, ``Plan``) and
exceptions from here.  To avoid circular imports, ``PlanningEngine`` imports
the concrete planners lazily when it is instantiated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .goal_manager import GoalManager
    from .strategic_planner import Strategy
    from .workflow_planner import Workflow

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class PlanningError(Exception):
    """Base class for all errors raised by the planning engine."""


class GoalError(PlanningError):
    """Invalid goal definition, hierarchy or goal operation."""


class TaskPlanningError(PlanningError):
    """Goal decomposition into tasks failed."""


class DependencyError(PlanningError):
    """Missing or otherwise invalid task dependency."""


class DependencyCycleError(DependencyError):
    """A circular dependency exists; ``cycle`` lists the offending ids."""

    def __init__(self, cycle: Sequence[str]) -> None:
        self.cycle: List[str] = list(cycle)
        super().__init__("Circular dependency: " + " -> ".join(self.cycle))


class ResourceError(PlanningError):
    """Unknown resource or impossible resource requirement."""


class SchedulingError(PlanningError):
    """A schedule could not be produced."""


class OptimizationError(PlanningError):
    """Optimization could not produce a valid result."""


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Priority(IntEnum):
    """Importance level; higher values are more important."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class GoalStatus(str, Enum):
    """Lifecycle state of a goal."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Lifecycle state of a task."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    DONE = "done"


# --------------------------------------------------------------------------- #
# Domain models
# --------------------------------------------------------------------------- #
def to_hours(delta: timedelta) -> float:
    """Convert a ``timedelta`` to fractional hours."""
    return delta.total_seconds() / 3600.0


@dataclass
class Goal:
    """A desired outcome, optionally nested under a parent goal.

    Attributes:
        id: Unique identifier.
        title: Short human-readable name.
        priority: Importance of the goal.
        deadline: Optional due date.
        parent_id: Parent goal id for hierarchical goals.
        depends_on: Ids of goals that must finish before this one starts.
        success_criteria: Each criterion becomes an execution task by default.
        metadata: Free-form hints (e.g. ``estimate_hours``, ``resources``).
    """

    id: str
    title: str
    description: str = ""
    priority: Priority = Priority.MEDIUM
    deadline: Optional[datetime] = None
    parent_id: Optional[str] = None
    depends_on: Set[str] = field(default_factory=set)
    success_criteria: List[str] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise GoalError("Goal id must be a non-empty string")
        if not self.title or not self.title.strip():
            raise GoalError(f"Goal '{self.id}' needs a title")
        if self.parent_id == self.id or self.id in self.depends_on:
            raise GoalError(f"Goal '{self.id}' cannot reference itself")
        self.priority = Priority(self.priority)
        self.depends_on = set(self.depends_on)


@dataclass
class Task:
    """A unit of schedulable work.

    Attributes:
        resources: Mapping ``resource_id -> units`` needed for the whole duration.
        earliest_start: Optional release time.
        deadline: Optional due time (violations are reported, not enforced).
    """

    id: str
    name: str
    goal_id: Optional[str] = None
    duration: timedelta = timedelta(hours=1)
    priority: Priority = Priority.MEDIUM
    dependencies: Set[str] = field(default_factory=set)
    resources: Dict[str, float] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    earliest_start: Optional[datetime] = None
    deadline: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise TaskPlanningError("Task id must be a non-empty string")
        if self.duration < timedelta(0):
            raise TaskPlanningError(f"Task '{self.id}' has a negative duration")
        for rid, units in self.resources.items():
            if units <= 0:
                raise TaskPlanningError(
                    f"Task '{self.id}' requests non-positive units of '{rid}'"
                )
        self.priority = Priority(self.priority)
        self.dependencies = set(self.dependencies)


@dataclass
class Resource:
    """A pooled, renewable resource (person, machine, budget line...)."""

    id: str
    name: str = ""
    capacity: float = 1.0
    cost_per_hour: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ResourceError(f"Resource '{self.id}' needs positive capacity")
        if self.cost_per_hour < 0:
            raise ResourceError(f"Resource '{self.id}' has negative cost")
        if not self.name:
            self.name = self.id


@dataclass
class ScheduledTask:
    """A task placed on the timeline."""

    task_id: str
    start: datetime
    end: datetime
    resources: Dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> timedelta:
        """Length of the scheduled interval."""
        return self.end - self.start


@dataclass
class Schedule:
    """A timeline of scheduled tasks plus quality metrics."""

    start: datetime
    entries: Dict[str, ScheduledTask] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def end(self) -> datetime:
        """Finish time of the last task (or ``start`` if empty)."""
        return max((e.end for e in self.entries.values()), default=self.start)

    @property
    def makespan(self) -> timedelta:
        """Total elapsed time of the schedule."""
        return self.end - self.start

    def ordered(self) -> List[ScheduledTask]:
        """Entries sorted chronologically."""
        return sorted(self.entries.values(), key=lambda e: (e.start, e.end, e.task_id))


@dataclass
class Plan:
    """The complete output of :meth:`PlanningEngine.create_plan`."""

    goal_ids: List[str]
    tasks: Dict[str, Task]
    schedule: Schedule
    strategy: Optional["Strategy"] = None
    workflow: Optional["Workflow"] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a readable multi-line summary of the plan."""
        lines = [
            f"Goals: {', '.join(self.goal_ids)}",
            f"Tasks: {len(self.tasks)} | makespan: {to_hours(self.schedule.makespan):.1f}h",
        ]
        for e in self.schedule.ordered():
            t = self.tasks[e.task_id]
            lines.append(f"  {e.start:%Y-%m-%d %H:%M} -> {e.end:%H:%M}  {t.name} [{t.id}]")
        for v in self.schedule.violations:
            lines.append(f"  ! {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class PlanningEngine:
    """Facade that turns goals into an optimized, resource-aware schedule.

    Pipeline: validate goals -> strategy -> task decomposition -> dependency
    and resource validation -> workflow analysis -> (optimized) scheduling.
    Every component can be injected; defaults are created lazily.
    """

    def __init__(
        self,
        goal_manager: Optional["GoalManager"] = None,
        task_planner: Any = None,
        strategic_planner: Any = None,
        workflow_planner: Any = None,
        resource_planner: Any = None,
        dependency_resolver: Any = None,
        schedule_manager: Any = None,
        plan_optimizer: Any = None,
    ) -> None:
        # Lazy imports keep planning_core free of import cycles.
        from .dependency_resolver import DependencyResolver
        from .goal_manager import GoalManager
        from .optimization.path_optimizer import PathOptimizer
        from .optimization.plan_optimizer import PlanOptimizer
        from .resource_planner import ResourcePlanner
        from .schedule_manager import ScheduleManager
        from .strategic_planner import StrategicPlanner
        from .task_planner import TaskPlanner
        from .workflow_planner import WorkflowPlanner

        self.goals = goal_manager or GoalManager()
        self.dependencies = dependency_resolver or DependencyResolver()
        self.resources = resource_planner or ResourcePlanner()
        self.task_planner = task_planner or TaskPlanner()
        self.strategic_planner = strategic_planner or StrategicPlanner()
        path_optimizer = PathOptimizer(self.dependencies)
        self.workflow_planner = workflow_planner or WorkflowPlanner(
            self.dependencies, path_optimizer
        )
        self.schedule_manager = schedule_manager or ScheduleManager(
            self.resources, self.dependencies
        )
        self.optimizer = plan_optimizer or PlanOptimizer(
            self.schedule_manager, path_optimizer
        )

    def create_plan(
        self,
        goal_ids: Optional[Sequence[str]] = None,
        start: Optional[datetime] = None,
        optimize: bool = True,
        objective: Any = None,
        commit: bool = False,
    ) -> Plan:
        """Build a full plan.

        Args:
            goal_ids: Goals to plan (descendants included); ``None`` = all goals.
            start: Project start; defaults to now.
            optimize: Search for a better schedule instead of a single pass.
            objective: Optional ``OptimizationObjective``.
            commit: Reserve the resulting allocations in the resource planner.

        Raises:
            PlanningError: (or a subclass) for any planning failure.
        """
        start = start or datetime.now().replace(microsecond=0)
        try:
            self.goals.validate()
            selected = self.goals.select(goal_ids)
            if not selected:
                raise GoalError("There are no goals to plan")
            strategy = self.strategic_planner.build_strategy(self.goals, goal_ids, start)
            tasks = self.task_planner.plan_goals(self.goals, goal_ids)
            self.dependencies.validate(tasks)
            self.resources.validate_requirements(tasks.values())
            workflow = self.workflow_planner.build_workflow(tasks)

            metrics: Dict[str, Any] = {}
            if optimize:
                result = self.optimizer.optimize(tasks, start, objective)
                schedule = result.schedule
                metrics.update(
                    optimizer_strategy=result.strategy_name,
                    optimizer_score=result.score,
                    candidates_evaluated=len(result.candidates),
                )
            else:
                schedule = self.schedule_manager.build_schedule(tasks, start)
            if commit:
                self.schedule_manager.commit_schedule(schedule)
            metrics.update(schedule.metrics)
            logger.info("Planned %d tasks for %d goals", len(tasks), len(selected))
            return Plan([g.id for g in selected], tasks, schedule, strategy, workflow, metrics)
        except PlanningError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise PlanningError(f"Unexpected planning failure: {exc}") from exc
