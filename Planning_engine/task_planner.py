"""Decomposition of goals into dependency-linked tasks."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Sequence, Set

from .goal_manager import GoalManager
from .planning_core import Goal, Task, TaskPlanningError

logger = logging.getLogger(__name__)

Decomposer = Callable[[Goal], List[Task]]


class TaskPlanner:
    """Turns goals into tasks.

    Default decomposition for a leaf goal: ``scope -> one task per success
    criterion (parallel) -> review``.  Register custom decomposers keyed by
    ``goal.metadata["type"]`` via :meth:`register_decomposer`.

    Recognised goal metadata: ``estimate_hours`` (split across execution
    tasks), ``scope_hours``, ``review_hours``, ``integrate_hours``, ``resources``.
    """

    def __init__(self, default_task_hours: float = 4.0) -> None:
        if default_task_hours <= 0:
            raise TaskPlanningError("default_task_hours must be positive")
        self.default_task_hours = default_task_hours
        self._decomposers: Dict[str, Decomposer] = {}

    def register_decomposer(self, goal_type: str, decomposer: Decomposer) -> None:
        """Use ``decomposer`` for goals whose ``metadata['type'] == goal_type``."""
        self._decomposers[goal_type] = decomposer

    # ---- single goal ------------------------------------------------------ #
    def decompose(self, goal: Goal) -> List[Task]:
        """Break one goal (ignoring its sub-goals) into tasks."""
        custom = self._decomposers.get(str(goal.metadata.get("type", "")))
        tasks = custom(goal) if custom else self._default(goal)
        if not tasks:
            raise TaskPlanningError(f"Decomposition of goal '{goal.id}' produced no tasks")
        ids = [t.id for t in tasks]
        if len(set(ids)) != len(ids):
            raise TaskPlanningError(f"Goal '{goal.id}' produced duplicate task ids")
        for t in tasks:
            t.goal_id = goal.id
        return tasks

    def _default(self, goal: Goal) -> List[Task]:
        md = goal.metadata
        hours = lambda key, dflt: timedelta(hours=float(md.get(key, dflt)))  # noqa: E731
        res = dict(md.get("resources", {}))
        scope = Task(f"{goal.id}.scope", f"Define scope: {goal.title}", goal.id,
                     hours("scope_hours", 1), goal.priority)
        names = goal.success_criteria or [goal.title]
        total = md.get("estimate_hours")
        each = timedelta(hours=float(total) / len(names)) if total else timedelta(
            hours=self.default_task_hours)
        work = [
            Task(f"{goal.id}.step{i}", f"Achieve: {name}", goal.id, each, goal.priority,
                 {scope.id}, dict(res))
            for i, name in enumerate(names, 1)
        ]
        review = Task(f"{goal.id}.review", f"Review: {goal.title}", goal.id,
                      hours("review_hours", 1), goal.priority, {t.id for t in work},
                      deadline=goal.deadline)
        return [scope, *work, review]

    # ---- goal hierarchy --------------------------------------------------- #
    def plan_goals(
        self, goal_manager: GoalManager, goal_ids: Optional[Sequence[str]] = None
    ) -> Dict[str, Task]:
        """Plan every selected goal, wiring parent/child and goal-level dependencies.

        * A goal with sub-goals gets one ``integrate`` task that waits for all
          of its children's final tasks.
        * A goal's entry tasks wait for the final tasks of goals in ``depends_on``.
        """
        selected = {g.id: g for g in goal_manager.select(goal_ids)}
        planned: Dict[str, List[Task]] = {}
        in_progress: Set[str] = set()

        def exits(tasks: List[Task]) -> Set[str]:
            used = {d for t in tasks for d in t.dependencies}
            return {t.id for t in tasks if t.id not in used}

        def plan(goal: Goal) -> List[Task]:
            if goal.id in planned:
                return planned[goal.id]
            if goal.id in in_progress:
                raise TaskPlanningError(f"Circular goal structure at '{goal.id}'")
            in_progress.add(goal.id)
            kids = [selected[k.id] for k in goal_manager.children(goal.id) if k.id in selected]
            prereqs = [selected[d] for d in sorted(goal.depends_on) if d in selected]
            prereq_exits = {x for p in prereqs for x in exits(plan(p))}
            if kids:
                child_exits = {x for k in kids for x in exits(plan(k))}
                hrs = float(goal.metadata.get("integrate_hours", 1))
                tasks = [Task(f"{goal.id}.integrate", f"Integrate: {goal.title}", goal.id,
                              timedelta(hours=hrs), goal.priority,
                              child_exits | prereq_exits, deadline=goal.deadline)]
            else:
                tasks = self.decompose(goal)
                own = {t.id for t in tasks}
                for t in tasks:
                    if not (t.dependencies & own):  # entry task of this goal
                        t.dependencies |= prereq_exits
            in_progress.discard(goal.id)
            planned[goal.id] = tasks
            return tasks

        result: Dict[str, Task] = {}
        for goal in goal_manager.ordered(selected.values()):
            for t in plan(goal):
                if t.id in result:
                    raise TaskPlanningError(f"Task id collision: '{t.id}'")
                result[t.id] = t
        logger.debug("Decomposed %d goals into %d tasks", len(selected), len(result))
        return result
