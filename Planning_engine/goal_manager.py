"""Goal registry with hierarchy, dependency validation and progress tracking."""
from __future__ import annotations

import heapq
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .planning_core import Goal, GoalError, GoalStatus, Priority

logger = logging.getLogger(__name__)


class GoalManager:
    """Stores goals and answers structural questions about them."""

    def __init__(self, goals: Optional[Iterable[Goal]] = None) -> None:
        self._goals: Dict[str, Goal] = {}
        self._counter = 0
        for goal in goals or []:
            self.add_goal(goal)

    # ---- CRUD ------------------------------------------------------------- #
    def add_goal(self, goal: Goal) -> Goal:
        """Register a goal. Raises :class:`GoalError` on duplicate ids."""
        if goal.id in self._goals:
            raise GoalError(f"Goal '{goal.id}' already exists")
        self._goals[goal.id] = goal
        return goal

    def create_goal(self, title: str, goal_id: Optional[str] = None, **kwargs: Any) -> Goal:
        """Create and register a goal, generating an id if none is given."""
        if goal_id is None:
            self._counter += 1
            goal_id = f"goal-{self._counter}"
        return self.add_goal(Goal(id=goal_id, title=title, **kwargs))

    def get_goal(self, goal_id: str) -> Goal:
        """Return a goal or raise :class:`GoalError` if unknown."""
        try:
            return self._goals[goal_id]
        except KeyError:
            raise GoalError(f"Unknown goal '{goal_id}'") from None

    def update_goal(self, goal_id: str, **changes: Any) -> Goal:
        """Update goal attributes in place (validated by :meth:`validate`)."""
        goal = self.get_goal(goal_id)
        for key, value in changes.items():
            if key == "id" or not hasattr(goal, key):
                raise GoalError(f"Cannot update field '{key}' of goal '{goal_id}'")
            setattr(goal, key, value)
        return goal

    def set_status(self, goal_id: str, status: GoalStatus) -> None:
        """Change a goal's lifecycle status."""
        self.get_goal(goal_id).status = GoalStatus(status)

    def remove_goal(self, goal_id: str, cascade: bool = False) -> None:
        """Remove a goal; children require ``cascade=True``."""
        self.get_goal(goal_id)
        kids = self.children(goal_id)
        if kids and not cascade:
            raise GoalError(f"Goal '{goal_id}' has children; use cascade=True")
        for kid in kids:
            self.remove_goal(kid.id, cascade=True)
        del self._goals[goal_id]
        for g in self._goals.values():
            g.depends_on.discard(goal_id)

    def all_goals(self) -> List[Goal]:
        """All goals in insertion order."""
        return list(self._goals.values())

    # ---- hierarchy -------------------------------------------------------- #
    def children(self, goal_id: str) -> List[Goal]:
        """Direct sub-goals."""
        return [g for g in self._goals.values() if g.parent_id == goal_id]

    def descendants(self, goal_id: str) -> List[Goal]:
        """All nested sub-goals (depth-first)."""
        out: List[Goal] = []
        for kid in self.children(goal_id):
            out.append(kid)
            out.extend(self.descendants(kid.id))
        return out

    def roots(self) -> List[Goal]:
        """Goals without a parent."""
        return [g for g in self._goals.values() if g.parent_id is None]

    def is_leaf(self, goal_id: str) -> bool:
        """True when the goal has no sub-goals."""
        return not self.children(goal_id)

    def progress(self, goal_id: str) -> float:
        """Fraction (0..1) of completed leaf goals under ``goal_id``."""
        goal = self.get_goal(goal_id)
        leaves = [g for g in self.descendants(goal_id) if self.is_leaf(g.id)] or [goal]
        done = sum(1 for g in leaves if g.status == GoalStatus.COMPLETED)
        return done / len(leaves)

    # ---- selection & ordering -------------------------------------------- #
    def select(self, goal_ids: Optional[Sequence[str]] = None) -> List[Goal]:
        """Return the requested goals plus all descendants (deduplicated).

        ``None`` selects every non-cancelled goal.  Cancelled goals are skipped.
        """
        if goal_ids is None:
            picked = self.all_goals()
        else:
            picked, seen = [], set()
            for gid in goal_ids:
                for g in [self.get_goal(gid), *self.descendants(gid)]:
                    if g.id not in seen:
                        seen.add(g.id)
                        picked.append(g)
        return [g for g in picked if g.status != GoalStatus.CANCELLED]

    def ordered(self, goals: Optional[Iterable[Goal]] = None) -> List[Goal]:
        """Topologically order goals by ``depends_on``, then priority/deadline."""
        pool = {g.id: g for g in (goals if goals is not None else self.all_goals())}
        indeg = {gid: len([d for d in g.depends_on if d in pool]) for gid, g in pool.items()}
        dependents: Dict[str, List[str]] = {gid: [] for gid in pool}
        for gid, g in pool.items():
            for d in g.depends_on:
                if d in pool:
                    dependents[d].append(gid)

        def key(gid: str) -> Any:
            g = pool[gid]
            return (-int(g.priority), g.deadline or datetime.max, gid)

        heap = [(key(g), g) for g, n in indeg.items() if n == 0]
        heapq.heapify(heap)
        out: List[Goal] = []
        while heap:
            _, gid = heapq.heappop(heap)
            out.append(pool[gid])
            for nxt in dependents[gid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    heapq.heappush(heap, (key(nxt), nxt))
        if len(out) != len(pool):
            raise GoalError("Circular goal dependencies detected")
        return out

    # ---- validation ------------------------------------------------------- #
    def validate(self) -> None:
        """Check references, parent cycles and dependency cycles."""
        for g in self._goals.values():
            if g.parent_id is not None and g.parent_id not in self._goals:
                raise GoalError(f"Goal '{g.id}' has unknown parent '{g.parent_id}'")
            missing = [d for d in g.depends_on if d not in self._goals]
            if missing:
                raise GoalError(f"Goal '{g.id}' depends on unknown goals {sorted(missing)}")
            seen: Set[str] = {g.id}
            cur = g.parent_id
            while cur is not None:
                if cur in seen:
                    raise GoalError(f"Circular goal hierarchy involving '{g.id}'")
                seen.add(cur)
                cur = self._goals[cur].parent_id
        self.ordered()  # raises on dependency cycles
