"""High-level strategy: goal ordering, phases, milestones and risks."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .goal_manager import GoalManager
from .planning_core import Goal


@dataclass
class Phase:
    """A group of goals that can be pursued in parallel."""

    name: str
    goal_ids: List[str]
    target_date: Optional[datetime] = None


@dataclass
class Milestone:
    """A dated checkpoint derived from a goal deadline."""

    goal_id: str
    title: str
    due: datetime


@dataclass
class Strategy:
    """Result of strategic analysis."""

    ordered_goal_ids: List[str] = field(default_factory=list)
    phases: List[Phase] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)


class StrategicPlanner:
    """Derives a strategy from the goal graph (no task-level detail)."""

    def __init__(self, urgency_horizon_days: int = 30) -> None:
        self.urgency_horizon_days = urgency_horizon_days

    def score(self, goal: Goal, now: datetime) -> float:
        """Priority-weighted score with an urgency bonus (0..10) for deadlines."""
        score = float(int(goal.priority)) * 10.0
        if goal.deadline is not None:
            days = (goal.deadline - now).total_seconds() / 86400.0
            score += 10.0 * max(0.0, min(1.0, 1.0 - days / self.urgency_horizon_days))
        return score

    def build_strategy(
        self,
        goal_manager: GoalManager,
        goal_ids: Optional[Sequence[str]] = None,
        start: Optional[datetime] = None,
    ) -> Strategy:
        """Order goals, split them into phases and flag schedule risks."""
        now = start or datetime.now()
        goals = {g.id: g for g in goal_manager.select(goal_ids)}
        ordered = goal_manager.ordered(goals.values())

        depth: Dict[str, int] = {}
        for g in ordered:  # dependency order guarantees prerequisites come first
            depth[g.id] = 1 + max((depth[d] for d in g.depends_on if d in depth), default=-1)
        phases: List[Phase] = []
        for level in range(max(depth.values(), default=-1) + 1):
            ids = sorted((i for i, d in depth.items() if d == level),
                         key=lambda i: -self.score(goals[i], now))
            dates = [goals[i].deadline for i in ids if goals[i].deadline]
            phases.append(Phase(f"Phase {level + 1}", ids, min(dates) if dates else None))

        milestones = sorted(
            (Milestone(g.id, g.title, g.deadline) for g in goals.values() if g.deadline),
            key=lambda m: m.due,
        )
        risks: List[str] = []
        for g in goals.values():
            if g.deadline and g.deadline <= now:
                risks.append(f"Goal '{g.id}' deadline is already past")
            for d in g.depends_on:
                pre = goals.get(d)
                if pre and pre.deadline and g.deadline and g.deadline < pre.deadline:
                    risks.append(f"Goal '{g.id}' is due before its prerequisite '{d}'")
        return Strategy([g.id for g in ordered], phases, milestones, risks,
                        {i: self.score(g, now) for i, g in goals.items()})
