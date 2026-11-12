"""PlanningAgent: goal decomposition, scheduling and critical-path analysis.

Payload actions
---------------
``plan`` (default)  ``goal``, ``max_steps``, ``constraints``   decompose a goal into dependent steps
``schedule``        ``steps`` (list of step dicts)             earliest/latest times, slack, critical path
``checkpoint``      (uses upstream results)                    aggregate upstream outputs

Use :func:`plan_to_tasks` to turn a plan into ``Task`` objects that the
``AgentOrchestrator`` can execute across the other agents.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..agent_core import AgentConfig, BaseAgent, Reasoner, Task, TaskPriority, call_reasoner

__all__ = ["PlanStep", "PlanningAgent", "compute_schedule", "plan_to_tasks"]

_RESEARCH = ("research", "investigate", "find", "explore", "study", "compare", "survey", "learn", "market")
_CODING = ("code", "build", "implement", "develop", "program", "script", "api", "app", "software", "fix", "bug", "deploy", "refactor", "feature")
_ANALYSIS = ("analy", "evaluate", "measure", "data", "metric", "statistic", "assess", "benchmark", "forecast", "trend")


@dataclass
class PlanStep:
    id: str
    title: str
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    estimate_minutes: float = 30.0
    earliest_start: float = 0.0
    earliest_finish: float = 0.0
    slack: float = 0.0
    critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanStep":
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or data["id"]),
            description=str(data.get("description") or ""),
            depends_on=[str(d) for d in data.get("depends_on", [])],
            capabilities=[str(c) for c in data.get("capabilities", [])],
            estimate_minutes=max(1.0, float(data.get("estimate_minutes", 30) or 30)),
        )


def compute_schedule(steps: Sequence[PlanStep]) -> Tuple[List[PlanStep], float, List[str]]:
    """Return (steps in topological order, total minutes, critical path ids)."""
    by_id = {s.id: s for s in steps}
    if len(by_id) != len(steps):
        raise ValueError("duplicate step ids")
    children: Dict[str, List[str]] = defaultdict(list)
    indegree: Dict[str, int] = {}
    for s in steps:
        deps = set(s.depends_on)
        for d in deps:
            if d not in by_id:
                raise ValueError(f"step {s.id!r} depends on unknown step {d!r}")
            if d == s.id:
                raise ValueError(f"step {s.id!r} depends on itself")
            children[d].append(s.id)
        indegree[s.id] = len(deps)
    queue = deque(s.id for s in steps if indegree[s.id] == 0)
    order: List[PlanStep] = []
    while queue:
        sid = queue.popleft()
        order.append(by_id[sid])
        for c in children[sid]:
            indegree[c] -= 1
            if indegree[c] == 0:
                queue.append(c)
    if len(order) != len(steps):
        raise ValueError("plan contains a dependency cycle")

    for s in order:
        s.earliest_start = max((by_id[d].earliest_finish for d in s.depends_on), default=0.0)
        s.earliest_finish = s.earliest_start + s.estimate_minutes
    total = max((s.earliest_finish for s in order), default=0.0)
    latest_finish: Dict[str, float] = {}
    for s in reversed(order):
        lf = min((latest_finish[c] - by_id[c].estimate_minutes for c in children[s.id]), default=total)
        latest_finish[s.id] = lf
        s.slack = round(lf - s.earliest_finish, 6)
        s.critical = s.slack <= 1e-9
    return order, total, [s.id for s in order if s.critical]


def _extract_json(text: str) -> Any:
    candidates = [text.strip()]
    candidates += re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i, j = text.find(open_c), text.rfind(close_c)
        if 0 <= i < j:
            candidates.append(text[i : j + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except (ValueError, TypeError):
            continue
    raise ValueError("no JSON found in model output")


def _heuristic_steps(goal: str) -> List[PlanStep]:
    g = goal.lower()
    research = any(k in g for k in _RESEARCH)
    coding = any(k in g for k in _CODING)
    analysis = any(k in g for k in _ANALYSIS)
    if not (research or coding or analysis):
        research = analysis = True
    steps: List[PlanStep] = []

    def add(title: str, desc: str, caps: List[str], minutes: float, deps: List[str]) -> str:
        sid = f"s{len(steps) + 1}"
        steps.append(PlanStep(sid, title, desc, list(deps), caps, minutes))
        return sid

    scope = add("Clarify scope and success criteria", f"Define what 'done' means for: {goal}", ["planning"], 15, [])
    gather = add("Gather information", f"Collect relevant information for: {goal}", ["research"], 45, [scope]) if research else None
    analyze = (
        add("Analyze findings and data", f"Analyze what was gathered for: {goal}", ["analysis"], 40, [gather or scope])
        if analysis
        else None
    )
    design = add("Design the approach", f"Decide the approach for: {goal}", ["planning"], 25, [x for x in (gather, analyze) if x] or [scope])
    if coding:
        impl = add("Implement solution", f"Write the code for: {goal}", ["code_generation"], 90, [design])
        add("Review and validate code", "Statically review the implementation", ["code_review"], 30, [impl])
    used = {d for s in steps for d in s.depends_on}
    add("Synthesize and deliver results", "Combine all outputs into the final deliverable", ["planning"], 20, [s.id for s in steps if s.id not in used])
    return steps


def _normalise(raw: Any, max_steps: int) -> List[PlanStep]:
    if isinstance(raw, Mapping):
        raw = raw.get("steps")
    if not isinstance(raw, list) or not raw:
        raise ValueError("model output has no steps")
    steps: List[PlanStep] = []
    for i, item in enumerate(raw[:max_steps], 1):
        if not isinstance(item, Mapping):
            raise ValueError("step is not an object")
        data = dict(item)
        data.setdefault("id", f"s{i}")
        steps.append(PlanStep.from_dict(data))
    ids = {s.id for s in steps}
    for s in steps:
        s.depends_on = [d for d in s.depends_on if d in ids and d != s.id]
    return steps


class PlanningAgent(BaseAgent):
    agent_type = "planning"
    DEFAULT_CAPABILITIES = frozenset({"planning", "task_decomposition", "scheduling", "estimation"})

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        agent_id: Optional[str] = None,
        llm: Optional[Reasoner] = None,
    ) -> None:
        config = config or AgentConfig(
            name="planning-agent", capabilities=self.DEFAULT_CAPABILITIES, max_concurrent_tasks=4
        )
        super().__init__(config, agent_id)
        self._llm = llm

    async def handle_task(self, task: Task) -> Dict[str, Any]:
        p = task.payload
        action = str(p.get("action", "plan")).lower()
        if action == "plan":
            return await self._plan(task)
        if action == "schedule":
            steps = [PlanStep.from_dict(s) for s in p.get("steps") or []]
            if not steps:
                raise ValueError("schedule requires a non-empty 'steps' list")
            order, total, critical = compute_schedule(steps)
            return {"steps": [s.to_dict() for s in order], "total_minutes": total, "critical_path": critical}
        if action == "checkpoint":
            upstream = task.context.get("dependency_results") or {}
            digest: Dict[str, str] = {}
            for key, value in upstream.items():
                if isinstance(value, Mapping) and isinstance(value.get("summary"), str):
                    digest[key] = value["summary"]
                else:
                    digest[key] = repr(value)[:500]
            return {"checkpoint": p.get("title", task.name), "upstream_count": len(upstream), "upstream": digest}
        raise ValueError(f"unknown planning action: {action!r}")

    async def _plan(self, task: Task) -> Dict[str, Any]:
        p = task.payload
        goal = str(p.get("goal") or p.get("instructions") or task.description or "").strip()
        if not goal:
            raise ValueError("plan requires a 'goal'")
        max_steps = max(2, int(p.get("max_steps", 8)))
        constraints = [str(c) for c in p.get("constraints", [])]
        steps: Optional[List[PlanStep]] = None
        source = "heuristic"

        if self._llm is not None:
            prompt = (
                f"Decompose this goal into at most {max_steps} steps. Reply with only a JSON array of objects with "
                'keys "id","title","description","depends_on" (list of ids),"capabilities" (subset of '
                '["research","analysis","code_generation","code_review","planning"]),"estimate_minutes".\n'
                f"Goal: {goal}\n" + (f"Constraints: {'; '.join(constraints)}\n" if constraints else "")
            )
            try:
                steps = _normalise(_extract_json(await call_reasoner(self._llm, prompt)), max_steps)
                compute_schedule(steps)  # validates graph
                source = "llm"
            except Exception as exc:  # noqa: BLE001
                self.log.warning("LLM plan rejected (%s); using heuristic plan", exc)
                steps = None
        if steps is None:
            steps = _heuristic_steps(goal)

        order, total, critical = compute_schedule(steps)
        return {
            "goal": goal,
            "source": source,
            "constraints": constraints,
            "steps": [s.to_dict() for s in order],
            "total_minutes": total,
            "critical_path": critical,
        }


def _default_payload(step: PlanStep, goal: str) -> Dict[str, Any]:
    caps = set(step.capabilities)
    text = f"{step.title}. {step.description}".strip()
    if "research" in caps:
        return {"action": "research", "query": text}
    if "code_review" in caps:
        return {"action": "review"}
    if "code_generation" in caps or "coding" in caps:
        return {"action": "generate", "spec": step.description or step.title}
    if "analysis" in caps:
        return {"action": "auto", "instructions": text}
    return {"action": "checkpoint", "title": step.title, "goal": goal}


def plan_to_tasks(
    plan: Any,
    *,
    goal: str = "",
    payload_builders: Optional[Mapping[str, Callable[[PlanStep], Dict[str, Any]]]] = None,
    timeout: Optional[float] = None,
    max_retries: int = 1,
) -> List[Task]:
    """Convert a plan (``PlanningAgent`` output dict or list of steps) into ``Task`` objects.

    ``payload_builders`` maps a capability name to a function building that
    step's payload, overriding the defaults. Critical-path steps get HIGH priority.
    """
    if isinstance(plan, Mapping):
        goal = goal or str(plan.get("goal", ""))
        raw_steps = plan.get("steps", [])
    else:
        raw_steps = plan
    steps = [s if isinstance(s, PlanStep) else PlanStep.from_dict(s) for s in raw_steps]
    critical_ids = {s["id"] if isinstance(s, Mapping) else s.id for s in raw_steps if (s.get("critical") if isinstance(s, Mapping) else s.critical)}
    tasks: Dict[str, Task] = {}
    for step in steps:
        builder = next((payload_builders[c] for c in step.capabilities if payload_builders and c in payload_builders), None)
        payload = builder(step) if builder else _default_payload(step, goal)
        tasks[step.id] = Task(
            name=step.title,
            description=step.description,
            payload=payload,
            required_capabilities=frozenset(step.capabilities),
            priority=TaskPriority.HIGH if step.id in critical_ids else TaskPriority.NORMAL,
            timeout=timeout,
            max_retries=max_retries,
            context={"plan_step_id": step.id},
        )
    for step in steps:
        tasks[step.id].dependencies = [tasks[d].id for d in step.depends_on]
    return [tasks[s.id] for s in steps]
