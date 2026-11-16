"""Reasoning inspection: structural and logical checks over a reasoning trace."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .self_monitoring import Severity


class StepKind(str, Enum):
    OBSERVATION = "observation"
    ASSUMPTION = "assumption"
    INFERENCE = "inference"
    CONCLUSION = "conclusion"


@dataclass(frozen=True)
class ReasoningStep:
    id: str
    content: str
    kind: StepKind = StepKind.INFERENCE
    premises: Tuple[str, ...] = ()
    confidence: float = 1.0
    evidence: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"step {self.id}: confidence must be in [0, 1]")


@dataclass
class ReasoningTrace:
    steps: List[ReasoningStep] = field(default_factory=list)
    goal: str = ""

    def add(self, step: ReasoningStep) -> "ReasoningTrace":
        if any(s.id == step.id for s in self.steps):
            raise ValueError(f"duplicate step id: {step.id}")
        self.steps.append(step)
        return self

    def by_id(self) -> Dict[str, ReasoningStep]:
        return {s.id: s for s in self.steps}


@dataclass(frozen=True)
class Flaw:
    kind: str
    message: str
    severity: Severity
    step_id: Optional[str] = None


@dataclass
class InspectionReport:
    flaws: List[Flaw]
    soundness: float  # 0..1, higher is better
    checked_steps: int

    @property
    def worst_severity(self) -> Severity:
        return max((f.severity for f in self.flaws), default=Severity.INFO)


class InspectionCheck(ABC):
    name: str = "check"

    @abstractmethod
    def run(self, trace: ReasoningTrace) -> List[Flaw]: ...


class DanglingPremiseCheck(InspectionCheck):
    name = "dangling_premise"

    def run(self, trace: ReasoningTrace) -> List[Flaw]:
        ids, out = trace.by_id(), []
        for s in trace.steps:
            for p in s.premises:
                if p not in ids:
                    out.append(Flaw(self.name, f"references unknown premise '{p}'", Severity.ERROR, s.id))
        return out


class UnsupportedClaimCheck(InspectionCheck):
    name = "unsupported_claim"

    def run(self, trace: ReasoningTrace) -> List[Flaw]:
        out = []
        for s in trace.steps:
            if s.kind in (StepKind.INFERENCE, StepKind.CONCLUSION) and not s.premises and not s.evidence:
                out.append(Flaw(self.name, "claim has neither premises nor evidence", Severity.ERROR, s.id))
        return out


class CircularReasoningCheck(InspectionCheck):
    name = "circular_reasoning"

    def run(self, trace: ReasoningTrace) -> List[Flaw]:
        graph = {s.id: [p for p in s.premises if p in trace.by_id()] for s in trace.steps}
        state: Dict[str, int] = {}
        flaws: List[Flaw] = []

        def dfs(n: str, stack: List[str]) -> None:
            state[n] = 1
            for m in graph[n]:
                if state.get(m) == 1:
                    cyc = stack[stack.index(m):] + [m] if m in stack else [n, m]
                    flaws.append(Flaw(self.name, "cycle: " + " -> ".join(cyc), Severity.CRITICAL, m))
                elif m not in state:
                    dfs(m, stack + [m])
            state[n] = 2

        for n in graph:
            if n not in state:
                dfs(n, [n])
        return flaws


class ContradictionCheck(InspectionCheck):
    """Baseline lexical contradiction detector (claim vs. its negation)."""
    name = "contradiction"
    _NEG = re.compile(r"\b(not|no|never|cannot|can't|isn't|aren't|doesn't|don't|won't)\b")

    def _norm(self, text: str) -> Tuple[str, bool]:
        t = text.lower().strip().rstrip(".!")
        negs = len(self._NEG.findall(t))
        core = re.sub(r"\s+", " ", self._NEG.sub("", t)).strip()
        return core, negs % 2 == 1

    def run(self, trace: ReasoningTrace) -> List[Flaw]:
        seen: Dict[str, Tuple[bool, str]] = {}
        out = []
        for s in trace.steps:
            core, neg = self._norm(s.content)
            if core in seen and seen[core][0] != neg:
                out.append(Flaw(self.name, f"contradicts step '{seen[core][1]}'", Severity.CRITICAL, s.id))
            seen.setdefault(core, (neg, s.id))
        return out


class ConfidenceInflationCheck(InspectionCheck):
    """A conclusion cannot be more certain than the weakest premise it depends on."""
    name = "confidence_inflation"

    def __init__(self, tolerance: float = 0.1) -> None:
        self.tolerance = tolerance

    def run(self, trace: ReasoningTrace) -> List[Flaw]:
        ids, out = trace.by_id(), []
        for s in trace.steps:
            ps = [ids[p].confidence for p in s.premises if p in ids]
            if ps and s.confidence > min(ps) + self.tolerance:
                out.append(Flaw(self.name, f"confidence {s.confidence:.2f} exceeds weakest premise {min(ps):.2f}",
                                Severity.WARNING, s.id))
        return out


class MissingConclusionCheck(InspectionCheck):
    name = "missing_conclusion"

    def run(self, trace: ReasoningTrace) -> List[Flaw]:
        if trace.steps and not any(s.kind is StepKind.CONCLUSION for s in trace.steps):
            return [Flaw(self.name, "trace has no conclusion step", Severity.WARNING)]
        return []


_WEIGHTS = {Severity.INFO: 0.0, Severity.WARNING: 0.15, Severity.ERROR: 0.35, Severity.CRITICAL: 0.8}


class ReasoningInspector:
    def __init__(self, checks: Optional[Sequence[InspectionCheck]] = None) -> None:
        self.checks: List[InspectionCheck] = list(checks) if checks is not None else [
            DanglingPremiseCheck(), UnsupportedClaimCheck(), CircularReasoningCheck(),
            ContradictionCheck(), ConfidenceInflationCheck(), MissingConclusionCheck()]
        self._last: Optional[InspectionReport] = None

    def add_check(self, check: InspectionCheck) -> None:
        self.checks.append(check)

    def inspect(self, trace: ReasoningTrace) -> InspectionReport:
        flaws: List[Flaw] = []
        for c in self.checks:
            try:
                flaws.extend(c.run(trace))
            except Exception as e:  # a broken check is itself reported, never fatal
                flaws.append(Flaw("inspector_failure", f"check '{c.name}' crashed: {e!r}", Severity.WARNING))
        n = max(1, len(trace.steps))
        penalty = sum(_WEIGHTS[f.severity] for f in flaws) / n
        self._last = InspectionReport(flaws, max(0.0, 1.0 - penalty), len(trace.steps))
        return self._last

    def introspect(self) -> Dict[str, Any]:
        r = self._last
        return {"checks": [c.name for c in self.checks],
                "last_soundness": r.soundness if r else None,
                "last_flaw_count": len(r.flaws) if r else None}
