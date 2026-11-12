"""problem_solver.py - combines every reasoner into one final problem-solving method."""
from __future__ import annotations

from typing import Any, Dict, List

from intelligence_interface import IntelligenceModule, Result, Signal

# which payload keys make each reasoner applicable
APPLICABLE = {
    "logical": ("facts",),
    "analytical": ("data",),
    "causal": ("effect", "cause"),
    "hypothesis": ("observation",),
    "multi_step": ("problem",),
    "deductive": ("premises",),
    "inductive": ("examples", "sequence"),
    "abductive": ("observations",),
}


class ProblemSolver(IntelligenceModule):
    name = "problem_solver"

    def __init__(self, reasoners: Dict[str, IntelligenceModule]) -> None:
        self.reasoners = reasoners

    def applicable(self, payload: Dict[str, Any]) -> List[str]:
        return [n for n, keys in APPLICABLE.items()
                if n in self.reasoners and any(payload.get(k) for k in keys)]

    def process(self, signal: Signal) -> Result:
        p = signal.payload
        if isinstance(p, str):
            p = {"problem": p}
        if not isinstance(p, dict):
            return Result(ok=False, error="problem must be text or a dict")
        names = self.applicable(p)
        if not names:
            return Result(ok=False, error="no applicable reasoner for this problem")

        findings: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        for n in names:
            res = self.reasoners[n].process(Signal(source=self.name, payload=p, kind="reason"))
            if res.ok:
                findings[n] = res.data
            else:
                errors[n] = res.error or "unknown error"
        if not findings:
            return Result(ok=False, error="all reasoners failed", meta={"errors": errors})

        best = max(findings, key=lambda n: findings[n].get("confidence", 0.0))
        conf = findings[best].get("confidence", 0.0)
        lines = [f"[{n}] {d.get('summary', '')}" for n, d in findings.items()]
        return Result(ok=True, data={
            "answer": findings[best].get("summary", ""),
            "best_reasoner": best,
            "confidence": conf,
            "summary": " || ".join(lines),
            "findings": findings,
            "errors": errors,
        })
