"""multi_step_reasoning.py - step-by-step (chain-of-thought) solving with a visible trace."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from intelligence_interface import IntelligenceModule, Result, Signal

Solver = Callable[[str], Optional[str]]
Step = Tuple[str, Callable[[Dict[str, Any]], None]]


class MultiStepReasoner(IntelligenceModule):
    """Pipeline: understand -> plan -> execute -> verify -> conclude.

    `solver` answers one sub-question (e.g. an LLM or other engine). Without a
    solver the chain still produces a plan, but steps stay unresolved.
    """
    name = "multi_step_reasoner"

    def __init__(self, solver: Optional[Solver] = None) -> None:
        self.solver = solver
        self._steps: List[Step] = [
            ("understand", self._understand), ("plan", self._plan), ("execute", self._execute),
            ("verify", self._verify), ("conclude", self._conclude),
        ]

    def set_steps(self, steps: List[Step]) -> None:
        self._steps = steps

    # ---- default steps -------------------------------------------------
    @staticmethod
    def _split(problem: str) -> List[str]:
        parts = [p.strip() for p in re.split(r"[.;?]\s*|\bthen\b", problem, flags=re.I) if p.strip()]
        return parts or [problem]

    def _understand(self, st: Dict[str, Any]) -> None:
        st["trace"].append(f"Understand: {st['problem']!r}")

    def _plan(self, st: Dict[str, Any]) -> None:
        st["subs"] = st["subs"] or self._split(st["problem"])
        for i, s in enumerate(st["subs"], 1):
            st["trace"].append(f"Plan {i}: {s}")

    def _execute(self, st: Dict[str, Any]) -> None:
        for i, s in enumerate(st["subs"], 1):
            ans = self.solver(s) if self.solver else None
            st["answers"][s] = ans
            st["trace"].append(f"Step {i}: {s} -> {ans if ans is not None else 'unresolved'}")

    def _verify(self, st: Dict[str, Any]) -> None:
        done = sum(a is not None for a in st["answers"].values())
        st["confidence"] = round(done / len(st["subs"]), 2) if st["subs"] else 0.0
        st["trace"].append(f"Verify: {done}/{len(st['subs'])} step(s) resolved")

    def _conclude(self, st: Dict[str, Any]) -> None:
        if st["confidence"] == 1.0:
            st["conclusion"] = " | ".join(str(a) for a in st["answers"].values())
        elif st["confidence"] > 0:
            st["conclusion"] = "Partial answer: " + " | ".join(str(a) for a in st["answers"].values() if a is not None)
        else:
            st["conclusion"] = f"Plan only ({len(st['subs'])} step(s)); no solver resolved them"
        st["trace"].append(f"Conclude: {st['conclusion']}")

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        p = signal.payload
        problem = p if isinstance(p, str) else (p or {}).get("problem") if isinstance(p, dict) else None
        if not problem:
            return Result(ok=False, error="multi-step reasoning needs a 'problem' text")
        st: Dict[str, Any] = {"problem": problem, "subs": list(p.get("steps", [])) if isinstance(p, dict) else [],
                              "answers": {}, "trace": [], "confidence": 0.0, "conclusion": ""}
        try:
            for _, fn in self._steps:
                fn(st)
        except Exception as exc:
            return Result(ok=False, error=f"chain broke: {exc}", data={"trace": st["trace"]})
        return Result(ok=True, data={"trace": st["trace"], "answers": st["answers"], "confidence": st["confidence"],
                                     "summary": st["conclusion"]})
