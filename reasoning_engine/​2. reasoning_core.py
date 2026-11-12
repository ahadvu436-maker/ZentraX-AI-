"""reasoning_core.py - main controller of the reasoning module.

All reasoning is managed from here. It owns the reasoners, picks the mode,
and can fall back to an external reasoning engine through the ExternalGateway.
Requires the 01_core_intelligence folder on sys.path (for intelligence_interface).
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, Optional

from intelligence_interface import ExternalGateway, IntelligenceModule, Result, Signal
from logical_reasoner import LogicalReasoner
from analytical_reasoner import AnalyticalReasoner
from causal_reasoning import CausalReasoner
from multi_step_reasoning import MultiStepReasoner
from hypothesis_engine import HypothesisEngine
from problem_solver import ProblemSolver
from strategies import AbductiveStrategy, DeductiveStrategy, InductiveStrategy

log = logging.getLogger("reasoning_core")


class ReasoningCore(IntelligenceModule):
    name = "reasoning_core"

    def __init__(self, gateway: Optional[ExternalGateway] = None) -> None:
        self.gateway = gateway
        self.reasoners: Dict[str, IntelligenceModule] = {
            "logical": LogicalReasoner(),
            "analytical": AnalyticalReasoner(),
            "causal": CausalReasoner(),
            "multi_step": MultiStepReasoner(solver=self._external_solve),
            "hypothesis": HypothesisEngine(),
            "deductive": DeductiveStrategy(),
            "inductive": InductiveStrategy(),
            "abductive": AbductiveStrategy(),
        }
        self.solver = ProblemSolver(self.reasoners)
        self._history: deque = deque(maxlen=100)

    # ---- external reasoning engine hook --------------------------------
    def _external_solve(self, question: str) -> Optional[str]:
        if self.gateway and self.gateway.has("reasoning"):
            res = self.gateway.call("reasoning", "reason", question)
            return str(res.data) if res.ok and res.data is not None else None
        return None

    def register_reasoner(self, mode: str, module: IntelligenceModule) -> None:
        self.reasoners[mode] = module

    # ---- main entry ----------------------------------------------------
    def reason(self, problem: Any, mode: str = "auto") -> Result:
        payload = {"problem": problem} if isinstance(problem, str) else dict(problem)
        mode = payload.pop("mode", mode)
        sig = Signal(source=self.name, payload=payload, kind="reason")
        try:
            if mode == "auto":
                res = self.solver.process(sig)
                if not res.ok and self.gateway and self.gateway.has("reasoning"):
                    ext = self.gateway.call("reasoning", "reason", str(payload.get("problem", payload)), context=payload)
                    if ext.ok:
                        res = Result(ok=True, data={"answer": ext.data, "best_reasoner": "external",
                                                    "confidence": 0.5, "summary": str(ext.data)})
            elif mode in self.reasoners:
                res = self.reasoners[mode].process(sig)
            else:
                res = Result(ok=False, error=f"unknown mode '{mode}', choose from {['auto', *self.reasoners]}")
        except Exception as exc:  # the controller must not crash its caller
            log.exception("reasoning error")
            res = Result(ok=False, error=str(exc))
        self._history.append({"mode": mode, "ok": res.ok, "error": res.error})
        return res

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        return self.reason(signal.payload)

    def get_state(self) -> Dict[str, Any]:
        return {"reasoners": list(self.reasoners), "calls": len(self._history),
                "failures": sum(not h["ok"] for h in self._history)}

    def health_check(self) -> bool:
        return all(r.health_check() for r in self.reasoners.values())
