"""decision_engine.py - filters options for safety and picks the final decision."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from intelligence_interface import IntelligenceModule, Result, Signal


class DecisionEngine(IntelligenceModule):
    name = "decision_engine"

    def __init__(self, max_risk: float = 0.5, min_score: float = 0.2,
                 blocked_actions: Iterable[str] = ()) -> None:
        self.max_risk = max_risk
        self.min_score = min_score
        self.blocked_actions = set(blocked_actions)
        self._decisions = 0
        self._rejections = 0

    def _check(self, option: Dict[str, Any]) -> str | None:
        """Return a rejection reason, or None if the option is acceptable."""
        if option.get("action") in self.blocked_actions:
            return "blocked action"
        if option.get("risk", 1.0) > self.max_risk:
            return f"risk above {self.max_risk}"
        if option.get("score", 0.0) < self.min_score:
            return f"score below {self.min_score}"
        return None

    @staticmethod
    def _utility(option: Dict[str, Any]) -> float:
        return option.get("score", 0.0) * (1 - option.get("risk", 1.0))

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        payload = signal.payload
        options: List[Dict[str, Any]] = (payload.get("insight") or {}).get("options", []) if isinstance(payload, dict) else []
        if not options:
            return Result(ok=False, error="no options to decide on")

        safe, rejected = [], []
        for opt in options:
            reason = self._check(opt)
            (rejected.append({**opt, "reason": reason}) if reason else safe.append(opt))
        self._rejections += len(rejected)

        if not safe:
            return Result(ok=False, error="no safe option available", meta={"rejected": rejected})

        # Best utility wins; ties go to the lower-risk option.
        best = max(safe, key=lambda o: (self._utility(o), -o.get("risk", 1.0)))
        self._decisions += 1
        return Result(ok=True, data={
            "action": best["action"],
            "utility": round(self._utility(best), 3),
            "risk": best.get("risk"),
            "rejected": rejected,
        })

    def get_state(self) -> Dict[str, Any]:
        return {"decisions": self._decisions, "rejections": self._rejections,
                "max_risk": self.max_risk}
 
