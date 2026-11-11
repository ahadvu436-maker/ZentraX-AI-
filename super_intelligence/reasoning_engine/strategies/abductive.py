"""abductive.py - incomplete observations -> the most plausible explanation."""
from __future__ import annotations

from typing import Any, Dict, List

from intelligence_interface import IntelligenceModule, Result, Signal

_DEFAULT_KB = {
    "it rained": {"explains": ["wet grass", "wet street", "umbrellas out"], "prior": 0.4},
    "sprinkler was on": {"explains": ["wet grass"], "prior": 0.3},
    "pipe burst": {"explains": ["wet street", "low water pressure"], "prior": 0.05},
    "router is down": {"explains": ["no wifi", "no internet", "smart devices offline"], "prior": 0.3},
    "isp outage": {"explains": ["no internet", "neighbors offline"], "prior": 0.15},
    "device wifi disabled": {"explains": ["no wifi"], "prior": 0.2},
}


def _match(obs: str, known: str) -> bool:
    a, b = obs.lower().strip(), known.lower().strip()
    return a in b or b in a


class AbductiveStrategy(IntelligenceModule):
    name = "abductive_strategy"

    def __init__(self) -> None:
        self._kb: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in _DEFAULT_KB.items()}

    def add_explanation(self, name: str, explains: List[str], prior: float = 0.3) -> None:
        self._kb[name.lower()] = {"explains": [e.lower() for e in explains], "prior": max(0.0, min(1.0, prior))}

    def _explained(self, name: str, observations: List[str]) -> List[str]:
        return [o for o in observations if any(_match(o, x) for x in self._kb[name]["explains"])]

    def explain(self, observations: List[str], top: int = 5) -> Dict[str, Any]:
        obs = [o.lower().strip() for o in observations if o.strip()]
        scored = []
        for name, info in self._kb.items():
            got = self._explained(name, obs)
            if got:
                coverage = len(got) / len(obs)
                scored.append((info["prior"] * coverage, name, coverage, got))
        if not scored:
            return {"explanations": [], "confidence": 0.0, "summary": "No known explanation fits these observations"}
        scored.sort(key=lambda t: -t[0])
        total = sum(s for s, *_ in scored) or 1.0
        ranked = [{"explanation": n, "score": round(s, 3), "share": round(s / total, 3),
                   "explains": got, "coverage": round(c, 2)} for s, n, c, got in scored[:top]]
        best = ranked[0]
        unexplained = [o for o in obs if o not in best["explains"]]

        # if the best single cause leaves gaps, greedily add causes that cover them
        combined, remaining = [best["explanation"]], list(unexplained)
        while remaining:
            pick = max(self._kb, key=lambda n: len([o for o in remaining if any(_match(o, x) for x in self._kb[n]["explains"])]))
            covered = [o for o in remaining if any(_match(o, x) for x in self._kb[pick]["explains"])]
            if not covered:
                break
            combined.append(pick)
            remaining = [o for o in remaining if o not in covered]

        to_confirm = [x for x in self._kb[best["explanation"]]["explains"] if not any(_match(o, x) for o in obs)]
        conf = round(min(0.9, best["share"] * best["coverage"]), 3)
        note = f"; still unexplained: {remaining}" if remaining else ""
        return {"explanations": ranked, "best": best["explanation"], "combined": combined if unexplained else None,
                "unexplained": remaining, "to_confirm": to_confirm, "confidence": conf,
                "summary": f"Best explanation: {best['explanation']} (explains {len(best['explains'])}/{len(obs)}){note}"
                           + (f"; check: {to_confirm[0]}" if to_confirm else "")}

    def process(self, signal: Signal) -> Result:
        p = signal.payload
        if not isinstance(p, dict) or not p.get("observations"):
            return Result(ok=False, error="abduction needs 'observations' (list of strings)")
        for name, info in (p.get("explanations") or {}).items():
            self.add_explanation(name, info.get("explains", []), info.get("prior", 0.3))
        return Result(ok=True, data=self.explain(p["observations"], p.get("top", 5)))

    def get_state(self) -> Dict[str, Any]:
        return {"known_explanations": len(self._kb)}
