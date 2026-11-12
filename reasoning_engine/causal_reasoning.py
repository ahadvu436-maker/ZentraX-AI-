"""causal_reasoning.py - cause-and-effect graph: find *why* something happened."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from intelligence_interface import IntelligenceModule, Result, Signal


class CausalReasoner(IntelligenceModule):
    name = "causal_reasoner"

    def __init__(self) -> None:
        self._edges: Dict[str, Dict[str, float]] = defaultdict(dict)  # cause -> {effect: strength 0..1}

    def add_cause(self, cause: str, effect: str, strength: float = 0.5) -> None:
        self._edges[cause.lower()][effect.lower()] = max(0.0, min(1.0, float(strength)))

    def _neighbors(self, node: str, forward: bool):
        if forward:
            return list(self._edges.get(node, {}).items())
        return [(c, effs[node]) for c, effs in self._edges.items() if node in effs]

    def _walk(self, start: str, forward: bool, max_depth: int = 10) -> Dict[str, Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}

        def go(node: str, path: List[str], strength: float) -> None:
            if len(path) > max_depth:
                return
            for nxt, s in self._neighbors(node, forward):
                if nxt in path:  # cycle guard
                    continue
                st, p = strength * s, path + [nxt]
                if nxt not in best or st > best[nxt]["strength"]:
                    best[nxt] = {"strength": round(st, 3), "path": p if forward else p[::-1]}
                go(nxt, p, st)

        go(start, [start], 1.0)
        return best

    def why(self, effect: str) -> List[Dict[str, Any]]:
        """All upstream causes of `effect`, strongest first; roots are flagged."""
        effect = effect.lower()
        causes = self._walk(effect, forward=False)
        ranked = [{"cause": c, "is_root": not self._neighbors(c, False), **info} for c, info in causes.items()]
        return sorted(ranked, key=lambda r: (-r["strength"], len(r["path"])))

    def consequences(self, cause: str) -> List[Dict[str, Any]]:
        effects = self._walk(cause.lower(), forward=True)
        ranked = [{"effect": e, **info} for e, info in effects.items()]
        return sorted(ranked, key=lambda r: -r["strength"])

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        p = signal.payload
        if not isinstance(p, dict):
            return Result(ok=False, error="payload must be a dict")
        for edge in p.get("edges", []):
            self.add_cause(*edge)
        if "effect" in p:
            ranked = self.why(p["effect"])
            roots = [r for r in ranked if r["is_root"]]
            if not ranked:
                return Result(ok=True, data={"causes": [], "confidence": 0.0,
                                             "summary": f"No known cause for '{p['effect']}'"})
            top = (roots or ranked)[0]
            arrow = " -> ".join(top["path"])
            return Result(ok=True, data={"causes": ranked, "root_causes": roots, "confidence": top["strength"],
                                         "summary": f"Most likely root cause of '{p['effect']}': {top['cause']} ({arrow}, strength {top['strength']})"})
        if "cause" in p:
            ranked = self.consequences(p["cause"])
            conf = ranked[0]["strength"] if ranked else 0.0
            return Result(ok=True, data={"effects": ranked, "confidence": conf,
                                         "summary": f"'{p['cause']}' leads to {len(ranked)} downstream effect(s)"})
        return Result(ok=bool(p.get("edges")), data={"confidence": 0.0, "summary": "graph updated"},
                      error=None if p.get("edges") else "causal reasoning needs 'effect' or 'cause'")

    def get_state(self) -> Dict[str, Any]:
        return {"edges": sum(len(v) for v in self._edges.values())}
