"""hypothesis_engine.py - generates and ranks hypotheses for a new problem."""
from __future__ import annotations

from typing import Any, Dict, List

from intelligence_interface import IntelligenceModule, Result, Signal

# hypothesis = {"hypothesis": str, "prior": float, "predicts": [evidence...], "contradicts": [evidence...]}
_GENERIC = [
    {"hypothesis": "bad or unexpected input data", "prior": 0.25},
    {"hypothesis": "configuration problem", "prior": 0.25},
    {"hypothesis": "external dependency failing", "prior": 0.25},
    {"hypothesis": "logic / code bug", "prior": 0.25},
]
_DEFAULT_PATTERNS = [
    (["slow", "latency", "timeout", "lag"], [
        {"hypothesis": "resource exhaustion (CPU/memory)", "prior": 0.3, "predicts": ["high cpu", "high memory"]},
        {"hypothesis": "network congestion", "prior": 0.3, "predicts": ["packet loss", "high ping"], "contradicts": ["local only"]},
        {"hypothesis": "inefficient query or algorithm", "prior": 0.4, "predicts": ["grows with data size"]},
    ]),
    (["error", "fail", "crash", "exception"], [
        {"hypothesis": "bad or unexpected input data", "prior": 0.3, "predicts": ["only some inputs"]},
        {"hypothesis": "dependency or service failure", "prior": 0.3, "predicts": ["service down", "connection refused"]},
        {"hypothesis": "recent code or config change", "prior": 0.4, "predicts": ["started after update"]},
    ]),
    (["offline", "disconnect", "network"], [
        {"hypothesis": "connectivity lost", "prior": 0.6, "predicts": ["no internet", "connection refused"]},
        {"hypothesis": "DNS or proxy problem", "prior": 0.4, "predicts": ["name resolution failed"], "contradicts": ["ping by ip works"]},
    ]),
]


class HypothesisEngine(IntelligenceModule):
    name = "hypothesis_engine"

    def __init__(self) -> None:
        self._patterns: List[tuple] = list(_DEFAULT_PATTERNS)

    def register_pattern(self, keywords: List[str], hypotheses: List[Dict[str, Any]]) -> None:
        self._patterns.append(([k.lower() for k in keywords], hypotheses))

    def generate(self, observation: str, evidence: List[str] | None = None, top: int = 5) -> List[Dict[str, Any]]:
        obs, evidence = observation.lower(), [e.lower() for e in (evidence or [])]
        pool: Dict[str, Dict[str, Any]] = {}
        for keywords, hyps in self._patterns:
            if any(k in obs for k in keywords):
                for h in hyps:
                    if h["hypothesis"] not in pool or h["prior"] > pool[h["hypothesis"]]["prior"]:
                        pool[h["hypothesis"]] = h
        if not pool:
            pool = {h["hypothesis"]: h for h in _GENERIC}

        scored = []
        for h in pool.values():
            score = float(h["prior"])
            for ev in evidence:
                if any(ev in x.lower() or x.lower() in ev for x in h.get("predicts", [])):
                    score *= 2.0
                if any(ev in x.lower() or x.lower() in ev for x in h.get("contradicts", [])):
                    score *= 0.2
            scored.append((score, h))
        total = sum(s for s, _ in scored) or 1.0
        ranked = []
        for s, h in sorted(scored, key=lambda t: -t[0])[:top]:
            unseen = [x for x in h.get("predicts", []) if x.lower() not in evidence]
            ranked.append({"hypothesis": h["hypothesis"], "probability": round(s / total, 3), "check_next": unseen})
        return ranked

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        p = signal.payload
        if not isinstance(p, dict) or not p.get("observation"):
            return Result(ok=False, error="hypothesis engine needs an 'observation'")
        ranked = self.generate(p["observation"], p.get("evidence"), p.get("top", 5))
        best = ranked[0]
        hint = f"; check next: {best['check_next'][0]}" if best["check_next"] else ""
        return Result(ok=True, data={"hypotheses": ranked, "confidence": best["probability"],
                                     "summary": f"Leading hypothesis: {best['hypothesis']} ({best['probability']:.0%}){hint}"})
