"""logical_reasoner.py - true/false reasoning via forward chaining.

Facts are strings; negation is written "not X". Rules: {"if": [...], "then": "..."}.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from intelligence_interface import IntelligenceModule, Result, Signal


class LogicalReasoner(IntelligenceModule):
    name = "logical_reasoner"

    @staticmethod
    def _norm(p: str) -> str:
        return " ".join(str(p).lower().split())

    @classmethod
    def _neg(cls, p: str) -> str:
        return p[4:].strip() if p.startswith("not ") else f"not {p}"

    def infer(self, facts: Iterable[str], rules: List[Dict[str, Any]], max_iter: int = 100) -> Dict[str, Any]:
        known = {self._norm(f) for f in facts}
        derivations: List[Dict[str, Any]] = []
        for _ in range(max_iter):
            changed = False
            for rule in rules:
                premises = [self._norm(p) for p in rule.get("if", [])]
                concl = self._norm(rule.get("then", ""))
                if concl and concl not in known and all(p in known for p in premises):
                    known.add(concl)
                    derivations.append({"conclusion": concl, "because": premises})
                    changed = True
            if not changed:
                break
        contradictions = sorted(p for p in known if not p.startswith("not ") and self._neg(p) in known)
        return {"known": sorted(known), "derived": derivations, "contradictions": contradictions}

    def evaluate(self, query: str, known: Iterable[str]) -> Optional[bool]:
        q, known = self._norm(query), set(known)
        if q in known:
            return True
        if self._neg(q) in known:
            return False
        return None  # unknown

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        p = signal.payload
        if not isinstance(p, dict) or "facts" not in p:
            return Result(ok=False, error="logical reasoning needs 'facts' (and optionally 'rules', 'query')")
        out = self.infer(p["facts"], p.get("rules", []))
        out["consistent"] = not out["contradictions"]
        answer = None
        if "query" in p:
            answer = self.evaluate(p["query"], out["known"])
            out["query"], out["answer"] = p["query"], answer
        if not out["consistent"]:
            confidence, summary = 0.0, f"Contradiction found: {out['contradictions']}"
        elif "query" in p and answer is None:
            confidence, summary = 0.4, f"Cannot determine '{p['query']}' from the given facts"
        elif "query" in p:
            confidence, summary = 1.0, f"'{p['query']}' is {'TRUE' if answer else 'FALSE'}"
        else:
            confidence, summary = 0.9, f"Derived {len(out['derived'])} new fact(s)"
        out.update(confidence=confidence, summary=summary)
        return Result(ok=True, data=out)
