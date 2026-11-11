"""inductive.py - specific examples -> general theory (always probable, never certain)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

from intelligence_interface import IntelligenceModule, Result, Signal


def _h(v: Any) -> Any:
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


class InductiveStrategy(IntelligenceModule):
    name = "inductive_strategy"

    def __init__(self, min_support: int = 2, min_accuracy: float = 0.75) -> None:
        self.min_support = min_support
        self.min_accuracy = min_accuracy

    # ---- rules from labeled examples -----------------------------------
    def learn_rules(self, examples: List[Dict[str, Any]], label_key: str = "label") -> List[Dict[str, Any]]:
        total: Counter = Counter()
        hits: Counter = Counter()
        for e in examples:
            if label_key not in e:
                continue
            for k, v in e.items():
                if k != label_key:
                    total[(k, _h(v))] += 1
                    hits[(k, _h(v), _h(e[label_key]))] += 1
        rules = []
        for (k, v, lab), h in hits.items():
            t = total[(k, v)]
            if t >= self.min_support and h / t >= self.min_accuracy:
                rules.append({"if": {k: v}, "then": {label_key: lab}, "support": t, "accuracy": round(h / t, 3),
                              "counterexamples": t - h, "confidence": round((h + 1) / (t + 2), 3)})
        return sorted(rules, key=lambda r: (-r["confidence"], -r["support"]))

    @staticmethod
    def predict(rules: List[Dict[str, Any]], features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for r in rules:  # rules are sorted best first
            if all(features.get(k) == v for k, v in r["if"].items()):
                return {"prediction": r["then"], "rule": r, "confidence": r["confidence"]}
        return None

    # ---- patterns in number sequences ----------------------------------
    @staticmethod
    def detect_sequence(seq: List[Any]) -> Optional[Dict[str, Any]]:
        nums = [float(x) for x in seq if isinstance(x, (int, float)) and not isinstance(x, bool)]
        n = len(nums)
        if n < 3:
            return None
        same = lambda xs: all(math.isclose(x, xs[0], rel_tol=1e-9, abs_tol=1e-9) for x in xs)
        d = [nums[i + 1] - nums[i] for i in range(n - 1)]
        conf = round(min(0.95, 1 - 1 / n), 2)
        if same(d):
            return {"pattern": "arithmetic", "rule": f"add {d[0]:g} each step", "next": nums[-1] + d[0], "confidence": conf}
        if all(x != 0 for x in nums[:-1]):
            r = [nums[i + 1] / nums[i] for i in range(n - 1)]
            if same(r):
                return {"pattern": "geometric", "rule": f"multiply by {r[0]:g} each step", "next": nums[-1] * r[0], "confidence": conf}
        if n >= 4:
            dd = [d[i + 1] - d[i] for i in range(len(d) - 1)]
            if same(dd):
                return {"pattern": "quadratic", "rule": f"differences grow by {dd[0]:g}",
                        "next": nums[-1] + d[-1] + dd[0], "confidence": round(conf - 0.1, 2)}
        return None

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        p = signal.payload
        if not isinstance(p, dict) or not (p.get("examples") or p.get("sequence")):
            return Result(ok=False, error="induction needs 'examples' (list of dicts) and/or 'sequence' (numbers)")
        out: Dict[str, Any] = {}
        notes: List[str] = []
        conf = 0.0
        if p.get("examples"):
            rules = self.learn_rules(p["examples"], p.get("label_key", "label"))
            out["rules"] = rules
            if rules:
                top = rules[0]
                notes.append(f"Theory: if {top['if']} then {top['then']} "
                             f"({top['support']} cases, {top['counterexamples']} counterexample(s))")
                conf = max(conf, top["confidence"])
            else:
                notes.append("No general rule found in the examples")
            if p.get("predict"):
                out["prediction"] = self.predict(rules, p["predict"])
        if p.get("sequence"):
            seq = self.detect_sequence(p["sequence"])
            out["sequence"] = seq
            if seq:
                notes.append(f"Sequence is {seq['pattern']} ({seq['rule']}); next is {seq['next']:g}")
                conf = max(conf, seq["confidence"])
            else:
                notes.append("No simple pattern in the sequence")
        out.update(confidence=conf, summary="; ".join(notes))
        return Result(ok=True, data=out)
