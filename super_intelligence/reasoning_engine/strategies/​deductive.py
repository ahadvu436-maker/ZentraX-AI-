"""deductive.py - general truths -> specific truths.

Premises are plain sentences:
    "all men are mortal"   "no fish are mammals"   "socrates is a man"   "rex is not a cat"
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from intelligence_interface import IntelligenceModule, Result, Signal

_IRREGULAR = {"men": "man", "women": "woman", "children": "child", "people": "person", "mice": "mouse"}
_ALL = re.compile(r"^all (.+?) (?:are|is) (.+)$", re.I)
_NO = re.compile(r"^no (.+?) (?:are|is) (.+)$", re.I)
_IS = re.compile(r"^(.+?) (?:is|are) (not )?(?:an? |the )?(.+)$", re.I)


def _cat(word: str) -> str:
    w = " ".join(word.lower().split())
    for art in ("a ", "an ", "the "):
        if w.startswith(art):
            w = w[len(art):]
    w = _IRREGULAR.get(w, w)
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


class DeductiveStrategy(IntelligenceModule):
    name = "deductive_strategy"

    def deduce(self, premises: List[str], query: Optional[str] = None) -> Dict[str, Any]:
        universals: Set[Tuple[str, str, bool]] = set()  # (A, B, negative)
        pos: Dict[str, Set[str]] = {}
        neg: Dict[str, Set[str]] = {}
        conclusions: List[Dict[str, Any]] = []
        unparsed: List[str] = []

        for s in premises:
            s = s.strip()
            if m := _ALL.match(s):
                universals.add((_cat(m[1]), _cat(m[2]), False))
            elif m := _NO.match(s):
                universals.add((_cat(m[1]), _cat(m[2]), True))
            elif m := _IS.match(s):
                (neg if m[2] else pos).setdefault(m[1].lower().strip(), set()).add(_cat(m[3]))
            else:
                unparsed.append(s)

        changed = True
        while changed:
            changed = False
            for a, b, negative in sorted(universals):
                for c, d, n2 in sorted(universals):  # chain: all A are B + all B are C => all A are C
                    if not negative and b == c and (a, d, n2) not in universals and a != d:
                        universals.add((a, d, n2))
                        conclusions.append({"statement": f"{'no' if n2 else 'all'} {a} are {d}",
                                            "because": [f"all {a} are {b}", f"{'no' if n2 else 'all'} {b} are {d}"]})
                        changed = True
                for ent, cats in list(pos.items()):
                    if a in cats:
                        target = neg if negative else pos
                        if b not in target.setdefault(ent, set()):
                            target[ent].add(b)
                            conclusions.append({"statement": f"{ent} is {'not ' if negative else ''}a {b}",
                                                "because": [f"{ent} is a {a}", f"{'no' if negative else 'all'} {a} are {b}"]})
                            changed = True

        contradictions = sorted(e for e in pos if pos[e] & neg.get(e, set()))
        out: Dict[str, Any] = {"conclusions": conclusions, "contradictions": contradictions, "unparsed": unparsed}

        answer: Optional[bool] = None
        if query:
            m = _IS.match(query.strip())
            if m:
                ent, negated, cat = m[1].lower().strip(), bool(m[2]), _cat(m[3])
                holds, excluded = cat in pos.get(ent, set()), cat in neg.get(ent, set())
                answer = (holds if not negated else excluded) if (holds or excluded) else None
                if negated and holds:
                    answer = False
            out["query"], out["answer"] = query, answer

        if contradictions:
            conf, summary = 0.0, f"Premises contradict each other about: {contradictions}"
        elif query and answer is None:
            conf, summary = 0.3, f"'{query}' does not follow from the premises"
        elif query:
            conf, summary = 1.0, f"'{query}' is {'TRUE' if answer else 'FALSE'} (valid if the premises are true)"
        else:
            conf, summary = 0.9, f"Deduced {len(conclusions)} conclusion(s)"
        out.update(confidence=conf, summary=summary)
        return out

    def process(self, signal: Signal) -> Result:
        p = signal.payload
        if not isinstance(p, dict) or not p.get("premises"):
            return Result(ok=False, error="deduction needs 'premises' (list of sentences)")
        return Result(ok=True, data=self.deduce(p["premises"], p.get("query")))
