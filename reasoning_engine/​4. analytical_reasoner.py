"""analytical_reasoner.py - breaks big data / complex problems into pieces and analyzes them."""
from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional

from intelligence_interface import IntelligenceModule, Result, Signal


class AnalyticalReasoner(IntelligenceModule):
    name = "analytical_reasoner"

    @staticmethod
    def decompose(problem: str) -> List[str]:
        """Split a complex problem statement into smaller sub-problems."""
        parts = re.split(r"\s*(?:;|,|\.|\bthen\b|\balso\b|\band\b)\s*", problem, flags=re.I)
        return [p.strip() for p in parts if len(p.strip()) > 2]

    @staticmethod
    def _summ(vals: List[float]) -> Dict[str, float]:
        return {"count": len(vals), "mean": round(statistics.fmean(vals), 3),
                "min": min(vals), "max": max(vals)}

    def analyze_data(self, data: List[Any], field: Optional[str] = None,
                     chunk_size: Optional[int] = None, z_limit: float = 2.0) -> Dict[str, Any]:
        vals: List[float] = []
        for x in data:
            v = x.get(field) if (field and isinstance(x, dict)) else x
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
        if not vals:
            raise ValueError("no numeric values to analyze")
        out: Dict[str, Any] = self._summ(vals)
        out["median"] = statistics.median(vals)
        sd = statistics.pstdev(vals)
        out["stdev"] = round(sd, 3)
        mean = out["mean"]
        out["outliers"] = ([{"index": i, "value": v} for i, v in enumerate(vals)
                            if abs(v - mean) / sd > z_limit] if sd > 0 and len(vals) >= 3 else [])
        out["trend"] = "n/a"
        if len(vals) >= 4:
            half = len(vals) // 2
            a, b = statistics.fmean(vals[:half]), statistics.fmean(vals[half:])
            change = (b - a) / abs(a) if a else 0.0
            out["trend"] = "rising" if change > 0.05 else "falling" if change < -0.05 else "flat"
        if chunk_size and chunk_size > 0:
            out["chunks"] = [{"range": [i, min(i + chunk_size, len(vals))], **self._summ(vals[i:i + chunk_size])}
                             for i in range(0, len(vals), chunk_size)]
        return out

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        p = signal.payload
        if not isinstance(p, dict) or not (p.get("data") or p.get("problem")):
            return Result(ok=False, error="analysis needs 'data' and/or 'problem'")
        out: Dict[str, Any] = {}
        notes: List[str] = []
        confidence = 0.0
        if p.get("problem"):
            out["sub_problems"] = self.decompose(p["problem"])
            notes.append(f"split into {len(out['sub_problems'])} sub-problem(s)")
            confidence = 0.6
        if p.get("data"):
            try:
                out["stats"] = self.analyze_data(p["data"], p.get("field"), p.get("chunk_size"))
            except ValueError as exc:
                return Result(ok=False, error=str(exc))
            s = out["stats"]
            notes.append(f"{s['count']} values, mean {s['mean']}, trend {s['trend']}, {len(s['outliers'])} outlier(s)")
            confidence = 0.9
        out.update(confidence=confidence, summary="; ".join(notes))
        return Result(ok=True, data=out)
