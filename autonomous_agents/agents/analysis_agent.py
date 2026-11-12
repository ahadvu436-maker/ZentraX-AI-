"""AnalysisAgent: statistics, trends, correlations, outliers and text analysis.

Payload actions
---------------
``describe``  ``values``                       summary statistics
``outliers``  ``values``, ``k`` (IQR factor)   IQR outlier detection
``trend``     ``values``, ``x``, ``forecast``  linear regression and forecast
``correlate`` ``x``, ``y``                     Pearson correlation
``compare``   ``a``, ``b``                     two-sample comparison (Welch t, Cohen's d)
``text``      ``text``                         lexical statistics
``insights``  ``datasets`` {name: values}      combined findings (+ optional LLM narrative)
``auto``      (default) picks an action from the payload keys
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from ..agent_core import AgentConfig, BaseAgent, Reasoner, Task, call_reasoner

__all__ = ["AnalysisAgent"]

_STOP = frozenset(
    "a an and are as at be but by for from has have he her his i in is it its of on or our she that the their them "
    "they this to was we were what when where which who will with you your not".split()
)


def _floats(values: Any, name: str = "values") -> List[float]:
    if values is None or isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"'{name}' must be a list of numbers")
    out: List[float] = []
    for v in values:
        if isinstance(v, bool):
            raise ValueError(f"'{name}' contains a boolean")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' contains non-numeric value {v!r}") from None
        if math.isnan(f) or math.isinf(f):
            raise ValueError(f"'{name}' contains NaN or infinity")
        out.append(f)
    if not out:
        raise ValueError(f"'{name}' is empty")
    return out


def _r(x: Optional[float], digits: int = 6) -> Optional[float]:
    return None if x is None else round(x, digits)


def _quantile(sorted_vals: List[float], q: float) -> float:
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _variance(v: List[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


def describe(values: Any) -> Dict[str, Any]:
    v = _floats(values)
    s = sorted(v)
    var = _variance(v)
    q1, q3 = _quantile(s, 0.25), _quantile(s, 0.75)
    return {
        "count": len(v),
        "sum": _r(sum(v)),
        "mean": _r(sum(v) / len(v)),
        "median": _r(_quantile(s, 0.5)),
        "stdev": _r(math.sqrt(var)),
        "variance": _r(var),
        "min": s[0],
        "max": s[-1],
        "range": _r(s[-1] - s[0]),
        "q1": _r(q1),
        "q3": _r(q3),
        "iqr": _r(q3 - q1),
    }


def outliers(values: Any, k: float = 1.5) -> Dict[str, Any]:
    v = _floats(values)
    s = sorted(v)
    q1, q3 = _quantile(s, 0.25), _quantile(s, 0.75)
    low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    found = [{"index": i, "value": x} for i, x in enumerate(v) if x < low or x > high]
    return {"method": "iqr", "k": k, "lower_bound": _r(low), "upper_bound": _r(high), "outliers": found}


def trend(values: Any, x: Any = None, forecast: int = 0) -> Dict[str, Any]:
    y = _floats(values)
    if len(y) < 2:
        raise ValueError("trend needs at least 2 values")
    xs = _floats(x, "x") if x is not None else [float(i) for i in range(len(y))]
    if len(xs) != len(y):
        raise ValueError("'x' and 'values' must have the same length")
    mx, my = sum(xs) / len(xs), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in xs)
    if sxx == 0:
        raise ValueError("'x' has no variance")
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, y)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((b - my) ** 2 for b in y)
    ss_res = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(xs, y))
    r2 = 1.0 if ss_tot == 0 else max(0.0, 1 - ss_res / ss_tot)
    step = (xs[-1] - xs[0]) / (len(xs) - 1)
    direction = "flat" if abs(slope) < 1e-12 else ("increasing" if slope > 0 else "decreasing")
    return {
        "slope": _r(slope),
        "intercept": _r(intercept),
        "r_squared": _r(r2),
        "direction": direction,
        "forecast": [_r(intercept + slope * (xs[-1] + step * i)) for i in range(1, max(0, int(forecast)) + 1)],
    }


def correlate(x: Any, y: Any) -> Dict[str, Any]:
    a, b = _floats(x, "x"), _floats(y, "y")
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("'x' and 'y' must have equal length >= 2")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    saa = sum((p - ma) ** 2 for p in a)
    sbb = sum((q - mb) ** 2 for q in b)
    if saa == 0 or sbb == 0:
        return {"pearson_r": None, "r_squared": None, "strength": "undefined", "note": "a series is constant"}
    r = max(-1.0, min(1.0, sum((p - ma) * (q - mb) for p, q in zip(a, b)) / math.sqrt(saa * sbb)))
    mag = abs(r)
    strength = (
        "negligible" if mag < 0.1 else "weak" if mag < 0.3 else "moderate" if mag < 0.5 else "strong" if mag < 0.7 else "very strong"
    )
    return {"pearson_r": _r(r), "r_squared": _r(r * r), "strength": strength, "n": len(a)}


def compare(a: Any, b: Any) -> Dict[str, Any]:
    xa, xb = _floats(a, "a"), _floats(b, "b")
    ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
    va, vb = _variance(xa), _variance(xb)
    out: Dict[str, Any] = {
        "a": describe(xa),
        "b": describe(xb),
        "mean_difference": _r(ma - mb),
        "percent_change_a_to_b": _r((mb - ma) / ma * 100) if ma != 0 else None,
        "welch_t": None,
        "degrees_of_freedom": None,
        "cohens_d": None,
    }
    if len(xa) > 1 and len(xb) > 1:
        se2 = va / len(xa) + vb / len(xb)
        if se2 > 0:
            out["welch_t"] = _r((ma - mb) / math.sqrt(se2))
            denom = (va / len(xa)) ** 2 / (len(xa) - 1) + (vb / len(xb)) ** 2 / (len(xb) - 1)
            out["degrees_of_freedom"] = _r(se2**2 / denom) if denom else None
        pooled = ((len(xa) - 1) * va + (len(xb) - 1) * vb) / (len(xa) + len(xb) - 2)
        if pooled > 0:
            out["cohens_d"] = _r((ma - mb) / math.sqrt(pooled))
    return out


def analyze_text(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("'text' is empty")
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    terms = Counter(w for w in words if w not in _STOP and len(w) > 2)
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": _r(len(words) / len(sentences), 2) if sentences else 0.0,
        "unique_words": len(set(words)),
        "lexical_diversity": _r(len(set(words)) / len(words), 4) if words else 0.0,
        "reading_minutes": _r(len(words) / 200, 2),
        "top_terms": [{"term": t, "count": c} for t, c in terms.most_common(10)],
    }


class AnalysisAgent(BaseAgent):
    agent_type = "analysis"
    DEFAULT_CAPABILITIES = frozenset({"analysis", "statistics", "data_analysis", "text_analysis"})

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        agent_id: Optional[str] = None,
        llm: Optional[Reasoner] = None,
    ) -> None:
        config = config or AgentConfig(
            name="analysis-agent", capabilities=self.DEFAULT_CAPABILITIES, max_concurrent_tasks=4
        )
        super().__init__(config, agent_id)
        self._llm = llm

    async def handle_task(self, task: Task) -> Dict[str, Any]:
        p = task.payload
        action = str(p.get("action", "auto")).lower()
        if action == "auto":
            action = self._infer_action(task)
        if action == "describe":
            return await asyncio.to_thread(describe, p.get("values"))
        if action == "outliers":
            return await asyncio.to_thread(outliers, p.get("values"), float(p.get("k", 1.5)))
        if action == "trend":
            return await asyncio.to_thread(trend, p.get("values"), p.get("x"), int(p.get("forecast", 0)))
        if action == "correlate":
            return await asyncio.to_thread(correlate, p.get("x"), p.get("y"))
        if action == "compare":
            return await asyncio.to_thread(compare, p.get("a"), p.get("b"))
        if action == "text":
            return await asyncio.to_thread(analyze_text, self._text_of(task))
        if action == "insights":
            return await self._insights(p.get("datasets"))
        raise ValueError(f"unknown analysis action: {action!r}")

    @staticmethod
    def _text_of(task: Task) -> str:
        p = task.payload
        for key in ("text", "instructions"):
            if isinstance(p.get(key), str) and p[key].strip():
                return p[key]
        upstreams = list((task.context.get("dependency_results") or {}).values())
        upstreams.append(task.context.get("previous_output"))
        upstreams.append(task.context.get("artifact"))
        for upstream in upstreams:
            if isinstance(upstream, str) and upstream.strip():
                return upstream
            if isinstance(upstream, dict) and isinstance(upstream.get("summary"), str) and upstream["summary"].strip():
                return upstream["summary"]
        raise ValueError("no analyzable text found")

    def _infer_action(self, task: Task) -> str:
        p = task.payload
        if "datasets" in p:
            return "insights"
        if "x" in p and "y" in p:
            return "correlate"
        if "a" in p and "b" in p:
            return "compare"
        if "values" in p:
            return "describe"
        return "text"

    async def _insights(self, datasets: Any) -> Dict[str, Any]:
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError("'datasets' must be a non-empty mapping of name -> values")

        def work() -> Dict[str, Any]:
            report: Dict[str, Any] = {}
            findings: List[str] = []
            for name, values in datasets.items():
                entry: Dict[str, Any] = {"summary": describe(values)}
                n = entry["summary"]["count"]
                finding = f"{name}: n={n}, mean={entry['summary']['mean']}, stdev={entry['summary']['stdev']}"
                if n >= 4:
                    entry["outliers"] = outliers(values)
                    if entry["outliers"]["outliers"]:
                        finding += f", {len(entry['outliers']['outliers'])} outlier(s)"
                if n >= 3:
                    entry["trend"] = trend(values)
                    finding += f", {entry['trend']['direction']} trend (R²={entry['trend']['r_squared']})"
                report[name] = entry
                findings.append(finding)
            return {"datasets": report, "findings": findings}

        result = await asyncio.to_thread(work)
        if self._llm is not None:
            prompt = "Write a short, factual analyst narrative for these findings:\n" + "\n".join(result["findings"])
            try:
                result["narrative"] = (await call_reasoner(self._llm, prompt)).strip()
            except Exception:  # noqa: BLE001
                self.log.exception("narrative generation failed")
        return result
