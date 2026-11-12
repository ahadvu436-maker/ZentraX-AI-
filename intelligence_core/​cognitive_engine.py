"""cognitive_engine.py - turns raw data into higher-level insight and candidate options."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Dict

from intelligence_interface import IntelligenceModule, Result, Signal

Strategy = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]

_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "on", "for", "it", "this", "that", "i", "you", "me", "my"}


class CognitiveEngine(IntelligenceModule):
    name = "cognitive_engine"

    def __init__(self) -> None:
        self._strategies: Dict[str, Strategy] = {}
        self._processed = 0

    def register_strategy(self, route: str, strategy: Strategy) -> None:
        """Plug in a smarter strategy (e.g. an LLM or vector search) for a route."""
        self._strategies[route] = strategy

    # ---- feature extraction -------------------------------------------
    @staticmethod
    def extract_features(request: Any) -> Dict[str, Any]:
        text = request if isinstance(request, str) else str(request)
        tokens = re.findall(r"\w+", text.lower())
        keywords = [w for w, _ in Counter(t for t in tokens if t not in _STOP).most_common(5)]
        return {
            "text": text,
            "length": len(tokens),
            "keywords": keywords,
            "is_question": "?" in text,
        }

    # ---- default strategy ---------------------------------------------
    @staticmethod
    def _default_strategy(features: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        route = context.get("_route", "general")
        confidence = 0.4 + 0.1 * min(len(features["keywords"]), 4)
        if context.get("history_size"):
            confidence += 0.1
        confidence = round(min(confidence, 0.95), 2)
        act_risk = 0.4 if route == "command" else 0.1
        return {
            "summary": f"{route} request with keywords {features['keywords']}",
            "confidence": confidence,
            "options": [
                {"action": "execute" if route == "command" else "respond", "score": confidence, "risk": act_risk},
                {"action": "clarify", "score": round(1 - confidence, 2), "risk": 0.05},
                {"action": "defer", "score": 0.2, "risk": 0.0},
            ],
        }

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        payload = signal.payload
        if not isinstance(payload, dict) or "request" not in payload:
            return Result(ok=False, error="payload must be a dict containing 'request'")
        route = (payload.get("route") or {}).get("route", "general")
        context = dict(payload.get("context") or {})
        context["_route"] = route
        features = self.extract_features(payload["request"])
        strategy = self._strategies.get(route, self._default_strategy)
        try:
            insight = strategy(features, context)
        except Exception as exc:
            return Result(ok=False, error=f"strategy failed: {exc}")
        insight["features"] = features
        self._processed += 1
        return Result(ok=True, data=insight)

    def get_state(self) -> Dict[str, Any]:
        return {"processed": self._processed, "strategies": list(self._strategies)}
