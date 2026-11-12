"""intelligence_router.py - routes requests to the right subsystem by input nature."""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

from intelligence_interface import IntelligenceModule, Result, Signal

Predicate = Callable[[Any], bool]


def _text(request: Any) -> str:
    return request.lower() if isinstance(request, str) else str(request).lower()


def keyword_rule(*words: str) -> Predicate:
    return lambda req: any(w in _text(req).split() or w in _text(req) for w in words)


class IntelligenceRouter(IntelligenceModule):
    name = "intelligence_router"
    DEFAULT_ROUTE = "general"

    def __init__(self) -> None:
        self._rules: List[Tuple[str, Predicate]] = []
        self._counts: Counter = Counter()
        # Built-in rules, checked in order. Add your own with add_rule().
        self.add_rule("explicit", lambda r: isinstance(r, dict) and "route" in r)
        self.add_rule("command", keyword_rule("run", "start", "stop", "create", "delete", "open"))
        self.add_rule("query", lambda r: "?" in _text(r) or keyword_rule("what", "how", "why", "find", "search")(r))

    def add_rule(self, route: str, predicate: Predicate) -> None:
        self._rules.append((route, predicate))

    def classify(self, request: Any) -> str:
        for route, predicate in self._rules:
            try:
                if predicate(request):
                    return request["route"] if route == "explicit" else route
            except Exception:
                continue  # a broken rule must not break routing
        return self.DEFAULT_ROUTE

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        payload = signal.payload
        request = payload.get("request") if isinstance(payload, dict) and "request" in payload else payload
        route = self.classify(request)
        self._counts[route] += 1
        return Result(ok=True, data={"route": route})

    def get_state(self) -> Dict[str, Any]:
        return {"rules": len(self._rules), "route_counts": dict(self._counts)}
