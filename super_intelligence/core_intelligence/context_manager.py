"""context_manager.py - live tracking of work context, history and environment."""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict

from intelligence_interface import IntelligenceModule, Result, Signal


class ContextManager(IntelligenceModule):
    name = "context_manager"

    def __init__(self, max_history: int = 100) -> None:
        self._history: deque = deque(maxlen=max_history)
        self._environment: Dict[str, Any] = {}
        self._current_task: Any = None

    # ---- environment / task ------------------------------------------
    def update_environment(self, **values: Any) -> None:
        """E.g. update_environment(network="online", device="mobile")."""
        self._environment.update(values)

    def set_task(self, task: Any) -> None:
        self._current_task = task

    # ---- snapshot ------------------------------------------------------
    def snapshot(self, recent: int = 5) -> Dict[str, Any]:
        return {
            "task": self._current_task,
            "environment": dict(self._environment),
            "recent_history": list(self._history)[-recent:],
            "history_size": len(self._history),
            "timestamp": time.time(),
        }

    def _record(self, kind: str, data: Any) -> None:
        self._history.append({"kind": kind, "data": data, "t": time.time()})

    # ---- IntelligenceModule -------------------------------------------
    def process(self, signal: Signal) -> Result:
        if signal.kind == "request":
            snap = self.snapshot()  # context *before* this request
            self._record("request", signal.payload)
            return Result(ok=True, data=snap)
        if signal.kind == "outcome":
            self._record("outcome", signal.payload)
            return Result(ok=True, data=self.snapshot())
        return Result(ok=False, error=f"unsupported signal kind '{signal.kind}'")

    def get_state(self) -> Dict[str, Any]:
        return {"history_size": len(self._history), "environment": dict(self._environment)}
