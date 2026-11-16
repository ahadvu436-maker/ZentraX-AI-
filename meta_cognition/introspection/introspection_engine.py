"""Introspection engine: snapshot, query, diff and explain registered internal state."""
from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional

StateProvider = Callable[[], Mapping[str, Any]]
_SENSITIVE = ("password", "secret", "token", "api_key", "apikey", "credential")


def to_plain(obj: Any, depth: int = 6) -> Any:
    """Convert arbitrary state into JSON-friendly primitives, redacting sensitive keys."""
    if depth <= 0:
        return "<max-depth>"
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): ("<redacted>" if any(s in str(k).lower() for s in _SENSITIVE)
                         else to_plain(v, depth - 1)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset, deque)):
        return [to_plain(v, depth - 1) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


@dataclass(frozen=True)
class Snapshot:
    state: Dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Change:
    path: str
    before: Any
    after: Any


class IntrospectionEngine:
    def __init__(self, history: int = 100) -> None:
        self._providers: Dict[str, StateProvider] = {}
        self._history: Deque[Snapshot] = deque(maxlen=history)
        self._lock = threading.RLock()

    def register(self, name: str, provider: StateProvider) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        with self._lock:
            self._providers[name] = provider

    def unregister(self, name: str) -> None:
        with self._lock:
            self._providers.pop(name, None)

    def snapshot(self) -> Snapshot:
        with self._lock:
            providers = dict(self._providers)
        state: Dict[str, Any] = {}
        for name, prov in providers.items():
            try:
                state[name] = to_plain(prov())
            except Exception as e:  # introspection must never take the agent down
                state[name] = {"_introspection_error": repr(e)}
        snap = Snapshot(state)
        with self._lock:
            self._history.append(snap)
        return snap

    def history(self) -> List[Snapshot]:
        with self._lock:
            return list(self._history)

    @staticmethod
    def get(snapshot: Snapshot, path: str, default: Any = None) -> Any:
        cur: Any = snapshot.state
        for part in path.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
                cur = cur[int(part)]
            else:
                return default
        return cur

    def diff(self, a: Snapshot, b: Snapshot) -> List[Change]:
        out: List[Change] = []

        def walk(x: Any, y: Any, path: str) -> None:
            if isinstance(x, Mapping) and isinstance(y, Mapping):
                for k in sorted(set(x) | set(y)):
                    walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
            elif x != y:
                out.append(Change(path, x, y))

        walk(a.state, b.state, "")
        return out

    def explain(self, snapshot: Optional[Snapshot] = None) -> str:
        snap = snapshot or self.snapshot()
        lines = [f"Introspection snapshot {snap.id}:"]
        for comp, st in snap.state.items():
            if isinstance(st, Mapping):
                brief = ", ".join(f"{k}={v}" for k, v in list(st.items())[:4])
            else:
                brief = str(st)
            lines.append(f"- {comp}: {brief[:160]}")
        return "\n".join(lines)
