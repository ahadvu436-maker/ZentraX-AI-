"""intelligence_interface.py - contract/blueprint for 01_core_intelligence.

Two kinds of contracts live here:
  1. IntelligenceModule  - what every module inside this folder implements.
  2. ExternalSystem      - what any outside system (memory, reasoning engine, ...)
                           implements, so modules talk to it in one standard way.
Modules never import a concrete external system; they ask the ExternalGateway.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Shared data types
# --------------------------------------------------------------------------
@dataclass
class Signal:
    """A unit of information flowing between modules."""
    source: str
    payload: Any
    kind: str = "generic"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Result:
    """Standard return type for every call, internal or external."""
    ok: bool
    data: Any = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 1. Internal contract
# --------------------------------------------------------------------------
class IntelligenceModule(ABC):
    """Every module in this folder implements this."""

    name: str = "unnamed"

    @abstractmethod
    def process(self, signal: Signal) -> Result:
        """Handle a signal and return a Result."""

    def get_state(self) -> Dict[str, Any]:
        """Live state snapshot, collected by the core."""
        return {}

    def health_check(self) -> bool:
        """True if the module can work right now."""
        return True


# --------------------------------------------------------------------------
# 2. External-system contracts
# --------------------------------------------------------------------------
class ExternalSystem(ABC):
    """Base contract for anything outside this folder."""

    name: str = "external"
    capability: str = "generic"  # e.g. "memory", "reasoning"

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def health_check(self) -> bool:
        return True


class MemoryProvider(ExternalSystem):
    """Contract for a memory system (vector DB, SQLite, files, ...)."""
    capability = "memory"

    @abstractmethod
    def store(self, key: str, value: Any, meta: Optional[Dict[str, Any]] = None) -> Result: ...

    @abstractmethod
    def recall(self, query: str, limit: int = 5) -> Result: ...

    @abstractmethod
    def forget(self, key: str) -> Result: ...


class ReasoningProvider(ExternalSystem):
    """Contract for a reasoning engine (LLM, rule engine, solver, ...)."""
    capability = "reasoning"

    @abstractmethod
    def reason(self, question: str, context: Optional[Dict[str, Any]] = None) -> Result: ...


# --------------------------------------------------------------------------
# Gateway: one standard door to all external systems
# --------------------------------------------------------------------------
class ExternalGateway:
    def __init__(self) -> None:
        self._systems: Dict[str, ExternalSystem] = {}

    def register(self, system: ExternalSystem) -> bool:
        if not system.connect():
            return False
        self._systems[system.capability] = system
        return True

    def has(self, capability: str) -> bool:
        return capability in self._systems

    def get(self, capability: str) -> Optional[ExternalSystem]:
        return self._systems.get(capability)

    def call(self, capability: str, method: str, *args: Any, **kwargs: Any) -> Result:
        """Safe call: never raises, always returns a Result."""
        system = self._systems.get(capability)
        if system is None:
            return Result(ok=False, error=f"no '{capability}' system registered")
        if not system.health_check():
            return Result(ok=False, error=f"'{capability}' system unhealthy")
        try:
            out = getattr(system, method)(*args, **kwargs)
            return out if isinstance(out, Result) else Result(ok=True, data=out)
        except Exception as exc:
            return Result(ok=False, error=f"{capability}.{method} failed: {exc}")

    def health(self) -> Dict[str, bool]:
        return {cap: s.health_check() for cap, s in self._systems.items()}


# --------------------------------------------------------------------------
# Tiny reference implementation (useful for tests)
# --------------------------------------------------------------------------
class InMemoryMemory(MemoryProvider):
    name = "in_memory"

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def store(self, key: str, value: Any, meta: Optional[Dict[str, Any]] = None) -> Result:
        self._store[key] = value
        return Result(ok=True, data=key)

    def recall(self, query: str, limit: int = 5) -> Result:
        hits: List[Any] = [v for k, v in self._store.items() if query.lower() in k.lower()]
        return Result(ok=True, data=hits[:limit])

    def forget(self, key: str) -> Result:
        return Result(ok=self._store.pop(key, None) is not None)
