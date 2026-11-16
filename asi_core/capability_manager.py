"""Capability registry with permission-checked invocation and usage stats."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapabilityDescriptor:
    name: str
    description: str = ""
    domains: frozenset[str] = frozenset()
    required_permission: str = "internal"


class Capability(ABC):
    descriptor: CapabilityDescriptor

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any: ...


class CapabilityManager:
    def __init__(self) -> None:
        self._caps: dict[str, Capability] = {}
        self._stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "errors": 0, "total_s": 0.0}
        )

    def register(self, cap: Capability) -> None:
        n = cap.descriptor.name
        if n in self._caps:
            raise ValueError(f"capability already registered: {n}")
        self._caps[n] = cap

    def unregister(self, name: str) -> None:
        self._caps.pop(name, None)

    def list(self) -> list[CapabilityDescriptor]:
        return [c.descriptor for c in self._caps.values()]

    def find_by_domain(self, domain: str) -> list[Capability]:
        return [c for c in self._caps.values()
                if not c.descriptor.domains or domain in c.descriptor.domains]

    async def invoke(self, name: str, permissions: frozenset[str] | set[str], **kwargs: Any) -> Any:
        cap = self._caps.get(name)
        if cap is None:
            raise KeyError(f"unknown capability: {name}")
        need = cap.descriptor.required_permission
        if need not in permissions and "*" not in permissions:
            raise PermissionError(f"'{need}' permission required for '{name}'")
        s, t0 = self._stats[name], time.perf_counter()
        s["calls"] += 1
        try:
            return await cap.execute(**kwargs)
        except Exception:
            s["errors"] += 1
            raise
        finally:
            s["total_s"] += time.perf_counter() - t0

    def stats(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self._stats.items()}
