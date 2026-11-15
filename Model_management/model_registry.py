"""Thread-safe registry of model specs, lifecycle status and runtime metrics."""
from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional

from .model_core import (
    ModelAlreadyRegisteredError,
    ModelNotFoundError,
    ModelSpec,
    ModelStatus,
    ModelTask,
    ProviderType,
)


@dataclass
class ModelMetrics:
    """Rolling counters for one model. Latency is tracked for successes only."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    total_tokens: int = 0

    @property
    def avg_latency_ms(self) -> Optional[float]:
        return self.total_latency_ms / self.successes if self.successes else None

    @property
    def success_rate(self) -> float:
        """Optimistic estimate: unknown models score 1.0."""
        return (self.successes + 1) / (self.calls + 1)

    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0


@dataclass
class RegistryEntry:
    spec: ModelSpec
    status: ModelStatus = ModelStatus.REGISTERED
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    registered_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    last_error: Optional[str] = None


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, RegistryEntry] = {}

    def register(self, spec: ModelSpec, *, replace: bool = False) -> None:
        with self._lock:
            if spec.model_id in self._entries and not replace:
                raise ModelAlreadyRegisteredError(spec.model_id)
            old = self._entries.get(spec.model_id)
            entry = RegistryEntry(spec=spec)
            if old is not None:
                entry.metrics = old.metrics
            self._entries[spec.model_id] = entry

    def unregister(self, model_id: str) -> None:
        with self._lock:
            if self._entries.pop(model_id, None) is None:
                raise ModelNotFoundError(model_id)

    def get(self, model_id: str) -> ModelSpec:
        return self._entry(model_id).spec

    def get_entry(self, model_id: str) -> RegistryEntry:
        """Return a snapshot copy of the entry."""
        with self._lock:
            e = self._entry(model_id)
            return dataclasses.replace(e, metrics=dataclasses.replace(e.metrics))

    def exists(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._entries

    __contains__ = exists

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def list_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._entries)

    def find(
        self,
        *,
        task: Optional[ModelTask] = None,
        provider: Optional[ProviderType] = None,
        capabilities: Iterable[str] = (),
        tags: Iterable[str] = (),
        include_deprecated: bool = False,
    ) -> List[ModelSpec]:
        caps: FrozenSet[str] = frozenset(capabilities)
        want_tags: FrozenSet[str] = frozenset(tags)
        with self._lock:
            out: List[ModelSpec] = []
            for e in self._entries.values():
                s = e.spec
                if not include_deprecated and e.status is ModelStatus.DEPRECATED:
                    continue
                if task is not None and s.task is not task:
                    continue
                if provider is not None and s.provider is not provider:
                    continue
                if not caps <= s.capabilities or not want_tags <= s.tags:
                    continue
                out.append(s)
            return sorted(out, key=lambda s: s.model_id)

    def set_status(self, model_id: str, status: ModelStatus, error: Optional[str] = None) -> None:
        with self._lock:
            e = self._entries.get(model_id)
            if e is None:
                return  # unregistered while loading; ignore
            e.status = status
            e.last_error = error if status is ModelStatus.FAILED else None

    def record_result(
        self, model_id: str, *, success: bool, latency_ms: float = 0.0, cost: float = 0.0, tokens: int = 0,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            e = self._entries.get(model_id)
            if e is None:
                return
            m = e.metrics
            m.calls += 1
            e.last_used = time.time()
            if success:
                m.successes += 1
                m.total_latency_ms += latency_ms
                m.total_cost += cost
                m.total_tokens += tokens
            else:
                m.failures += 1
                e.last_error = error

    def metrics(self, model_id: str) -> ModelMetrics:
        with self._lock:
            return dataclasses.replace(self._entry(model_id).metrics)

    def all_metrics(self, model_ids: Iterable[str]) -> Dict[str, ModelMetrics]:
        with self._lock:
            return {i: dataclasses.replace(self._entries[i].metrics) for i in model_ids if i in self._entries}

    def _entry(self, model_id: str) -> RegistryEntry:
        with self._lock:
            try:
                return self._entries[model_id]
            except KeyError:
                raise ModelNotFoundError(model_id) from None
