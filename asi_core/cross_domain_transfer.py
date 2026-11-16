"""Cross-domain knowledge store and transfer strategies."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .intelligence_abstraction import AbstractionLayer, Concept, KeywordAbstractionLayer


@dataclass
class KnowledgeUnit:
    domain: str
    payload: Any
    concept: Concept
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class TransferResult:
    source_id: str
    source_domain: str
    score: float
    grounded: dict[str, Any]


class TransferStrategy(ABC):
    @abstractmethod
    def score(self, unit: KnowledgeUnit, query: Concept, target_domain: str) -> float: ...


class AnalogicalTransfer(TransferStrategy):
    """Scores by abstract-feature overlap; only foreign-domain knowledge counts."""

    def score(self, unit: KnowledgeUnit, query: Concept, target_domain: str) -> float:
        if unit.domain == target_domain:
            return 0.0
        return unit.concept.similarity(query)


class CrossDomainTransferEngine:
    def __init__(self, layer: AbstractionLayer | None = None,
                 strategy: TransferStrategy | None = None) -> None:
        self.layer = layer or KeywordAbstractionLayer()
        self.strategy = strategy or AnalogicalTransfer()
        self._store: dict[str, KnowledgeUnit] = {}

    def add_knowledge(self, domain: str, payload: Any) -> KnowledgeUnit:
        u = KnowledgeUnit(domain, payload, self.layer.abstract(domain, payload))
        self._store[u.id] = u
        return u

    def transfer(self, target_domain: str, query: Any, top_k: int = 3,
                 min_score: float = 0.05) -> list[TransferResult]:
        q = self.layer.abstract(target_domain, query)
        results = []
        for u in self._store.values():
            s = self.strategy.score(u, q, target_domain)
            if s >= min_score:
                g = self.layer.ground(u.concept, target_domain)
                g["payload"] = u.payload
                results.append(TransferResult(u.id, u.domain, s, g))
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    def __len__(self) -> int:
        return len(self._store)
