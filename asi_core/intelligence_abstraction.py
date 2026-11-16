"""Domain-independent concept abstraction."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class AbstractionLevel(IntEnum):
    CONCRETE = 0
    PATTERN = 1
    PRINCIPLE = 2


@dataclass(frozen=True)
class Concept:
    domain: str
    features: frozenset[str]
    level: AbstractionLevel = AbstractionLevel.PATTERN

    def similarity(self, other: "Concept") -> float:
        """Jaccard similarity over abstract features."""
        if not self.features and not other.features:
            return 0.0
        return len(self.features & other.features) / len(self.features | other.features)


class AbstractionLayer(ABC):
    @abstractmethod
    def abstract(self, domain: str, payload: Any) -> Concept: ...

    @abstractmethod
    def ground(self, concept: Concept, target_domain: str) -> dict[str, Any]: ...


class KeywordAbstractionLayer(AbstractionLayer):
    """Baseline layer; swap for embedding / neural abstraction in production."""
    _STOP = frozenset({"the", "a", "an", "of", "to", "and", "in", "is", "for", "on", "with"})

    def abstract(self, domain: str, payload: Any) -> Concept:
        toks = re.findall(r"[a-z0-9_]+", str(payload).lower())
        feats = frozenset(t for t in toks if t not in self._STOP and len(t) > 2)
        return Concept(domain=domain, features=feats)

    def ground(self, concept: Concept, target_domain: str) -> dict[str, Any]:
        return {"target_domain": target_domain, "hints": sorted(concept.features),
                "source_domain": concept.domain}
