"""Validation of entities, relationships and whole-graph integrity."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .knowledge_core import Entity, Relationship, normalize_name
from .knowledge_graph import KnowledgeGraph
from .ontology_manager import OntologyManager

__all__ = ["Severity", "ValidationIssue", "ValidationReport", "KnowledgeValidator"]


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    subject_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"severity": self.severity.value, "code": self.code,
                "message": self.message, "subject_id": self.subject_id}


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    def extend(self, issues: List[ValidationIssue]) -> None:
        self.issues.extend(issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = defaultdict(int)
        for issue in self.issues:
            counts[issue.code] += 1
        return {"ok": self.ok, "errors": len(self.errors), "warnings": len(self.warnings),
                "total": len(self.issues), "by_code": dict(counts)}

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "issues": [i.to_dict() for i in self.issues]}


class KnowledgeValidator:
    """Checks data against the ontology and basic quality rules."""

    def __init__(
        self,
        ontology: OntologyManager,
        graph: Optional[KnowledgeGraph] = None,
        *,
        min_entity_confidence: float = 0.2,
        min_relationship_confidence: float = 0.2,
        max_name_length: int = 200,
        max_bucket_size: int = 500,
    ) -> None:
        self.ontology = ontology
        self.graph = graph
        self.min_entity_confidence = min_entity_confidence
        self.min_relationship_confidence = min_relationship_confidence
        self.max_name_length = max_name_length
        self.max_bucket_size = max_bucket_size

    # ------------------------------------------------------------------ #
    def validate_entity(self, entity: Entity) -> List[ValidationIssue]:
        E, W = Severity.ERROR, Severity.WARNING
        issues: List[ValidationIssue] = []
        name = entity.name.strip()
        if not name or not normalize_name(name):
            issues.append(ValidationIssue(E, "empty_name", "Entity name is empty", entity.id))
        elif len(name) > self.max_name_length:
            issues.append(ValidationIssue(E, "name_too_long", f"Name exceeds {self.max_name_length} chars", entity.id))
        if self.ontology.resolve_entity_type(entity.type) != entity.type:
            issues.append(ValidationIssue(E, "unknown_type", f"Unknown entity type {entity.type!r}", entity.id))
        if not 0.0 <= entity.confidence <= 1.0:
            issues.append(ValidationIssue(E, "bad_confidence", "Confidence must be within [0, 1]", entity.id))
        elif entity.confidence < self.min_entity_confidence:
            issues.append(ValidationIssue(E, "low_confidence", f"Confidence {entity.confidence:.2f} below threshold", entity.id))
        if entity.type == self.ontology.ROOT:
            issues.append(ValidationIssue(W, "untyped", "Entity has no specific type", entity.id))
        return issues

    def validate_relationship(self, rel: Relationship, graph: Optional[KnowledgeGraph] = None) -> List[ValidationIssue]:
        E, W = Severity.ERROR, Severity.WARNING
        graph = graph or self.graph
        issues: List[ValidationIssue] = []
        if rel.source_id == rel.target_id:
            issues.append(ValidationIssue(E, "self_loop", "Relationship connects an entity to itself", rel.id))
        if not 0.0 <= rel.confidence <= 1.0:
            issues.append(ValidationIssue(E, "bad_confidence", "Confidence must be within [0, 1]", rel.id))
        elif rel.confidence < self.min_relationship_confidence:
            issues.append(ValidationIssue(E, "low_confidence", f"Confidence {rel.confidence:.2f} below threshold", rel.id))
        if self.ontology.resolve_relation(rel.type) is None:
            issues.append(ValidationIssue(E, "unknown_relation", f"Unknown relation {rel.type!r}", rel.id))
            return issues
        if graph is None:
            return issues
        src, dst = graph.get_entity(rel.source_id), graph.get_entity(rel.target_id)
        for label, ent, eid in (("source", src, rel.source_id), ("target", dst, rel.target_id)):
            if ent is None:
                issues.append(ValidationIssue(E, "dangling", f"Relationship {label} {eid} does not exist", rel.id))
        if src is not None and dst is not None:
            ok, reason = self.ontology.validate_relationship(rel.type, src.type, dst.type)
            if not ok:
                issues.append(ValidationIssue(E, "ontology_violation", reason, rel.id))
            elif src.type == self.ontology.ROOT or dst.type == self.ontology.ROOT:
                issues.append(ValidationIssue(W, "untyped_endpoint", "An endpoint is untyped", rel.id))
        return issues

    # ------------------------------------------------------------------ #
    def validate_graph(self, graph: Optional[KnowledgeGraph] = None) -> ValidationReport:
        graph = graph or self.graph
        if graph is None:
            raise ValueError("No graph to validate")
        report = ValidationReport()
        for entity in graph.iter_entities():
            report.extend(self.validate_entity(entity))
            if graph.degree(entity.id) == 0:
                report.extend([ValidationIssue(Severity.INFO, "orphan", "Entity has no relationships", entity.id)])
        for rel in graph.iter_relationships():
            report.extend(self.validate_relationship(rel, graph))
        report.extend(self._type_conflicts(graph))
        for a, b, score in self.find_duplicate_candidates(graph):
            report.extend([ValidationIssue(
                Severity.INFO, "duplicate_candidate",
                f"Entities {a} and {b} look alike (similarity {score:.2f})", a)])
        return report

    def _type_conflicts(self, graph: KnowledgeGraph) -> List[ValidationIssue]:
        by_name: Dict[str, List[Entity]] = defaultdict(list)
        for e in graph.iter_entities():
            by_name[normalize_name(e.name)].append(e)
        return [
            ValidationIssue(Severity.WARNING, "same_name_different_type",
                            f"'{group[0].name}' exists as types {sorted({e.type for e in group})}", group[0].id)
            for group in by_name.values() if len({e.type for e in group}) > 1
        ]

    def find_duplicate_candidates(self, graph: Optional[KnowledgeGraph] = None, *,
                                  threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """Return ``(keep_id, drop_id, similarity)`` for same-type look-alike names.

        ``keep`` is the entity with higher confidence (ties: more provenance).
        Candidates are bucketed by type and first letter to bound the cost.
        """
        graph = graph or self.graph
        if graph is None:
            raise ValueError("No graph to inspect")
        buckets: Dict[Tuple[str, str], List[Tuple[str, Entity]]] = defaultdict(list)
        for e in graph.iter_entities():
            norm = normalize_name(e.name)
            if norm:
                buckets[(e.type, norm[0])].append((norm, e))
        out: List[Tuple[str, str, float]] = []
        for items in buckets.values():
            items = items[: self.max_bucket_size]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    na, a = items[i]
                    nb, b = items[j]
                    if na == nb or abs(len(na) - len(nb)) > 0.3 * max(len(na), len(nb)):
                        continue
                    score = SequenceMatcher(None, na, nb).ratio()
                    if score >= threshold:
                        keep, drop = sorted((a, b), key=lambda e: (-e.confidence, -len(e.provenance), e.id))
                        out.append((keep.id, drop.id, score))
        return sorted(out, key=lambda t: -t[2])
