"""Ontology: entity class hierarchy and typed relation constraints."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple, Union

from .knowledge_core import KnowledgeError

__all__ = ["EntityClass", "RelationClass", "OntologyManager"]


@dataclass
class EntityClass:
    name: str
    parent: Optional[str] = None
    description: str = ""
    aliases: List[str] = field(default_factory=list)


@dataclass
class RelationClass:
    name: str
    domain: FrozenSet[str] = frozenset()   # allowed source classes (empty = any)
    range: FrozenSet[str] = frozenset()    # allowed target classes (empty = any)
    symmetric: bool = False
    inverse: Optional[str] = None
    description: str = ""
    aliases: List[str] = field(default_factory=list)


def _rel_key(raw: str) -> str:
    return re.sub(r"[\s\-]+", "_", raw.strip().casefold())


class OntologyManager:
    """Registry of entity classes and relations with subclass-aware validation.

    ``Thing`` is the root class. An entity typed exactly ``Thing`` is treated as
    *untyped* and is compatible with any relation endpoint (it can later be
    refined from relation context).
    """

    ROOT = "Thing"

    def __init__(self, load_defaults: bool = True) -> None:
        self._lock = threading.RLock()
        self._classes: Dict[str, EntityClass] = {}
        self._relations: Dict[str, RelationClass] = {}
        self._class_lookup: Dict[str, str] = {}
        self._rel_lookup: Dict[str, str] = {}
        self.register_entity_class(EntityClass(self.ROOT, None, "Root class (untyped)"))
        if load_defaults:
            self._load_defaults()

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register_entity_class(self, cls: EntityClass) -> None:
        with self._lock:
            if cls.name != self.ROOT:
                if cls.parent is None:
                    cls.parent = self.ROOT
                if cls.parent not in self._classes:
                    raise KnowledgeError(f"Unknown parent class {cls.parent!r} for {cls.name!r}")
                if cls.name in self._classes and self.is_subclass(cls.parent, cls.name):
                    raise KnowledgeError(f"Cycle: {cls.parent!r} is a descendant of {cls.name!r}")
            self._classes[cls.name] = cls
            for key in [cls.name, *cls.aliases]:
                self._class_lookup[key.strip().casefold()] = cls.name

    def register_relation(self, rel: RelationClass) -> None:
        rel.name = _rel_key(rel.name)
        with self._lock:
            for side in (rel.domain, rel.range):
                unknown = [c for c in side if c not in self._classes]
                if unknown:
                    raise KnowledgeError(f"Relation {rel.name!r} uses unknown classes: {unknown}")
            self._relations[rel.name] = rel
            for key in [rel.name, *rel.aliases]:
                self._rel_lookup[_rel_key(key)] = rel.name

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def resolve_entity_type(self, raw: str) -> Optional[str]:
        with self._lock:
            return self._class_lookup.get(raw.strip().casefold())

    def resolve_relation(self, raw: str) -> Optional[str]:
        with self._lock:
            return self._rel_lookup.get(_rel_key(raw))

    def get_relation(self, name: str) -> Optional[RelationClass]:
        resolved = self.resolve_relation(name)
        return self._relations.get(resolved) if resolved else None

    def entity_classes(self) -> List[str]:
        with self._lock:
            return sorted(self._classes)

    def relations(self) -> List[str]:
        with self._lock:
            return sorted(self._relations)

    def is_symmetric(self, relation: str) -> bool:
        rel = self.get_relation(relation)
        return bool(rel and rel.symmetric)

    # ------------------------------------------------------------------ #
    # Hierarchy
    # ------------------------------------------------------------------ #
    def ancestors(self, name: str) -> List[str]:
        with self._lock:
            chain: List[str] = []
            current = self._classes.get(name)
            while current is not None and current.parent is not None:
                chain.append(current.parent)
                current = self._classes.get(current.parent)
            return chain

    def descendants(self, name: str) -> List[str]:
        with self._lock:
            return sorted(c for c in self._classes if c != name and name in self.ancestors(c))

    def is_subclass(self, child: str, ancestor: str) -> bool:
        return child == ancestor or ancestor in self.ancestors(child)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def single_class(self, relation: str, side: str) -> Optional[str]:
        """The only allowed class on ``side`` ('domain'/'range'), if exactly one."""
        rel = self.get_relation(relation)
        if rel is None:
            return None
        allowed = rel.domain if side == "domain" else rel.range
        return next(iter(allowed)) if len(allowed) == 1 and self.ROOT not in allowed else None

    def _fits(self, cls: str, allowed: FrozenSet[str]) -> bool:
        if cls == self.ROOT or not allowed:
            return True
        return any(self.is_subclass(cls, a) for a in allowed)

    def validate_relationship(self, relation: str, source_type: str, target_type: str) -> Tuple[bool, str]:
        rel = self.get_relation(relation)
        if rel is None:
            return False, f"Unknown relation {relation!r}"
        for label, cls in (("source", source_type), ("target", target_type)):
            if cls not in self._classes:
                return False, f"Unknown {label} class {cls!r}"
        if not self._fits(source_type, rel.domain):
            return False, f"{source_type} not allowed as source of {rel.name} (expected {sorted(rel.domain)})"
        if not self._fits(target_type, rel.range):
            return False, f"{target_type} not allowed as target of {rel.name} (expected {sorted(rel.range)})"
        return True, "ok"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "classes": [vars(c) for c in self._classes.values()],
                "relations": [
                    {**vars(r), "domain": sorted(r.domain), "range": sorted(r.range)}
                    for r in self._relations.values()
                ],
            }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OntologyManager":
        onto = cls(load_defaults=False)
        pending = [c for c in data.get("classes", []) if c["name"] != cls.ROOT]
        while pending:  # parents may appear after children
            progressed = False
            for raw in list(pending):
                if raw.get("parent") in onto._classes:
                    onto.register_entity_class(EntityClass(**raw))
                    pending.remove(raw)
                    progressed = True
            if not progressed:
                raise KnowledgeError(f"Unresolvable class parents: {[c['name'] for c in pending]}")
        for raw in data.get("relations", []):
            onto.register_relation(RelationClass(
                **{**raw, "domain": frozenset(raw.get("domain", [])), "range": frozenset(raw.get("range", []))}
            ))
        return onto

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OntologyManager":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ #
    def _load_defaults(self) -> None:
        classes = [
            ("Person", "Thing", ["people", "individual", "human"]),
            ("Organization", "Thing", ["org", "company", "institution", "organisation"]),
            ("Location", "Thing", ["place", "gpe", "loc", "city", "country"]),
            ("Event", "Thing", ["occasion"]),
            ("Concept", "Thing", ["topic", "idea"]),
            ("Technology", "Thing", ["tech", "software", "tool", "language"]),
            ("Product", "Thing", ["item"]),
            ("Document", "Thing", ["doc", "file", "article"]),
            ("Date", "Thing", ["time", "datetime"]),
            ("Identifier", "Thing", []),
            ("Email", "Identifier", ["email address"]),
            ("URL", "Identifier", ["link", "uri"]),
        ]
        for name, parent, aliases in classes:
            self.register_entity_class(EntityClass(name, parent, aliases=aliases))
        P, O, L, E, T, PR, D, DT, TH = (
            "Person", "Organization", "Location", "Event", "Technology",
            "Product", "Document", "Date", "Thing",
        )
        defs = [
            ("works_for", {P}, {O}, False, None, ["employed_by"]),
            ("founded", {P, O}, {O, PR, T}, False, "founded_by", []),
            ("founded_by", {O, PR, T}, {P, O}, False, "founded", []),
            ("acquired", {O}, {O, PR}, False, "acquired_by", []),
            ("acquired_by", {O, PR}, {O}, False, "acquired", []),
            ("located_in", {P, O, L, E, PR}, {L}, False, None, ["based_in"]),
            ("part_of", {TH}, {TH}, False, "has_part", ["member_of"]),
            ("has_part", {TH}, {TH}, False, "part_of", []),
            ("uses", {P, O, T, PR}, {T, PR}, False, None, ["depends_on"]),
            ("created_by", {T, PR, D}, {P, O}, False, None, ["developed_by", "authored_by"]),
            ("collaborates_with", {P, O}, {P, O}, True, None, ["partners_with"]),
            ("occurred_on", {E}, {DT}, False, None, []),
            ("mentions", {D}, {TH}, False, None, []),
            ("related_to", {TH}, {TH}, True, None, ["associated_with"]),
        ]
        for name, dom, rng, sym, inv, aliases in defs:
            self.register_relation(RelationClass(
                name, frozenset(dom), frozenset(rng), sym, inv, aliases=aliases,
            ))
