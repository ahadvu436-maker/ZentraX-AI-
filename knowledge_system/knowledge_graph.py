"""Thread-safe, in-memory knowledge graph with JSON persistence."""
from __future__ import annotations

import copy
import json
import os
import threading
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

from .knowledge_core import (
    Entity, KnowledgeError, Relationship, make_relationship_id, normalize_name,
)

__all__ = ["KnowledgeGraph"]

_FORMAT_VERSION = 1


class KnowledgeGraph:
    """Directed, typed multigraph of entities and relationships."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._out: Dict[str, Set[str]] = defaultdict(set)
        self._in: Dict[str, Set[str]] = defaultdict(set)
        self._names: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------ #
    # Indexing helpers
    # ------------------------------------------------------------------ #
    def _index(self, entity: Entity) -> None:
        for name in entity.all_names():
            key = normalize_name(name)
            if key:
                self._names[key].add(entity.id)

    def _unindex(self, entity: Entity) -> None:
        for name in entity.all_names():
            key = normalize_name(name)
            ids = self._names.get(key)
            if ids:
                ids.discard(entity.id)
                if not ids:
                    del self._names[key]

    def _remove_relationship(self, rel_id: str) -> None:
        rel = self._relationships.pop(rel_id, None)
        if rel is not None:
            self._out[rel.source_id].discard(rel_id)
            self._in[rel.target_id].discard(rel_id)

    # ------------------------------------------------------------------ #
    # Entities
    # ------------------------------------------------------------------ #
    def add_entity(self, entity: Entity) -> Tuple[Entity, bool]:
        """Insert or merge. Returns ``(stored_entity, created)``."""
        with self._lock:
            existing = self._entities.get(entity.id)
            if existing is None:
                self._entities[entity.id] = entity
                self._index(entity)
                return entity, True
            self._unindex(existing)
            existing.merge(entity)
            self._index(existing)
            return existing, False

    def has_entity(self, entity_id: str) -> bool:
        with self._lock:
            return entity_id in self._entities

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        with self._lock:
            return self._entities.get(entity_id)

    def find_entities(self, name: str, entity_type: Optional[str] = None) -> List[Entity]:
        """Exact (normalised) match on name or alias."""
        with self._lock:
            ids = self._names.get(normalize_name(name), set())
            found = [self._entities[i] for i in ids if i in self._entities]
        if entity_type:
            found = [e for e in found if e.type == entity_type]
        return sorted(found, key=lambda e: (-e.confidence, e.id))

    def search_entities(self, text: str, limit: int = 20) -> List[Entity]:
        """Substring match over names and aliases."""
        needle = normalize_name(text)
        if not needle:
            return []
        with self._lock:
            hits = [self._entities[i] for key, ids in self._names.items() if needle in key
                    for i in ids if i in self._entities]
        unique = {e.id: e for e in hits}
        return sorted(unique.values(), key=lambda e: (len(e.name), e.id))[:limit]

    def remove_entity(self, entity_id: str) -> bool:
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                return False
            for rel_id in list(self._out.get(entity_id, ())) + list(self._in.get(entity_id, ())):
                self._remove_relationship(rel_id)
            self._unindex(entity)
            del self._entities[entity_id]
            self._out.pop(entity_id, None)
            self._in.pop(entity_id, None)
            return True

    def merge_entities(self, keep_id: str, drop_id: str) -> Entity:
        """Merge ``drop_id`` into ``keep_id``, rewiring relationships."""
        with self._lock:
            keep = self._entities.get(keep_id)
            drop = self._entities.get(drop_id)
            if keep is None or drop is None:
                raise KnowledgeError("Both entities must exist to merge")
            if keep_id == drop_id:
                return keep
            affected = [self._relationships[r] for r in (self._out[drop_id] | self._in[drop_id])
                        if r in self._relationships]
            for rel in affected:
                self._remove_relationship(rel.id)
            self._unindex(keep)
            self._unindex(drop)
            del self._entities[drop_id]
            self._out.pop(drop_id, None)
            self._in.pop(drop_id, None)
            keep.merge(drop)
            self._index(keep)
            for rel in affected:
                src = keep_id if rel.source_id == drop_id else rel.source_id
                dst = keep_id if rel.target_id == drop_id else rel.target_id
                if src == dst:
                    continue
                self.add_relationship(Relationship(
                    id=make_relationship_id(src, rel.type, dst), source_id=src, target_id=dst,
                    type=rel.type, confidence=rel.confidence, properties=rel.properties,
                    provenance=rel.provenance, evidence=rel.evidence,
                ))
            return keep

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #
    def add_relationship(self, rel: Relationship) -> Tuple[Relationship, bool]:
        with self._lock:
            if rel.source_id not in self._entities or rel.target_id not in self._entities:
                raise KnowledgeError(f"Relationship {rel.id} references a missing entity")
            existing = self._relationships.get(rel.id)
            if existing is not None:
                existing.merge(rel)
                return existing, False
            self._relationships[rel.id] = rel
            self._out[rel.source_id].add(rel.id)
            self._in[rel.target_id].add(rel.id)
            return rel, True

    def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        with self._lock:
            return self._relationships.get(rel_id)

    def relationships_of(self, entity_id: str, direction: str = "both",
                         relation_type: Optional[str] = None) -> List[Relationship]:
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be 'out', 'in' or 'both'")
        with self._lock:
            ids: Set[str] = set()
            if direction in ("out", "both"):
                ids |= self._out.get(entity_id, set())
            if direction in ("in", "both"):
                ids |= self._in.get(entity_id, set())
            rels = [self._relationships[i] for i in ids if i in self._relationships]
        if relation_type:
            rels = [r for r in rels if r.type == relation_type]
        return sorted(rels, key=lambda r: (-r.confidence, r.id))

    def neighbors(self, entity_id: str, direction: str = "both",
                  relation_type: Optional[str] = None) -> List[Entity]:
        seen: Dict[str, Entity] = {}
        for rel in self.relationships_of(entity_id, direction, relation_type):
            other = rel.target_id if rel.source_id == entity_id else rel.source_id
            entity = self.get_entity(other)
            if entity is not None and other != entity_id:
                seen[other] = entity
        return list(seen.values())

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #
    def shortest_path(self, source_id: str, target_id: str, max_depth: int = 6,
                      directed: bool = False) -> Optional[List[str]]:
        """BFS shortest path as a list of entity ids, or ``None``."""
        with self._lock:
            if source_id not in self._entities or target_id not in self._entities:
                return None
            if source_id == target_id:
                return [source_id]
            parents: Dict[str, Optional[str]] = {source_id: None}
            queue = deque([(source_id, 0)])
            while queue:
                node, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for rel in self.relationships_of(node, "out" if directed else "both"):
                    nxt = rel.target_id if rel.source_id == node else rel.source_id
                    if nxt in parents:
                        continue
                    parents[nxt] = node
                    if nxt == target_id:
                        path = [nxt]
                        while parents[path[-1]] is not None:
                            path.append(parents[path[-1]])  # type: ignore[arg-type]
                        return path[::-1]
                    queue.append((nxt, depth + 1))
        return None

    def subgraph(self, seed_ids: List[str], depth: int = 1) -> "KnowledgeGraph":
        """Copy of the neighbourhood around ``seed_ids`` up to ``depth`` hops."""
        with self._lock:
            nodes: Set[str] = {s for s in seed_ids if s in self._entities}
            frontier = set(nodes)
            for _ in range(max(0, depth)):
                nxt: Set[str] = set()
                for node in frontier:
                    for rel in self.relationships_of(node):
                        nxt.add(rel.target_id if rel.source_id == node else rel.source_id)
                frontier = nxt - nodes
                nodes |= nxt
            sub = KnowledgeGraph()
            for nid in nodes:
                sub._entities[nid] = copy.deepcopy(self._entities[nid])
                sub._index(sub._entities[nid])
            for rel in self._relationships.values():
                if rel.source_id in nodes and rel.target_id in nodes:
                    sub.add_relationship(copy.deepcopy(rel))
            return sub

    def connected_components(self) -> List[Set[str]]:
        with self._lock:
            unseen = set(self._entities)
            components: List[Set[str]] = []
            while unseen:
                root = unseen.pop()
                comp = {root}
                queue = deque([root])
                while queue:
                    node = queue.popleft()
                    for rel in self.relationships_of(node):
                        other = rel.target_id if rel.source_id == node else rel.source_id
                        if other not in comp:
                            comp.add(other)
                            unseen.discard(other)
                            queue.append(other)
                components.append(comp)
            return components

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def iter_entities(self) -> Iterator[Entity]:
        with self._lock:
            return iter(list(self._entities.values()))

    def iter_relationships(self) -> Iterator[Relationship]:
        with self._lock:
            return iter(list(self._relationships.values()))

    def degree(self, entity_id: str) -> int:
        with self._lock:
            return len(self._out.get(entity_id, ())) + len(self._in.get(entity_id, ()))

    def __len__(self) -> int:
        return len(self._entities)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n, m = len(self._entities), len(self._relationships)
            return {
                "entities": n,
                "relationships": m,
                "entity_types": dict(Counter(e.type for e in self._entities.values())),
                "relationship_types": dict(Counter(r.type for r in self._relationships.values())),
                "components": len(self.connected_components()) if n else 0,
                "avg_degree": round(2 * m / n, 3) if n else 0.0,
            }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": _FORMAT_VERSION,
                "entities": [e.to_dict() for e in self._entities.values()],
                "relationships": [r.to_dict() for r in self._relationships.values()],
            }

    def _replace(self, data: Mapping[str, Any]) -> None:
        if data.get("version") != _FORMAT_VERSION:
            raise KnowledgeError(f"Unsupported graph format version: {data.get('version')!r}")
        with self._lock:
            self._entities.clear()
            self._relationships.clear()
            self._out.clear()
            self._in.clear()
            self._names.clear()
            for raw in data.get("entities", []):
                entity = Entity.from_dict(raw)
                self._entities[entity.id] = entity
                self._index(entity)
            for raw in data.get("relationships", []):
                self.add_relationship(Relationship.from_dict(raw))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeGraph":
        graph = cls()
        graph._replace(data)
        return graph

    def save(self, path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)

    def load(self, path: Union[str, Path]) -> None:
        """Replace this graph's contents in place (keeps external references valid)."""
        self._replace(json.loads(Path(path).read_text(encoding="utf-8")))
