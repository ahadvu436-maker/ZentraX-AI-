"""Semantic memory: de-duplicated facts plus a knowledge graph of triples."""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

from .long_term_memory import LongTermMemory
from .memory_core import MemoryItem, MemoryType, ScoredMemory, clamp

__all__ = ["SemanticMemory"]

Edge = Tuple[str, str, str]  # (subject, predicate, object)


def normalize_concept(name: str) -> str:
    return " ".join(name.lower().split())


class SemanticMemory(LongTermMemory):
    memory_type = MemoryType.SEMANTIC

    def _payload(self, item: MemoryItem) -> Dict[str, Any]:
        payload = super()._payload(item)
        subject = item.metadata.get("subject")
        payload["subject"] = normalize_concept(subject) if subject else None
        return payload

    # --------------------------------- facts --------------------------------- #
    async def add_fact(
        self,
        statement: str,
        *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        confidence: float = 0.7,
        importance: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        source: str = "semantic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Store a fact. Re-stating a known fact reinforces it instead of duplicating it.

        If ``subject``/``predicate``/``obj`` are all given, the triple is also
        added to the knowledge graph.
        """
        confidence = clamp(confidence)
        existing = await self.find_duplicate(statement)
        if existing is not None:
            md = dict(existing.metadata)
            old = float(md.get("confidence", 0.5))
            md["confidence"] = min(1.0, old + (1.0 - old) * 0.2)
            await self.update_fields(
                existing.id, importance=min(1.0, existing.importance + 0.05), metadata=md
            )
            await self.rel.touch(existing.id)
            refreshed = await self.get(existing.id, touch=False)
            return refreshed if refreshed is not None else existing

        md = dict(metadata or {})
        md.update(kind=md.get("kind", "fact"), confidence=confidence, subject=subject, predicate=predicate, object=obj)
        item = MemoryItem(
            content=statement, memory_type=MemoryType.SEMANTIC,
            importance=clamp(importance if importance is not None else 0.5 + 0.3 * confidence),
            tags=list(tags or []), metadata=md, source=source,
        )
        await self.add(item)
        if subject and predicate and obj:
            await self.add_relation(subject, predicate, obj, weight=confidence, metadata={"fact_id": item.id})
        return item

    # -------------------------------- relations ------------------------------- #
    async def add_relation(
        self, subject: str, predicate: str, obj: str, weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        return await self.rel.add_relation(
            normalize_concept(subject), normalize_concept(predicate), normalize_concept(obj),
            clamp(weight), metadata,
        )

    async def relations(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        *,
        min_weight: float = 0.0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return await self.rel.get_relations(
            normalize_concept(subject) if subject else None,
            normalize_concept(predicate) if predicate else None,
            normalize_concept(obj) if obj else None,
            min_weight=min_weight, limit=limit,
        )

    async def _edges_of(self, node: str, limit: int) -> List[Dict[str, Any]]:
        out = await self.rel.get_relations(subject=node, limit=limit)
        inc = await self.rel.get_relations(obj=node, limit=limit)
        return out + inc

    async def neighbors(self, concept: str, depth: int = 1, limit_per_node: int = 50) -> List[Edge]:
        """All edges within ``depth`` hops of ``concept`` (both directions), de-duplicated."""
        start = normalize_concept(concept)
        seen_nodes: Set[str] = {start}
        edges: Dict[Edge, None] = {}
        frontier = [start]
        for _ in range(max(0, depth)):
            nxt: List[str] = []
            for node in frontier:
                for rel in await self._edges_of(node, limit_per_node):
                    edge = (rel["subject"], rel["predicate"], rel["object"])
                    edges[edge] = None
                    for other in (rel["subject"], rel["object"]):
                        if other not in seen_nodes:
                            seen_nodes.add(other)
                            nxt.append(other)
            frontier = nxt
            if not frontier:
                break
        return list(edges)

    async def find_path(self, source: str, target: str, max_depth: int = 4, limit_per_node: int = 50) -> Optional[List[Edge]]:
        """Shortest chain of edges connecting two concepts (edges may be traversed either way)."""
        src, dst = normalize_concept(source), normalize_concept(target)
        if src == dst:
            return []
        queue: Deque[Tuple[str, List[Edge]]] = deque([(src, [])])
        visited: Set[str] = {src}
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for rel in await self._edges_of(node, limit_per_node):
                edge = (rel["subject"], rel["predicate"], rel["object"])
                other = rel["object"] if rel["subject"] == node else rel["subject"]
                if other in visited:
                    continue
                if other == dst:
                    return path + [edge]
                visited.add(other)
                queue.append((other, path + [edge]))
        return None

    async def facts_about(self, concept: str, top_k: int = 10) -> List[ScoredMemory]:
        """Facts whose subject is ``concept`` first, then semantically similar facts."""
        key = normalize_concept(concept)
        direct = await self.search(concept, top_k, filter={"subject": key})
        seen = {h.item.id for h in direct}
        similar = [h for h in await self.search(concept, top_k) if h.item.id not in seen]
        return (direct + similar)[:top_k]
