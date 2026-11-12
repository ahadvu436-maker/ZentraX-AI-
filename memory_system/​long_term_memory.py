"""Long-term memory: durable storage backed by the relational + vector stores.

``EpisodicMemory`` and ``SemanticMemory`` subclass this and only change the
``memory_type`` and add domain-specific helpers, so all three share the same
persistence, decay and retrieval behaviour.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence

from .memory_core import (
    BaseMemory, Embedder, MemoryItem, MemoryType, ScoredMemory, Vector, content_hash,
)
from .memory_index import MemoryIndex
from .storage.relational_store import RelationalStore
from .storage.vector_store import VectorStore

logger = logging.getLogger("zentrax.memory.long_term")

__all__ = ["LongTermMemory"]


class LongTermMemory(BaseMemory):
    memory_type = MemoryType.LONG_TERM

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        relational_store: RelationalStore,
        index: Optional[MemoryIndex] = None,
        *,
        half_life: float = 7 * 86400.0,
        forget_threshold: float = 0.05,
        protect_importance: float = 0.9,
        memory_type: Optional[MemoryType] = None,
    ) -> None:
        super().__init__(embedder)
        if memory_type is not None:
            self.memory_type = MemoryType(memory_type)
        self.vectors = vector_store
        self.rel = relational_store
        self.index = index
        self.half_life = half_life
        self.forget_threshold = forget_threshold
        self.protect_importance = protect_importance

    # ------------------------------ lifecycle ------------------------------- #
    def _payload(self, item: MemoryItem) -> Dict[str, Any]:
        """Vector-store payload used for filtering; subclasses may extend."""
        return {"memory_type": self.memory_type.value, "tags": list(item.tags), "created_at": item.created_at}

    async def initialize(self) -> None:
        await self._sync_vectors()
        if self.index is not None:
            items = await self.rel.query_memories(self.memory_type, limit=1_000_000, order_by="created_at ASC")
            await self.index.add_many(items)

    async def _sync_vectors(self) -> None:
        """Rebuild missing vectors from the relational store (e.g. lost snapshot)."""
        stored = await self.rel.count(self.memory_type)
        indexed = await self.vectors.count({"memory_type": self.memory_type.value})
        if indexed >= stored:
            return
        rows = await self.rel.query_memories(
            self.memory_type, limit=1_000_000, order_by="created_at ASC", include_embedding=True
        )
        rebuilt = 0
        for item in rows:
            if item.embedding is not None and await self.vectors.get(item.id) is None:
                await self.vectors.upsert(item.id, item.embedding, self._payload(item))
                rebuilt += 1
        if rebuilt:
            logger.info("%s: rebuilt %d vectors", self.memory_type.value, rebuilt)

    # -------------------------------- writes -------------------------------- #
    async def add(self, item: MemoryItem) -> str:
        item.memory_type = self.memory_type
        if item.embedding is None:
            item.embedding = await self.embedder.embed(item.content)
        await self.rel.upsert_memory(item)
        await self.vectors.upsert(item.id, item.embedding, self._payload(item))
        if self.index is not None:
            await self.index.add(item)
        return item.id

    async def add_many(self, items: Sequence[MemoryItem]) -> List[str]:
        missing = [i for i in items if i.embedding is None]
        if missing:
            for item, vec in zip(missing, await self.embedder.embed_batch([i.content for i in missing])):
                item.embedding = vec
        for item in items:
            item.memory_type = self.memory_type
        await self.rel.upsert_memories(items)
        await self.vectors.upsert_many([(i.id, i.embedding, self._payload(i)) for i in items])  # type: ignore[misc]
        if self.index is not None:
            await self.index.add_many(items)
        return [i.id for i in items]

    async def update_fields(
        self,
        memory_id: str,
        *,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        access_count: Optional[int] = None,
    ) -> bool:
        fields: Dict[str, Any] = {}
        if importance is not None:
            fields["importance"] = max(0.0, min(1.0, importance))
        if metadata is not None:
            fields["metadata"] = metadata
        if tags is not None:
            fields["tags"] = tags
        if access_count is not None:
            fields["access_count"] = access_count
        if not fields or not await self.rel.update_memory(memory_id, **fields):
            return False
        if "tags" in fields or "metadata" in fields:
            item = await self.rel.get_memory(memory_id)
            stored = await self.vectors.get(memory_id)
            if item is not None and stored is not None:
                await self.vectors.upsert(memory_id, stored[0], self._payload(item))
            if item is not None and self.index is not None and "tags" in fields:
                await self.index.add(item)
        return True

    async def reinforce(self, memory_id: str, boost: float = 0.05) -> bool:
        """Rehearsal: bump importance slightly and record an access."""
        item = await self.rel.get_memory(memory_id)
        if item is None or item.memory_type is not self.memory_type:
            return False
        await self.rel.update_memory(memory_id, importance=min(1.0, item.importance + boost))
        await self.rel.touch(memory_id)
        return True

    async def delete(self, memory_id: str) -> bool:
        item = await self.rel.get_memory(memory_id)
        if item is None or item.memory_type is not self.memory_type:
            return False
        await self._remove_ids([memory_id])
        return True

    async def _remove_ids(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        await self.rel.delete_memories(ids)
        for memory_id in ids:
            await self.vectors.delete(memory_id)
            if self.index is not None:
                await self.index.remove(memory_id)

    async def clear(self) -> None:
        items = await self.rel.query_memories(self.memory_type, limit=1_000_000)
        await self._remove_ids([i.id for i in items])

    # -------------------------------- reads --------------------------------- #
    async def get(self, memory_id: str, *, touch: bool = True) -> Optional[MemoryItem]:
        item = await self.rel.get_memory(memory_id)
        if item is None or item.memory_type is not self.memory_type or item.is_expired():
            return None
        if touch:
            await self.rel.touch(memory_id)
            item.touch()
        return item

    async def mark_accessed(self, memory_id: str) -> None:
        await self.rel.touch(memory_id)

    async def count(self) -> int:
        return await self.rel.count(self.memory_type)

    async def find_duplicate(self, content: str) -> Optional[MemoryItem]:
        return await self.rel.find_by_hash(content_hash(content), self.memory_type)

    async def list_items(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        since: Optional[float] = None,
        until: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        min_importance: Optional[float] = None,
        order_by: str = "created_at DESC",
        include_embedding: bool = False,
    ) -> List[MemoryItem]:
        return await self.rel.query_memories(
            self.memory_type, since=since, until=until, tags=tags, min_importance=min_importance,
            limit=limit, offset=offset, order_by=order_by, include_embedding=include_embedding,
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        query_vector: Optional[Vector] = None,
        tags: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredMemory]:
        if top_k <= 0:
            return []
        qv = query_vector if query_vector is not None else await self.embedder.embed(query)
        vfilter: Dict[str, Any] = {"memory_type": self.memory_type.value}
        if tags:
            vfilter["tags"] = list(tags)
        if filter:
            vfilter.update(filter)
        hits = await self.vectors.search(qv, top_k=top_k * 3 + 5, filter=vfilter, min_score=min_score)
        items = {i.id: i for i in await self.rel.get_memories([h.id for h in hits])}
        now = time.time()
        results: List[ScoredMemory] = []
        for hit in hits:
            item = items.get(hit.id)
            if item is None or item.is_expired(now):
                continue
            item.embedding = hit.vector
            retention = self.retention(item, now)
            results.append(
                ScoredMemory(
                    item=item,
                    score=hit.score * (0.7 + 0.3 * retention),
                    source=self.memory_type.value,
                    components={"similarity": hit.score, "retention": retention},
                )
            )
        results.sort(key=lambda s: s.score, reverse=True)
        return results[:top_k]

    # -------------------------------- decay --------------------------------- #
    def retention(self, item: MemoryItem, now: Optional[float] = None) -> float:
        """Ebbinghaus-style retention in [0, 1]: time decay x importance x rehearsal."""
        now = time.time() if now is None else now
        age = max(0.0, now - item.last_accessed)
        decay = 0.5 ** (age / self.half_life)
        strength = 0.5 + 0.5 * item.importance
        rehearsal = 1.0 + 0.1 * math.log1p(item.access_count)
        return max(0.0, min(1.0, decay * strength * rehearsal))

    def is_protected(self, item: MemoryItem) -> bool:
        return item.importance >= self.protect_importance or bool(item.metadata.get("pinned"))

    async def apply_decay(self, scan_limit: int = 1000) -> int:
        """Delete TTL-expired items and long-unused, low-value ones. Returns count removed."""
        now = time.time()
        doomed = set(await self.rel.find_expired_ids(now, self.memory_type))
        stalest = await self.rel.query_memories(self.memory_type, limit=scan_limit, order_by="last_accessed ASC")
        for item in stalest:
            if not self.is_protected(item) and self.retention(item, now) < self.forget_threshold:
                doomed.add(item.id)
        await self._remove_ids(list(doomed))
        return len(doomed)
