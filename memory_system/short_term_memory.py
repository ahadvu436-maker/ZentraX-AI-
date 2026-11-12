"""Short-term memory: a bounded, TTL-governed buffer of recent items.

Items that are evicted or expire are handed to registered callbacks (the
consolidator uses this to promote valuable ones to long-term storage).
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Awaitable, Callable, List, Optional, Sequence

from .memory_core import BaseMemory, Embedder, MemoryItem, MemoryType, ScoredMemory, Vector, cosine_similarity

logger = logging.getLogger("zentrax.memory.short_term")

EvictionCallback = Callable[[MemoryItem], Awaitable[None]]

__all__ = ["ShortTermMemory", "EvictionCallback"]


class ShortTermMemory(BaseMemory):
    memory_type = MemoryType.SHORT_TERM

    def __init__(
        self,
        embedder: Embedder,
        capacity: int = 200,
        default_ttl: Optional[float] = 1800.0,
        on_evict: Optional[EvictionCallback] = None,
    ) -> None:
        super().__init__(embedder)
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._items: "OrderedDict[str, MemoryItem]" = OrderedDict()
        self._lock = asyncio.Lock()
        self._callbacks: List[EvictionCallback] = [on_evict] if on_evict else []

    def add_eviction_callback(self, callback: EvictionCallback) -> None:
        self._callbacks.append(callback)

    async def _notify(self, items: Sequence[MemoryItem]) -> None:
        for item in items:
            for cb in self._callbacks:
                try:
                    await cb(item)
                except Exception:  # noqa: BLE001 - a bad callback must not break memory
                    logger.exception("eviction callback failed for %s", item.id)

    def _keep_score(self, item: MemoryItem, now: float) -> float:
        horizon = self.default_ttl or 1800.0
        recency = 0.5 ** (max(0.0, now - item.last_accessed) / horizon)
        return 0.6 * item.importance + 0.25 * recency + 0.15 * min(1.0, item.access_count / 5.0)

    # -------------------------------- writes -------------------------------- #
    async def add(self, item: MemoryItem) -> str:
        item.memory_type = MemoryType.SHORT_TERM
        if item.ttl is None and self.default_ttl:
            item.ttl = self.default_ttl
        if item.embedding is None:
            item.embedding = await self.embedder.embed(item.content)
        evicted: List[MemoryItem] = []
        async with self._lock:
            self._items[item.id] = item
            self._items.move_to_end(item.id)
            now = time.time()
            for stale_id in [i.id for i in self._items.values() if i.is_expired(now) and i.id != item.id]:
                evicted.append(self._items.pop(stale_id))
            while len(self._items) > self.capacity:
                victim = min(
                    (i for i in self._items.values() if i.id != item.id),
                    key=lambda i: self._keep_score(i, now),
                )
                evicted.append(self._items.pop(victim.id))
        await self._notify(evicted)
        return item.id

    async def delete(self, memory_id: str) -> bool:
        async with self._lock:
            return self._items.pop(memory_id, None) is not None

    async def remove_many(self, ids: Sequence[str]) -> int:
        """Drop items silently (no eviction callbacks), e.g. after promotion."""
        async with self._lock:
            return sum(1 for i in ids if self._items.pop(i, None) is not None)

    async def clear(self) -> None:
        async with self._lock:
            self._items.clear()

    async def expire(self) -> List[MemoryItem]:
        """Remove expired items, notify callbacks, and return them."""
        now = time.time()
        async with self._lock:
            expired = [self._items.pop(i.id) for i in list(self._items.values()) if i.is_expired(now)]
        await self._notify(expired)
        return expired

    # -------------------------------- reads --------------------------------- #
    async def get(self, memory_id: str, *, touch: bool = True) -> Optional[MemoryItem]:
        async with self._lock:
            item = self._items.get(memory_id)
            if item is None or item.is_expired():
                return None
            if touch:
                item.touch()
                self._items.move_to_end(memory_id)
            return item

    async def mark_accessed(self, memory_id: str) -> None:
        async with self._lock:
            item = self._items.get(memory_id)
            if item is not None:
                item.touch()

    async def count(self) -> int:
        return len(self._items)

    async def recent(self, n: int = 10) -> List[MemoryItem]:
        async with self._lock:
            return list(reversed(list(self._items.values())))[:n]

    async def candidates(self, min_importance: float, min_access: int) -> List[MemoryItem]:
        """Live items worthy of promotion to long-term memory."""
        now = time.time()
        async with self._lock:
            return [
                i for i in self._items.values()
                if not i.is_expired(now)
                and not i.metadata.get("ephemeral")
                and (i.importance >= min_importance or i.access_count >= min_access or i.metadata.get("promote") is True)
            ]

    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        query_vector: Optional[Vector] = None,
        tags: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
    ) -> List[ScoredMemory]:
        if top_k <= 0:
            return []
        qv = query_vector if query_vector is not None else await self.embedder.embed(query)
        now = time.time()
        async with self._lock:
            pool = [i for i in self._items.values() if not i.is_expired(now)]
        wanted = set(tags or ())
        results: List[ScoredMemory] = []
        for item in pool:
            if wanted and not wanted.intersection(item.tags):
                continue
            if item.embedding is None:
                item.embedding = await self.embedder.embed(item.content)
            sim = cosine_similarity(qv, item.embedding)
            if sim >= min_score:
                results.append(ScoredMemory(item, sim, self.memory_type.value, {"similarity": sim}))
        results.sort(key=lambda s: s.score, reverse=True)
        return results[:top_k]
