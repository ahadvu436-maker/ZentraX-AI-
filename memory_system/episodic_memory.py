"""Episodic memory: time-stamped events with actors, place, outcome and emotion."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from .long_term_memory import LongTermMemory
from .memory_core import MemoryItem, MemoryType, ScoredMemory, clamp

__all__ = ["EpisodicMemory"]


class EpisodicMemory(LongTermMemory):
    memory_type = MemoryType.EPISODIC

    def _payload(self, item: MemoryItem) -> Dict[str, Any]:
        payload = super()._payload(item)
        payload["session_id"] = item.metadata.get("session_id")
        return payload

    async def record_event(
        self,
        content: str,
        *,
        actors: Optional[Sequence[str]] = None,
        location: Optional[str] = None,
        outcome: Optional[str] = None,
        valence: float = 0.0,
        session_id: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "episode",
    ) -> MemoryItem:
        """Record an event. ``valence`` in [-1, 1] (emotional charge) boosts default importance."""
        valence = clamp(valence, -1.0, 1.0)
        if importance is None:
            importance = clamp(0.4 + 0.4 * abs(valence) + (0.1 if outcome else 0.0))
        md: Dict[str, Any] = dict(metadata or {})
        md.update(
            kind="event", actors=list(actors or []), location=location, outcome=outcome,
            valence=valence, session_id=session_id, consolidated=False,
        )
        now = time.time()
        item = MemoryItem(
            content=content, memory_type=MemoryType.EPISODIC, importance=importance,
            created_at=timestamp if timestamp is not None else now, last_accessed=now,
            tags=list(tags or []), metadata=md, source=source,
        )
        await self.add(item)
        return item

    async def timeline(
        self,
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        ascending: bool = True,
        scan_limit: int = 5000,
    ) -> List[MemoryItem]:
        """Events in a time window (optionally one session), in chronological order."""
        items = await self.rel.query_memories(
            self.memory_type, since=start, until=end, limit=scan_limit if session_id else limit,
            order_by="created_at ASC" if ascending else "created_at DESC",
        )
        if session_id is not None:
            items = [i for i in items if i.metadata.get("session_id") == session_id]
        return items[:limit]

    async def recent(self, n: int = 10) -> List[MemoryItem]:
        return await self.rel.query_memories(self.memory_type, limit=n, order_by="created_at DESC")

    async def by_session(self, session_id: str, limit: int = 100) -> List[MemoryItem]:
        return await self.timeline(session_id=session_id, limit=limit)

    async def context_around(self, memory_id: str, window: int = 3) -> List[MemoryItem]:
        """The event plus up to ``window`` events before/after it (same session if it has one)."""
        anchor = await self.get(memory_id, touch=False)
        if anchor is None:
            return []
        session_id = anchor.metadata.get("session_id")
        before = await self.timeline(end=anchor.created_at, session_id=session_id, limit=window + 1, ascending=False)
        after = await self.timeline(start=anchor.created_at, session_id=session_id, limit=window + 1)
        merged = {i.id: i for i in list(reversed(before)) + after}
        return sorted(merged.values(), key=lambda i: (i.created_at, i.id))

    async def recall_similar(
        self,
        query: str,
        top_k: int = 5,
        *,
        session_id: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> List[ScoredMemory]:
        flt = {"session_id": session_id} if session_id is not None else None
        hits = await self.search(query, top_k * 3 if (since or until) else top_k, tags=tags, filter=flt)
        hits = [
            h for h in hits
            if (since is None or h.item.created_at >= since) and (until is None or h.item.created_at <= until)
        ]
        return hits[:top_k]

    async def unconsolidated(self, limit: int = 500) -> List[MemoryItem]:
        """Episodes not yet summarised into semantic memory (embeddings included)."""
        items = await self.rel.query_memories(
            self.memory_type, limit=limit * 2, order_by="created_at ASC", include_embedding=True
        )
        return [i for i in items if not i.metadata.get("consolidated")][:limit]

    async def mark_consolidated(self, items: Sequence[MemoryItem], summary_id: str) -> None:
        for item in items:
            md = {**item.metadata, "consolidated": True, "summary_id": summary_id, "consolidated_at": time.time()}
            await self.update_fields(item.id, metadata=md)
