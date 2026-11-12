"""Working memory: a few attention slots with decaying activation, goals and a scratchpad."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .memory_core import (
    BaseMemory, Embedder, MemoryCapacityError, MemoryItem, MemoryType, ScoredMemory, Vector,
    clamp, cosine_similarity,
)

logger = logging.getLogger("zentrax.memory.working")

EvictionCallback = Callable[[MemoryItem], Awaitable[None]]

__all__ = ["WorkingMemory", "WorkingSlot"]


@dataclass
class WorkingSlot:
    item: MemoryItem
    activation: float = 1.0
    pinned: bool = False
    updated_at: float = field(default_factory=time.monotonic)


class WorkingMemory(BaseMemory):
    memory_type = MemoryType.WORKING

    def __init__(
        self,
        embedder: Embedder,
        capacity: int = 7,
        half_life: float = 120.0,
        on_evict: Optional[EvictionCallback] = None,
    ) -> None:
        super().__init__(embedder)
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.half_life = half_life
        self._on_evict = on_evict
        self._slots: Dict[str, WorkingSlot] = {}
        self._goals: List[str] = []
        self._notes: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def _effective(self, slot: WorkingSlot, now: float) -> float:
        return slot.activation * 0.5 ** (max(0.0, now - slot.updated_at) / self.half_life)

    async def _notify(self, items: Sequence[MemoryItem]) -> None:
        if self._on_evict is None:
            return
        for item in items:
            try:
                await self._on_evict(item)
            except Exception:  # noqa: BLE001
                logger.exception("working-memory eviction callback failed for %s", item.id)

    # --------------------------------- slots --------------------------------- #
    async def add(self, item: MemoryItem) -> str:
        item.memory_type = MemoryType.WORKING
        await self.attend(item)
        return item.id

    async def attend(self, item: MemoryItem, activation: float = 1.0, pin: bool = False) -> MemoryItem:
        """Bring ``item`` into the focus of attention (re-attending boosts it)."""
        if item.embedding is None:
            item.embedding = await self.embedder.embed(item.content)
        evicted: List[MemoryItem] = []
        async with self._lock:
            now = time.monotonic()
            slot = self._slots.get(item.id)
            if slot is None:
                slot = WorkingSlot(item=item, activation=clamp(activation), pinned=pin, updated_at=now)
                self._slots[item.id] = slot
            else:
                slot.activation = min(1.0, self._effective(slot, now) + 0.5 * clamp(activation))
                slot.updated_at = now
                slot.pinned = slot.pinned or pin
            while len(self._slots) > self.capacity:
                candidates = [s for s in self._slots.values() if not s.pinned and s.item.id != item.id]
                if not candidates:
                    if slot.item.id in self._slots and len(self._slots) > self.capacity:
                        del self._slots[item.id]
                    raise MemoryCapacityError("working memory is full of pinned items")
                victim = min(candidates, key=lambda s: self._effective(s, now))
                evicted.append(self._slots.pop(victim.item.id).item)
        await self._notify(evicted)
        return item

    async def pin(self, memory_id: str, pinned: bool = True) -> bool:
        async with self._lock:
            slot = self._slots.get(memory_id)
            if slot is None:
                return False
            slot.pinned = pinned
            return True

    async def focus(self) -> Optional[MemoryItem]:
        """The item with the highest current activation."""
        now = time.monotonic()
        async with self._lock:
            if not self._slots:
                return None
            return max(self._slots.values(), key=lambda s: self._effective(s, now)).item

    async def items(self) -> List[MemoryItem]:
        """Contents ordered by current activation (strongest first)."""
        now = time.monotonic()
        async with self._lock:
            ranked = sorted(self._slots.values(), key=lambda s: self._effective(s, now), reverse=True)
            return [s.item for s in ranked]

    async def release(self, memory_id: str) -> Optional[MemoryItem]:
        async with self._lock:
            slot = self._slots.pop(memory_id, None)
            return slot.item if slot else None

    async def decay_and_prune(self, min_activation: float = 0.05) -> List[MemoryItem]:
        """Drop unpinned slots whose activation has faded; evicted items go to the callback."""
        now = time.monotonic()
        async with self._lock:
            faded = [s for s in self._slots.values() if not s.pinned and self._effective(s, now) < min_activation]
            evicted = [self._slots.pop(s.item.id).item for s in faded]
        await self._notify(evicted)
        return evicted

    # ------------------------------ goals & notes ----------------------------- #
    async def push_goal(self, goal: str) -> None:
        async with self._lock:
            self._goals.append(goal)

    async def pop_goal(self) -> Optional[str]:
        async with self._lock:
            return self._goals.pop() if self._goals else None

    @property
    def goals(self) -> List[str]:
        return list(self._goals)

    @property
    def current_goal(self) -> Optional[str]:
        return self._goals[-1] if self._goals else None

    async def set_note(self, key: str, value: Any) -> None:
        async with self._lock:
            self._notes[key] = value

    async def get_note(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._notes.get(key, default)

    @property
    def notes(self) -> Dict[str, Any]:
        return dict(self._notes)

    async def reset(self) -> None:
        """Clear slots, goals and notes."""
        async with self._lock:
            self._slots.clear()
            self._goals.clear()
            self._notes.clear()

    async def context(self, max_chars: int = 2000) -> str:
        """Prompt-ready snapshot: goals, notes, then attended items by activation."""
        lines: List[str] = [f"GOAL: {g}" for g in reversed(self.goals)]
        lines += [f"NOTE {k}: {v}" for k, v in self.notes.items()]
        lines += [f"- {i.content}" for i in await self.items()]
        out, used = [], 0
        for line in lines:
            if used + len(line) + 1 > max_chars:
                remaining = max_chars - used - 2
                if remaining > 20:
                    out.append(line[:remaining] + "…")
                break
            out.append(line)
            used += len(line) + 1
        return "\n".join(out)

    async def snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        async with self._lock:
            slots = [
                {"id": s.item.id, "content": s.item.content, "activation": round(self._effective(s, now), 4), "pinned": s.pinned}
                for s in sorted(self._slots.values(), key=lambda s: self._effective(s, now), reverse=True)
            ]
            return {"slots": slots, "goals": list(self._goals), "notes": dict(self._notes), "capacity": self.capacity}

    # ------------------------------- BaseMemory ------------------------------- #
    async def get(self, memory_id: str, *, touch: bool = True) -> Optional[MemoryItem]:
        async with self._lock:
            slot = self._slots.get(memory_id)
            if slot is None:
                return None
            if touch:
                slot.item.touch()
            return slot.item

    async def mark_accessed(self, memory_id: str) -> None:
        async with self._lock:
            slot = self._slots.get(memory_id)
            if slot is not None:
                slot.item.touch()

    async def delete(self, memory_id: str) -> bool:
        return (await self.release(memory_id)) is not None

    async def count(self) -> int:
        return len(self._slots)

    async def clear(self) -> None:
        async with self._lock:
            self._slots.clear()

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
        now = time.monotonic()
        async with self._lock:
            pool = [(s.item, self._effective(s, now)) for s in self._slots.values()]
        wanted = set(tags or ())
        results: List[ScoredMemory] = []
        for item, activation in pool:
            if wanted and not wanted.intersection(item.tags):
                continue
            sim = cosine_similarity(qv, item.embedding)
            if sim >= min_score:
                results.append(
                    ScoredMemory(item, sim * (0.5 + 0.5 * activation), self.memory_type.value,
                                 {"similarity": sim, "activation": activation})
                )
        results.sort(key=lambda s: s.score, reverse=True)
        return results[:top_k]
