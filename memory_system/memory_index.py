"""Lexical / structural index over persistent memories.

Provides BM25 keyword search plus tag, type and time-range lookups. Every
method is synchronous in effect (no awaits inside), hence atomic under
asyncio; they are ``async`` only to match the rest of the module API.
"""
from __future__ import annotations

import bisect
import heapq
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .memory_core import MemoryItem, MemoryType, tokenize

__all__ = ["MemoryIndex"]

_INF = float("inf")


class MemoryIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._doc_terms: Dict[str, List[str]] = {}
        self._doc_len: Dict[str, int] = {}
        self._doc_type: Dict[str, str] = {}
        self._doc_time: Dict[str, float] = {}
        self._doc_tags: Dict[str, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._type_index: Dict[str, Set[str]] = defaultdict(set)
        self._time: List[Tuple[float, str]] = []
        self._total_len = 0

    # ------------------------------ properties ------------------------------ #
    @property
    def size(self) -> int:
        return len(self._doc_len)

    def contains(self, memory_id: str) -> bool:
        return memory_id in self._doc_len

    def type_of(self, memory_id: str) -> Optional[str]:
        return self._doc_type.get(memory_id)

    # -------------------------------- writes -------------------------------- #
    async def add(self, item: MemoryItem) -> None:
        if item.id in self._doc_len:
            self._remove_sync(item.id)
        terms = tokenize(item.content + " " + " ".join(item.tags))
        counts = Counter(terms)
        for term, tf in counts.items():
            self._postings[term][item.id] = tf
        self._doc_terms[item.id] = list(counts)
        self._doc_len[item.id] = len(terms)
        self._total_len += len(terms)
        self._doc_type[item.id] = item.memory_type.value
        self._doc_time[item.id] = item.created_at
        self._doc_tags[item.id] = set(item.tags)
        for tag in item.tags:
            self._tag_index[tag].add(item.id)
        self._type_index[item.memory_type.value].add(item.id)
        bisect.insort(self._time, (item.created_at, item.id))

    async def add_many(self, items: Iterable[MemoryItem]) -> None:
        for item in items:
            await self.add(item)

    async def remove(self, memory_id: str) -> bool:
        return self._remove_sync(memory_id)

    def _remove_sync(self, memory_id: str) -> bool:
        if memory_id not in self._doc_len:
            return False
        for term in self._doc_terms.pop(memory_id, []):
            posting = self._postings.get(term)
            if posting is not None:
                posting.pop(memory_id, None)
                if not posting:
                    del self._postings[term]
        self._total_len -= self._doc_len.pop(memory_id)
        mtype = self._doc_type.pop(memory_id)
        self._type_index[mtype].discard(memory_id)
        for tag in self._doc_tags.pop(memory_id, set()):
            bucket = self._tag_index.get(tag)
            if bucket is not None:
                bucket.discard(memory_id)
                if not bucket:
                    del self._tag_index[tag]
        created = self._doc_time.pop(memory_id)
        pos = bisect.bisect_left(self._time, (created, memory_id))
        if pos < len(self._time) and self._time[pos] == (created, memory_id):
            del self._time[pos]
        return True

    async def clear(self) -> None:
        self.__init__(self.k1, self.b)  # type: ignore[misc]

    # -------------------------------- lookups ------------------------------- #
    def ids_by_tag(self, tag: str) -> Set[str]:
        return set(self._tag_index.get(tag, ()))

    def ids_by_type(self, memory_type: MemoryType) -> Set[str]:
        return set(self._type_index.get(MemoryType(memory_type).value, ()))

    def ids_between(self, since: Optional[float] = None, until: Optional[float] = None) -> Set[str]:
        lo = bisect.bisect_left(self._time, (-_INF if since is None else since, ""))
        hi = bisect.bisect_right(self._time, (_INF if until is None else until, "\U0010ffff"))
        return {i for _, i in self._time[lo:hi]}

    def _allowed(
        self,
        memory_types: Optional[Sequence[MemoryType]],
        tags: Optional[Sequence[str]],
        since: Optional[float],
        until: Optional[float],
    ) -> Optional[Set[str]]:
        allowed: Optional[Set[str]] = None

        def narrow(ids: Set[str]) -> None:
            nonlocal allowed
            allowed = ids if allowed is None else allowed & ids

        if memory_types:
            narrow(set().union(*(self._type_index.get(MemoryType(t).value, set()) for t in memory_types)))
        if tags:
            narrow(set().union(*(self._tag_index.get(t, set()) for t in tags)))
        if since is not None or until is not None:
            narrow(self.ids_between(since, until))
        return allowed

    async def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        memory_types: Optional[Sequence[MemoryType]] = None,
        tags: Optional[Sequence[str]] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """BM25 search. Returns ``[(memory_id, score)]`` sorted by descending score."""
        terms = list(dict.fromkeys(tokenize(query)))
        n = len(self._doc_len)
        if not terms or n == 0 or top_k <= 0:
            return []
        allowed = self._allowed(memory_types, tags, since, until)
        if allowed is not None and not allowed:
            return []
        avgdl = (self._total_len / n) or 1.0
        scores: Dict[str, float] = defaultdict(float)
        for term in terms:
            posting = self._postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tf in posting.items():
                if allowed is not None and doc_id not in allowed:
                    continue
                norm = tf + self.k1 * (1.0 - self.b + self.b * self._doc_len[doc_id] / avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1.0) / norm
        return heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
