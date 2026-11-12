"""Hybrid retrieval across all memory systems.

Pipeline: embed query -> per-source vector search (concurrent) -> BM25 lexical
boost -> fuse with recency + importance -> optional MMR diversification ->
rehearsal (touch winners) -> short-TTL result cache.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Union

from .memory_core import (
    BaseMemory, Embedder, MemoryItem, MemoryType, ScoredMemory, Vector, cosine_similarity, tokenize,
)
from .memory_index import MemoryIndex
from .storage.cache_store import CacheStore

logger = logging.getLogger("zentrax.memory.retrieval")

_PERSISTENT = {MemoryType.LONG_TERM, MemoryType.EPISODIC, MemoryType.SEMANTIC}

__all__ = ["RetrievalWeights", "RetrievalQuery", "MemoryRetriever"]


@dataclass(frozen=True)
class RetrievalWeights:
    semantic: float = 0.55
    lexical: float = 0.20
    recency: float = 0.15
    importance: float = 0.10

    @property
    def total(self) -> float:
        return (self.semantic + self.lexical + self.recency + self.importance) or 1.0


@dataclass
class RetrievalQuery:
    text: str
    top_k: int = 5
    memory_types: Optional[Sequence[MemoryType]] = None
    tags: Optional[Sequence[str]] = None
    min_score: float = 0.0
    since: Optional[float] = None
    until: Optional[float] = None
    use_mmr: bool = True
    mmr_lambda: float = 0.7  # 1.0 = pure relevance, 0.0 = pure diversity
    weights: RetrievalWeights = field(default_factory=RetrievalWeights)
    touch: bool = True

    def cache_key(self) -> str:
        types = sorted(MemoryType(t).value for t in self.memory_types) if self.memory_types else None
        raw = repr((
            self.text.strip(), self.top_k, types, sorted(self.tags) if self.tags else None,
            self.min_score, self.since, self.until, self.use_mmr, self.mmr_lambda, self.weights, self.touch,
        ))
        return "q:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class _Candidate:
    item: MemoryItem
    source: MemoryType
    similarity: float = 0.0
    lexical: float = 0.0
    score: float = 0.0


class MemoryRetriever:
    def __init__(
        self,
        sources: Mapping[MemoryType, BaseMemory],
        embedder: Embedder,
        index: Optional[MemoryIndex] = None,
        cache: Optional[CacheStore] = None,
        recency_half_life: float = 86400.0,
        overfetch: int = 3,
    ) -> None:
        self.sources = dict(sources)
        self.embedder = embedder
        self.index = index
        self.cache = cache
        self.recency_half_life = recency_half_life
        self.overfetch = max(1, overfetch)

    async def invalidate(self) -> None:
        """Drop cached results (call after any write)."""
        if self.cache is not None:
            await self.cache.invalidate_prefix("q:")

    # --------------------------------- retrieve -------------------------------- #
    async def retrieve(self, query: Union[str, RetrievalQuery], **overrides: object) -> List[ScoredMemory]:
        if isinstance(query, str):
            query = RetrievalQuery(text=query, **{k: v for k, v in overrides.items() if v is not None})  # type: ignore[arg-type]
        elif overrides:
            query = dataclasses.replace(query, **overrides)  # type: ignore[arg-type]
        text = query.text.strip()
        if not text or query.top_k <= 0:
            return []

        key = query.cache_key()
        if self.cache is not None:
            cached = await self.cache.get(key)
            if cached is not None:
                return list(cached)

        types = [MemoryType(t) for t in (query.memory_types or self.sources.keys()) if MemoryType(t) in self.sources]
        qv = await self.embedder.embed(text)
        fetch_k = max(query.top_k * self.overfetch, query.top_k + 5)

        batches = await asyncio.gather(
            *(self.sources[t].search(text, fetch_k, query_vector=qv, tags=query.tags) for t in types),
            return_exceptions=True,
        )
        cands: Dict[str, _Candidate] = {}
        for mtype, batch in zip(types, batches):
            if isinstance(batch, BaseException):
                logger.warning("retrieval from %s failed: %r", mtype.value, batch)
                continue
            for hit in batch:
                sim = hit.components.get("similarity", hit.score)
                current = cands.get(hit.item.id)
                if current is None or sim > current.similarity:
                    cands[hit.item.id] = _Candidate(hit.item, mtype, sim)

        await self._add_lexical(cands, query, types, fetch_k)
        ranked = self._score(cands, query)
        if query.use_mmr and len(ranked) > query.top_k:
            ranked = await self._mmr(ranked, query)
        chosen = ranked[: query.top_k]

        if query.touch and chosen:
            await asyncio.gather(
                *(self.sources[c.source].mark_accessed(c.item.id) for c in chosen), return_exceptions=True
            )
        results = [
            ScoredMemory(
                item=c.item, score=c.score, source=c.source.value,
                components={"similarity": c.similarity, "lexical": c.lexical, "score": c.score},
            )
            for c in chosen
        ]
        if self.cache is not None:
            await self.cache.set(key, list(results))
        return results

    # ---------------------------------- internals ------------------------------- #
    async def _add_lexical(
        self, cands: Dict[str, _Candidate], query: RetrievalQuery, types: Sequence[MemoryType], fetch_k: int
    ) -> None:
        q_tokens = set(tokenize(query.text))
        persistent = [t for t in types if t in _PERSISTENT]
        lex: Dict[str, float] = {}
        if self.index is not None and persistent:
            pairs = await self.index.search(
                query.text, fetch_k, memory_types=persistent, tags=query.tags, since=query.since, until=query.until
            )
            if pairs:
                top = pairs[0][1] or 1.0
                lex = {doc_id: score / top for doc_id, score in pairs}

        for doc_id, score in lex.items():
            if doc_id in cands:
                cands[doc_id].lexical = score
                continue
            mtype_name = self.index.type_of(doc_id) if self.index else None
            if mtype_name is None:
                continue
            mtype = MemoryType(mtype_name)
            source = self.sources.get(mtype)
            item = await source.get(doc_id, touch=False) if source else None
            if item is None:
                continue
            emb = item.embedding or await self.embedder.embed(item.content)
            item.embedding = emb
            qv = await self.embedder.embed(query.text)
            cands[doc_id] = _Candidate(item, mtype, cosine_similarity(qv, emb), score)

        # Token-overlap fallback for items the BM25 index does not cover (STM / working memory).
        if q_tokens:
            for cand in cands.values():
                if cand.lexical == 0.0 and (self.index is None or cand.source not in _PERSISTENT):
                    doc = set(tokenize(cand.item.content))
                    cand.lexical = len(q_tokens & doc) / len(q_tokens)

    def _score(self, cands: Dict[str, _Candidate], query: RetrievalQuery) -> List[_Candidate]:
        w = query.weights
        now = time.time()
        wanted_tags = set(query.tags or ())
        kept: List[_Candidate] = []
        for cand in cands.values():
            item = cand.item
            if query.since is not None and item.created_at < query.since:
                continue
            if query.until is not None and item.created_at > query.until:
                continue
            if wanted_tags and not wanted_tags.intersection(item.tags):
                continue
            age = max(0.0, now - max(item.created_at, item.last_accessed))
            recency = 0.5 ** (age / self.recency_half_life)
            cand.score = (
                w.semantic * max(cand.similarity, 0.0) + w.lexical * cand.lexical
                + w.recency * recency + w.importance * item.importance
            ) / w.total
            if cand.score >= query.min_score:
                kept.append(cand)
        kept.sort(key=lambda c: c.score, reverse=True)
        return kept

    async def _mmr(self, ranked: List[_Candidate], query: RetrievalQuery) -> List[_Candidate]:
        """Maximal Marginal Relevance: trade relevance against redundancy."""
        pool = ranked[: max(query.top_k * 3, query.top_k + 5)]
        missing = [c for c in pool if c.item.embedding is None]
        if missing:
            for cand, vec in zip(missing, await self.embedder.embed_batch([c.item.content for c in missing])):
                cand.item.embedding = vec
        lam = min(1.0, max(0.0, query.mmr_lambda))
        selected: List[_Candidate] = []
        remaining = list(pool)
        while remaining and len(selected) < query.top_k:
            def mmr_score(c: _Candidate) -> float:
                redundancy = max((cosine_similarity(c.item.embedding, s.item.embedding) for s in selected), default=0.0)
                return lam * c.score - (1.0 - lam) * redundancy

            best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        return selected

    # ------------------------------------ context ------------------------------- #
    async def build_context(
        self, query: str, max_chars: int = 2000, top_k: int = 8, include_working: bool = True
    ) -> str:
        """Prompt-ready text: working-memory state followed by the best-matching memories."""
        parts: List[str] = []
        working = self.sources.get(MemoryType.WORKING)
        if include_working and working is not None and hasattr(working, "context"):
            ctx = await working.context(max_chars // 2)  # type: ignore[attr-defined]
            if ctx:
                parts.append("[working memory]\n" + ctx)
        hits = await self.retrieve(query, top_k=top_k, memory_types=[
            t for t in self.sources if t is not MemoryType.WORKING
        ])
        if hits:
            parts.append("[relevant memories]")
            parts.extend(f"- ({h.source}) {h.item.content}" for h in hits)
        text = "\n".join(parts)
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
