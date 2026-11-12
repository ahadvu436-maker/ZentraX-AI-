"""Hybrid retrieval: BM25 text ranking + entity matching + graph expansion."""
from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

from .knowledge_core import Chunk, KnowledgeError
from .knowledge_graph import KnowledgeGraph

__all__ = ["KnowledgeRetriever", "SearchResult", "tokenize"]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.&'\-]*")
_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have he her his how i if
in into is it its me my no not of on or our she so than that the their them then there these they
this to was we were what when where which who why will with would you your about after also any
all more most other some such only over under up out
""".split())


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    return [
        _stem(t) for t in _TOKEN_RE.findall(text.casefold())
        if len(t) > 1 and t not in _STOPWORDS
    ]


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    text_score: float = 0.0
    entity_score: float = 0.0
    matched_entities: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.id, "doc_id": self.chunk.doc_id, "uri": self.chunk.source_uri,
            "text": self.chunk.text, "score": round(self.score, 4),
            "text_score": round(self.text_score, 4), "entity_score": round(self.entity_score, 4),
            "matched_entities": self.matched_entities, "reasons": self.reasons,
        }


class KnowledgeRetriever:
    """In-memory inverted index (BM25) fused with knowledge-graph signals.

    Final score = normalised BM25 + ``entity_boost`` per query entity found in a
    chunk + ``graph_boost`` per graph neighbour of a query entity found in it.
    """

    def __init__(self, graph: KnowledgeGraph, *, k1: float = 1.5, b: float = 0.75,
                 entity_boost: float = 1.0, graph_boost: float = 0.4, max_ngram: int = 4) -> None:
        self.graph = graph
        self.k1, self.b = k1, b
        self.entity_boost, self.graph_boost, self.max_ngram = entity_boost, graph_boost, max_ngram
        self._lock = threading.RLock()
        self._chunks: Dict[str, Chunk] = {}
        self._postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._terms: Dict[str, List[str]] = {}
        self._lengths: Dict[str, int] = {}
        self._total_len = 0
        self._by_doc: Dict[str, Set[str]] = defaultdict(set)
        self._entity_chunks: Dict[str, Set[str]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------ #
    # Index maintenance
    # ------------------------------------------------------------------ #
    def index_chunk(self, chunk: Chunk) -> None:
        with self._lock:
            if chunk.id in self._chunks:
                self._remove_chunk(chunk.id)
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self._chunks[chunk.id] = chunk
            self._lengths[chunk.id] = len(tokens)
            self._total_len += len(tokens)
            self._terms[chunk.id] = list(counts)
            for term, freq in counts.items():
                self._postings[term][chunk.id] = freq
            self._by_doc[chunk.doc_id].add(chunk.id)
            for eid in chunk.entity_ids:
                self._entity_chunks[eid].add(chunk.id)

    def _remove_chunk(self, chunk_id: str) -> None:
        chunk = self._chunks.pop(chunk_id, None)
        if chunk is None:
            return
        self._total_len -= self._lengths.pop(chunk_id, 0)
        for term in self._terms.pop(chunk_id, []):
            posting = self._postings.get(term)
            if posting is not None:
                posting.pop(chunk_id, None)
                if not posting:
                    del self._postings[term]
        self._by_doc[chunk.doc_id].discard(chunk_id)
        for eid in chunk.entity_ids:
            self._entity_chunks[eid].discard(chunk_id)

    def remove_document(self, doc_id: str) -> int:
        with self._lock:
            ids = list(self._by_doc.pop(doc_id, set()))
            for cid in ids:
                self._remove_chunk(cid)
            return len(ids)

    def remap_entity(self, old_id: str, new_id: str) -> None:
        """Re-point chunk entity references after an entity merge."""
        with self._lock:
            for cid in list(self._entity_chunks.pop(old_id, set())):
                chunk = self._chunks.get(cid)
                if chunk is None:
                    continue
                chunk.entity_ids = [new_id if e == old_id else e for e in chunk.entity_ids]
                chunk.entity_ids = list(dict.fromkeys(chunk.entity_ids))
                self._entity_chunks[new_id].add(cid)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            return self._chunks.get(chunk_id)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def _match_query_entities(self, query: str) -> List[str]:
        words = _WORD_RE.findall(query)
        used: Set[int] = set()
        matched: List[str] = []
        for n in range(min(self.max_ngram, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                if any(j in used for j in range(i, i + n)):
                    continue
                phrase = " ".join(words[i:i + n])
                if n == 1 and (len(phrase) < 3 or phrase.casefold() in _STOPWORDS):
                    continue
                entities = self.graph.find_entities(phrase)
                if entities:
                    used.update(range(i, i + n))
                    matched.extend(e.id for e in entities)
        return list(dict.fromkeys(matched))

    def search(self, query: str, top_k: int = 5, expand_graph: bool = True,
               filters: Optional[Mapping[str, Any]] = None) -> List[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        with self._lock:
            n_docs = len(self._chunks)
            if n_docs == 0 or not query.strip():
                return []
            avg_len = (self._total_len / n_docs) or 1.0
            text_scores: Dict[str, float] = defaultdict(float)
            for term in set(tokenize(query)):
                posting = self._postings.get(term)
                if not posting:
                    continue
                idf = math.log(1.0 + (n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
                for cid, tf in posting.items():
                    norm = tf + self.k1 * (1 - self.b + self.b * self._lengths[cid] / avg_len)
                    text_scores[cid] += idf * tf * (self.k1 + 1) / norm
            max_text = max(text_scores.values(), default=0.0)

            matched = self._match_query_entities(query)
            ent_scores: Dict[str, float] = defaultdict(float)
            ent_hits: Dict[str, List[str]] = defaultdict(list)
            reasons: Dict[str, List[str]] = defaultdict(list)
            for eid in matched:
                entity = self.graph.get_entity(eid)
                label = entity.name if entity else eid
                for cid in self._entity_chunks.get(eid, ()):
                    ent_scores[cid] += self.entity_boost
                    ent_hits[cid].append(label)
                    reasons[cid].append(f"mentions query entity '{label}'")
            if expand_graph:
                for eid in matched:
                    for nb in self.graph.neighbors(eid):
                        for cid in self._entity_chunks.get(nb.id, ()):
                            ent_scores[cid] += self.graph_boost
                            reasons[cid].append(f"mentions '{nb.name}', related to a query entity")

            results: List[SearchResult] = []
            for cid in set(text_scores) | set(ent_scores):
                chunk = self._chunks[cid]
                if filters and any(chunk.metadata.get(k) != v for k, v in filters.items()):
                    continue
                t = text_scores.get(cid, 0.0) / max_text if max_text else 0.0
                e = ent_scores.get(cid, 0.0)
                if t > 0:
                    reasons[cid].insert(0, "keyword match")
                results.append(SearchResult(
                    chunk=chunk, score=t + e, text_score=t, entity_score=e,
                    matched_entities=list(dict.fromkeys(ent_hits.get(cid, []))),
                    reasons=reasons.get(cid, []),
                ))
        results.sort(key=lambda r: (-r.score, r.chunk.id))
        return results[:top_k]

    # ------------------------------------------------------------------ #
    # Context & profiles
    # ------------------------------------------------------------------ #
    def build_context(self, query: str, top_k: int = 5, max_chars: int = 4000) -> str:
        """Numbered, source-attributed context block for downstream reasoning/LLMs."""
        parts: List[str] = []
        used = 0
        for i, res in enumerate(self.search(query, top_k=top_k), 1):
            block = f"[{i}] ({res.chunk.source_uri})\n{res.chunk.text.strip()}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts)

    def entity_profile(self, entity_id: str) -> Dict[str, Any]:
        entity = self.graph.get_entity(entity_id)
        if entity is None:
            raise KnowledgeError(f"Unknown entity {entity_id!r}")
        relations = []
        for rel in self.graph.relationships_of(entity_id):
            outgoing = rel.source_id == entity_id
            other = self.graph.get_entity(rel.target_id if outgoing else rel.source_id)
            relations.append({
                "relation": rel.type, "direction": "out" if outgoing else "in",
                "other": other.name if other else None,
                "other_id": other.id if other else None, "confidence": rel.confidence,
            })
        with self._lock:
            chunk_ids = sorted(self._entity_chunks.get(entity_id, ()))
        return {
            "entity": {"id": entity.id, "name": entity.name, "type": entity.type,
                       "aliases": entity.aliases, "confidence": entity.confidence,
                       "properties": entity.properties},
            "relations": relations,
            "chunk_ids": chunk_ids,
        }

    # ------------------------------------------------------------------ #
    # Persistence (index is rebuilt from chunks on load)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"chunks": [c.to_dict() for c in self._chunks.values()]}

    def load_dict(self, data: Mapping[str, Any]) -> None:
        with self._lock:
            for cid in list(self._chunks):
                self._remove_chunk(cid)
            self._by_doc.clear()
            self._entity_chunks.clear()
            for raw in data.get("chunks", []):
                self.index_chunk(Chunk.from_dict(raw))
