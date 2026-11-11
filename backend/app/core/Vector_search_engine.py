"""
Vector_search_engine.py

A small, dependency-free semantic search pipeline with:
  1. Threshold & fallback  - never answers below a similarity cutoff
                             (protects against hallucinated answers).
  2. Separation of concerns - Embedder / VectorStore / VectorSearchEngine /
                             SuperIntelligenceEngine each have one job.
  3. Scalability           - the in-memory store and the toy embedder sit
                             behind interfaces, so a real embedding model or
                             an external vector DB (FAISS, Qdrant, pgvector,
                             Pinecone...) can be dropped in without touching
                             the engines.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

Vector = List[float]

FALLBACK_MESSAGE = "I don't know. I couldn't find reliable information on that."


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float


# --------------------------------------------------------------------------- #
# Embedding layer (swap for a real model in production)
# --------------------------------------------------------------------------- #
class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> Vector:
        """Convert text into a fixed-length, L2-normalised vector."""


class HashingEmbedder(Embedder):
    """
    Deterministic bag-of-words hashing embedder.

    It is a stand-in that simulates real embeddings without any external
    dependency. Replace it with e.g. sentence-transformers or an API-based
    embedder by implementing the same `embed` method.
    """

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> Vector:
        vec = [0.0] * self.dim
        for token in self._TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return self._normalise(vec)

    @staticmethod
    def _normalise(vec: Vector) -> Vector:
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


# --------------------------------------------------------------------------- #
# Storage layer (swap for an external vector DB in production)
# --------------------------------------------------------------------------- #
class VectorStore(ABC):
    @abstractmethod
    def add(self, document: Document, vector: Vector) -> None: ...

    @abstractmethod
    def query(self, vector: Vector, top_k: int) -> List[SearchResult]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine-similarity store held entirely in memory."""

    def __init__(self) -> None:
        self._items: List[tuple[Document, Vector]] = []

    def add(self, document: Document, vector: Vector) -> None:
        self._items.append((document, vector))

    def query(self, vector: Vector, top_k: int) -> List[SearchResult]:
        # Vectors are L2-normalised, so the dot product equals cosine similarity.
        scored = [
            SearchResult(doc, self._dot(vector, vec)) for doc, vec in self._items
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# Retrieval engine: knows HOW to search
# --------------------------------------------------------------------------- #
class VectorSearchEngine:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        threshold: float = 0.35,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.embedder = embedder or HashingEmbedder()
        self.store = store or InMemoryVectorStore()
        self.threshold = threshold

    def index(self, documents: Sequence[Document]) -> int:
        for doc in documents:
            self.store.add(doc, self.embedder.embed(doc.text))
        return len(documents)

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Return only results whose similarity >= threshold (may be empty)."""
        if not query.strip() or len(self.store) == 0:
            return []
        cutoff = self.threshold if threshold is None else threshold
        candidates = self.store.query(self.embedder.embed(query), top_k)
        return [r for r in candidates if r.score >= cutoff]


# --------------------------------------------------------------------------- #
# Answering engine: knows WHAT to say (and when to stay silent)
# --------------------------------------------------------------------------- #
class SuperIntelligenceEngine:
    def __init__(
        self,
        search_engine: VectorSearchEngine,
        fallback_message: str = FALLBACK_MESSAGE,
    ) -> None:
        self.search_engine = search_engine
        self.fallback_message = fallback_message

    def ask(self, question: str, top_k: int = 3) -> str:
        results = self.search_engine.search(question, top_k=top_k)
        if not results:
            # Nothing is similar enough: refuse instead of guessing.
            return self.fallback_message
        best = results[0]
        return f"{best.document.text}\n(confidence: {best.score:.2f}, source: {best.document.doc_id})"


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def build_demo_engine() -> SuperIntelligenceEngine:
    docs = [
        Document("kb-1", "Python is a high level programming language known for readability."),
        Document("kb-2", "FastAPI is a modern Python web framework for building APIs."),
        Document("kb-3", "Vector databases store embeddings and support similarity search."),
        Document("kb-4", "SQLite is a lightweight embedded relational database."),
    ]
    search = VectorSearchEngine(threshold=0.30)
    search.index(docs)
    return SuperIntelligenceEngine(search)


if __name__ == "__main__":
    engine = build_demo_engine()
    for q in [
        "What is FastAPI used for?",
        "How do vector databases do similarity search?",
        "Who won the 1998 football world cup?",  # should hit the fallback
    ]:
        print(f"Q: {q}\nA: {engine.ask(q)}\n")
