"""
backend/app/core/vector_search.py

Vector embedding storage and similarity search for ZentraX AI.

Provides an async-first `VectorSearch` service that:
  - Generates embeddings via a pluggable `EmbeddingProvider`.
  - Persists vectors via a pluggable `VectorStore` backend (an in-memory
    reference implementation is included; swap in pgvector/Qdrant/Pinecone/
    Weaviate etc. by implementing the same interface).
  - Enforces per-namespace (tenant/user) isolation so one user's context
    can never leak into another user's similarity search results.
  - Exposes a clean, narrow surface (`store`, `query`, `delete`) intended
    to be called directly from the AI gateway or chat routers.

Security notes:
  - `namespace` is required on every call and is treated as the isolation
    boundary. Callers should derive it from the authenticated user/tenant,
    never from unauthenticated client input.
  - Raw vectors are never logged; only ids, namespaces, and scores are.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger("zentrax.vector_search")


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VectorRecord:
    id: str
    namespace: str
    vector: tuple[float, ...]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ScoredMatch:
    id: str
    text: str
    score: float  # cosine similarity, higher is more similar
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": round(self.score, 6),
            "metadata": self.metadata,
        }


class VectorSearchError(Exception):
    """Base error for vector search failures."""


class NamespaceRequiredError(VectorSearchError):
    """Raised when a call is made without a valid namespace (tenant/user id)."""


# --------------------------------------------------------------------------- #
# Pluggable embedding provider
# --------------------------------------------------------------------------- #

class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> tuple[float, ...]:
        ...

    async def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        ...


class CallableEmbeddingProvider:
    """
    Adapts a single async embedding function (e.g. a call to your model
    provider's embeddings endpoint) into the EmbeddingProvider interface.

    Example:
        async def embed_fn(text: str) -> list[float]:
            resp = await openai_client.embeddings.create(model="...", input=text)
            return resp.data[0].embedding

        provider = CallableEmbeddingProvider(embed_fn)
    """

    def __init__(self, embed_fn: Callable[[str], Awaitable[list[float]]]):
        self._embed_fn = embed_fn

    async def embed(self, text: str) -> tuple[float, ...]:
        vec = await self._embed_fn(text)
        return tuple(vec)

    async def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        results = await asyncio.gather(*(self.embed(t) for t in texts))
        return list(results)


# --------------------------------------------------------------------------- #
# Pluggable vector store backend
# --------------------------------------------------------------------------- #

class VectorStore(Protocol):
    async def upsert(self, record: VectorRecord) -> None:
        ...

    async def upsert_batch(self, records: list[VectorRecord]) -> None:
        ...

    async def query(
        self, namespace: str, vector: tuple[float, ...], top_k: int
    ) -> list[ScoredMatch]:
        ...

    async def delete(self, namespace: str, ids: list[str]) -> int:
        ...

    async def delete_namespace(self, namespace: str) -> int:
        ...


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise VectorSearchError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """
    Reference VectorStore implementation, suitable for local dev, tests, or
    small deployments. Not durable across restarts and O(n) per query —
    replace with pgvector/Qdrant/Pinecone/Weaviate for production scale by
    implementing the same `VectorStore` protocol.

    Data is partitioned by namespace at the dict level, so a query for one
    namespace can structurally never scan another namespace's vectors.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, VectorRecord]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, record: VectorRecord) -> None:
        async with self._lock:
            self._data.setdefault(record.namespace, {})[record.id] = record

    async def upsert_batch(self, records: list[VectorRecord]) -> None:
        async with self._lock:
            for record in records:
                self._data.setdefault(record.namespace, {})[record.id] = record

    async def query(
        self, namespace: str, vector: tuple[float, ...], top_k: int
    ) -> list[ScoredMatch]:
        async with self._lock:
            bucket = self._data.get(namespace, {})
            records = list(bucket.values())

        scored = [
            ScoredMatch(
                id=r.id,
                text=r.text,
                score=_cosine_similarity(vector, r.vector),
                metadata=r.metadata,
            )
            for r in records
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    async def delete(self, namespace: str, ids: list[str]) -> int:
        async with self._lock:
            bucket = self._data.get(namespace, {})
            deleted = 0
            for record_id in ids:
                if bucket.pop(record_id, None) is not None:
                    deleted += 1
            return deleted

    async def delete_namespace(self, namespace: str) -> int:
        async with self._lock:
            bucket = self._data.pop(namespace, {})
            return len(bucket)


# --------------------------------------------------------------------------- #
# Core service
# --------------------------------------------------------------------------- #

class VectorSearch:
    """
    Async vector embedding + similarity search service.

    Usage:
        vector_search = VectorSearch(
            embedding_provider=CallableEmbeddingProvider(embed_fn),
            store=InMemoryVectorStore(),  # or your production backend
        )

        await vector_search.store(namespace=user_id, text="...", metadata={...})
        matches = await vector_search.query(namespace=user_id, text="...", top_k=5)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: Optional[VectorStore] = None,
        default_top_k: int = 5,
        min_score: float = 0.0,
    ):
        self._embeddings = embedding_provider
        self._store = store or InMemoryVectorStore()
        self._default_top_k = default_top_k
        self._min_score = min_score

    async def store(
        self,
        *,
        namespace: str,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        record_id: Optional[str] = None,
    ) -> str:
        """Embed and persist a single piece of text. Returns the record id."""
        self._require_namespace(namespace)
        if not text or not text.strip():
            raise VectorSearchError("Cannot store empty text")

        vector = await self._embeddings.embed(text)
        record = VectorRecord(
            id=record_id or str(uuid.uuid4()),
            namespace=namespace,
            vector=vector,
            text=text,
            metadata=metadata or {},
        )
        await self._store.upsert(record)
        logger.info("Stored vector id=%s namespace=%s", record.id, namespace)
        return record.id

    async def store_batch(
        self,
        *,
        namespace: str,
        items: list[dict[str, Any]],
    ) -> list[str]:
        """
        Embed and persist multiple items concurrently.
        Each item: {"text": str, "metadata": dict (optional), "id": str (optional)}
        """
        self._require_namespace(namespace)
        if not items:
            return []

        texts = [item["text"] for item in items]
        vectors = await self._embeddings.embed_batch(texts)

        records = [
            VectorRecord(
                id=item.get("id") or str(uuid.uuid4()),
                namespace=namespace,
                vector=vector,
                text=item["text"],
                metadata=item.get("metadata") or {},
            )
            for item, vector in zip(items, vectors)
        ]
        await self._store.upsert_batch(records)
        logger.info("Stored %d vectors in namespace=%s", len(records), namespace)
        return [r.id for r in records]

    async def query(
        self,
        *,
        namespace: str,
        text: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[ScoredMatch]:
        """Embed the query text and return the most similar stored records."""
        self._require_namespace(namespace)
        if not text or not text.strip():
            raise VectorSearchError("Cannot query with empty text")

        vector = await self._embeddings.embed(text)
        matches = await self._store.query(namespace, vector, top_k or self._default_top_k)

        threshold = self._min_score if min_score is None else min_score
        filtered = [m for m in matches if m.score >= threshold]
        return filtered

    async def query_many(
        self,
        *,
        namespace: str,
        texts: list[str],
        top_k: Optional[int] = None,
    ) -> list[list[ScoredMatch]]:
        """Run multiple queries concurrently against the same namespace."""
        self._require_namespace(namespace)
        return await asyncio.gather(
            *(self.query(namespace=namespace, text=t, top_k=top_k) for t in texts)
        )

    async def delete(self, *, namespace: str, ids: list[str]) -> int:
        self._require_namespace(namespace)
        return await self._store.delete(namespace, ids)

    async def delete_namespace(self, *, namespace: str) -> int:
        """Purge all vectors for a namespace (e.g. on account/user deletion)."""
        self._require_namespace(namespace)
        count = await self._store.delete_namespace(namespace)
        logger.info("Deleted namespace=%s (%d records)", namespace, count)
        return count

    @staticmethod
    def _require_namespace(namespace: str) -> None:
        if not namespace or not namespace.strip():
            raise NamespaceRequiredError(
                "A non-empty namespace (tenant/user id) is required for vector operations"
            )


# --------------------------------------------------------------------------- #
# Integration helpers
# --------------------------------------------------------------------------- #

_service_instance: Optional[VectorSearch] = None


def configure_vector_search(embedding_provider: EmbeddingProvider, store: Optional[VectorStore] = None) -> VectorSearch:
    """
    Call once at app startup (e.g. in a FastAPI lifespan/startup handler) to
    wire in your real embedding provider and store backend.
    """
    global _service_instance
    _service_instance = VectorSearch(embedding_provider=embedding_provider, store=store)
    return _service_instance


def get_vector_search() -> VectorSearch:
    """
    FastAPI dependency provider.

    Example:
        from fastapi import Depends
        from backend.app.core.vector_search import get_vector_search, VectorSearch

        @router.post("/chat")
        async def chat(
            payload: ChatRequest,
            current_user: User = Depends(get_current_user),
            vector_search: VectorSearch = Depends(get_vector_search),
        ):
            context = await vector_search.query(
                namespace=current_user.id,
                text=payload.message,
                top_k=5,
            )
            ...
    """
    if _service_instance is None:
        raise VectorSearchError(
            "Vector search not configured. Call configure_vector_search() during app startup."
        )
    return _service_instance
