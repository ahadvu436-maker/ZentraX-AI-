"""ZentraX-AI :: 03_memory_system :: memory_core

Foundation of the cognitive memory architecture:

* shared data types (``MemoryItem``, ``ScoredMemory``, ``MemoryType``)
* ``MemoryConfig`` and the exception hierarchy
* the ``Embedder`` protocol plus dependency-free / optional embedders
* the ``BaseMemory`` contract every memory subsystem implements
* ``MemoryCore`` - the facade that wires every subsystem together

Requires Python 3.10+. Only the standard library is mandatory; ``numpy`` and
``sentence-transformers`` are optional accelerators.
"""
from __future__ import annotations

import abc
import asyncio
import contextlib
import hashlib
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids circular imports
    from .episodic_memory import EpisodicMemory
    from .long_term_memory import LongTermMemory
    from .memory_consolidation import ConsolidationReport, MemoryConsolidator
    from .memory_index import MemoryIndex
    from .memory_retrieval import MemoryRetriever, RetrievalQuery
    from .semantic_memory import SemanticMemory
    from .short_term_memory import ShortTermMemory
    from .storage.cache_store import CacheStore
    from .storage.relational_store import RelationalStore
    from .storage.vector_store import VectorStore
    from .working_memory import WorkingMemory

logger = logging.getLogger("zentrax.memory")

Vector = List[float]

__all__ = [
    "Vector", "MemoryType", "MemoryItem", "ScoredMemory", "MemoryConfig",
    "MemorySystemError", "MemoryNotFoundError", "MemoryCapacityError", "StorageError",
    "Embedder", "HashingEmbedder", "SentenceTransformerEmbedder", "BaseMemory", "MemoryCore",
    "clamp", "l2_normalize", "cosine_similarity", "tokenize", "content_hash",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class MemorySystemError(RuntimeError):
    """Base class for all memory-system errors."""


class MemoryNotFoundError(MemorySystemError):
    """Requested memory does not exist."""


class MemoryCapacityError(MemorySystemError):
    """A capacity-limited store cannot accept more items."""


class StorageError(MemorySystemError):
    """A storage backend failed."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be been but by for from has have he her his i in is it its "
    "me my of on or our she so that the their them they this to was we were what "
    "when where which who will with you your".split()
)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS] if drop_stopwords else tokens


def content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def l2_normalize(vec: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0.0 else list(vec)


def cosine_similarity(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryItem:
    """A single unit of memory, shared by every subsystem."""

    content: str
    memory_type: MemoryType = MemoryType.SHORT_TERM
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # seconds since created_at; None = never expires
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Vector] = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("MemoryItem.content must be a non-empty string")
        self.memory_type = MemoryType(self.memory_type)
        self.importance = clamp(self.importance)

    @property
    def content_hash(self) -> str:
        return content_hash(self.content)

    def touch(self, now: Optional[float] = None) -> None:
        self.last_accessed = now if now is not None else time.time()
        self.access_count += 1

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl is None:
            return False
        return (now if now is not None else time.time()) > self.created_at + self.ttl

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["memory_type"] = self.memory_type.value
        if not include_embedding:
            data["embedding"] = None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ScoredMemory:
    """A retrieval hit: the memory, its final score and score breakdown."""

    item: MemoryItem
    score: float
    source: str
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    data_dir: Optional[Union[str, Path]] = Path("./zentrax_data/memory")  # None => in-memory only
    stm_capacity: int = 200
    stm_ttl: Optional[float] = 1800.0
    wm_capacity: int = 7
    wm_half_life: float = 120.0
    cache_size: int = 2048
    cache_ttl: Optional[float] = 30.0
    auto_consolidate: bool = True
    consolidation_interval: float = 300.0
    promote_importance: float = 0.65
    promote_access_count: int = 3
    decay_half_life: float = 7 * 86400.0
    forget_threshold: float = 0.05
    protect_importance: float = 0.9
    cluster_similarity: float = 0.72
    min_cluster_size: int = 3
    duplicate_similarity: float = 0.97
    dedupe_scan_limit: int = 100
    recency_half_life: float = 86400.0

    def __post_init__(self) -> None:
        if self.data_dir is not None:
            self.data_dir = Path(self.data_dir)
        if self.stm_capacity < 1 or self.wm_capacity < 1:
            raise ValueError("capacities must be >= 1")
        if self.decay_half_life <= 0 or self.wm_half_life <= 0 or self.recency_half_life <= 0:
            raise ValueError("half-lives must be > 0")
        if self.consolidation_interval <= 0:
            raise ValueError("consolidation_interval must be > 0")

    @property
    def db_path(self) -> str:
        return str(Path(self.data_dir) / "memory.sqlite3") if self.data_dir else ":memory:"

    @property
    def vector_path(self) -> Optional[Path]:
        return Path(self.data_dir) / "vectors.jsonl" if self.data_dir else None


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    """Anything that turns text into fixed-size, L2-normalised vectors."""

    dim: int

    async def embed(self, text: str) -> Vector: ...

    async def embed_batch(self, texts: Sequence[str]) -> List[Vector]: ...


class HashingEmbedder:
    """Deterministic, dependency-free feature-hashing embedder (unigrams+bigrams).

    Captures lexical overlap only. It is a robust default for tests/offline use;
    swap in ``SentenceTransformerEmbedder`` or any ``Embedder`` for semantics.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def _embed_sync(self, text: str) -> Vector:
        tokens = tokenize(text, drop_stopwords=False)
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for feat in feats:
            h = int.from_bytes(hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return l2_normalize(vec)

    async def embed(self, text: str) -> Vector:
        return self._embed_sync(text)

    async def embed_batch(self, texts: Sequence[str]) -> List[Vector]:
        return [self._embed_sync(t) for t in texts]


class SentenceTransformerEmbedder:
    """Optional neural embedder (``pip install sentence-transformers``)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise MemorySystemError("pip install sentence-transformers to use this embedder") from exc
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    async def embed(self, text: str) -> Vector:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: Sequence[str]) -> List[Vector]:
        def run() -> List[Vector]:
            arr = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
            return [row.tolist() for row in arr]

        return await asyncio.to_thread(run)


# --------------------------------------------------------------------------- #
# BaseMemory contract
# --------------------------------------------------------------------------- #
class BaseMemory(abc.ABC):
    """Contract implemented by every memory subsystem."""

    memory_type: MemoryType

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    async def initialize(self) -> None:
        """Optional async setup hook."""

    async def close(self) -> None:
        """Optional async teardown hook."""

    @abc.abstractmethod
    async def add(self, item: MemoryItem) -> str: ...

    @abc.abstractmethod
    async def get(self, memory_id: str, *, touch: bool = True) -> Optional[MemoryItem]: ...

    @abc.abstractmethod
    async def delete(self, memory_id: str) -> bool: ...

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        query_vector: Optional[Vector] = None,
        tags: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
    ) -> List[ScoredMemory]: ...

    @abc.abstractmethod
    async def count(self) -> int: ...

    @abc.abstractmethod
    async def clear(self) -> None: ...

    async def mark_accessed(self, memory_id: str) -> None:
        """Record that a memory was used (rehearsal). Default: no-op."""


# --------------------------------------------------------------------------- #
# MemoryCore - orchestrator
# --------------------------------------------------------------------------- #
class MemoryCore:
    """Facade wiring STM, working, long-term, episodic and semantic memory
    together with retrieval, indexing, consolidation and storage.

    Usage::

        async with MemoryCore(MemoryConfig(data_dir=None)) as mem:
            await mem.remember("User prefers concise answers", importance=0.8)
            hits = await mem.recall("how should I answer?")
    """

    stm: "ShortTermMemory"
    working: "WorkingMemory"
    ltm: "LongTermMemory"
    episodic: "EpisodicMemory"
    semantic: "SemanticMemory"
    index: "MemoryIndex"
    retriever: "MemoryRetriever"
    consolidator: "MemoryConsolidator"
    vector_store: "VectorStore"
    relational_store: "RelationalStore"
    cache: "CacheStore"

    def __init__(self, config: Optional[MemoryConfig] = None, embedder: Optional[Embedder] = None) -> None:
        self.config = config or MemoryConfig()
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._task: Optional["asyncio.Task[None]"] = None

    # ------------------------------ lifecycle ------------------------------ #
    async def initialize(self) -> "MemoryCore":
        async with self._init_lock:
            if self._initialized:
                return self
            # Local imports: submodules import this module's base types.
            from .episodic_memory import EpisodicMemory
            from .long_term_memory import LongTermMemory
            from .memory_consolidation import MemoryConsolidator
            from .memory_index import MemoryIndex
            from .memory_retrieval import MemoryRetriever
            from .semantic_memory import SemanticMemory
            from .short_term_memory import ShortTermMemory
            from .storage.cache_store import CacheStore
            from .storage.relational_store import RelationalStore
            from .storage.vector_store import InMemoryVectorStore
            from .working_memory import WorkingMemory

            cfg = self.config
            if cfg.data_dir is not None:
                Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)

            self.vector_store = InMemoryVectorStore(self.embedder.dim, persist_path=cfg.vector_path)
            self.relational_store = RelationalStore(cfg.db_path)
            self.cache = CacheStore(max_size=cfg.cache_size, default_ttl=cfg.cache_ttl, namespace="retrieval")
            await self.vector_store.initialize()
            await self.relational_store.initialize()
            self.index = MemoryIndex()

            common: Dict[str, Any] = dict(
                vector_store=self.vector_store,
                relational_store=self.relational_store,
                index=self.index,
                forget_threshold=cfg.forget_threshold,
                protect_importance=cfg.protect_importance,
            )
            self.ltm = LongTermMemory(self.embedder, half_life=cfg.decay_half_life, **common)
            self.episodic = EpisodicMemory(self.embedder, half_life=cfg.decay_half_life, **common)
            self.semantic = SemanticMemory(self.embedder, half_life=cfg.decay_half_life * 4, **common)
            for mem in (self.ltm, self.episodic, self.semantic):
                await mem.initialize()

            self.stm = ShortTermMemory(self.embedder, capacity=cfg.stm_capacity, default_ttl=cfg.stm_ttl)
            self.working = WorkingMemory(
                self.embedder, capacity=cfg.wm_capacity, half_life=cfg.wm_half_life,
                on_evict=self._on_working_evict,
            )
            self.consolidator = MemoryConsolidator(
                stm=self.stm, working=self.working, ltm=self.ltm, episodic=self.episodic,
                semantic=self.semantic, embedder=self.embedder, config=cfg,
            )
            self.stm.add_eviction_callback(self.consolidator.handle_evicted)

            self.retriever = MemoryRetriever(
                sources={
                    MemoryType.WORKING: self.working,
                    MemoryType.SHORT_TERM: self.stm,
                    MemoryType.LONG_TERM: self.ltm,
                    MemoryType.EPISODIC: self.episodic,
                    MemoryType.SEMANTIC: self.semantic,
                },
                embedder=self.embedder,
                index=self.index,
                cache=self.cache,
                recency_half_life=cfg.recency_half_life,
            )
            self._initialized = True
            if cfg.auto_consolidate:
                self._task = asyncio.create_task(self._maintenance_loop(), name="zentrax-memory-maintenance")
            logger.info("MemoryCore initialised (dim=%d, persistent=%s)", self.embedder.dim, cfg.data_dir is not None)
            return self

    async def shutdown(self, *, final_consolidation: bool = True) -> None:
        async with self._init_lock:
            if not self._initialized:
                return
            if self._task is not None:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
                self._task = None
            if final_consolidation:
                try:
                    await self.consolidator.run()
                except Exception:  # noqa: BLE001
                    logger.exception("final consolidation failed")
            for mem in (self.stm, self.working, self.ltm, self.episodic, self.semantic):
                with contextlib.suppress(Exception):
                    await mem.close()
            await self.vector_store.close()
            await self.relational_store.close()
            self._initialized = False

    async def __aenter__(self) -> "MemoryCore":
        return await self.initialize()

    async def __aexit__(self, *exc: object) -> None:
        await self.shutdown()

    def _ensure_ready(self) -> None:
        if not self._initialized:
            raise MemorySystemError("MemoryCore is not initialised; await initialize() first")

    def _target(self, memory_type: MemoryType) -> BaseMemory:
        return {
            MemoryType.SHORT_TERM: self.stm,
            MemoryType.WORKING: self.working,
            MemoryType.LONG_TERM: self.ltm,
            MemoryType.EPISODIC: self.episodic,
            MemoryType.SEMANTIC: self.semantic,
        }[MemoryType(memory_type)]

    async def _on_working_evict(self, item: MemoryItem) -> None:
        # Working-only items fall back to short-term instead of vanishing.
        if item.memory_type is MemoryType.WORKING:
            await self.stm.add(item)

    async def _maintenance_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.consolidation_interval)
            try:
                await self.consolidate()
                await self.vector_store.flush()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("memory maintenance cycle failed")

    # ------------------------------ public API ----------------------------- #
    async def remember(
        self,
        content: str,
        *,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        importance: float = 0.5,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
        source: str = "agent",
        attend: bool = False,
    ) -> MemoryItem:
        """Store ``content`` in the chosen subsystem; optionally pin it to attention."""
        self._ensure_ready()
        item = MemoryItem(
            content=content, memory_type=memory_type, importance=importance,
            tags=list(tags or []), metadata=dict(metadata or {}), ttl=ttl, source=source,
        )
        await self._target(item.memory_type).add(item)
        if attend and item.memory_type is not MemoryType.WORKING:
            await self.working.attend(item)
        await self.retriever.invalidate()
        return item

    async def remember_event(self, content: str, **kwargs: Any) -> MemoryItem:
        """Record an episode (see ``EpisodicMemory.record_event``)."""
        self._ensure_ready()
        item = await self.episodic.record_event(content, **kwargs)
        await self.retriever.invalidate()
        return item

    async def learn_fact(self, statement: str, **kwargs: Any) -> MemoryItem:
        """Store a fact / triple (see ``SemanticMemory.add_fact``)."""
        self._ensure_ready()
        item = await self.semantic.add_fact(statement, **kwargs)
        await self.retriever.invalidate()
        return item

    async def recall(
        self,
        query: Union[str, "RetrievalQuery"],
        top_k: int = 5,
        *,
        memory_types: Optional[Sequence[MemoryType]] = None,
        tags: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[ScoredMemory]:
        """Hybrid retrieval across (a subset of) all memory systems."""
        self._ensure_ready()
        if isinstance(query, str):
            return await self.retriever.retrieve(query, top_k=top_k, memory_types=memory_types, tags=tags, **kwargs)
        return await self.retriever.retrieve(query)

    async def context(self, query: str, max_chars: int = 2000, top_k: int = 8) -> str:
        """Prompt-ready text: working-memory state + the most relevant memories."""
        self._ensure_ready()
        return await self.retriever.build_context(query, max_chars=max_chars, top_k=top_k)

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        self._ensure_ready()
        for mem in (self.working, self.stm, self.ltm, self.episodic, self.semantic):
            item = await mem.get(memory_id)
            if item is not None:
                return item
        return None

    async def forget(self, memory_id: str) -> bool:
        self._ensure_ready()
        results = [await m.delete(memory_id) for m in (self.working, self.stm, self.ltm, self.episodic, self.semantic)]
        await self.retriever.invalidate()
        return any(results)

    async def consolidate(self) -> "ConsolidationReport":
        self._ensure_ready()
        report = await self.consolidator.run()
        await self.retriever.invalidate()
        return report

    async def stats(self) -> Dict[str, Any]:
        self._ensure_ready()
        last = self.consolidator.last_report
        return {
            "short_term": await self.stm.count(),
            "working": await self.working.count(),
            "long_term": await self.ltm.count(),
            "episodic": await self.episodic.count(),
            "semantic": await self.semantic.count(),
            "relations": await self.relational_store.count_relations(),
            "vectors": await self.vector_store.count(),
            "index_docs": self.index.size,
            "cache": self.cache.stats.as_dict(),
            "last_consolidation": last.as_dict() if last else None,
        }
