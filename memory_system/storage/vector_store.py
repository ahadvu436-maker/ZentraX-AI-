"""Vector storage: abstract interface + a persistent in-process implementation.

``InMemoryVectorStore`` does exact cosine search (numpy-accelerated when
available), supports payload filtering and atomic JSONL snapshots. Implement
``VectorStore`` to plug in FAISS / Qdrant / pgvector for larger deployments.
"""
from __future__ import annotations

import abc
import asyncio
import heapq
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..memory_core import StorageError

try:  # optional acceleration
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

logger = logging.getLogger("zentrax.memory.vector_store")

Vector = List[float]
Filter = Mapping[str, Any]

__all__ = ["VectorHit", "VectorStore", "InMemoryVectorStore"]


@dataclass
class VectorHit:
    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[Vector] = None


def _match(payload: Mapping[str, Any], flt: Optional[Filter]) -> bool:
    """Equality filter; a list/set condition means 'any of'; list payloads match on overlap."""
    if not flt:
        return True
    for key, cond in flt.items():
        val = payload.get(key)
        cond_multi = isinstance(cond, (list, tuple, set, frozenset))
        val_multi = isinstance(val, (list, tuple, set, frozenset))
        if cond_multi and val_multi:
            ok = bool(set(cond) & set(val))
        elif cond_multi:
            ok = val in cond
        elif val_multi:
            ok = cond in val
        else:
            ok = val == cond
        if not ok:
            return False
    return True


class VectorStore(abc.ABC):
    """Async vector-store interface."""

    dim: int

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def flush(self) -> None: ...

    @abc.abstractmethod
    async def upsert(self, id: str, vector: Sequence[float], payload: Optional[Dict[str, Any]] = None) -> None: ...

    async def upsert_many(self, entries: Sequence[Tuple[str, Sequence[float], Optional[Dict[str, Any]]]]) -> None:
        for id_, vec, payload in entries:
            await self.upsert(id_, vec, payload)

    @abc.abstractmethod
    async def delete(self, id: str) -> bool: ...

    @abc.abstractmethod
    async def get(self, id: str) -> Optional[Tuple[Vector, Dict[str, Any]]]: ...

    @abc.abstractmethod
    async def search(
        self, vector: Sequence[float], top_k: int = 5, filter: Optional[Filter] = None, min_score: float = -1.0
    ) -> List[VectorHit]: ...

    @abc.abstractmethod
    async def count(self, filter: Optional[Filter] = None) -> int: ...

    @abc.abstractmethod
    async def clear(self) -> None: ...


class InMemoryVectorStore(VectorStore):
    def __init__(self, dim: int, persist_path: Optional[Union[str, Path]] = None) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.persist_path = Path(persist_path) if persist_path else None
        self._vectors: Dict[str, Any] = {}
        self._payloads: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._ids: List[str] = []
        self._matrix: Any = None
        self._matrix_valid = False

    # ------------------------------- helpers ------------------------------- #
    def _prepare(self, vector: Sequence[float]) -> Any:
        if len(vector) != self.dim:
            raise StorageError(f"vector dim {len(vector)} != store dim {self.dim}")
        norm = math.sqrt(sum(x * x for x in vector))
        scaled = [x / norm for x in vector] if norm > 0 else list(vector)
        return np.asarray(scaled, dtype=np.float32) if np is not None else scaled

    @staticmethod
    def _as_list(vec: Any) -> Vector:
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def _ensure_matrix(self) -> None:
        if self._matrix_valid:
            return
        self._ids = list(self._vectors.keys())
        if self._ids:
            self._matrix = np.stack([self._vectors[i] for i in self._ids])
        else:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._matrix_valid = True

    # -------------------------------- writes -------------------------------- #
    async def upsert(self, id: str, vector: Sequence[float], payload: Optional[Dict[str, Any]] = None) -> None:
        self._vectors[id] = self._prepare(vector)
        self._payloads[id] = dict(payload or {})
        self._matrix_valid = False
        self._dirty = True

    async def delete(self, id: str) -> bool:
        existed = self._vectors.pop(id, None) is not None
        self._payloads.pop(id, None)
        if existed:
            self._matrix_valid = False
            self._dirty = True
        return existed

    async def clear(self) -> None:
        self._vectors.clear()
        self._payloads.clear()
        self._matrix_valid = False
        self._dirty = True

    # -------------------------------- reads --------------------------------- #
    async def get(self, id: str) -> Optional[Tuple[Vector, Dict[str, Any]]]:
        vec = self._vectors.get(id)
        return None if vec is None else (self._as_list(vec), dict(self._payloads[id]))

    async def count(self, filter: Optional[Filter] = None) -> int:
        if not filter:
            return len(self._vectors)
        return sum(1 for p in self._payloads.values() if _match(p, filter))

    async def search(
        self, vector: Sequence[float], top_k: int = 5, filter: Optional[Filter] = None, min_score: float = -1.0
    ) -> List[VectorHit]:
        if top_k <= 0 or not self._vectors:
            return []
        q = self._prepare(vector)
        pairs: List[Tuple[float, str]]
        if np is not None:
            self._ensure_matrix()
            scores = self._matrix @ q
            if filter:
                mask = np.fromiter((_match(self._payloads[i], filter) for i in self._ids), dtype=bool, count=len(self._ids))
                scores = np.where(mask, scores, -np.inf)
            k = min(top_k, len(self._ids))
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            pairs = [(float(scores[i]), self._ids[i]) for i in idx if np.isfinite(scores[i])]
        else:
            def gen() -> Any:
                for id_, vec in self._vectors.items():
                    if _match(self._payloads[id_], filter):
                        yield sum(a * b for a, b in zip(q, vec)), id_

            pairs = heapq.nlargest(top_k, gen())
        return [
            VectorHit(id_, score, dict(self._payloads[id_]), self._as_list(self._vectors[id_]))
            for score, id_ in pairs
            if score >= min_score
        ]

    # ------------------------------ persistence ----------------------------- #
    async def initialize(self) -> None:
        if self.persist_path is None or not self.persist_path.exists():
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        assert self.persist_path is not None
        with open(self.persist_path, "r", encoding="utf-8") as fh:
            header = json.loads(fh.readline() or "{}")
            if header.get("dim") != self.dim:
                raise StorageError(
                    f"persisted vector dim {header.get('dim')} != configured dim {self.dim}; "
                    "use a fresh data_dir or the original embedder"
                )
            loaded = 0
            for line_no, line in enumerate(fh, start=2):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._vectors[row["id"]] = self._prepare(row["v"])
                    self._payloads[row["id"]] = row.get("p") or {}
                    loaded += 1
                except (ValueError, KeyError, StorageError):
                    logger.warning("skipping corrupt vector row at line %d", line_no)
        self._matrix_valid = False
        logger.info("loaded %d vectors from %s", loaded, self.persist_path)

    async def flush(self) -> None:
        if self.persist_path is None or not self._dirty:
            return
        snapshot = [(i, self._as_list(v), dict(self._payloads[i])) for i, v in self._vectors.items()]
        self._dirty = False
        try:
            await asyncio.to_thread(self._write_sync, snapshot)
        except Exception as exc:
            self._dirty = True
            raise StorageError(f"vector snapshot failed: {exc}") from exc

    def _write_sync(self, snapshot: List[Tuple[str, Vector, Dict[str, Any]]]) -> None:
        assert self.persist_path is not None
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"dim": self.dim, "version": 1}) + "\n")
            for id_, vec, payload in snapshot:
                row = {"id": id_, "v": [round(x, 6) for x in vec], "p": payload}
                fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.persist_path)

    async def close(self) -> None:
        await self.flush()
