"""SQLite-backed relational store (stdlib only) with an async facade.

A single connection is shared and guarded by a lock; every call runs in a
worker thread via ``asyncio.to_thread`` so the event loop is never blocked.
Holds the durable ``memories`` table and the ``relations`` (knowledge-graph
triples) table.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from array import array
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from ..memory_core import MemoryItem, MemoryType, StorageError

logger = logging.getLogger("zentrax.memory.relational_store")
T = TypeVar("T")

__all__ = ["RelationalStore"]

_COLS = "id, memory_type, content, importance, created_at, last_accessed, access_count, ttl, tags, metadata, source"
_ORDERS = {
    "created_at DESC", "created_at ASC", "last_accessed DESC", "last_accessed ASC",
    "importance DESC", "importance ASC", "access_count DESC", "access_count ASC",
}
_UPDATABLE = {"importance", "metadata", "tags", "access_count", "last_accessed", "ttl"}
_CHUNK = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    ttl REAL,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'unknown',
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_mem_type_created ON memories(memory_type, created_at);
CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_mem_accessed ON memories(memory_type, last_accessed);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE(subject, predicate, object)
);
CREATE INDEX IF NOT EXISTS idx_rel_subject ON relations(subject);
CREATE INDEX IF NOT EXISTS idx_rel_object ON relations(object);
"""

_UPSERT = """
INSERT INTO memories (id, memory_type, content, content_hash, importance, created_at, last_accessed,
                      access_count, ttl, tags, metadata, source, embedding)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(id) DO UPDATE SET
    memory_type=excluded.memory_type, content=excluded.content, content_hash=excluded.content_hash,
    importance=excluded.importance, created_at=excluded.created_at, last_accessed=excluded.last_accessed,
    access_count=excluded.access_count, ttl=excluded.ttl, tags=excluded.tags, metadata=excluded.metadata,
    source=excluded.source, embedding=COALESCE(excluded.embedding, memories.embedding)
"""


def _pack(vec: Optional[Sequence[float]]) -> Optional[bytes]:
    return array("f", vec).tobytes() if vec else None


def _unpack(blob: Optional[bytes]) -> Optional[List[float]]:
    if not blob:
        return None
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


def _row_to_item(row: Sequence[Any]) -> MemoryItem:
    return MemoryItem(
        id=row[0], memory_type=MemoryType(row[1]), content=row[2], importance=row[3],
        created_at=row[4], last_accessed=row[5], access_count=row[6], ttl=row[7],
        tags=json.loads(row[8] or "[]"), metadata=json.loads(row[9] or "{}"), source=row[10],
        embedding=_unpack(row[11]) if len(row) > 11 else None,
    )


def _chunks(seq: Sequence[T], size: int = _CHUNK) -> List[Sequence[T]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _like_pattern(tag: str) -> str:
    esc = json.dumps(tag).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{esc}%"


class RelationalStore:
    def __init__(self, path: Union[str, Path] = ":memory:") -> None:
        self.path = str(path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    # ------------------------------ plumbing ------------------------------- #
    async def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if self._conn is None:
            raise StorageError("RelationalStore is not initialised")

        def call() -> T:
            with self._lock:
                if self._conn is None:
                    raise StorageError("RelationalStore is closed")
                try:
                    return fn(self._conn, *args, **kwargs)
                except sqlite3.Error as exc:
                    raise StorageError(f"sqlite error: {exc}") from exc

        return await asyncio.to_thread(call)

    async def initialize(self) -> None:
        if self._conn is not None:
            return

        def setup() -> sqlite3.Connection:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False)
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.commit()
            return conn

        try:
            self._conn = await asyncio.to_thread(setup)
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open database {self.path}: {exc}") from exc

    async def close(self) -> None:
        if self._conn is None:
            return

        def shutdown() -> None:
            with self._lock:
                if self._conn is not None:
                    self._conn.commit()
                    self._conn.close()
                    self._conn = None

        await asyncio.to_thread(shutdown)

    # ------------------------------- memories ------------------------------ #
    async def upsert_memory(self, item: MemoryItem) -> None:
        await self.upsert_memories([item])

    async def upsert_memories(self, items: Sequence[MemoryItem]) -> None:
        if not items:
            return
        rows = [
            (
                i.id, i.memory_type.value, i.content, i.content_hash, i.importance, i.created_at,
                i.last_accessed, i.access_count, i.ttl, json.dumps(i.tags),
                json.dumps(i.metadata, default=str), i.source, _pack(i.embedding),
            )
            for i in items
        ]

        def op(conn: sqlite3.Connection) -> None:
            with conn:
                conn.executemany(_UPSERT, rows)

        await self._run(op)

    async def get_memory(self, memory_id: str, *, include_embedding: bool = False) -> Optional[MemoryItem]:
        found = await self.get_memories([memory_id], include_embedding=include_embedding)
        return found[0] if found else None

    async def get_memories(self, ids: Sequence[str], *, include_embedding: bool = False) -> List[MemoryItem]:
        """Fetch by id, preserving the order of ``ids`` and skipping missing ones."""
        if not ids:
            return []
        cols = _COLS + (", embedding" if include_embedding else "")

        def op(conn: sqlite3.Connection) -> List[MemoryItem]:
            found: Dict[str, MemoryItem] = {}
            for chunk in _chunks(list(ids)):
                marks = ",".join("?" * len(chunk))
                for row in conn.execute(f"SELECT {cols} FROM memories WHERE id IN ({marks})", list(chunk)):
                    found[row[0]] = _row_to_item(row)
            return [found[i] for i in ids if i in found]

        return await self._run(op)

    async def delete_memory(self, memory_id: str) -> bool:
        return (await self.delete_memories([memory_id])) > 0

    async def delete_memories(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0

        def op(conn: sqlite3.Connection) -> int:
            total = 0
            with conn:
                for chunk in _chunks(list(ids)):
                    marks = ",".join("?" * len(chunk))
                    total += conn.execute(f"DELETE FROM memories WHERE id IN ({marks})", list(chunk)).rowcount
            return total

        return await self._run(op)

    async def touch(self, memory_id: str) -> None:
        await self.touch_many([memory_id])

    async def touch_many(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        now = time.time()

        def op(conn: sqlite3.Connection) -> None:
            with conn:
                conn.executemany(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    [(now, i) for i in ids],
                )

        await self._run(op)

    async def update_memory(self, memory_id: str, **fields: Any) -> bool:
        """Patch whitelisted columns: importance, metadata, tags, access_count, last_accessed, ttl."""
        bad = set(fields) - _UPDATABLE
        if bad:
            raise ValueError(f"cannot update fields: {sorted(bad)}")
        if not fields:
            return False
        sets, params = [], []
        for key, val in fields.items():
            if key in ("metadata", "tags"):
                val = json.dumps(val, default=str)
            sets.append(f"{key} = ?")
            params.append(val)
        params.append(memory_id)

        def op(conn: sqlite3.Connection) -> bool:
            with conn:
                return conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params).rowcount > 0

        return await self._run(op)

    async def query_memories(
        self,
        memory_type: Optional[Union[MemoryType, str]] = None,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
        min_importance: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at DESC",
        include_embedding: bool = False,
    ) -> List[MemoryItem]:
        if order_by not in _ORDERS:
            raise ValueError(f"order_by must be one of {sorted(_ORDERS)}")
        where: List[str] = []
        params: List[Any] = []
        if memory_type is not None:
            where.append("memory_type = ?")
            params.append(MemoryType(memory_type).value)
        if since is not None:
            where.append("created_at >= ?")
            params.append(since)
        if until is not None:
            where.append("created_at <= ?")
            params.append(until)
        if min_importance is not None:
            where.append("importance >= ?")
            params.append(min_importance)
        if tags:
            # any-of semantics, consistent with the vector store / index filters
            where.append("(" + " OR ".join("tags LIKE ? ESCAPE '\\'" for _ in tags) + ")")
            params.extend(_like_pattern(t) for t in tags)
        cols = _COLS + (", embedding" if include_embedding else "")
        sql = f"SELECT {cols} FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([max(0, limit), max(0, offset)])

        def op(conn: sqlite3.Connection) -> List[MemoryItem]:
            return [_row_to_item(r) for r in conn.execute(sql, params)]

        return await self._run(op)

    async def find_by_hash(self, content_hash: str, memory_type: Optional[Union[MemoryType, str]] = None) -> Optional[MemoryItem]:
        sql = f"SELECT {_COLS} FROM memories WHERE content_hash = ?"
        params: List[Any] = [content_hash]
        if memory_type is not None:
            sql += " AND memory_type = ?"
            params.append(MemoryType(memory_type).value)
        sql += " LIMIT 1"

        def op(conn: sqlite3.Connection) -> Optional[MemoryItem]:
            row = conn.execute(sql, params).fetchone()
            return _row_to_item(row) if row else None

        return await self._run(op)

    async def find_expired_ids(self, now: Optional[float] = None, memory_type: Optional[Union[MemoryType, str]] = None) -> List[str]:
        now = time.time() if now is None else now
        sql = "SELECT id FROM memories WHERE ttl IS NOT NULL AND created_at + ttl < ?"
        params: List[Any] = [now]
        if memory_type is not None:
            sql += " AND memory_type = ?"
            params.append(MemoryType(memory_type).value)

        def op(conn: sqlite3.Connection) -> List[str]:
            return [r[0] for r in conn.execute(sql, params)]

        return await self._run(op)

    async def count(self, memory_type: Optional[Union[MemoryType, str]] = None) -> int:
        sql, params = "SELECT COUNT(*) FROM memories", []
        if memory_type is not None:
            sql += " WHERE memory_type = ?"
            params.append(MemoryType(memory_type).value)

        def op(conn: sqlite3.Connection) -> int:
            return int(conn.execute(sql, params).fetchone()[0])

        return await self._run(op)

    # ------------------------------- relations ------------------------------ #
    async def add_relation(
        self, subject: str, predicate: str, obj: str, weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        meta = json.dumps(metadata or {}, default=str)
        now = time.time()

        def op(conn: sqlite3.Connection) -> int:
            with conn:
                conn.execute(
                    """INSERT INTO relations (subject, predicate, object, weight, created_at, metadata)
                       VALUES (?,?,?,?,?,?)
                       ON CONFLICT(subject, predicate, object) DO UPDATE SET
                           weight = MAX(relations.weight, excluded.weight), metadata = excluded.metadata""",
                    (subject, predicate, obj, weight, now, meta),
                )
                row = conn.execute(
                    "SELECT id FROM relations WHERE subject=? AND predicate=? AND object=?", (subject, predicate, obj)
                ).fetchone()
            return int(row[0])

        return await self._run(op)

    async def get_relations(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        *,
        min_weight: float = 0.0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        where, params = ["weight >= ?"], [min_weight]
        for col, val in (("subject", subject), ("predicate", predicate), ("object", obj)):
            if val is not None:
                where.append(f"{col} = ?")
                params.append(val)
        sql = (
            "SELECT id, subject, predicate, object, weight, created_at, metadata FROM relations WHERE "
            + " AND ".join(where) + " ORDER BY weight DESC, id ASC LIMIT ?"
        )
        params.append(max(0, limit))

        def op(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            return [
                {"id": r[0], "subject": r[1], "predicate": r[2], "object": r[3], "weight": r[4],
                 "created_at": r[5], "metadata": json.loads(r[6] or "{}")}
                for r in conn.execute(sql, params)
            ]

        return await self._run(op)

    async def delete_relations(
        self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None
    ) -> int:
        where, params = [], []
        for col, val in (("subject", subject), ("predicate", predicate), ("object", obj)):
            if val is not None:
                where.append(f"{col} = ?")
                params.append(val)
        if not where:
            raise ValueError("delete_relations requires at least one filter")

        def op(conn: sqlite3.Connection) -> int:
            with conn:
                return conn.execute("DELETE FROM relations WHERE " + " AND ".join(where), params).rowcount

        return await self._run(op)

    async def count_relations(self) -> int:
        def op(conn: sqlite3.Connection) -> int:
            return int(conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0])

        return await self._run(op)
