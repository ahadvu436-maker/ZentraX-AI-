"""SQLite-backed local storage: a durable outbox queue and a small TTL cache.

* Uses only the standard library (``sqlite3`` run in a worker thread via
  ``asyncio.to_thread``) so it never blocks the event loop.
* WAL mode + ``synchronous=FULL`` so queued requests survive crashes/power loss.
* Every outbox row carries an idempotency key; the cloud API should use it to
  de-duplicate replays (a request may be delivered twice if the response is lost).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key  TEXT    NOT NULL UNIQUE,
    operation        TEXT    NOT NULL,
    payload          TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'pending',  -- pending | in_flight | dead
    attempts         INTEGER NOT NULL DEFAULT 0,
    next_attempt_at  REAL    NOT NULL DEFAULT 0,
    last_error       TEXT,
    created_at       REAL    NOT NULL,
    updated_at       REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outbox_status_id ON outbox (status, id);

CREATE TABLE IF NOT EXISTS cache (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    expires_at  REAL,
    updated_at  REAL NOT NULL
);
"""


class LocalStoreError(Exception):
    """Raised for any local persistence failure."""


@dataclass(frozen=True)
class OutboxItem:
    id: int
    idempotency_key: str
    operation: str
    payload: dict[str, Any]
    attempts: int = 0
    next_attempt_at: float = 0.0
    created_at: float = 0.0


class LocalStore:
    def __init__(self, path: str | Path = "data/zentrax_local.db", max_attempts: int = 10) -> None:
        self._path = Path(path)
        self._max_attempts = max_attempts
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    async def open(self) -> None:
        if self._conn is not None:
            return
        await asyncio.to_thread(self._open)

    def _open(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(_SCHEMA)
            # Crash recovery: anything left in flight was never confirmed.
            conn.execute("UPDATE outbox SET status='pending' WHERE status='in_flight'")
            conn.commit()
            self._conn = conn
        except (sqlite3.Error, OSError) as exc:
            raise LocalStoreError(f"Cannot open local store at {self._path}: {exc}") from exc

    async def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            def _close() -> None:
                with self._lock:
                    conn.close()
            await asyncio.to_thread(_close)

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        conn = self._conn
        if conn is None:
            raise LocalStoreError("LocalStore is not open")

        def _call() -> T:
            with self._lock:
                try:
                    with conn:  # one transaction: commit on success, rollback on error
                        return fn(conn)
                except sqlite3.Error as exc:
                    raise LocalStoreError(str(exc)) from exc

        return await asyncio.to_thread(_call)

    # --------------------------------------------------------------- outbox
    async def enqueue(
        self, operation: str, payload: dict[str, Any], idempotency_key: str | None = None
    ) -> OutboxItem:
        """Persist a request. Re-enqueueing the same key returns the existing row."""
        key = idempotency_key or uuid.uuid4().hex
        try:
            body = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise LocalStoreError(f"Payload is not JSON serialisable: {exc}") from exc
        now = time.time()

        def _do(conn: sqlite3.Connection) -> OutboxItem:
            conn.execute(
                "INSERT INTO outbox (idempotency_key, operation, payload, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(idempotency_key) DO NOTHING",
                (key, operation, body, now, now),
            )
            row = conn.execute(
                "SELECT * FROM outbox WHERE idempotency_key = ?", (key,)
            ).fetchone()
            return self._to_item(row)

        return await self._run(_do)

    async def peek(self, limit: int = 50) -> list[OutboxItem]:
        """Oldest pending items first (FIFO)."""
        def _do(conn: sqlite3.Connection) -> list[OutboxItem]:
            rows = conn.execute(
                "SELECT * FROM outbox WHERE status='pending' ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
            return [self._to_item(r) for r in rows]

        return await self._run(_do)

    async def claim(self, item_id: int) -> bool:
        """Mark in-flight. False if someone else already took it."""
        def _do(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE outbox SET status='in_flight', updated_at=? WHERE id=? AND status='pending'",
                (time.time(), item_id),
            )
            return cur.rowcount == 1

        return await self._run(_do)

    async def release(self, item_id: int) -> None:
        """Put an in-flight item back to pending without counting an attempt."""
        await self._run(
            lambda c: c.execute(
                "UPDATE outbox SET status='pending', updated_at=? WHERE id=? AND status='in_flight'",
                (time.time(), item_id),
            )
        )

    async def complete(self, item_id: int) -> None:
        await self._run(lambda c: c.execute("DELETE FROM outbox WHERE id=?", (item_id,)))

    async def retry_later(self, item_id: int, error: str, delay: float) -> bool:
        """Record a failed attempt. Returns True if the item became dead."""
        now = time.time()

        def _do(conn: sqlite3.Connection) -> bool:
            row = conn.execute("SELECT attempts FROM outbox WHERE id=?", (item_id,)).fetchone()
            if row is None:
                return False
            attempts = row["attempts"] + 1
            dead = attempts >= self._max_attempts
            conn.execute(
                "UPDATE outbox SET status=?, attempts=?, next_attempt_at=?, last_error=?, updated_at=? "
                "WHERE id=?",
                ("dead" if dead else "pending", attempts, now + delay, error[:1000], now, item_id),
            )
            return dead

        return await self._run(_do)

    async def mark_dead(self, item_id: int, error: str) -> None:
        now = time.time()
        await self._run(
            lambda c: c.execute(
                "UPDATE outbox SET status='dead', last_error=?, updated_at=? WHERE id=?",
                (error[:1000], now, item_id),
            )
        )

    async def backlog_count(self) -> int:
        """Items still waiting to reach the cloud (pending + in flight)."""
        return await self._run(
            lambda c: c.execute(
                "SELECT COUNT(*) FROM outbox WHERE status IN ('pending','in_flight')"
            ).fetchone()[0]
        )

    async def stats(self) -> dict[str, int]:
        def _do(conn: sqlite3.Connection) -> dict[str, int]:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM outbox GROUP BY status").fetchall()
            out = {"pending": 0, "in_flight": 0, "dead": 0}
            out.update({r["status"]: r["n"] for r in rows})
            return out

        return await self._run(_do)

    # ---------------------------------------------------------------- cache
    async def cache_set(self, key: str, value: Any, ttl: float | None = None) -> None:
        now = time.time()
        expires = now + ttl if ttl else None
        body = json.dumps(value, ensure_ascii=False)
        await self._run(
            lambda c: c.execute(
                "INSERT INTO cache (key, value, expires_at, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "expires_at=excluded.expires_at, updated_at=excluded.updated_at",
                (key, body, expires, now),
            )
        )

    async def cache_get(self, key: str, default: Any = None) -> Any:
        def _do(conn: sqlite3.Connection) -> Any:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return default
            if row["expires_at"] is not None and row["expires_at"] < time.time():
                conn.execute("DELETE FROM cache WHERE key=?", (key,))
                return default
            return json.loads(row["value"])

        return await self._run(_do)

    async def cache_delete(self, key: str) -> None:
        await self._run(lambda c: c.execute("DELETE FROM cache WHERE key=?", (key,)))

    async def purge_expired(self) -> int:
        return await self._run(
            lambda c: c.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?", (time.time(),)
            ).rowcount
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _to_item(row: sqlite3.Row) -> OutboxItem:
        return OutboxItem(
            id=row["id"],
            idempotency_key=row["idempotency_key"],
            operation=row["operation"],
            payload=json.loads(row["payload"]),
            attempts=row["attempts"],
            next_attempt_at=row["next_attempt_at"],
            created_at=row["created_at"],
        )
