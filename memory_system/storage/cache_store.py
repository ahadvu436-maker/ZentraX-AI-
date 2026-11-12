"""In-process async cache with LRU eviction, per-entry TTL and single-flight loading.

Methods are ``async`` so the class can later be swapped for a networked
backend (e.g. Redis) without touching callers. Because the in-memory
operations never ``await`` mid-mutation, they are atomic under asyncio.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypeVar

T = TypeVar("T")
_MISSING: Any = object()

__all__ = ["CacheStats", "CacheStore"]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "hits": self.hits, "misses": self.misses, "sets": self.sets,
            "evictions": self.evictions, "expirations": self.expirations,
            "hit_rate": round(self.hit_rate, 4),
        }


class CacheStore:
    def __init__(
        self,
        max_size: int = 1024,
        default_ttl: Optional[float] = 60.0,
        namespace: str = "default",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.namespace = namespace
        self._clock = clock
        self._data: "OrderedDict[str, Tuple[Any, Optional[float]]]" = OrderedDict()
        self._inflight: Dict[str, "asyncio.Future[Any]"] = {}
        self.stats = CacheStats()

    def _k(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def __len__(self) -> int:
        return len(self._data)

    @property
    def size(self) -> int:
        return len(self._data)

    async def get(self, key: str, default: Any = None) -> Any:
        k = self._k(key)
        entry = self._data.get(k)
        if entry is None:
            self.stats.misses += 1
            return default
        value, expires = entry
        if expires is not None and self._clock() >= expires:
            del self._data[k]
            self.stats.expirations += 1
            self.stats.misses += 1
            return default
        self._data.move_to_end(k)
        self.stats.hits += 1
        return value

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store ``value``. ``ttl=None`` uses the default TTL (None default = no expiry)."""
        ttl = self.default_ttl if ttl is None else ttl
        expires = self._clock() + ttl if ttl is not None and ttl > 0 else None
        k = self._k(key)
        self._data[k] = (value, expires)
        self._data.move_to_end(k)
        self.stats.sets += 1
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)
            self.stats.evictions += 1

    async def delete(self, key: str) -> bool:
        return self._data.pop(self._k(key), None) is not None

    async def exists(self, key: str) -> bool:
        return await self.get(key, _MISSING) is not _MISSING

    async def clear(self) -> None:
        self._data.clear()

    async def invalidate_prefix(self, prefix: str) -> int:
        full = self._k(prefix)
        doomed = [k for k in self._data if k.startswith(full)]
        for k in doomed:
            del self._data[k]
        return len(doomed)

    async def purge_expired(self) -> int:
        now = self._clock()
        doomed = [k for k, (_, exp) in self._data.items() if exp is not None and now >= exp]
        for k in doomed:
            del self._data[k]
        self.stats.expirations += len(doomed)
        return len(doomed)

    async def get_or_set(self, key: str, factory: Callable[[], Awaitable[T]], ttl: Optional[float] = None) -> T:
        """Return the cached value or compute it once, even under concurrent callers."""
        value = await self.get(key, _MISSING)
        if value is not _MISSING:
            return value  # type: ignore[no-any-return]
        k = self._k(key)
        pending = self._inflight.get(k)
        if pending is not None:
            return await asyncio.shield(pending)  # type: ignore[no-any-return]
        future: "asyncio.Future[Any]" = asyncio.get_running_loop().create_future()
        self._inflight[k] = future
        try:
            result = await factory()
            await self.set(key, result, ttl)
            future.set_result(result)
            return result
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved: avoids "never retrieved" warnings
            raise
        finally:
            self._inflight.pop(k, None)
