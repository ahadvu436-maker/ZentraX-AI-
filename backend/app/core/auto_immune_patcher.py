"""Self-healing quarantine: anomaly windows, circuit breaking, dependency isolation, integrity checks."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Iterable, Optional, TypeVar

from fastapi import HTTPException
from pydantic import ValidationError

log = logging.getLogger("zentrax.immune")
T = TypeVar("T")


class State(str, Enum):
    HEALTHY = "healthy"
    QUARANTINED = "quarantined"
    PROBING = "probing"


class ModuleQuarantined(RuntimeError):
    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"module '{name}' quarantined: {reason}")
        self.module, self.reason = name, reason


@dataclass
class ModulePolicy:
    window: float = 60.0
    max_anomalies: int = 5
    base_cooldown: float = 30.0
    max_cooldown: float = 900.0
    probe_successes: int = 3
    depends_on: tuple[str, ...] = ()
    fallback: Optional[Callable[..., Awaitable[Any]]] = None
    integrity_files: tuple[Path, ...] = ()
    anomaly_types: tuple[type[BaseException], ...] = (ValidationError, ValueError, KeyError, LookupError)


@dataclass
class _Module:
    name: str
    policy: ModulePolicy
    state: State = State.HEALTHY
    anomalies: Deque[float] = field(default_factory=deque)
    reason: str = ""
    until: float = 0.0
    strikes: int = 0
    probe_ok: int = 0
    baseline: dict[Path, str] = field(default_factory=dict)


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class ImmuneSystem:
    def __init__(self, on_event: Optional[Callable[[str, str, str], Awaitable[None]]] = None, scan_interval: float = 30.0) -> None:
        self._mods: dict[str, _Module] = {}
        self.on_event = on_event
        self.scan_interval = scan_interval
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ----------------------------------------------------------------- registry
    def register(self, name: str, policy: Optional[ModulePolicy] = None) -> None:
        pol = policy or ModulePolicy()
        m = _Module(name, pol)
        for f in pol.integrity_files:
            m.baseline[f] = _digest(f)
        self._mods[name] = m

    def status(self) -> dict[str, dict]:
        now = time.time()
        return {
            n: {"state": m.state.value, "reason": m.reason, "retry_in": max(0.0, m.until - now), "anomalies": len(m.anomalies)}
            for n, m in self._mods.items()
        }

    async def _emit(self, name: str, event: str, detail: str) -> None:
        log.warning("immune[%s] %s: %s", name, event, detail)
        if self.on_event:
            with contextlib.suppress(Exception):
                await self.on_event(name, event, detail)

    # --------------------------------------------------------------- transitions
    async def quarantine(self, name: str, reason: str, *, cascade: bool = True) -> None:
        async with self._lock:
            await self._quarantine_locked(name, reason, cascade, seen=set())

    async def _quarantine_locked(self, name: str, reason: str, cascade: bool, seen: set[str]) -> None:
        m = self._mods[name]
        if name in seen:
            return
        seen.add(name)
        if m.state is not State.QUARANTINED:
            m.strikes += 1
            m.state, m.reason = State.QUARANTINED, reason
            m.until = time.time() + min(m.policy.max_cooldown, m.policy.base_cooldown * 2 ** (m.strikes - 1))
            m.probe_ok = 0
            await self._emit(name, "quarantined", reason)
        if cascade:
            for other in self._mods.values():
                if name in other.policy.depends_on:
                    await self._quarantine_locked(other.name, f"dependency '{name}' quarantined", True, seen)

    async def release(self, name: str) -> None:
        async with self._lock:
            m = self._mods[name]
            m.state, m.reason, m.strikes, m.until = State.HEALTHY, "", 0, 0.0
            m.anomalies.clear()
            for f in m.policy.integrity_files:
                m.baseline[f] = _digest(f)
        await self._emit(name, "released", "manual")

    async def report_anomaly(self, name: str, kind: str, detail: str = "") -> None:
        m = self._mods[name]
        now = time.time()
        m.anomalies.append(now)
        while m.anomalies and now - m.anomalies[0] > m.policy.window:
            m.anomalies.popleft()
        if m.state is State.PROBING:
            await self.quarantine(name, f"probe failed: {kind} {detail}".strip())
        elif len(m.anomalies) >= m.policy.max_anomalies:
            await self.quarantine(name, f"{len(m.anomalies)} anomalies/{int(m.policy.window)}s (last: {kind} {detail})".strip())

    async def _success(self, name: str) -> None:
        m = self._mods[name]
        if m.state is State.PROBING:
            m.probe_ok += 1
            if m.probe_ok >= m.policy.probe_successes:
                m.state, m.reason, m.strikes = State.HEALTHY, "", 0
                m.anomalies.clear()
                await self._emit(name, "recovered", f"{m.probe_ok} clean probes")

    def _admit(self, m: _Module) -> None:
        if m.state is State.QUARANTINED:
            if time.time() >= m.until and all(self._mods[d].state is State.HEALTHY for d in m.policy.depends_on if d in self._mods):
                m.state, m.probe_ok = State.PROBING, 0
            else:
                raise ModuleQuarantined(m.name, m.reason)

    # ------------------------------------------------------------------ guarding
    def guard(self, name: str, *, validate: Optional[Callable[[Any], bool]] = None):
        def deco(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @functools.wraps(fn)
            async def wrapper(*a: Any, **kw: Any) -> T:
                m = self._mods[name]
                try:
                    self._admit(m)
                except ModuleQuarantined:
                    if m.policy.fallback:
                        return await m.policy.fallback(*a, **kw)
                    raise
                try:
                    result = await fn(*a, **kw)
                except m.policy.anomaly_types as e:
                    await self.report_anomaly(name, type(e).__name__, str(e)[:200])
                    raise
                if validate and not validate(result):
                    await self.report_anomaly(name, "invalid_result")
                    raise ValueError(f"{name}: result failed validation")
                await self._success(name)
                return result

            return wrapper

        return deco

    def gate(self, name: str) -> Callable[[], None]:
        """FastAPI dependency: Depends(immune.gate('billing'))."""

        def dep() -> None:
            try:
                self._admit(self._mods[name])
            except ModuleQuarantined as e:
                raise HTTPException(status_code=503, detail="Service temporarily isolated", headers={"Retry-After": "30"}) from e

        return dep

    # ---------------------------------------------------------------- integrity
    async def _scan(self) -> None:
        while True:
            await asyncio.sleep(self.scan_interval)
            for m in list(self._mods.values()):
                for f, base in m.baseline.items():
                    try:
                        cur = await asyncio.to_thread(_digest, f)
                    except OSError:
                        cur = "missing"
                    if cur != base and m.state is not State.QUARANTINED:
                        await self.quarantine(m.name, f"integrity mismatch: {f.name}")

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._scan(), name="immune-scan")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
