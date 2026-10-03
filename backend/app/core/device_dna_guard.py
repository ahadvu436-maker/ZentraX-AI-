"""Device fingerprint guard: salted component hashes, fuzzy matching, persistent ban registry.

Only salted hashes are stored (never raw client signals). Bans support expiry, review notes and unban.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger("zentrax.dna")

# component -> weight (higher = more identifying / stable)
WEIGHTS: dict[str, float] = {
    "canvas": 3.0, "webgl": 3.0, "audio": 2.5, "fonts": 2.0, "hw": 2.0, "screen": 1.0, "tz": 0.5,
    "ja3": 2.5, "ua": 1.0, "lang": 0.5, "hdr_order": 1.0, "ch_ua": 1.0, "net": 0.8,
}
STRONG = ("canvas", "webgl", "audio", "ja3")
CLIENT_KEYS = ("canvas", "webgl", "audio", "fonts", "hw", "screen", "tz")


@dataclass
class Fingerprint:
    components: dict[str, str]
    stable_id: str


@dataclass
class BanEntry:
    stable_id: str
    components: dict[str, str]
    reason: str
    created: float
    expires: float = 0.0  # 0 = permanent
    notes: str = ""


class JsonBanStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def load(self) -> list[BanEntry]:
        if not self.path.exists():
            return []
        raw = await asyncio.to_thread(self.path.read_text)
        return [BanEntry(**d) for d in json.loads(raw or "[]")]

    async def save(self, entries: list[BanEntry]) -> None:
        async with self._lock:
            tmp = self.path.with_suffix(".tmp")
            data = json.dumps([asdict(e) for e in entries])

            def _w() -> None:
                tmp.write_text(data)
                os.replace(tmp, self.path)

            await asyncio.to_thread(_w)


class DeviceDNAGuard:
    def __init__(self, secret: bytes, store: JsonBanStore, *, match_threshold: float = 0.72, min_strong: int = 1) -> None:
        self._secret, self.store = secret, store
        self.threshold, self.min_strong = match_threshold, min_strong
        self._bans: dict[str, BanEntry] = {}
        self._index: dict[tuple[str, str], set[str]] = defaultdict(set)

    async def load(self) -> None:
        for e in await self.store.load():
            self._add_local(e)

    def _add_local(self, e: BanEntry) -> None:
        self._bans[e.stable_id] = e
        for k in STRONG:
            if k in e.components:
                self._index[(k, e.components[k])].add(e.stable_id)

    def _drop_local(self, sid: str) -> None:
        e = self._bans.pop(sid, None)
        if e:
            for k in STRONG:
                if k in e.components:
                    self._index[(k, e.components[k])].discard(sid)

    # -------------------------------------------------------------- fingerprint
    def _h(self, name: str, value: str) -> str:
        return hmac.new(self._secret, f"{name}|{value}".encode(), hashlib.sha256).hexdigest()[:16]

    @staticmethod
    def _net_prefix(ip: str) -> str:
        try:
            a = ipaddress.ip_address(ip)
            return str(ipaddress.ip_network(f"{ip}/{24 if a.version == 4 else 48}", strict=False))
        except ValueError:
            return ""

    def fingerprint(self, scope: Scope, client_signals: Optional[Mapping[str, str]] = None, ip: str = "") -> Fingerprint:
        h = {k.decode().lower(): v.decode("latin-1") for k, v in scope["headers"]}
        raw: dict[str, str] = {
            "ua": h.get("user-agent", ""),
            "lang": h.get("accept-language", ""),
            "ja3": h.get("x-ja3-hash", ""),
            "ch_ua": "|".join(h.get(k, "") for k in ("sec-ch-ua", "sec-ch-ua-platform", "sec-ch-ua-mobile")),
            "hdr_order": ",".join(k.decode().lower() for k, _ in scope["headers"] if k not in (b"cookie", b"content-length", b"host")),
            "net": self._net_prefix(ip or (scope.get("client") or ("", 0))[0]),
        }
        for k in CLIENT_KEYS:
            if client_signals and client_signals.get(k):
                raw[k] = str(client_signals[k])[:512]
        comps = {k: self._h(k, v) for k, v in raw.items() if v}
        core = sorted((k, v) for k, v in comps.items() if k in ("canvas", "webgl", "audio", "fonts", "hw", "ja3", "ua"))
        stable = hmac.new(self._secret, json.dumps(core).encode(), hashlib.sha256).hexdigest()[:32]
        return Fingerprint(comps, stable)

    @staticmethod
    def similarity(a: Mapping[str, str], b: Mapping[str, str]) -> tuple[float, int]:
        total = match = 0.0
        strong = 0
        for k, w in WEIGHTS.items():
            if k in a and k in b:
                total += w
                if a[k] == b[k]:
                    match += w
                    strong += k in STRONG
        return (match / total if total else 0.0), strong

    # ---------------------------------------------------------------------- ban
    async def ban(self, fp: Fingerprint, reason: str, ttl: Optional[int] = None, notes: str = "") -> BanEntry:
        now = time.time()
        e = BanEntry(fp.stable_id, fp.components, reason, now, now + ttl if ttl else 0.0, notes)
        self._add_local(e)
        await self.store.save(list(self._bans.values()))
        log.warning("device banned %s (%s)", e.stable_id, reason)
        return e

    async def unban(self, stable_id: str) -> bool:
        if stable_id not in self._bans:
            return False
        self._drop_local(stable_id)
        await self.store.save(list(self._bans.values()))
        return True

    def check(self, fp: Fingerprint) -> Optional[BanEntry]:
        now = time.time()
        hit = self._bans.get(fp.stable_id)
        if hit and (not hit.expires or hit.expires > now):
            return hit
        candidates: set[str] = set()
        for k in STRONG:
            if k in fp.components:
                candidates |= self._index.get((k, fp.components[k]), set())
        for sid in candidates:
            e = self._bans.get(sid)
            if not e:
                continue
            if e.expires and e.expires < now:
                self._drop_local(sid)
                continue
            sim, strong = self.similarity(fp.components, e.components)
            if sim >= self.threshold and strong >= self.min_strong:
                return e
        return None


class DeviceDNAMiddleware:
    """Reads optional client signals from header `X-Device-Signals` (JSON of pre-hashed values)."""

    def __init__(self, app: ASGIApp, guard: DeviceDNAGuard) -> None:
        self.app, self.guard = app, guard

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            signals: dict[str, str] = {}
            for k, v in scope["headers"]:
                if k == b"x-device-signals" and len(v) < 4096:
                    try:
                        parsed = json.loads(v)
                        signals = {a: str(b) for a, b in parsed.items() if a in CLIENT_KEYS} if isinstance(parsed, dict) else {}
                    except ValueError:
                        pass
            ip = scope.get("state", {}).get("client_ip", "")
            fp = self.guard.fingerprint(scope, signals, ip)
            scope.setdefault("state", {})["device_fp"] = fp
            if self.guard.check(fp):
                return await JSONResponse({"detail": "Forbidden"}, status_code=403)(scope, receive, send)
        await self.app(scope, receive, send)
