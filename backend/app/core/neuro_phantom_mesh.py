"""neuro_phantom_mesh.py - Deception ("Hallucination Tunnel") middleware for FastAPI.

Detects scanner/probe traffic and serves plausible but entirely fabricated
success responses. No real data is ever touched or exposed.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import secrets
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Iterable, Optional, Pattern

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("zentrax.phantom_mesh")

_SQLI = re.compile(
    r"(?i)(\bunion\b.{0,20}\bselect\b|\bor\b\s+1\s*=\s*1|sleep\s*\(\s*\d+|benchmark\s*\(|"
    r"information_schema|--\s*$|;\s*drop\s+table|xp_cmdshell|load_file\s*\()"
)
_XSS = re.compile(r"(?i)(<\s*script|javascript:|onerror\s*=|onload\s*=|<\s*svg[^>]*on\w+\s*=)")
_TRAVERSAL = re.compile(r"(?i)(\.\./|\.\.\\|%2e%2e%2f|%252e%252e|/etc/passwd|boot\.ini)")
_ADMIN_PATH = re.compile(
    r"(?i)(/wp-admin|/wp-login|/phpmyadmin|/\.env|/\.git|/admin(/|$)|/administrator|"
    r"/actuator|/server-status|/console|/manager/html|/cgi-bin|/backup\.(zip|sql|tar))"
)
_SCANNER_UA = re.compile(
    r"(?i)(sqlmap|nikto|nmap|masscan|acunetix|nessus|burp|zgrab|dirbuster|gobuster|"
    r"wfuzz|ffuf|hydra|nuclei|havij|openvas|w3af)"
)
_BYPASS_HEADERS = ("x-original-url", "x-rewrite-url", "x-forwarded-host-override")


@dataclass(slots=True)
class PhantomConfig:
    enabled: bool = True
    score_threshold: int = 3
    session_ttl: float = 900.0
    max_sessions: int = 20_000
    latency_range: tuple = (0.12, 0.65)
    protected_prefixes: tuple = ("/health", "/docs", "/openapi.json")
    extra_patterns: Iterable[Pattern] = field(default_factory=tuple)


@dataclass(slots=True)
class _Trapped:
    first_seen: float
    last_seen: float
    hits: int = 0
    fake_records: int = 0
    fake_bytes: int = 0


class _SessionBook:
    """Bounded LRU of trapped sources keyed by client IP."""

    def __init__(self, ttl: float, cap: int) -> None:
        self._ttl, self._cap = ttl, cap
        self._data: "OrderedDict[str, _Trapped]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def touch(self, key: str) -> _Trapped:
        now = time.monotonic()
        async with self._lock:
            rec = self._data.get(key)
            if rec is None or now - rec.last_seen > self._ttl:
                rec = _Trapped(first_seen=now, last_seen=now)
                self._data[key] = rec
            rec.last_seen = now
            rec.hits += 1
            self._data.move_to_end(key)
            while len(self._data) > self._cap:
                self._data.popitem(last=False)
            return rec

    async def is_trapped(self, key: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            rec = self._data.get(key)
            return bool(rec and now - rec.last_seen <= self._ttl)


class PhantomFactory:
    """Generates fabricated, internally consistent decoy payloads."""

    _FIRST = ("alex", "maria", "jun", "sara", "omar", "lena", "ravi", "chloe", "ivan", "nora")
    _LAST = ("keller", "santos", "park", "novak", "haddad", "ross", "iyer", "moreau", "volkov")
    _TABLES = ("users", "sessions", "api_keys", "billing", "audit_log", "tokens")

    @classmethod
    def user_rows(cls, n: int) -> list:
        rows = []
        for i in range(n):
            f, l = random.choice(cls._FIRST), random.choice(cls._LAST)
            rows.append(
                {
                    "id": random.randint(1000, 99999),
                    "username": f"{f}.{l}{random.randint(1, 99)}",
                    "email": f"{f}.{l}@example.invalid",
                    "role": random.choice(("user", "user", "user", "editor", "admin")),
                    "password_hash": "$2b$12$" + secrets.token_urlsafe(40)[:53],
                    "created_at": int(time.time()) - random.randint(10**6, 10**8),
                }
            )
        return rows

    @classmethod
    def sqli_success(cls) -> dict:
        n = random.randint(18, 240)
        return {
            "status": "ok",
            "rows_affected": n,
            "extraction_rate_kbps": round(random.uniform(180.0, 2400.0), 1),
            "table": random.choice(cls._TABLES),
            "data": cls.user_rows(min(n, random.randint(3, 8))),
            "cursor": uuid.uuid4().hex,
        }

    @classmethod
    def admin_success(cls) -> dict:
        return {
            "authenticated": True,
            "privilege": "superuser",
            "session": secrets.token_hex(24),
            "modules": ["users", "billing", "secrets", "deploy"],
            "last_login": int(time.time()) - random.randint(60, 86400),
        }

    @classmethod
    def file_success(cls) -> Response:
        body = "\n".join(
            f"{secrets.token_hex(4)}_{random.choice(('KEY','TOKEN','DSN'))}={secrets.token_urlsafe(24)}"
            for _ in range(random.randint(4, 9))
        )
        return Response(body, media_type="text/plain")

    @classmethod
    def generic_success(cls) -> dict:
        return {
            "success": True,
            "records_exfiltrated": random.randint(500, 90000),
            "throughput_mb_s": round(random.uniform(1.2, 48.0), 2),
            "transfer_id": uuid.uuid4().hex,
            "progress": f"{random.randint(61, 99)}%",
        }


class NeuroPhantomMesh(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[PhantomConfig] = None,
        on_trap: Optional[Callable[[Request, int, Dict], Awaitable[None]]] = None,
    ) -> None:
        super().__init__(app)
        self.cfg = config or PhantomConfig()
        self.book = _SessionBook(self.cfg.session_ttl, self.cfg.max_sessions)
        self.on_trap = on_trap

    @staticmethod
    def _client(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _score(self, request: Request) -> tuple:
        score, reasons = 0, []
        path = request.url.path
        target = f"{path}?{request.url.query}" if request.url.query else path
        from urllib.parse import unquote

        decoded = unquote(unquote(target))

        if _SQLI.search(decoded):
            score += 3; reasons.append("sqli")
        if _XSS.search(decoded):
            score += 3; reasons.append("xss")
        if _TRAVERSAL.search(target) or _TRAVERSAL.search(decoded):
            score += 3; reasons.append("traversal")
        if _ADMIN_PATH.search(path):
            score += 2; reasons.append("admin_path")
        if _SCANNER_UA.search(request.headers.get("user-agent", "")):
            score += 3; reasons.append("scanner_ua")
        if any(h in request.headers for h in _BYPASS_HEADERS):
            score += 2; reasons.append("bypass_header")
        for pat in self.cfg.extra_patterns:
            if pat.search(decoded):
                score += 2; reasons.append("custom")
        return score, reasons

    def _craft(self, reasons: list, path: str) -> Response:
        if "sqli" in reasons:
            return JSONResponse(PhantomFactory.sqli_success())
        if "admin_path" in reasons and re.search(r"(?i)(\.env|\.git|backup)", path):
            return PhantomFactory.file_success()
        if "admin_path" in reasons or "bypass_header" in reasons:
            return JSONResponse(PhantomFactory.admin_success())
        return JSONResponse(PhantomFactory.generic_success())

    async def dispatch(self, request: Request, call_next):
        if not self.cfg.enabled or request.url.path.startswith(self.cfg.protected_prefixes):
            return await call_next(request)

        ip = self._client(request)
        score, reasons = self._score(request)
        already = await self.book.is_trapped(ip)

        if score < self.cfg.score_threshold and not (already and score > 0):
            return await call_next(request)

        rec = await self.book.touch(ip)
        response = self._craft(reasons or ["generic"], request.url.path)
        rec.fake_bytes += len(response.body or b"")
        rec.fake_records += 1

        await asyncio.sleep(random.uniform(*self.cfg.latency_range))
        response.headers["Server"] = random.choice(("nginx/1.18.0", "Apache/2.4.41", "gunicorn/20.1.0"))
        response.headers["X-Request-Id"] = uuid.uuid4().hex

        logger.warning("phantom_trap ip=%s path=%s reasons=%s hits=%d", ip, request.url.path, reasons, rec.hits)
        if self.on_trap:
            try:
                await self.on_trap(request, score, {"reasons": reasons, "hits": rec.hits, "ip": ip})
            except Exception:  # noqa: BLE001 - hook must never break the tunnel
                logger.exception("on_trap hook failed")
        return response


def install(app, config: Optional[PhantomConfig] = None, on_trap=None) -> None:
    app.add_middleware(NeuroPhantomMesh, config=config, on_trap=on_trap)
