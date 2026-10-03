"""autonomous_counter_intel.py - Counter-intelligence middleware for FastAPI.

Intercepts malicious user agents and probe headers, tracks repeat offenders,
and returns deterrent payloads (simulated containment notices and decoy
"locked telemetry" blobs). All payloads are inert, fabricated text/JSON.
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
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("zentrax.counter_intel")

_BAD_UA = re.compile(
    r"(?i)(sqlmap|nikto|nmap|masscan|zgrab|acunetix|nessus|burp|dirbuster|gobuster|"
    r"wfuzz|ffuf|hydra|nuclei|metasploit|python-requests/0|libwww-perl|curl/7\.[0-4]\d?\b|"
    r"go-http-client/1\.1$|scrapy|havij|openvas)"
)
_PROBE_HEADERS = {
    "x-forwarded-for-spoof", "x-originating-ip", "x-remote-addr", "x-remote-ip",
    "x-original-url", "x-rewrite-url", "x-custom-ip-authorization", "x-scanner",
    "x-sqlmap", "x-probe",
}
_SUSPECT_VALUE = re.compile(r"(?i)(\$\{jndi:|<script|union\s+select|\.\./|;\s*cat\s|\|\s*nc\s)")


@dataclass(slots=True)
class CounterIntelConfig:
    enabled: bool = True
    strike_ttl: float = 1800.0
    max_tracked: int = 50_000
    escalate_after: int = 3
    exempt_prefixes: tuple = ("/health", "/docs", "/openapi.json")
    block_status: int = 403
    delay_range: tuple = (0.2, 1.1)


@dataclass(slots=True)
class _Offender:
    strikes: int
    last_seen: float
    tag: str


class _OffenderLedger:
    def __init__(self, ttl: float, cap: int) -> None:
        self._ttl, self._cap = ttl, cap
        self._d: "OrderedDict[str, _Offender]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def strike(self, key: str) -> _Offender:
        now = time.monotonic()
        async with self._lock:
            o = self._d.get(key)
            if o is None or now - o.last_seen > self._ttl:
                o = _Offender(0, now, secrets.token_hex(4).upper())
                self._d[key] = o
            o.strikes += 1
            o.last_seen = now
            self._d.move_to_end(key)
            while len(self._d) > self._cap:
                self._d.popitem(last=False)
            return o


class DeterrentComposer:
    _PHASES = ("OBSERVE", "FINGERPRINT", "CONTAIN", "ISOLATE")
    _NOTICES = (
        "Automated defense layer has flagged this session.",
        "Request pattern matched a known hostile signature.",
        "Session telemetry has been sealed pending review.",
        "This endpoint is monitored; further probing is logged.",
    )

    @classmethod
    def containment(cls, offender: _Offender, reasons: list, escalate_after: int) -> dict:
        level = min(len(cls._PHASES) - 1, offender.strikes // max(1, escalate_after))
        return {
            "status": "contained",
            "case_id": f"ZX-{offender.tag}",
            "phase": cls._PHASES[level],
            "strikes": offender.strikes,
            "notice": random.choice(cls._NOTICES),
            "indicators": reasons,
            "containment_progress": f"{min(99, 20 + offender.strikes * random.randint(8, 17))}%",
            "reference": uuid.uuid4().hex,
        }

    @classmethod
    def locked_telemetry(cls, offender: _Offender) -> dict:
        return {
            "telemetry": {
                "state": "LOCKED",
                "cipher": "AES-256-GCM",
                "blob": secrets.token_urlsafe(random.randint(96, 192)),
                "unlock_requires": "authority-held key",
                "decoy_checksum": secrets.token_hex(16),
            },
            "message": "Source attribution package assembled and escrowed.",
            "case_id": f"ZX-{offender.tag}",
        }


class AutonomousCounterIntel(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[CounterIntelConfig] = None,
        on_incident: Optional[Callable[[Request, Dict], Awaitable[None]]] = None,
    ) -> None:
        super().__init__(app)
        self.cfg = config or CounterIntelConfig()
        self.ledger = _OffenderLedger(self.cfg.strike_ttl, self.cfg.max_tracked)
        self.on_incident = on_incident

    @staticmethod
    def _ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    @staticmethod
    def _inspect(request: Request) -> list:
        reasons = []
        ua = request.headers.get("user-agent", "")
        if not ua:
            reasons.append("empty_user_agent")
        elif _BAD_UA.search(ua):
            reasons.append("malicious_user_agent")
        for name, value in request.headers.items():
            lname = name.lower()
            if lname in _PROBE_HEADERS:
                reasons.append(f"probe_header:{lname}")
            if _SUSPECT_VALUE.search(value):
                reasons.append(f"hostile_value:{lname}")
        return reasons

    async def dispatch(self, request: Request, call_next):
        if not self.cfg.enabled or request.url.path.startswith(self.cfg.exempt_prefixes):
            return await call_next(request)

        reasons = self._inspect(request)
        if not reasons:
            return await call_next(request)

        offender = await self.ledger.strike(self._ip(request))
        body = DeterrentComposer.containment(offender, reasons, self.cfg.escalate_after)
        if offender.strikes >= self.cfg.escalate_after:
            body.update(DeterrentComposer.locked_telemetry(offender))

        logger.warning("counter_intel ip=%s case=%s strikes=%d reasons=%s",
                       self._ip(request), offender.tag, offender.strikes, reasons)
        if self.on_incident:
            try:
                await self.on_incident(request, {"case": offender.tag, "strikes": offender.strikes, "reasons": reasons})
            except Exception:  # noqa: BLE001
                logger.exception("on_incident hook failed")

        await asyncio.sleep(random.uniform(*self.cfg.delay_range))
        return JSONResponse(body, status_code=self.cfg.block_status,
                            headers={"X-Case-Id": f"ZX-{offender.tag}", "Cache-Control": "no-store"})


def install(app, config: Optional[CounterIntelConfig] = None, on_incident=None) -> None:
    app.add_middleware(AutonomousCounterIntel, config=config, on_incident=on_incident)
