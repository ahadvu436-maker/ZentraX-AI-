"""Decoy identity swarm and canary tripwires.

Every decoy record carries an HMAC `decoy_tag` so defenders can filter synthetic
entries out of forensics. Decoy credentials presented in real traffic trip an alarm.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import random
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, Optional, Protocol

from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger("zentrax.swarm")

_FIRST = ("alex", "sam", "jordan", "taylor", "morgan", "riley", "casey", "jamie", "devon", "robin")
_LAST = ("nguyen", "garcia", "khan", "smith", "silva", "ivanov", "okafor", "tanaka", "mueller", "rossi")
_DOMAINS = ("example.org", "example.net", "example.com")
_ACTIONS = ("login", "view_dashboard", "list_projects", "export_report", "update_profile", "refresh_token")
_UAS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
)


class AuditSink(Protocol):
    async def write(self, record: dict) -> None: ...


class ListSink:
    def __init__(self, maxlen: int = 50_000) -> None:
        self.records: list[dict] = []
        self.maxlen = maxlen

    async def write(self, record: dict) -> None:
        self.records.append(record)
        if len(self.records) > self.maxlen:
            del self.records[: self.maxlen // 10]


@dataclass(slots=True)
class DecoyAgent:
    agent_id: str
    email: str
    session_token: str
    api_key: str
    user_agent: str
    ip: str
    created: float = field(default_factory=time.time)


class SyntheticAgentSwarm:
    def __init__(
        self,
        sink: AuditSink,
        *,
        secret: Optional[bytes] = None,
        swarm_size: int = 300,
        events_per_second: float = 20.0,
        on_trip: Optional[Callable[[str, dict], Awaitable[None]]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.sink = sink
        self._secret = secret or secrets.token_bytes(32)
        self.size = swarm_size
        self.eps = events_per_second
        self.on_trip = on_trip
        self.rng = rng or random.Random()
        self.agents: dict[str, DecoyAgent] = {}
        self._by_secret: dict[str, str] = {}
        self._task: Optional[asyncio.Task] = None

    # --------------------------------------------------------------- identities
    def tag(self, agent_id: str) -> str:
        return hmac.new(self._secret, agent_id.encode(), hashlib.sha256).hexdigest()[:24]

    def is_decoy_record(self, record: dict) -> bool:
        aid, t = record.get("agent_id"), record.get("decoy_tag")
        return bool(aid and t and hmac.compare_digest(self.tag(aid), t))

    def _spawn(self) -> DecoyAgent:
        r = self.rng
        name = f"{r.choice(_FIRST)}.{r.choice(_LAST)}{r.randint(1, 999)}"
        a = DecoyAgent(
            agent_id=str(uuid.UUID(int=r.getrandbits(128), version=4)),
            email=f"{name}@{r.choice(_DOMAINS)}",
            session_token="zxs_" + secrets.token_urlsafe(24),
            api_key="zxk_" + secrets.token_urlsafe(32),
            user_agent=r.choice(_UAS),
            ip=f"10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}",
        )
        self.agents[a.agent_id] = a
        self._by_secret[a.session_token] = a.agent_id
        self._by_secret[a.api_key] = a.agent_id
        return a

    def _retire(self, agent_id: str) -> None:
        a = self.agents.pop(agent_id, None)
        if a:
            self._by_secret.pop(a.session_token, None)
            self._by_secret.pop(a.api_key, None)

    # ------------------------------------------------------------------- events
    async def _emit(self, a: DecoyAgent) -> None:
        await self.sink.write(
            {
                "ts": time.time(),
                "agent_id": a.agent_id,
                "actor": a.email,
                "session": a.session_token[:12],
                "ip": a.ip,
                "user_agent": a.user_agent,
                "action": self.rng.choice(_ACTIONS),
                "status": self.rng.choices((200, 204, 401, 403), (80, 8, 8, 4))[0],
                "decoy_tag": self.tag(a.agent_id),
                "synthetic": True,
            }
        )

    async def _run(self) -> None:
        while True:
            while len(self.agents) < self.size:
                self._spawn()
            if self.rng.random() < 0.02:  # churn identities
                self._retire(self.rng.choice(list(self.agents)))
            await self._emit(self.rng.choice(list(self.agents.values())))
            await asyncio.sleep(self.rng.expovariate(self.eps))

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="decoy-swarm")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ----------------------------------------------------------------- tripwire
    async def check_credentials(self, presented: list[str], context: dict) -> Optional[str]:
        for value in presented:
            aid = self._by_secret.get(value)
            if aid:
                evt = {**context, "decoy_agent": aid, "ts": time.time()}
                log.warning("decoy credential used: %s", evt)
                if self.on_trip:
                    await self.on_trip(aid, evt)
                return aid
        return None


class CanaryTripwireMiddleware:
    def __init__(self, app: ASGIApp, swarm: SyntheticAgentSwarm) -> None:
        self.app, self.swarm = app, swarm

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode("latin-1") for k, v in scope["headers"]}
            presented = [headers.get("x-api-key", "")]
            auth = headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                presented.append(auth[7:].strip())
            for part in headers.get("cookie", "").split(";"):
                presented.append(part.partition("=")[2].strip())
            client = scope.get("client") or ("", 0)
            hit = await self.swarm.check_credentials(
                [p for p in presented if p],
                {"ip": client[0], "path": scope["path"], "ua": headers.get("user-agent", "")},
            )
            if hit:
                from starlette.responses import JSONResponse

                return await JSONResponse({"detail": "Unauthorized"}, status_code=401)(scope, receive, send)
        await self.app(scope, receive, send)
