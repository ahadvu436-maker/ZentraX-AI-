"""
trap_decoy_engine.py

Honeypot / decoy endpoint generator for FastAPI.

Purpose
-------
Registers a configurable set of fake API routes (admin panels, legacy
versions, debug endpoints, credential-looking paths, etc.) that do not
exist in the real application. Any traffic hitting these routes is, by
definition, automated scanning or malicious probing -- a legitimate
user never requests them.

The engine:
  * Serves plausible-looking but fake responses (never real data).
  * Introduces randomized, bounded latency to slow down scanners
    without tying up the event loop (no blocking sleeps).
  * Records structured events (IP, path, headers, UA, timestamp) for
    the alerting/intrusion-tracking module to consume.
  * Never executes, evals, or reflects attacker-supplied input --
    all responses are templated, not derived from the request.

This module is purely defensive: it does not attack, probe, or send
any payload back to the caller beyond a static, inert decoy response.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers

DecoyEventHandler = Callable[["DecoyEvent"], Awaitable[None]]


@dataclass(slots=True)
class DecoyEvent:
    event_id: str
    timestamp: float
    client_ip: str
    path: str
    method: str
    user_agent: str
    headers: dict[str, str]
    query_params: dict[str, str]


@dataclass(slots=True)
class DecoyRoute:
    path: str
    methods: tuple[str, ...] = ("GET", "POST")
    status_code: int = 404
    payload: dict[str, Any] = field(default_factory=dict)
    min_delay_ms: int = 50
    max_delay_ms: int = 400


DEFAULT_DECOY_ROUTES: tuple[DecoyRoute, ...] = (
    DecoyRoute("/admin", status_code=403, payload={"detail": "Forbidden"}),
    DecoyRoute("/admin/login", status_code=200,
               payload={"status": "ok", "session": "disabled"}),
    DecoyRoute("/.env", status_code=404, payload={"detail": "Not Found"}),
    DecoyRoute("/wp-login.php", status_code=404, payload={"detail": "Not Found"}),
    DecoyRoute("/api/v0/debug", status_code=500,
               payload={"detail": "Internal Server Error"}),
    DecoyRoute("/api/internal/users", status_code=200, payload={"users": []}),
    DecoyRoute("/config.json", status_code=404, payload={"detail": "Not Found"}),
    DecoyRoute("/backup.sql", status_code=404, payload={"detail": "Not Found"}),
    DecoyRoute("/.git/config", status_code=404, payload={"detail": "Not Found"}),
    DecoyRoute("/actuator/health", status_code=200, payload={"status": "UP"}),
)


class TrapDecoyEngine:
    """
    Registers decoy routes on a FastAPI app and reports every hit to a
    pluggable async event handler (e.g. the intrusion-alerting module).
    """

    def __init__(
        self,
        routes: tuple[DecoyRoute, ...] = DEFAULT_DECOY_ROUTES,
        on_event: DecoyEventHandler | None = None,
        max_concurrent_delays: int = 500,
    ) -> None:
        self._routes = routes
        self._on_event = on_event
        self._router = APIRouter()
        self._delay_semaphore = asyncio.Semaphore(max_concurrent_delays)
        self._build_router()

    def _build_router(self) -> None:
        for decoy in self._routes:
            self._router.add_api_route(
                decoy.path,
                self._make_handler(decoy),
                methods=list(decoy.methods),
                include_in_schema=False,
            )

    def _make_handler(self, decoy: DecoyRoute):
        async def handler(request: Request) -> JSONResponse:
            await self._simulate_latency(decoy)
            await self._record_hit(request, decoy)
            return JSONResponse(status_code=decoy.status_code, content=decoy.payload)

        return handler

    async def _simulate_latency(self, decoy: DecoyRoute) -> None:
        delay_ms = random.randint(decoy.min_delay_ms, decoy.max_delay_ms)
        async with self._delay_semaphore:
            await asyncio.sleep(delay_ms / 1000)

    async def _record_hit(self, request: Request, decoy: DecoyRoute) -> None:
        if self._on_event is None:
            return
        headers: Headers = request.headers
        event = DecoyEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            client_ip=_client_ip(request),
            path=decoy.path,
            method=request.method,
            user_agent=headers.get("user-agent", "unknown"),
            headers=dict(headers),
            query_params=dict(request.query_params),
        )
        try:
            await self._on_event(event)
        except Exception:
            # Decoy hits must never raise into the response path.
            pass

    def attach(self, app: FastAPI) -> None:
        app.include_router(self._router)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def install_decoys(
    app: FastAPI,
    on_event: DecoyEventHandler | None = None,
    routes: tuple[DecoyRoute, ...] = DEFAULT_DECOY_ROUTES,
) -> TrapDecoyEngine:
    engine = TrapDecoyEngine(routes=routes, on_event=on_event)
    engine.attach(app)
    return engine
