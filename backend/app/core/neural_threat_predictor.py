"""Online threat predictor: per-client behavioural features -> logistic model -> adaptive rate policy."""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, Optional

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger("zentrax.predictor")

_SUSPICIOUS = re.compile(
    r"(\.\./|%2e%2e|union\s+select|<script|etc/passwd|\bor\s+1=1|\$\{jndi|;\s*cat\s|/\.env|wp-login|phpmyadmin|\.git/)",
    re.I,
)
FEATURES = ("bias", "rate", "err_ratio", "not_found", "path_div", "path_entropy", "payload_hits", "no_ua", "burst")
_PRIORS = dict(bias=-4.0, rate=2.5, err_ratio=2.0, not_found=3.0, path_div=1.0, path_entropy=0.6, payload_hits=4.0, no_ua=1.0, burst=1.5)


@dataclass(slots=True)
class _Event:
    ts: float
    path: str
    status: int
    hits: int
    no_ua: bool


@dataclass
class _Client:
    events: Deque[_Event] = field(default_factory=lambda: deque(maxlen=200))
    score: float = 0.0
    last: float = 0.0


@dataclass(slots=True)
class Policy:
    rate_limit: int  # requests per window
    challenge: bool
    ban: bool


class ThreatPredictor:
    def __init__(
        self,
        *,
        window: float = 30.0,
        base_limit: int = 120,
        max_clients: int = 100_000,
        lr: float = 0.05,
        tighten_at: float = 0.4,
        challenge_at: float = 0.75,
        ban_at: float = 0.95,
        on_ban: Optional[Callable[[str, float, str], Awaitable[None]]] = None,
    ) -> None:
        self.window, self.base_limit, self.max_clients, self.lr = window, base_limit, max_clients, lr
        self.t_tighten, self.t_challenge, self.t_ban = tighten_at, challenge_at, ban_at
        self.on_ban = on_ban
        self.w = dict(_PRIORS)
        self._clients: "OrderedDict[str, _Client]" = OrderedDict()

    # ----------------------------------------------------------------- features
    @staticmethod
    def _entropy(items: list[str]) -> float:
        if not items:
            return 0.0
        n, c = len(items), Counter(items)
        return -sum(v / n * math.log2(v / n) for v in c.values())

    def _features(self, c: _Client, now: float) -> dict[str, float]:
        ev = [e for e in c.events if now - e.ts <= self.window]
        n = len(ev)
        if n == 0:
            return dict.fromkeys(FEATURES, 0.0) | {"bias": 1.0}
        paths = [e.path for e in ev]
        recent = sum(1 for e in ev if now - e.ts <= 2.0)
        return {
            "bias": 1.0,
            "rate": min(n / self.base_limit, 3.0),
            "err_ratio": sum(e.status >= 400 for e in ev) / n,
            "not_found": min(sum(e.status == 404 for e in ev) / 10.0, 3.0),
            "path_div": len(set(paths)) / n if n > 5 else 0.0,
            "path_entropy": min(self._entropy(paths) / 5.0, 2.0),
            "payload_hits": min(sum(e.hits for e in ev), 5),
            "no_ua": sum(e.no_ua for e in ev) / n,
            "burst": min(recent / 20.0, 3.0),
        }

    def _predict(self, x: dict[str, float]) -> float:
        z = sum(self.w[k] * v for k, v in x.items())
        return 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))

    # ---------------------------------------------------------------- interface
    def _get(self, key: str) -> _Client:
        c = self._clients.get(key)
        if c is None:
            c = self._clients[key] = _Client()
            if len(self._clients) > self.max_clients:
                self._clients.popitem(last=False)
        else:
            self._clients.move_to_end(key)
        return c

    async def observe(self, client: str, path: str, query: str, ua: str, status: int) -> float:
        now = time.time()
        c = self._get(client)
        hits = len(_SUSPICIOUS.findall(f"{path}?{query}"))
        c.events.append(_Event(now, path, status, hits, not ua))
        c.last = now
        c.score = self._predict(self._features(c, now))
        if c.score >= self.t_ban and self.on_ban:
            await self.on_ban(client, c.score, "predicted_attack")
        return c.score

    def feedback(self, client: str, malicious: bool) -> None:
        """Supervised SGD step from confirmed incidents / false-positive reviews."""
        c = self._clients.get(client)
        if not c:
            return
        x = self._features(c, time.time())
        err = (1.0 if malicious else 0.0) - self._predict(x)
        for k, v in x.items():
            self.w[k] += self.lr * err * v

    def policy_for(self, client: str) -> Policy:
        c = self._clients.get(client)
        s = c.score if c else 0.0
        limit = self.base_limit
        if s >= self.t_tighten:
            limit = max(5, int(self.base_limit * (1.0 - s) ** 2))
        return Policy(limit, s >= self.t_challenge, s >= self.t_ban)

    def allow(self, client: str) -> bool:
        c = self._clients.get(client)
        if not c:
            return True
        now = time.time()
        n = sum(1 for e in c.events if now - e.ts <= self.window)
        return n < self.policy_for(client).rate_limit


class ThreatPredictorMiddleware:
    def __init__(self, app: ASGIApp, predictor: ThreatPredictor) -> None:
        self.app, self.pred = app, predictor

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        client = scope.get("state", {}).get("client_ip") or (scope.get("client") or ("unknown", 0))[0]
        pol = self.pred.policy_for(client)
        if pol.ban or not self.pred.allow(client):
            return await JSONResponse(
                {"detail": "Too Many Requests"}, status_code=429, headers={"Retry-After": "30"}
            )(scope, receive, send)
        headers = dict(scope["headers"])
        ua = headers.get(b"user-agent", b"").decode("latin-1")
        status = {"v": 500}

        async def _send(m: Message) -> None:
            if m["type"] == "http.response.start":
                status["v"] = m["status"]
            await send(m)

        try:
            await self.app(scope, receive, _send)
        finally:
            await self.pred.observe(client, scope["path"], scope.get("query_string", b"").decode("latin-1"), ua, status["v"])
