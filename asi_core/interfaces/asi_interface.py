"""Standardized, secure contract for talking to the ASI core."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..general_intelligence import Problem

if TYPE_CHECKING:
    from ..asi_engine import ASIEngine


@dataclass(frozen=True)
class ASIRequest:
    token: str
    domain: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    priority: int = 5


@dataclass(frozen=True)
class ASIResponse:
    ok: bool
    output: Any = None
    error: str | None = None
    request_id: str = ""


class ASIInterface(ABC):
    @abstractmethod
    async def submit(self, request: ASIRequest) -> ASIResponse: ...
    @abstractmethod
    async def health(self, token: str) -> dict[str, Any]: ...


class AuthError(Exception): ...


class TokenAuthority:
    """Stateless HMAC-signed tokens: base64(payload).signature."""

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)

    def issue(self, subject: str, scopes: set[str], ttl_s: int = 3600) -> str:
        body = base64.urlsafe_b64encode(json.dumps(
            {"sub": subject, "scopes": sorted(scopes), "exp": time.time() + ttl_s}).encode())
        return f"{body.decode()}.{self._sign(body)}"

    def verify(self, token: str) -> tuple[str, frozenset[str]]:
        try:
            body, sig = token.split(".", 1)
            if not hmac.compare_digest(sig, self._sign(body.encode())):
                raise AuthError("bad signature")
            data = json.loads(base64.urlsafe_b64decode(body))
        except AuthError:
            raise
        except Exception as e:
            raise AuthError("malformed token") from e
        if data["exp"] < time.time():
            raise AuthError("token expired")
        return data["sub"], frozenset(data["scopes"])

    def _sign(self, body: bytes) -> str:
        return hmac.new(self._secret, body, hashlib.sha256).hexdigest()


class _TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate, self.burst, self.tokens, self.t = rate, burst, float(burst), time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class SecureASIGateway(ASIInterface):
    """Auth + scope check + rate limit + input validation + audit trail."""

    def __init__(self, engine: "ASIEngine", authority: TokenAuthority,
                 rate_per_s: float = 5.0, burst: int = 10, max_input_chars: int = 20_000) -> None:
        self.engine, self.auth = engine, authority
        self.rate, self.burst, self.max_chars = rate_per_s, burst, max_input_chars
        self._buckets: dict[str, _TokenBucket] = {}
        self.audit_log: list[dict[str, Any]] = []

    def _audit(self, sub: str, action: str, ok: bool, detail: str = "") -> None:
        self.audit_log.append({"t": time.time(), "sub": sub, "action": action,
                               "ok": ok, "detail": detail})
        del self.audit_log[:-10_000]

    async def submit(self, request: ASIRequest) -> ASIResponse:
        try:
            sub, scopes = self.auth.verify(request.token)
        except AuthError as e:
            self._audit("?", "submit", False, str(e))
            return ASIResponse(False, error="unauthorized")
        if "asi:submit" not in scopes:
            self._audit(sub, "submit", False, "missing scope")
            return ASIResponse(False, error="forbidden")
        if not self._buckets.setdefault(sub, _TokenBucket(self.rate, self.burst)).allow():
            self._audit(sub, "submit", False, "rate limited")
            return ASIResponse(False, error="rate_limited")
        if not request.description or len(request.description) > self.max_chars:
            self._audit(sub, "submit", False, "invalid input")
            return ASIResponse(False, error="invalid_input")
        # Caller scopes are never mapped to "*": external callers get "internal"-level only if granted.
        perms = frozenset(s.removeprefix("cap:") for s in scopes if s.startswith("cap:"))
        problem = Problem(request.domain, request.description, dict(request.params), perms)
        sol = await self.engine.solve(problem, request.priority)
        self._audit(sub, "submit", sol.success, problem.id)
        return ASIResponse(sol.success, sol.output, sol.error, problem.id)

    async def health(self, token: str) -> dict[str, Any]:
        try:
            sub, scopes = self.auth.verify(token)
        except AuthError:
            return {"error": "unauthorized"}
        if "asi:health" not in scopes:
            return {"error": "forbidden"}
        self._audit(sub, "health", True)
        return self.engine.health()
