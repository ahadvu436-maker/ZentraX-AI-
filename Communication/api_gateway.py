"""Framework-agnostic HTTP gateway: routing, auth, rate limiting, error mapping.

``ApiGateway.handle`` turns an :class:`ApiRequest` into an :class:`ApiResponse` and
never raises. :func:`create_fastapi_app` mounts it (plus a ``/ws`` endpoint) on
FastAPI/Starlette; any other ASGI/WSGI framework can call ``handle`` directly.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Pattern, Protocol, Sequence

from .communication_core import (
    AuthenticationError, AuthorizationError, CommunicationError, CommunicationTimeoutError,
    RateLimitError, RoutingError, TokenBucket, ValidationError,
)

if TYPE_CHECKING:  # pragma: no cover
    from .communication_core import CommunicationCore

logger = logging.getLogger(__name__)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


@dataclass(frozen=True)
class Principal:
    id: str
    roles: FrozenSet[str] = frozenset()


@dataclass(frozen=True, eq=False)
class ApiRequest:
    method: str
    path: str
    headers: Mapping[str, str] = field(default_factory=dict)  # lower-case keys
    query: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    client_ip: str = "unknown"
    path_params: Mapping[str, str] = field(default_factory=dict)
    principal: Optional[Principal] = None
    request_id: str = ""

    def json(self) -> Any:
        try:
            return json.loads(self.body or b"null")
        except (ValueError, RecursionError):
            raise ValidationError("request body is not valid JSON", code="invalid_json", status=400) from None

    def json_object(self) -> Dict[str, Any]:
        data = self.json()
        if not isinstance(data, dict):
            raise ValidationError("JSON object expected", code="invalid_json", status=400)
        return data


@dataclass
class ApiResponse:
    status: int = 200
    body: bytes = b""
    headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def json(cls, data: Any, status: int = 200, headers: Optional[Dict[str, str]] = None) -> "ApiResponse":
        h = {"content-type": "application/json"}
        h.update(headers or {})
        return cls(status, json.dumps(data, default=str, separators=(",", ":")).encode(), h)


ApiHandler = Callable[[ApiRequest], Awaitable[ApiResponse]]
ApiMiddleware = Callable[[ApiRequest, Callable[[ApiRequest], Awaitable[ApiResponse]]], Awaitable[ApiResponse]]


class Authenticator(Protocol):
    async def authenticate(self, request: ApiRequest) -> Optional[Principal]: ...


class ApiKeyAuthenticator:
    """``X-API-Key`` or ``Authorization: Bearer`` keys. Only SHA-256 digests are stored."""

    def __init__(self) -> None:
        self._keys: Dict[str, Principal] = {}

    def add_key(self, api_key: str, principal: Principal) -> None:
        if len(api_key) < 16:
            raise ValueError("API keys must be at least 16 characters")
        self._keys[hashlib.sha256(api_key.encode()).hexdigest()] = principal

    async def authenticate(self, request: ApiRequest) -> Optional[Principal]:
        key = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        if not key:
            return None
        digest = hashlib.sha256(key.encode()).hexdigest()
        for known, principal in self._keys.items():
            if hmac.compare_digest(known, digest):
                return principal
        return None


@dataclass(frozen=True, eq=False)
class _Route:
    method: str
    template: str
    regex: Pattern[str]
    handler: ApiHandler
    auth: bool
    roles: FrozenSet[str]
    timeout_s: Optional[float]


class ApiGateway:
    """Request pipeline: size limit -> rate limit -> CORS -> route -> auth/roles ->
    middleware -> handler (with timeout) -> error mapping and security headers."""

    def __init__(self, authenticator: Optional[Authenticator] = None, *, max_body_bytes: int = 1_048_576,
                 rate_per_s: float = 20.0, burst: int = 40, request_timeout_s: float = 30.0,
                 cors_origins: Sequence[str] = (), max_tracked_clients: int = 10_000) -> None:
        self._auth = authenticator
        self.max_body_bytes = max_body_bytes
        self._rate, self._burst = rate_per_s, burst
        self._timeout = request_timeout_s
        self._cors = frozenset(cors_origins)
        self._max_clients = max_tracked_clients
        self._routes: List[_Route] = []
        self._middleware: List[ApiMiddleware] = []
        self._buckets: Dict[str, TokenBucket] = {}

    # -- configuration ------------------------------------------------------ #
    def add_route(self, method: str, path: str, handler: ApiHandler, *, auth: bool = True,
                  roles: Iterable[str] = (), timeout_s: Optional[float] = None) -> None:
        if not re.fullmatch(r"(/(\{\w+\}|[A-Za-z0-9._\-~]+))+|/", path):
            raise ValueError(f"invalid route path {path!r}")
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", path)
        self._routes.append(_Route(method.upper(), path, re.compile(f"^{pattern}$"), handler, auth,
                                   frozenset(roles), timeout_s))

    def route(self, method: str, path: str, **kwargs: Any) -> Callable[[ApiHandler], ApiHandler]:
        def deco(fn: ApiHandler) -> ApiHandler:
            self.add_route(method, path, fn, **kwargs)
            return fn
        return deco

    def use(self, middleware: ApiMiddleware) -> None:
        self._middleware.append(middleware)

    @property
    def require_auth(self) -> bool:
        return self._auth is not None

    async def authenticate_headers(self, headers: Mapping[str, str], client_ip: str = "unknown") -> Optional[Principal]:
        """Authenticate from headers only (used for WebSocket handshakes)."""
        if self._auth is None:
            return None
        req = ApiRequest("GET", "/ws", {k.lower(): v for k, v in headers.items()}, client_ip=client_ip)
        return await self._auth.authenticate(req)

    # -- request handling --------------------------------------------------- #
    async def handle(self, request: ApiRequest) -> ApiResponse:
        rid = request.headers.get("x-request-id", "")
        rid = rid if _REQUEST_ID_RE.match(rid) else uuid.uuid4().hex
        request = ApiRequest(request.method.upper(), request.path, request.headers, request.query,
                             request.body, request.client_ip, request.path_params, request.principal, rid)
        try:
            response = await self._pipeline(request)
        except CommunicationError as exc:
            response = self._error(exc, rid)
        except Exception:
            logger.exception("unhandled gateway error (request %s)", rid)
            response = self._error(CommunicationError("internal error", code="internal_error"), rid)
        response.headers.setdefault("x-request-id", rid)
        response.headers.setdefault("x-content-type-options", "nosniff")
        response.headers.setdefault("cache-control", "no-store")
        origin = request.headers.get("origin")
        if origin and origin in self._cors:
            response.headers["access-control-allow-origin"] = origin
            response.headers["vary"] = "Origin"
        return response

    async def _pipeline(self, request: ApiRequest) -> ApiResponse:
        if len(request.body) > self.max_body_bytes:
            raise ValidationError("request body too large", code="payload_too_large", status=413)
        self._check_rate(request.client_ip)
        origin = request.headers.get("origin")
        if request.method == "OPTIONS" and origin in self._cors:
            return ApiResponse(204, headers={
                "access-control-allow-origin": origin, "access-control-allow-methods": "GET,POST,PUT,DELETE",
                "access-control-allow-headers": "authorization,content-type,x-api-key,x-request-id",
                "access-control-max-age": "600"})
        route, params, other_methods = self._match(request)
        if route is None:
            if other_methods:
                resp_err = CommunicationError("method not allowed", code="method_not_allowed", status=405)
                resp = self._error(resp_err, request.request_id)
                resp.headers["allow"] = ", ".join(sorted(other_methods))
                return resp
            raise RoutingError("not found")
        principal = request.principal
        if route.auth:
            principal = await self._auth.authenticate(request) if self._auth else None
            if principal is None:
                raise AuthenticationError("authentication required")
            if route.roles and not (route.roles & principal.roles):
                raise AuthorizationError("insufficient role")
        request = ApiRequest(request.method, request.path, request.headers, request.query, request.body,
                             request.client_ip, params, principal, request.request_id)

        async def terminal(req: ApiRequest) -> ApiResponse:
            return await route.handler(req)

        call: Callable[[ApiRequest], Awaitable[ApiResponse]] = terminal
        for mw in reversed(self._middleware):
            call = self._link(mw, call)
        timeout = route.timeout_s or self._timeout
        try:
            return await asyncio.wait_for(call(request), timeout)
        except asyncio.TimeoutError:
            raise CommunicationTimeoutError("request timed out") from None

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _link(mw: ApiMiddleware, nxt: Callable[[ApiRequest], Awaitable[ApiResponse]]) -> Callable[[ApiRequest], Awaitable[ApiResponse]]:
        async def run(req: ApiRequest) -> ApiResponse:
            return await mw(req, nxt)
        return run

    def _match(self, request: ApiRequest) -> Any:
        others = set()
        for r in self._routes:
            m = r.regex.match(request.path)
            if not m:
                continue
            if r.method == request.method:
                return r, m.groupdict(), others
            others.add(r.method)
        return None, {}, others

    def _check_rate(self, key: str) -> None:
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self._max_clients:
                self._buckets.pop(next(iter(self._buckets)))  # evict oldest
            bucket = self._buckets[key] = TokenBucket(self._rate, self._burst)
        if not bucket.allow():
            raise RateLimitError(retry_after=bucket.retry_after())

    @staticmethod
    def _error(exc: CommunicationError, rid: str) -> ApiResponse:
        headers: Dict[str, str] = {}
        if isinstance(exc, RateLimitError):
            headers["retry-after"] = str(max(1, int(exc.retry_after + 0.999)))
        if isinstance(exc, AuthenticationError):
            headers["www-authenticate"] = "Bearer"
        message = str(exc) if exc.status < 500 or exc.code != "communication_error" else "internal error"
        if exc.status >= 500 and exc.code in ("handler_error", "internal_error"):
            message = "internal error"
        return ApiResponse.json({"error": {"code": exc.code, "message": message, "request_id": rid}},
                                exc.status, headers)

    # -- default routes ----------------------------------------------------- #
    def mount_core_routes(self, core: "CommunicationCore") -> None:
        """Expose health, event publishing, message routing and notifications."""
        from .notification_manager import Notification, NotificationLevel
        from .protocol_handler import MessageType
        from .message_router import RouteContext

        async def health(req: ApiRequest) -> ApiResponse:
            return ApiResponse.json({"status": "ok" if core.running else "stopped"},
                                    200 if core.running else 503)

        async def stats(req: ApiRequest) -> ApiResponse:
            return ApiResponse.json(core.health())

        async def publish(req: ApiRequest) -> ApiResponse:
            body = req.json_object()
            ev = await core.publish_event(req.path_params["topic"], body.get("payload"),
                                          source=f"http:{req.principal.id if req.principal else 'anon'}",
                                          correlation_id=req.request_id)
            return ApiResponse.json({"event_id": ev.id}, 202)

        async def message(req: ApiRequest) -> ApiResponse:
            body = req.json_object()
            topic = body.get("topic")
            if not isinstance(topic, str):
                raise ValidationError("topic is required")
            env = core.protocol.make(MessageType.REQUEST, topic=topic, payload=body.get("payload"),
                                     correlation_id=req.request_id)
            result = await core.send(env, RouteContext(principal=req.principal))
            return ApiResponse.json({"id": env.id, "result": result})

        async def notify(req: ApiRequest) -> ApiResponse:
            b = req.json_object()
            try:
                level = NotificationLevel(b.get("level", "info"))
            except ValueError:
                raise ValidationError("invalid level") from None
            channels = b.get("channels", [])
            if not isinstance(channels, list) or not all(isinstance(c, str) for c in channels):
                raise ValidationError("channels must be a list of strings")
            n = Notification(recipient=str(b.get("recipient", "")), title=str(b.get("title", "")),
                             body=str(b.get("body", "")), level=level, channels=tuple(channels),
                             data=b.get("data") if isinstance(b.get("data"), dict) else {},
                             dedupe_key=b.get("dedupe_key") if isinstance(b.get("dedupe_key"), str) else None)
            return ApiResponse.json({"id": await core.notify(n)}, 202)

        self.add_route("GET", "/health", health, auth=False)
        self.add_route("GET", "/v1/stats", stats, roles=["admin"])
        self.add_route("POST", "/v1/events/{topic}", publish)
        self.add_route("POST", "/v1/messages", message)
        self.add_route("POST", "/v1/notifications", notify)


def create_fastapi_app(core: "CommunicationCore") -> Any:
    """Build a FastAPI app exposing the gateway over HTTP and ``/ws`` for WebSockets.

    The core's lifecycle is tied to the app lifespan. Browsers' ``Origin`` header is
    checked against ``core.allowed_origins`` (cross-site WebSocket hijacking guard);
    clients that send no Origin (servers, CLIs) are unaffected. Credentials must be
    sent in headers, never in the URL.
    """
    try:
        from fastapi import FastAPI, Request, Response, WebSocket
    except ImportError as exc:  # pragma: no cover
        raise ImportError("create_fastapi_app requires 'fastapi' to be installed") from exc

    @asynccontextmanager
    async def lifespan(app: Any):  # type: ignore[no-untyped-def]
        await core.start()
        try:
            yield
        finally:
            await core.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def _http(request: Request, path: str) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > core.gateway.max_body_bytes:
            return Response(status_code=413)
        body = await request.body()
        ip = request.client.host if request.client else "unknown"
        resp = await core.gateway.handle(ApiRequest(
            request.method, request.url.path, {k.lower(): v for k, v in request.headers.items()},
            dict(request.query_params), body, ip))
        return Response(content=resp.body, status_code=resp.status, headers=resp.headers)

    @app.websocket("/ws")
    async def _ws(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin and origin not in core.allowed_origins:
            await websocket.close(code=1008)
            return
        ip = websocket.client.host if websocket.client else "unknown"
        principal = await core.gateway.authenticate_headers(dict(websocket.headers), ip)
        if principal is None and core.gateway.require_auth:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await core.ws.serve(websocket, principal=principal, remote=ip)

    return app
