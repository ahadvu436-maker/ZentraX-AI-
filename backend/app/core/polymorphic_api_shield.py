"""Polymorphic API shield: rotating, HMAC-derived path segments under a protected prefix."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import logging
import secrets
import time
from typing import Awaitable, Callable, Optional

from fastapi import APIRouter, Depends, FastAPI
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger("zentrax.shield")

ProbeHook = Callable[[Scope, str], Awaitable[None]]
BypassFn = Callable[[Scope], bool]


class _Node:
    __slots__ = ("static", "param", "path")

    def __init__(self, path: str) -> None:
        self.static: dict[str, _Node] = {}
        self.param: Optional[_Node] = None
        self.path = path


class PolymorphicAPIShield:
    def __init__(
        self,
        *,
        secret: Optional[bytes] = None,
        rotation_seconds: int = 300,
        grace_epochs: int = 1,
        protected_prefix: str = "/api",
        on_probe: Optional[ProbeHook] = None,
        bypass: Optional[BypassFn] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret = secret or secrets.token_bytes(32)
        self.rotation = rotation_seconds
        self.grace = grace_epochs
        self.prefix = "/" + protected_prefix.strip("/")
        self.on_probe = on_probe
        self.bypass = bypass
        self._clock = clock
        self._root = _Node("")
        self._cache: dict[tuple[int, str, str], str] = {}
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ routes
    def build_from_app(self, app: FastAPI) -> None:
        self._root = _Node("")
        for route in app.routes:
            if not isinstance(route, APIRoute) or not route.path.startswith(self.prefix + "/"):
                continue
            node = self._root
            for seg in route.path[len(self.prefix) :].strip("/").split("/"):
                if seg.startswith("{") and seg.endswith("}"):
                    node.param = node.param or _Node(node.path + "/{}")
                    node = node.param
                else:
                    node = node.static.setdefault(seg, _Node(node.path + "/" + seg))

    # ------------------------------------------------------------------ tokens
    def epoch(self, now: Optional[float] = None) -> int:
        return int((now or self._clock()) // self.rotation)

    def _token(self, epoch: int, parent: str, seg: str) -> str:
        key = (epoch, parent, seg)
        tok = self._cache.get(key)
        if tok is None:
            mac = hmac.new(self._secret, f"{epoch}|{parent}|{seg}".encode(), hashlib.sha256).digest()
            tok = "s" + base64.b32encode(mac)[:11].decode().lower()
            self._cache[key] = tok
        return tok

    def alias_for(self, real_path: str, epoch: Optional[int] = None) -> str:
        epoch = self.epoch() if epoch is None else epoch
        node, out = self._root, []
        for seg in real_path[len(self.prefix) :].strip("/").split("/"):
            child = node.static.get(seg)
            if child is not None:
                out.append(self._token(epoch, node.path, seg))
                node = child
            elif node.param is not None:
                out.append(seg)
                node = node.param
            else:
                raise KeyError(real_path)
        return self.prefix + "/" + "/".join(out)

    def resolve(self, path: str) -> Optional[str]:
        segs = path[len(self.prefix) :].strip("/").split("/")
        current = self.epoch()
        for ep in range(current, current - self.grace - 1, -1):
            node, real, ok = self._root, [], True
            for seg in segs:
                hit = next(
                    (s for s in node.static if hmac.compare_digest(self._token(ep, node.path, s), seg)),
                    None,
                )
                if hit is not None:
                    real.append(hit)
                    node = node.static[hit]
                elif node.param is not None:
                    real.append(seg)
                    node = node.param
                else:
                    ok = False
                    break
            if ok:
                return self.prefix + "/" + "/".join(real)
        return None

    def manifest(self) -> dict[str, str]:
        out: dict[str, str] = {}
        ep = self.epoch()

        def walk(node: _Node, real: list[str], alias: list[str]) -> None:
            if node is not self._root:
                out["/".join([self.prefix, *real])] = "/".join([self.prefix, *alias])
            for seg, child in node.static.items():
                walk(child, real + [seg], alias + [self._token(ep, node.path, seg)])
            if node.param is not None:
                walk(node.param, real + ["{param}"], alias + ["{param}"])

        walk(self._root, [], [])
        return out

    # ---------------------------------------------------------------- lifecycle
    async def _rotator(self) -> None:
        while True:
            now = self._clock()
            await asyncio.sleep(self.rotation - (now % self.rotation) + 0.05)
            cur = self.epoch()
            self._cache = {k: v for k, v in self._cache.items() if k[0] >= cur - self.grace}
            log.info("shield rotated to epoch %d", cur)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._rotator(), name="shield-rotator")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def install(self, app: FastAPI, manifest_auth: Callable, manifest_path: str = "/.well-known/shield-manifest") -> None:
        self.build_from_app(app)
        router = APIRouter()

        @router.get(manifest_path, include_in_schema=False, dependencies=[Depends(manifest_auth)])
        async def _manifest() -> dict:
            return {"epoch": self.epoch(), "rotation": self.rotation, "routes": self.manifest()}

        app.include_router(router)
        app.add_middleware(PolymorphicShieldMiddleware, shield=self)
        app.add_event_handler("startup", self.start)
        app.add_event_handler("shutdown", self.stop)


class PolymorphicShieldMiddleware:
    def __init__(self, app: ASGIApp, shield: PolymorphicAPIShield) -> None:
        self.app, self.shield = app, shield

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        sh, path = self.shield, scope["path"]
        if not (path == sh.prefix or path.startswith(sh.prefix + "/")) or (sh.bypass and sh.bypass(scope)):
            return await self.app(scope, receive, send)
        real = sh.resolve(path) if path != sh.prefix else path
        if real is None:
            if sh.on_probe:
                with contextlib.suppress(Exception):
                    await sh.on_probe(scope, path)
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            return await JSONResponse({"detail": "Not Found"}, status_code=404)(scope, receive, send)
        scope = dict(scope, path=real, raw_path=real.encode())
        await self.app(scope, receive, send)
