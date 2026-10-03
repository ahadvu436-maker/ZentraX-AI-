"""Cluster-wide IP blackhole: Redis pub/sub propagation with signed messages and local fast-path."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import ipaddress
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger("zentrax.blackhole")

IP = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
Net = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


@dataclass(slots=True)
class BanRecord:
    target: str
    reason: str
    expires: float  # 0 = permanent
    origin: str
    created: float


class BlackholeSync:
    def __init__(
        self,
        secret: bytes,
        *,
        redis: Any = None,  # redis.asyncio.Redis | None
        node_id: Optional[str] = None,
        channel: str = "zx:blackhole",
        key_prefix: str = "zx:bh:",
        allowlist: Iterable[str] = ("127.0.0.1", "::1"),
        default_ttl: int = 86_400,
        max_local: int = 250_000,
        max_skew: float = 60.0,
    ) -> None:
        self._secret = secret
        self.redis = redis
        self.node_id = node_id or uuid.uuid4().hex[:12]
        self.channel, self.prefix = channel, key_prefix
        self.allow: list[Net] = [ipaddress.ip_network(a, strict=False) for a in allowlist]
        self.default_ttl, self.max_local, self.max_skew = default_ttl, max_local, max_skew
        self._ips: dict[str, BanRecord] = {}
        self._nets: dict[Net, BanRecord] = {}
        self._tasks: list[asyncio.Task] = []

    # ---------------------------------------------------------------- utilities
    def _sign(self, body: str) -> str:
        return hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _parse(target: str) -> Net:
        return ipaddress.ip_network(target.strip(), strict=False)

    def _allowed(self, net: Net) -> bool:
        return any(net.version == a.version and net.overlaps(a) for a in self.allow)

    def _apply(self, rec: BanRecord) -> None:
        net = self._parse(rec.target)
        if net.num_addresses == 1:
            if len(self._ips) >= self.max_local:
                self._evict()
            self._ips[str(net.network_address)] = rec
        else:
            self._nets[net] = rec

    def _evict(self) -> None:
        now = time.time()
        for k in [k for k, r in self._ips.items() if r.expires and r.expires < now]:
            del self._ips[k]
        if len(self._ips) >= self.max_local:
            oldest = sorted(self._ips.items(), key=lambda kv: kv[1].created)[: self.max_local // 10]
            for k, _ in oldest:
                del self._ips[k]

    # --------------------------------------------------------------------- API
    def is_blocked(self, ip: str) -> Optional[BanRecord]:
        try:
            addr: IP = ipaddress.ip_address(ip)
        except ValueError:
            return None
        now = time.time()
        rec = self._ips.get(str(addr))
        if rec and (not rec.expires or rec.expires > now):
            return rec
        if rec:
            self._ips.pop(str(addr), None)
        for net, r in list(self._nets.items()):
            if r.expires and r.expires < now:
                del self._nets[net]
            elif addr.version == net.version and addr in net:
                return r
        return None

    async def ban(self, target: str, reason: str, ttl: Optional[int] = None) -> bool:
        net = self._parse(target)
        if self._allowed(net):
            log.warning("refused to ban allowlisted target %s", target)
            return False
        ttl = self.default_ttl if ttl is None else ttl
        now = time.time()
        rec = BanRecord(str(net), reason, now + ttl if ttl else 0.0, self.node_id, now)
        self._apply(rec)
        if self.redis:
            payload = json.dumps(rec.__dict__, separators=(",", ":"))
            await self.redis.set(self.prefix + rec.target, payload, ex=ttl or None)
            msg = json.dumps({"ts": now, "node": self.node_id, "op": "ban", "rec": payload}, separators=(",", ":"))
            await self.redis.publish(self.channel, json.dumps({"m": msg, "s": self._sign(msg)}))
        log.warning("blackholed %s (%s)", rec.target, reason)
        return True

    async def unban(self, target: str) -> None:
        net = self._parse(target)
        self._ips.pop(str(net.network_address), None) if net.num_addresses == 1 else self._nets.pop(net, None)
        if self.redis:
            await self.redis.delete(self.prefix + str(net))
            msg = json.dumps({"ts": time.time(), "node": self.node_id, "op": "unban", "target": str(net)})
            await self.redis.publish(self.channel, json.dumps({"m": msg, "s": self._sign(msg)}))

    # ------------------------------------------------------------------ syncing
    async def _hydrate(self) -> None:
        async for key in self.redis.scan_iter(match=self.prefix + "*", count=500):
            raw = await self.redis.get(key)
            if raw:
                with contextlib.suppress(Exception):
                    self._apply(BanRecord(**json.loads(raw)))

    async def _listen(self) -> None:
        while True:
            try:
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(self.channel)
                await self._hydrate()
                async for item in pubsub.listen():
                    if item.get("type") != "message":
                        continue
                    try:
                        env = json.loads(item["data"])
                        if not hmac.compare_digest(self._sign(env["m"]), env["s"]):
                            log.error("blackhole message with bad signature dropped")
                            continue
                        body = json.loads(env["m"])
                        if body["node"] == self.node_id or abs(time.time() - body["ts"]) > self.max_skew:
                            continue
                        if body["op"] == "ban":
                            rec = BanRecord(**json.loads(body["rec"]))
                            if not self._allowed(self._parse(rec.target)):
                                self._apply(rec)
                        elif body["op"] == "unban":
                            net = self._parse(body["target"])
                            self._ips.pop(str(net.network_address), None) if net.num_addresses == 1 else self._nets.pop(net, None)
                    except Exception:
                        log.exception("bad blackhole message")
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("blackhole listener lost connection; retrying")
                await asyncio.sleep(2)

    async def _janitor(self) -> None:
        while True:
            await asyncio.sleep(60)
            self._evict()

    async def start(self) -> None:
        if self.redis:
            self._tasks.append(asyncio.create_task(self._listen(), name="blackhole-listen"))
        self._tasks.append(asyncio.create_task(self._janitor(), name="blackhole-janitor"))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._tasks.clear()


def client_ip(scope: Scope, trusted_proxies: tuple[Net, ...] = ()) -> str:
    peer = (scope.get("client") or ("", 0))[0]
    if not trusted_proxies:
        return peer
    try:
        if not any(ipaddress.ip_address(peer) in n for n in trusted_proxies):
            return peer
    except ValueError:
        return peer
    for k, v in scope["headers"]:
        if k == b"x-forwarded-for":
            hops = [h.strip() for h in v.decode("latin-1").split(",")]
            for hop in reversed(hops):
                try:
                    if not any(ipaddress.ip_address(hop) in n for n in trusted_proxies):
                        return hop
                except ValueError:
                    break
    return peer


class BlackholeMiddleware:
    def __init__(self, app: ASGIApp, hole: BlackholeSync, trusted_proxies: Iterable[str] = ()) -> None:
        self.app, self.hole = app, hole
        self.proxies = tuple(ipaddress.ip_network(p, strict=False) for p in trusted_proxies)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            ip = client_ip(scope, self.proxies)
            scope.setdefault("state", {})["client_ip"] = ip
            if self.hole.is_blocked(ip):
                if scope["type"] == "websocket":
                    return await send({"type": "websocket.close", "code": 1008})
                return await Response(status_code=403)(scope, receive, send)
        await self.app(scope, receive, send)
