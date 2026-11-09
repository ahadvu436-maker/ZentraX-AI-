"""
ZentraX-AI  |  super_intelligence/nexus_bridge.py

Secure communication bridge between the Super Intelligence core and the rest of the
system (backend, frontend, local databases, ...). Local-first: the built-in network
transport only binds to loopback, and everything else is in-process.

What it provides
    * Envelope      versioned, validated JSON message (command / event / response / error)
    * Per-peer auth every peer has its own key; frames are HMAC-SHA256 signed, or
                    AES-256-GCM encrypted (optional, needs `cryptography`)
    * Replay guard  timestamp window + nonce cache
    * ACLs          each peer may only call routes matching its allowed patterns
                    (default: deny all)
    * Router        async handlers registered per route, with timeout + concurrency cap
    * Event bus     in-process publish/subscribe for internal fan-out
    * Transport     length-prefixed frames over loopback TCP (asyncio)

Wire frame
    "ZB" | version(1) | mode(1: 0=HMAC, 1=AES-GCM) | sender_len(1) | sender | body
      HMAC mode: body = tag(32) || plaintext,  tag = HMAC(sign_key, header || plaintext)
      GCM  mode: body = nonce(12) || ciphertext, AAD = header
    The header (incl. sender name) is authenticated in both modes. Sign/encrypt keys
    are derived from each peer's master key with HKDF-SHA256 (distinct labels).

Security notes
    * Unauthenticated / malformed / replayed frames are dropped silently (no oracle).
    * Handlers only ever see validated payloads; internal errors are never leaked to
      peers (only a stable error code, and safe messages from `HandlerError`).
    * Key material: use `generate_key()` once per peer pair, store it in a 0600 file
      (`load_key`) or an env var, and rotate by redeploying both sides.
    * HMAC mode gives integrity + authenticity, NOT confidentiality. Over loopback
      that is usually enough; enable `encrypt=True` for sensitive payloads.

Requires Python 3.10+. `cryptography` is optional (only for encrypt=True).
"""
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
import struct
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional

try:  # optional dependency
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover
    AESGCM = None  # type: ignore[assignment,misc]

log = logging.getLogger("zentrax.bridge")

MAGIC = b"ZB"
VERSION = 1
MODE_HMAC, MODE_GCM = 0, 1
MIN_KEY_BYTES = 32
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class BridgeError(Exception):
    pass


class ProtocolError(BridgeError):
    pass


class AuthenticationError(BridgeError):
    pass


class ReplayError(BridgeError):
    pass


class AuthorizationError(BridgeError):
    pass


class RouteNotFound(BridgeError):
    pass


class HandlerError(BridgeError):
    """Raise from a handler to return a *safe*, client-visible error."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code, self.message = code, message or code


class RemoteError(BridgeError):
    """Raised on the caller side when the peer answered with an error envelope."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def generate_key() -> bytes:
    """New 256-bit random master key (one per peer pair)."""
    return secrets.token_bytes(32)


def load_key(path: Path) -> bytes:
    """Load a hex-encoded key file; refuses group/world-accessible files on POSIX."""
    path = Path(path)
    if os.name == "posix" and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise BridgeError(f"key file {path} must not be accessible by group/others (chmod 600)")
    key = bytes.fromhex(path.read_text().strip())
    if len(key) < MIN_KEY_BYTES:
        raise BridgeError("key too short")
    return key


def save_key(path: Path, key: bytes) -> None:
    """Write a hex key file with 0600 permissions."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key.hex() + "\n")


def _hkdf(master: bytes, info: bytes) -> bytes:
    """HKDF-SHA256 (RFC 5869), single output block (32 bytes)."""
    prk = hmac.new(b"\x00" * 32, master, hashlib.sha256).digest()
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()


_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ROUTE_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


@dataclass(frozen=True)
class PeerConfig:
    name: str
    key: bytes = field(repr=False)
    allowed_routes: tuple = ()  # fnmatch patterns this peer may call; empty = none
    encrypt: bool = False

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"invalid peer name: {self.name!r}")
        if len(self.key) < MIN_KEY_BYTES:
            raise ValueError("peer key must be at least 32 bytes")
        if self.encrypt and AESGCM is None:
            raise ValueError("encrypt=True requires the 'cryptography' package")


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #

KINDS = frozenset({"command", "event", "response", "error"})


@dataclass(slots=True)
class Envelope:
    kind: str
    route: str
    payload: dict
    sender: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: Optional[str] = None
    ts: float = field(default_factory=time.time)
    nonce: str = field(default_factory=lambda: secrets.token_hex(12))
    v: int = VERSION

    def to_bytes(self) -> bytes:
        return json.dumps({"v": self.v, "kind": self.kind, "route": self.route, "payload": self.payload,
                           "sender": self.sender, "id": self.id, "correlation_id": self.correlation_id,
                           "ts": self.ts, "nonce": self.nonce},
                          separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes, max_bytes: int) -> "Envelope":
        if len(data) > max_bytes:
            raise ProtocolError("envelope too large")
        try:
            d = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ProtocolError("malformed envelope") from None
        if not isinstance(d, dict) or d.get("v") != VERSION:
            raise ProtocolError("unsupported envelope version")
        try:
            env = cls(kind=d["kind"], route=d["route"], payload=d["payload"], sender=d["sender"],
                      id=d["id"], correlation_id=d.get("correlation_id"), ts=d["ts"], nonce=d["nonce"])
        except KeyError as e:
            raise ProtocolError(f"missing field {e}") from None
        ok = (env.kind in KINDS and isinstance(env.route, str) and len(env.route) <= 128
              and (_ROUTE_RE.match(env.route) or env.kind == "error")
              and isinstance(env.payload, dict) and isinstance(env.sender, str)
              and isinstance(env.id, str) and len(env.id) <= 64
              and isinstance(env.nonce, str) and 8 <= len(env.nonce) <= 64
              and isinstance(env.ts, (int, float)) and not isinstance(env.ts, bool)
              and (env.correlation_id is None or isinstance(env.correlation_id, str)))
        if not ok:
            raise ProtocolError("invalid envelope fields")
        return env


class ReplayGuard:
    """Rejects stale timestamps and re-used nonces (per sender)."""

    def __init__(self, max_skew: float = 30.0, capacity: int = 50_000) -> None:
        self.max_skew, self.capacity = max_skew, capacity
        self._seen: "OrderedDict[tuple, float]" = OrderedDict()

    def check(self, sender: str, nonce: str, ts: float) -> None:
        now = time.time()
        if abs(now - ts) > self.max_skew:
            raise ReplayError("timestamp outside allowed window")
        while self._seen and (next(iter(self._seen.values())) < now or len(self._seen) >= self.capacity):
            self._seen.popitem(last=False)
        key = (sender, nonce)
        if key in self._seen:
            raise ReplayError("nonce already used")
        self._seen[key] = now + 2 * self.max_skew


# --------------------------------------------------------------------------- #
# The bridge
# --------------------------------------------------------------------------- #

RouteHandler = Callable[[dict, Envelope], Awaitable[Optional[dict]]]
EventCallback = Callable[[str, dict], Awaitable[None]]
Transport = Callable[[bytes], Awaitable[bytes]]


class NexusBridge:
    def __init__(self, local_name: str, peers: Iterable[PeerConfig] = (), *,
                 max_frame_bytes: int = 1_048_576, handler_timeout: float = 30.0,
                 max_concurrency: int = 64, max_skew: float = 30.0) -> None:
        if not _NAME_RE.match(local_name):
            raise ValueError("invalid local_name")
        self.local_name = local_name
        self.max_frame = max_frame_bytes
        self.handler_timeout = handler_timeout
        self._peers: dict[str, PeerConfig] = {}
        self._keys: dict[str, tuple] = {}
        for p in peers:
            self.add_peer(p)
        self._routes: dict[str, RouteHandler] = {}
        self._subs: dict[str, list] = {}
        self._sem = asyncio.Semaphore(max_concurrency)
        self._replay = ReplayGuard(max_skew)

    # ---- configuration ---------------------------------------------------- #

    def add_peer(self, peer: PeerConfig) -> None:
        if peer.name == self.local_name:
            raise ValueError("peer name must differ from local name")
        self._peers[peer.name] = peer
        self._keys[peer.name] = (_hkdf(peer.key, b"zentrax-bridge/v1/sign"),
                                 _hkdf(peer.key, b"zentrax-bridge/v1/enc"))

    def route(self, name: str):
        """Decorator: register an async handler `(payload, envelope) -> dict | None`."""
        if not _ROUTE_RE.match(name):
            raise ValueError(f"invalid route name: {name!r}")

        def decorator(fn: RouteHandler) -> RouteHandler:
            if name in self._routes:
                raise ValueError(f"route already registered: {name}")
            self._routes[name] = fn
            return fn
        return decorator

    # ---- sealing / opening frames ---------------------------------------- #

    def _header(self, mode: int) -> bytes:
        name = self.local_name.encode("ascii")
        return MAGIC + bytes([VERSION, mode, len(name)]) + name

    def _seal(self, peer: PeerConfig, plaintext: bytes) -> bytes:
        sign_key, enc_key = self._keys[peer.name]
        if peer.encrypt:
            header, nonce = self._header(MODE_GCM), os.urandom(12)
            return header + nonce + AESGCM(enc_key).encrypt(nonce, plaintext, header)
        header = self._header(MODE_HMAC)
        tag = hmac.new(sign_key, header + plaintext, hashlib.sha256).digest()
        return header + tag + plaintext

    def _open(self, frame: bytes) -> tuple:
        """Authenticate + decrypt. Returns (peer, plaintext). Raises BridgeError."""
        if len(frame) < 6 or len(frame) > self.max_frame + 512:
            raise ProtocolError("bad frame size")
        if frame[:2] != MAGIC or frame[2] != VERSION:
            raise ProtocolError("bad magic/version")
        mode, slen = frame[3], frame[4]
        try:
            sender = frame[5:5 + slen].decode("ascii")
        except UnicodeDecodeError:
            raise ProtocolError("bad sender") from None
        peer = self._peers.get(sender)
        if peer is None:
            raise AuthenticationError("unknown peer")
        if mode != (MODE_GCM if peer.encrypt else MODE_HMAC):
            raise AuthenticationError("mode mismatch (possible downgrade)")
        header, body = frame[:5 + slen], frame[5 + slen:]
        sign_key, enc_key = self._keys[sender]
        if mode == MODE_HMAC:
            if len(body) < 32:
                raise ProtocolError("truncated frame")
            tag, plaintext = body[:32], body[32:]
            expected = hmac.new(sign_key, header + plaintext, hashlib.sha256).digest()
            if not hmac.compare_digest(tag, expected):
                raise AuthenticationError("bad signature")
            return peer, plaintext
        if len(body) < 12 + 16:
            raise ProtocolError("truncated frame")
        try:
            return peer, AESGCM(enc_key).decrypt(body[:12], body[12:], header)
        except Exception:
            raise AuthenticationError("decryption failed") from None

    # ---- server side ------------------------------------------------------ #

    async def handle_frame(self, frame: bytes) -> bytes:
        """Process one inbound frame; returns the reply frame (b'' = drop silently)."""
        try:
            peer, plaintext = self._open(frame)
            env = Envelope.from_bytes(plaintext, self.max_frame)
            if env.sender != peer.name:
                raise AuthenticationError("sender mismatch")
            if env.kind not in ("command", "event"):
                raise ProtocolError("unexpected envelope kind")
            self._replay.check(peer.name, env.nonce, env.ts)
        except BridgeError as exc:
            log.warning("dropped frame: %s", exc)
            return b""

        try:
            if not any(fnmatch.fnmatchcase(env.route, pat) for pat in peer.allowed_routes):
                raise AuthorizationError(env.route)
            out = await self._dispatch(env.route, env.payload, env)
            reply = Envelope("response", env.route, out, self.local_name, correlation_id=env.id)
        except AuthorizationError:
            log.warning("peer '%s' denied route '%s'", peer.name, env.route)
            reply = self._error(env, "forbidden", "route not permitted")
        except RouteNotFound:
            reply = self._error(env, "not_found", "unknown route")
        except HandlerError as exc:
            reply = self._error(env, exc.code, exc.message)
        except asyncio.TimeoutError:
            reply = self._error(env, "timeout", "handler timed out")
        except Exception:
            log.exception("handler for '%s' crashed", env.route)
            reply = self._error(env, "internal_error", "internal error")
        return self._seal(peer, reply.to_bytes())

    def _error(self, req: Envelope, code: str, message: str) -> Envelope:
        return Envelope("error", req.route, {"code": code, "message": message}, self.local_name,
                        correlation_id=req.id)

    async def _dispatch(self, route: str, payload: dict, env: Envelope) -> dict:
        handler = self._routes.get(route)
        if handler is None:
            raise RouteNotFound(route)
        async with self._sem:
            out = await asyncio.wait_for(handler(payload, env), self.handler_timeout)
        if out is None:
            return {}
        if not isinstance(out, dict):
            raise TypeError("handlers must return a dict or None")
        return out

    # ---- internal (in-process) routing ------------------------------------ #

    async def call_local(self, route: str, payload: Optional[dict] = None) -> dict:
        """Call a registered route in-process (trusted caller, no crypto, no ACL)."""
        env = Envelope("command", route, payload or {}, self.local_name)
        return await self._dispatch(route, env.payload, env)

    def subscribe(self, topic: str, callback: EventCallback) -> None:
        self._subs.setdefault(topic, []).append(callback)

    async def publish(self, topic: str, payload: dict) -> None:
        """Fan an event out to in-process subscribers (errors are isolated + logged)."""
        results = await asyncio.gather(*(cb(topic, payload) for cb in self._subs.get(topic, [])),
                                       return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                log.error("subscriber for '%s' failed: %r", topic, r)

    # ---- client side ------------------------------------------------------ #

    def build_request(self, peer_name: str, route: str, payload: dict, *,
                      kind: str = "command") -> tuple:
        """Returns (frame, request_id) ready to hand to any transport."""
        if kind not in ("command", "event") or not _ROUTE_RE.match(route):
            raise ValueError("bad kind or route")
        env = Envelope(kind, route, payload, self.local_name)
        return self._seal(self._peers[peer_name], env.to_bytes()), env.id

    def parse_response(self, peer_name: str, frame: bytes, request_id: str) -> Envelope:
        if not frame:
            raise ProtocolError("empty reply (request rejected by peer)")
        peer, plaintext = self._open(frame)
        env = Envelope.from_bytes(plaintext, self.max_frame)
        if peer.name != peer_name or env.sender != peer_name:
            raise AuthenticationError("reply from unexpected peer")
        if env.kind not in ("response", "error") or env.correlation_id != request_id:
            raise ProtocolError("reply does not match request")
        self._replay.check(peer.name, env.nonce, env.ts)
        return env

    async def call(self, peer_name: str, route: str, payload: Optional[dict], transport: Transport,
                   *, timeout: float = 10.0) -> dict:
        """Send a command to a peer and return its response payload."""
        frame, rid = self.build_request(peer_name, route, payload or {})
        reply = await asyncio.wait_for(transport(frame), timeout)
        env = self.parse_response(peer_name, reply, rid)
        if env.kind == "error":
            raise RemoteError(str(env.payload.get("code", "error")), str(env.payload.get("message", "")))
        return env.payload

    # ---- loopback TCP transport ------------------------------------------ #

    async def serve_tcp(self, host: str = "127.0.0.1", port: int = 0, *, allow_remote: bool = False,
                        idle_timeout: float = 15.0) -> asyncio.AbstractServer:
        """Serve frames over TCP. Loopback only unless allow_remote=True (then use encrypt=True!)."""
        if host not in LOOPBACK_HOSTS and not allow_remote:
            raise BridgeError("refusing to bind a non-loopback address without allow_remote=True")

        async def on_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                while True:
                    hdr = await asyncio.wait_for(reader.readexactly(4), idle_timeout)
                    (n,) = struct.unpack(">I", hdr)
                    if n == 0 or n > self.max_frame + 512:
                        break
                    frame = await asyncio.wait_for(reader.readexactly(n), idle_timeout)
                    reply = await self.handle_frame(frame)
                    if not reply:
                        break  # drop the connection on any rejected frame
                    writer.write(struct.pack(">I", len(reply)) + reply)
                    await writer.drain()
            except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
                pass
            finally:
                writer.close()
                with_suppress = getattr(writer, "wait_closed", None)
                if with_suppress:
                    try:
                        await with_suppress()
                    except Exception:
                        pass

        server = await asyncio.start_server(on_connection, host, port)
        log.info("bridge '%s' listening on %s", self.local_name,
                 ", ".join(str(s.getsockname()) for s in server.sockets))
        return server


class TCPTransport:
    """Client transport for `NexusBridge.call` (one connection per request)."""

    def __init__(self, host: str, port: int, max_frame_bytes: int = 1_048_576) -> None:
        self.host, self.port, self.max = host, port, max_frame_bytes

    async def __call__(self, frame: bytes) -> bytes:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            writer.write(struct.pack(">I", len(frame)) + frame)
            await writer.drain()
            try:
                (n,) = struct.unpack(">I", await reader.readexactly(4))
            except asyncio.IncompleteReadError:
                return b""  # server dropped us
            if n > self.max + 512:
                raise ProtocolError("reply too large")
            return await reader.readexactly(n)
        finally:
            writer.close()


# --------------------------------------------------------------------------- #
# Integration with autonomous_executor (duck-typed, no hard import)
# --------------------------------------------------------------------------- #


def attach_executor(bridge: NexusBridge, executor: Any) -> None:
    """Expose an AutonomousExecutor through the bridge.

    Routes:  executor.submit {directive...}      -> {"id"}
             executor.result {"id", "wait"?}     -> ExecutionResult dict
             executor.stats  {}                  -> stats dict
    Topic:   "executor.result" is published to in-process subscribers for every result.
    Remote peers still need an ACL such as allowed_routes=("executor.*",).
    """

    def safe(exc: Exception) -> HandlerError:
        code = getattr(exc, "code", None)  # ExecutorError subclasses carry a stable code
        return HandlerError(code, str(exc)) if code else HandlerError("internal_error", "internal error")

    @bridge.route("executor.submit")
    async def _submit(payload: dict, env: Envelope) -> dict:
        try:  # `source` is taken from the authenticated sender, never from the payload
            return {"id": await executor.submit(payload, source=env.sender)}
        except Exception as exc:
            raise safe(exc) from exc

    @bridge.route("executor.result")
    async def _result(payload: dict, env: Envelope) -> dict:
        did, wait = payload.get("id"), payload.get("wait", 0)
        if not isinstance(did, str) or isinstance(wait, bool) or not isinstance(wait, (int, float)):
            raise HandlerError("invalid_request", "need string 'id' and numeric 'wait'")
        try:
            return (await executor.result(did, timeout=min(max(wait, 0), 25) or None)).to_dict()
        except asyncio.TimeoutError:
            raise HandlerError("pending", "result not ready") from None
        except Exception as exc:
            raise safe(exc) from exc

    @bridge.route("executor.stats")
    async def _stats(payload: dict, env: Envelope) -> dict:
        return executor.stats()

    executor.add_listener(lambda result: bridge.publish("executor.result", result.to_dict()))


# --------------------------------------------------------------------------- #
# Demo:  python -m super_intelligence.nexus_bridge
# --------------------------------------------------------------------------- #


async def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    key = generate_key()
    core = NexusBridge("core", [PeerConfig("backend", key, allowed_routes=("demo.*",), encrypt=AESGCM is not None)])
    backend = NexusBridge("backend", [PeerConfig("core", key, encrypt=AESGCM is not None)])

    @core.route("demo.echo")
    async def echo(payload: dict, env: Envelope) -> dict:
        return {"you_said": payload.get("text"), "from": env.sender}

    server = await core.serve_tcp()
    port = server.sockets[0].getsockname()[1]
    transport = TCPTransport("127.0.0.1", port)
    print(await backend.call("core", "demo.echo", {"text": "hello nexus"}, transport))
    try:
        await backend.call("core", "executor.submit", {}, transport)  # not in the ACL
    except RemoteError as e:
        print("denied as expected:", e)
    server.close()
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(_demo())
