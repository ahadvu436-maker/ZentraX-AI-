"""temporal_entropy_vault.py - Time-windowed HMAC signatures and rotating runtime tokens.

Keys are derived per epoch from a master secret via HKDF-style HMAC chaining.
Verification accepts a bounded sliding window of neighbouring epochs to tolerate
clock skew and in-flight requests, using constant-time comparison.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass(frozen=True, slots=True)
class VaultConfig:
    epoch_seconds: int = 5
    past_windows: int = 1
    future_windows: int = 1
    token_ttl_epochs: int = 6
    digest: str = "sha256"
    cache_size: int = 64


class TemporalEntropyVault:
    def __init__(self, master_secret: Optional[bytes] = None, config: Optional[VaultConfig] = None) -> None:
        secret = master_secret or (os.environ.get("ZENTRAX_VAULT_SECRET", "").encode() or None)
        if secret is None or len(secret) < 32:
            raise ValueError("master secret of at least 32 bytes is required (ZENTRAX_VAULT_SECRET)")
        self._master = secret
        self.cfg = config or VaultConfig()
        self._digest = getattr(hashlib, self.cfg.digest)
        self._key_cache: Dict[int, bytes] = {}
        self._lock = asyncio.Lock()
        self._rotation_task: Optional[asyncio.Task] = None

    # ---- epoch & key derivation -------------------------------------------------
    def epoch(self, at: Optional[float] = None) -> int:
        return int((at if at is not None else time.time()) // self.cfg.epoch_seconds)

    def _derive(self, epoch: int) -> bytes:
        cached = self._key_cache.get(epoch)
        if cached is not None:
            return cached
        prk = hmac.new(b"zentrax-temporal-vault-v1", self._master, self._digest).digest()
        key = hmac.new(prk, b"epoch:" + epoch.to_bytes(8, "big") + b"\x01", self._digest).digest()
        if len(self._key_cache) >= self.cfg.cache_size:
            for old in sorted(self._key_cache)[: len(self._key_cache) // 2]:
                self._key_cache.pop(old, None)
        self._key_cache[epoch] = key
        return key

    # ---- route signatures -------------------------------------------------------
    def sign(self, context: str, at: Optional[float] = None) -> str:
        """Signature for a route/context, valid for the current epoch window."""
        e = self.epoch(at)
        mac = hmac.new(self._derive(e), context.encode(), self._digest).digest()
        return f"{e}.{_b64(mac)}"

    def verify(self, context: str, signature: str, at: Optional[float] = None) -> bool:
        try:
            epoch_str, mac_b64 = signature.split(".", 1)
            claimed = int(epoch_str)
            provided = _unb64(mac_b64)
        except (ValueError, TypeError):
            return False
        now = self.epoch(at)
        if not (now - self.cfg.past_windows <= claimed <= now + self.cfg.future_windows):
            return False
        expected = hmac.new(self._derive(claimed), context.encode(), self._digest).digest()
        return hmac.compare_digest(expected, provided)

    # ---- runtime tokens ---------------------------------------------------------
    def issue_token(self, subject: str, scope: str = "default", at: Optional[float] = None) -> str:
        e = self.epoch(at)
        nonce = secrets.token_bytes(12)
        payload = f"{subject}|{scope}|{e}".encode()
        mac = hmac.new(self._derive(e), nonce + payload, self._digest).digest()
        return ".".join((_b64(nonce), _b64(payload), _b64(mac)))

    def validate_token(self, token: str, scope: str = "default", at: Optional[float] = None) -> Optional[str]:
        """Returns the subject if valid, else None."""
        try:
            nonce_b, payload_b, mac_b = token.split(".")
            nonce, payload, provided = _unb64(nonce_b), _unb64(payload_b), _unb64(mac_b)
            subject, tok_scope, epoch_str = payload.decode().rsplit("|", 2)
            issued = int(epoch_str)
        except (ValueError, TypeError, UnicodeDecodeError):
            return None
        now = self.epoch(at)
        if tok_scope != scope or issued > now + self.cfg.future_windows:
            return None
        if now - issued > self.cfg.token_ttl_epochs:
            return None
        expected = hmac.new(self._derive(issued), nonce + payload, self._digest).digest()
        return subject if hmac.compare_digest(expected, provided) else None

    # ---- background rotation ----------------------------------------------------
    async def start(self, on_rotate: Optional[Callable[[int], Awaitable[None]]] = None) -> None:
        if self._rotation_task and not self._rotation_task.done():
            return

        async def _loop() -> None:
            last = self.epoch()
            while True:
                nxt = (last + 1) * self.cfg.epoch_seconds
                await asyncio.sleep(max(0.0, nxt - time.time()))
                last = self.epoch()
                async with self._lock:
                    floor = last - (self.cfg.token_ttl_epochs + self.cfg.past_windows + 2)
                    for k in [k for k in self._key_cache if k < floor]:
                        self._key_cache.pop(k, None)
                if on_rotate:
                    try:
                        await on_rotate(last)
                    except Exception:  # noqa: BLE001
                        pass

        self._rotation_task = asyncio.create_task(_loop(), name="temporal-entropy-rotation")

    async def stop(self) -> None:
        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass
            self._rotation_task = None


# ---- FastAPI integration -------------------------------------------------------
_sig_header = APIKeyHeader(name="X-Entropy-Signature", auto_error=False)
_tok_header = APIKeyHeader(name="X-Runtime-Token", auto_error=False)


def get_vault(request: Request) -> TemporalEntropyVault:
    return request.app.state.entropy_vault


def require_route_signature(request: Request, sig: Optional[str] = Depends(_sig_header),
                            vault: TemporalEntropyVault = Depends(get_vault)) -> None:
    context = f"{request.method}:{request.url.path}"
    if not sig or not vault.verify(context, sig):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid or expired signature")


def require_runtime_token(scope: str = "default"):
    def _dep(token: Optional[str] = Depends(_tok_header),
             vault: TemporalEntropyVault = Depends(get_vault)) -> str:
        subject = vault.validate_token(token, scope) if token else None
        if subject is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token")
        return subject
    return _dep


def install(app, master_secret: Optional[bytes] = None, config: Optional[VaultConfig] = None) -> TemporalEntropyVault:
    vault = TemporalEntropyVault(master_secret, config)
    app.state.entropy_vault = vault

    @app.on_event("startup")
    async def _start() -> None:  # pragma: no cover
        await vault.start()

    @app.on_event("shutdown")
    async def _stop() -> None:  # pragma: no cover
        await vault.stop()

    return vault
