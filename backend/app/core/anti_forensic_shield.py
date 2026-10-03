"""
anti_forensic_shield.py

Memory-safe secret handling utilities for FastAPI / asyncio services.

This module does NOT attempt to detect debuggers, tamper with memory
inspection tools, or serve fake data to an "attacker inspecting
runtime memory." Techniques like that (anti-debugging, anti-dumping,
environment-fingerprinting evasion) are standard malware defenses
used to resist analysis by security researchers and incident
responders, and this module will not implement them regardless of
framing.

What it provides instead -- legitimate, standard practice for
reducing the value of a memory dump if one ever occurs:

  * SecretBox: holds sensitive values (API keys, tokens, passwords)
    in a mutable, explicitly-wipeable buffer instead of an immutable
    Python str, so the value can be overwritten rather than lingering
    in memory/garbage until GC.
  * Short-lived secret handling: context manager that wipes a secret
    as soon as its block of use completes.
  * Redaction helpers for logging so secrets never land in logs,
    tracebacks, or error responses.
  * A FastAPI exception handler that strips sensitive fields from
    error payloads before they're returned to clients.

Caveats (stated honestly): CPython strings/bytes can be copied
internally by the interpreter, and this module cannot guarantee a
secret never touches a swapped page or a core dump. For strong
guarantees, use an external secrets manager/KMS/HSM and avoid
holding raw secrets in process memory longer than necessary -- this
module minimizes exposure, it does not make memory dumps useless.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_REDACT_KEYS = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|authorization|bearer|"
    r"credential|private[_-]?key|session[_-]?id)",
    re.IGNORECASE,
)
_REDACTED = "***REDACTED***"


class SecretBox:
    """
    Holds a secret in a mutable bytearray so it can be explicitly
    zeroed out after use, instead of relying on Python's string
    immutability and garbage collection timing.
    """

    __slots__ = ("_buf", "_wiped")

    def __init__(self, value: str | bytes) -> None:
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._buf = bytearray(value)
        self._wiped = False

    def reveal(self) -> bytes:
        if self._wiped:
            raise ValueError("Secret has already been wiped.")
        return bytes(self._buf)

    def reveal_str(self) -> str:
        return self.reveal().decode("utf-8")

    def wipe(self) -> None:
        if self._wiped:
            return
        for i in range(len(self._buf)):
            self._buf[i] = 0
        self._wiped = True

    def __del__(self) -> None:
        try:
            self.wipe()
        except Exception:
            pass

    def __repr__(self) -> str:
        return "SecretBox(****)"

    def __str__(self) -> str:
        return "****"


@contextmanager
def use_secret(value: str | bytes) -> Iterator[SecretBox]:
    """Synchronous scoped secret: wiped automatically on exit."""
    box = SecretBox(value)
    try:
        yield box
    finally:
        box.wipe()


@asynccontextmanager
async def use_secret_async(value: str | bytes) -> AsyncIterator[SecretBox]:
    """Async scoped secret: wiped automatically on exit."""
    box = SecretBox(value)
    try:
        yield box
    finally:
        box.wipe()


def load_secret_from_env(var_name: str) -> SecretBox:
    raw = os.environ.get(var_name)
    if raw is None:
        raise KeyError(f"Environment variable '{var_name}' is not set.")
    return SecretBox(raw)


def redact(value: Any) -> Any:
    """Recursively redact dict/list structures for safe logging."""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _REDACT_KEYS.search(str(k)) else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


class RedactingFilter(logging.Filter):
    """Attach to loggers/handlers to scrub secret-like substrings."""

    _inline_pattern = re.compile(
        r'(?i)("(?:password|token|secret|api[_-]?key|authorization)"\s*:\s*")'
        r'[^"]*(")'
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._inline_pattern.sub(r"\1" + _REDACTED + r"\2", record.msg)
        if record.args:
            record.args = tuple(
                self._inline_pattern.sub(r"\1" + _REDACTED + r"\2", str(a))
                if isinstance(a, str) else a
                for a in record.args
            )
        return True


def install_redacting_logger(logger_name: str | None = None) -> None:
    logger = logging.getLogger(logger_name)
    logger.addFilter(RedactingFilter())


def install_safe_error_handler(app: FastAPI) -> None:
    """
    Ensures unhandled exceptions never leak secrets (env values,
    connection strings, header contents) into the HTTP response.
    """

    @app.exception_handler(Exception)
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("anti_forensic_shield").exception(
            "Unhandled exception", extra={"path": str(request.url.path)}
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )
