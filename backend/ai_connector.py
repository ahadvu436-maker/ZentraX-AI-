"""
backend/ai_connector.py — Bridge between the ZentraX-AI backend and the
Super Intelligence core (super_intelligence/).

Responsibilities
----------------
1. Path resolution: make `super_intelligence/` importable from the backend.
2. Core integration: lazily create and own `NexusBridge` and
   `AutonomousExecutor`.
3. Request handling: validate, route, execute and normalise every command
   into a safe, JSON-serialisable response, with logging and error handling.

Usage (sync)
------------
    from backend.ai_connector import get_ai_connector

    connector = get_ai_connector()
    result = connector.handle_incoming_request(
        {"command": "analyze_logs", "payload": {"hours": 24}}
    )

Usage (FastAPI / async)
-----------------------
    @app.post("/ai/command")
    async def ai_command(body: dict):
        return await get_ai_connector().handle_incoming_request_async(body)

Request format
--------------
    {
        "command": "string (required)",   # what the AI core should do
        "payload": {...},                 # optional arguments
        "target":  "bridge" | "executor", # optional, default "bridge"
        "request_id": "string"            # optional, generated if missing
    }

Response format (always returned, never raises)
-----------------------------------------------
    {"success": bool, "request_id": str, "target": str,
     "data": Any | None, "error": str | None, "duration_ms": float}

NOTE ON THE CORE API
--------------------
The exact public methods of NexusBridge / AutonomousExecutor are resolved by
name from BRIDGE_METHODS / EXECUTOR_METHODS below (first match wins). If your
core uses different method names, edit those tuples — nothing else changes.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("zentrax.ai_connector")

# --------------------------------------------------------------------------- #
# 1. Path resolution
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Override with the ZENTRAX_SI_PATH env var if the core lives elsewhere.
SUPER_INTELLIGENCE_DIR = Path(
    os.environ.get("ZENTRAX_SI_PATH", PROJECT_ROOT / "super_intelligence")
).resolve()


def _ensure_on_sys_path(path: Path) -> None:
    """Append `path` to sys.path once (idempotent)."""
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)
        logger.debug("Added to sys.path: %s", path_str)


def _setup_paths() -> None:
    if not SUPER_INTELLIGENCE_DIR.is_dir():
        raise FileNotFoundError(
            f"super_intelligence directory not found: {SUPER_INTELLIGENCE_DIR}"
        )
    _ensure_on_sys_path(SUPER_INTELLIGENCE_DIR)
    # Also expose the project root so core modules that use package-style
    # imports (e.g. `from super_intelligence.x import y`) keep working.
    _ensure_on_sys_path(PROJECT_ROOT)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class AIConnectorError(Exception):
    """Base error for connector failures."""


class InvalidCommandError(AIConnectorError):
    """The incoming command_data failed validation."""


class CoreUnavailableError(AIConnectorError):
    """The AI core could not be imported or initialised."""


# --------------------------------------------------------------------------- #
# 2. Connector
# --------------------------------------------------------------------------- #

class BackendAIConnector:
    """Thread-safe adapter around NexusBridge and AutonomousExecutor."""

    # Candidate method names on the core objects, checked in order.
    BRIDGE_METHODS: tuple[str, ...] = (
        "handle_request", "process_request", "process_command", "process", "send", "run",
    )
    EXECUTOR_METHODS: tuple[str, ...] = (
        "execute", "execute_task", "execute_command", "run_task", "run",
    )
    VALID_TARGETS = ("bridge", "executor")
    DEFAULT_TIMEOUT_S = 60.0  # used by the async handler only

    def __init__(
        self,
        bridge: Any = None,
        executor: Any = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """
        Args:
            bridge:   Optional pre-built NexusBridge (dependency injection, tests).
            executor: Optional pre-built AutonomousExecutor.
            timeout_s: Max seconds for async requests.
        Core objects are created lazily on first use unless injected.
        """
        self._bridge = bridge
        self._executor = executor
        self._timeout_s = timeout_s
        self._init_lock = threading.Lock()
        self._initialized = bridge is not None and executor is not None

    # ------------------------------------------------------------ init/health
    def initialize(self) -> None:
        """Import and construct the core components (safe to call repeatedly)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            try:
                _setup_paths()
                if self._bridge is None:
                    module = importlib.import_module("nexus_bridge")
                    self._bridge = module.NexusBridge()
                if self._executor is None:
                    module = importlib.import_module("autonomous_executor")
                    self._executor = module.AutonomousExecutor()
            except Exception as exc:  # import errors, bad constructors, etc.
                logger.exception("Failed to initialise Super Intelligence core")
                raise CoreUnavailableError(
                    f"AI core initialisation failed: {exc}"
                ) from exc
            self._initialized = True
            logger.info("Super Intelligence core initialised "
                        "(NexusBridge + AutonomousExecutor)")

    def health_check(self) -> dict[str, Any]:
        """Report connector status without raising."""
        try:
            self.initialize()
            return {"healthy": True, "bridge": type(self._bridge).__name__,
                    "executor": type(self._executor).__name__}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    def shutdown(self) -> None:
        """Call close()/shutdown()/stop() on core objects if they provide it."""
        for obj in (self._executor, self._bridge):
            for name in ("shutdown", "close", "stop"):
                fn = getattr(obj, name, None)
                if callable(fn):
                    try:
                        result = fn()
                        if inspect.isawaitable(result):
                            asyncio.run(_await(result))
                    except Exception:
                        logger.exception("Error while shutting down %s", type(obj).__name__)
                    break
        self._initialized = False

    # ------------------------------------------------------------ validation
    @staticmethod
    def _validate(command_data: Any) -> dict[str, Any]:
        """Check the request shape and fill in defaults."""
        if not isinstance(command_data, dict):
            raise InvalidCommandError("command_data must be a dict")
        command = command_data.get("command")
        if not isinstance(command, str) or not command.strip():
            raise InvalidCommandError("'command' is required and must be a non-empty string")
        target = command_data.get("target", "bridge")
        if target not in BackendAIConnector.VALID_TARGETS:
            raise InvalidCommandError(
                f"'target' must be one of {BackendAIConnector.VALID_TARGETS}"
            )
        payload = command_data.get("payload", {})
        if not isinstance(payload, dict):
            raise InvalidCommandError("'payload' must be a dict if provided")
        return {
            "command": command.strip(),
            "payload": payload,
            "target": target,
            "request_id": str(command_data.get("request_id") or uuid.uuid4()),
        }

    # ---------------------------------------------------------- core calling
    def _resolve_method(self, target: str) -> Callable[..., Any]:
        """Find the first matching public method on the chosen core object."""
        obj = self._bridge if target == "bridge" else self._executor
        names = self.BRIDGE_METHODS if target == "bridge" else self.EXECUTOR_METHODS
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn
        raise CoreUnavailableError(
            f"{type(obj).__name__} exposes none of the expected methods {names}. "
            f"Update {target.upper()}_METHODS in ai_connector.py."
        )

    @staticmethod
    def _build_call(fn: Callable[..., Any], req: dict[str, Any]) -> tuple[tuple, dict]:
        """
        Adapt to the core method's signature: pass the whole request dict if it
        takes a single parameter, else pass command/payload as keywords.
        """
        try:
            params = [p for p in inspect.signature(fn).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        except (TypeError, ValueError):
            return (req,), {}
        names = {p.name for p in params}
        if {"command", "payload"} <= names:
            return (), {"command": req["command"], "payload": req["payload"]}
        return ({"command": req["command"], "payload": req["payload"],
                 "request_id": req["request_id"]},), {}

    # --------------------------------------------------------- response utils
    @staticmethod
    def _safe_data(data: Any) -> Any:
        """Make arbitrary core output JSON-serialisable (falls back to str)."""
        try:
            return json.loads(json.dumps(data, default=str))
        except Exception:
            return str(data)

    @staticmethod
    def _response(req_id: str, target: str, started: float, *,
                  data: Any = None, error: Optional[str] = None) -> dict[str, Any]:
        return {
            "success": error is None,
            "request_id": req_id,
            "target": target,
            "data": BackendAIConnector._safe_data(data) if error is None else None,
            "error": error,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    # ------------------------------------------------------- 3. sync handler
    def handle_incoming_request(self, command_data: dict) -> dict[str, Any]:
        """
        Validate and forward a backend command to the AI core (blocking).
        Never raises; failures are returned as {"success": False, "error": ...}.
        Do not call from inside a running event loop; use the async variant.
        """
        started = time.perf_counter()
        req_id, target = "unknown", "bridge"
        try:
            req = self._validate(command_data)
            req_id, target = req["request_id"], req["target"]
            logger.info("[%s] -> %s: command=%r", req_id, target, req["command"])

            self.initialize()
            fn = self._resolve_method(target)
            args, kwargs = self._build_call(fn, req)
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):  # core method is async
                result = asyncio.run(_await(result))

            resp = self._response(req_id, target, started, data=result)
            logger.info("[%s] <- ok in %.1f ms", req_id, resp["duration_ms"])
            return resp
        except InvalidCommandError as exc:
            logger.warning("[%s] invalid request: %s", req_id, exc)
            return self._response(req_id, target, started, error=f"Invalid request: {exc}")
        except CoreUnavailableError as exc:
            logger.error("[%s] core unavailable: %s", req_id, exc)
            return self._response(req_id, target, started, error=str(exc))
        except Exception as exc:
            logger.exception("[%s] AI core raised an error", req_id)
            return self._response(req_id, target, started,
                                  error=f"AI core error: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------ 3. async handler
    async def handle_incoming_request_async(self, command_data: dict) -> dict[str, Any]:
        """
        Non-blocking variant for FastAPI/asyncio. Sync core methods run in a
        worker thread; async core methods are awaited directly. Applies a timeout.
        """
        started = time.perf_counter()
        req_id, target = "unknown", "bridge"
        try:
            req = self._validate(command_data)
            req_id, target = req["request_id"], req["target"]
            logger.info("[%s] -> %s (async): command=%r", req_id, target, req["command"])

            await asyncio.to_thread(self.initialize)  # import/init may block
            fn = self._resolve_method(target)
            args, kwargs = self._build_call(fn, req)

            async def _run() -> Any:
                if inspect.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                result = await asyncio.to_thread(fn, *args, **kwargs)
                return await result if inspect.isawaitable(result) else result

            result = await asyncio.wait_for(_run(), timeout=self._timeout_s)
            resp = self._response(req_id, target, started, data=result)
            logger.info("[%s] <- ok in %.1f ms", req_id, resp["duration_ms"])
            return resp
        except asyncio.TimeoutError:
            logger.error("[%s] timed out after %.0fs", req_id, self._timeout_s)
            return self._response(req_id, target, started,
                                  error=f"AI core timed out after {self._timeout_s:.0f}s")
        except InvalidCommandError as exc:
            logger.warning("[%s] invalid request: %s", req_id, exc)
            return self._response(req_id, target, started, error=f"Invalid request: {exc}")
        except CoreUnavailableError as exc:
            logger.error("[%s] core unavailable: %s", req_id, exc)
            return self._response(req_id, target, started, error=str(exc))
        except Exception as exc:
            logger.exception("[%s] AI core raised an error", req_id)
            return self._response(req_id, target, started,
                                  error=f"AI core error: {type(exc).__name__}: {exc}")


async def _await(awaitable: Any) -> Any:
    """Small helper so asyncio.run() can execute any awaitable."""
    return await awaitable


# --------------------------------------------------------------------------- #
# Shared instance (use as a FastAPI dependency or import directly)
# --------------------------------------------------------------------------- #

_connector: Optional[BackendAIConnector] = None
_connector_lock = threading.Lock()


def get_ai_connector() -> BackendAIConnector:
    """Return the process-wide BackendAIConnector, creating it on first call."""
    global _connector
    if _connector is None:
        with _connector_lock:
            if _connector is None:
                _connector = BackendAIConnector()
    return _connector
