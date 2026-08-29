"""
toolkit_service.py

Extensible tool/plugin management for ZentraX AI.

Responsibilities:
    - Provide a registry that modular "tools" (helper utilities, external
      integrations, AI-callable functions) plug into, without this service
      knowing anything about their internals.
    - Validate inputs against each tool's declared schema before execution.
    - Enforce per-tool permissions, timeouts, and concurrency limits so one
      misbehaving or slow tool can't degrade the rest of the platform.
    - Keep execution auditable (what ran, for whom, how long) without ever
      logging raw argument/result payloads, which may contain user data.

Concrete tools live in `app/services/tools/` and register themselves via
`ToolRegistry.register()` — this file defines the contract, not any specific
tool (web search, calculator, file lookup, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger("zentrax.toolkit_service")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_TOOL_TIMEOUT_SECONDS = 20
MAX_CONCURRENT_EXECUTIONS_PER_TOOL = 5
MAX_RESULT_SIZE_CHARS = 50_000  # guard against a runaway tool flooding a response


class ToolkitServiceError(Exception):
    """Base class for toolkit-related errors."""


class ToolNotFoundError(ToolkitServiceError):
    pass


class ToolAlreadyRegisteredError(ToolkitServiceError):
    pass


class ToolPermissionError(ToolkitServiceError):
    pass


class ToolValidationError(ToolkitServiceError):
    pass


class ToolExecutionError(ToolkitServiceError):
    """Wraps any exception raised by a tool's own implementation."""


class ToolTimeoutError(ToolkitServiceError):
    pass


# -----------------------------------------------------------------------------
# Tool contract
# -----------------------------------------------------------------------------

class PermissionLevel(str, Enum):
    """Coarse-grained gate on who may invoke a tool. Concrete authorization
    (does *this* user actually have PRIVILEGED access) is the caller's
    responsibility — this enum just lets a tool declare its own floor."""
    PUBLIC = "public"          # any authenticated user
    PRIVILEGED = "privileged"  # elevated / paid / admin users only
    INTERNAL = "internal"      # system-initiated calls only, never user-facing


@dataclass(slots=True)
class ToolContext:
    """
    Per-call context passed to every tool invocation. Carries only what a
    tool needs to act on a user's behalf — never raw credentials. `user_ref`
    is an opaque identifier, consistent with the rest of the platform's
    privacy-by-default approach.
    """
    user_ref: str
    permission_level: PermissionLevel = PermissionLevel.PUBLIC
    request_id: str = ""


class ToolHandler(Protocol):
    """Signature every tool implementation must satisfy."""

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> Any: ...


@dataclass(slots=True)
class ToolSpec:
    """
    Registration record for a single tool.

    `parameters_schema` is a JSON-Schema-like dict (draft-07 subset is
    plenty) describing accepted arguments — used for validation and can be
    handed directly to an LLM as a function-calling schema.
    """
    name: str
    description: str
    handler: ToolHandler
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    required_permission: PermissionLevel = PermissionLevel.PUBLIC
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS
    enabled: bool = True


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

class ToolRegistry:
    """Holds all known tools. Separate from ToolkitService so tools can be
    registered at import time / app startup independently of any single
    service instance's lifecycle."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, *, replace: bool = False) -> None:
        if spec.name in self._tools and not replace:
            raise ToolAlreadyRegisteredError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec
        logger.info("tool_registered name=%s permission=%s", spec.name, spec.required_permission.value)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolNotFoundError(f"No tool registered with name '{name}'.")
        return spec

    def list_available(self, permission_level: PermissionLevel) -> list[ToolSpec]:
        """Return tools an actor at the given permission level may call,
        ordered by name for stable output (e.g. when handed to an LLM)."""
        rank = {PermissionLevel.PUBLIC: 0, PermissionLevel.PRIVILEGED: 1, PermissionLevel.INTERNAL: 2}
        return sorted(
            (
                spec for spec in self._tools.values()
                if spec.enabled and rank[permission_level] >= rank[spec.required_permission]
            ),
            key=lambda s: s.name,
        )

    def as_llm_function_schema(self, permission_level: PermissionLevel) -> list[dict[str, Any]]:
        """Convenience export for AI-provider function-calling / tool-use APIs."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters_schema,
            }
            for spec in self.list_available(permission_level)
        ]


# -----------------------------------------------------------------------------
# Simple JSON-Schema-subset validator
# -----------------------------------------------------------------------------
# Intentionally minimal — covers type checking and required fields, which is
# enough for most tool argument shapes. Swap in `jsonschema` if a tool needs
# richer validation (patterns, enums, nested objects).

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _validate_against_schema(args: dict[str, Any], schema: dict[str, Any]) -> None:
    if not schema:
        return  # tool declared no schema -> no validation to do

    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    missing = [field_name for field_name in required if field_name not in args]
    if missing:
        raise ToolValidationError(f"Missing required argument(s): {', '.join(missing)}")

    for key, value in args.items():
        prop_schema = properties.get(key)
        if prop_schema is None:
            continue  # unknown args are ignored rather than rejected, for forward-compat
        expected_type = _TYPE_MAP.get(prop_schema.get("type", ""))
        if expected_type and not isinstance(value, expected_type):
            raise ToolValidationError(
                f"Argument '{key}' expected type '{prop_schema.get('type')}', got '{type(value).__name__}'."
            )


# -----------------------------------------------------------------------------
# Toolkit service
# -----------------------------------------------------------------------------

class ToolkitService:
    """
    Executes registered tools on behalf of the chat service / API layer,
    enforcing permissions, schema validation, timeouts, and concurrency caps.

    Usage:
        registry = ToolRegistry()
        registry.register(ToolSpec(name="get_weather", handler=get_weather, ...))
        toolkit = ToolkitService(registry)
        result = await toolkit.execute("get_weather", {"city": "Lisbon"}, ctx)
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def _semaphore_for(self, tool_name: str) -> asyncio.Semaphore:
        if tool_name not in self._semaphores:
            self._semaphores[tool_name] = asyncio.Semaphore(MAX_CONCURRENT_EXECUTIONS_PER_TOOL)
        return self._semaphores[tool_name]

    async def execute(self, tool_name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """
        Run a single tool call end-to-end: authorize, validate, execute with
        timeout, and normalize the outcome into a ToolResult (never raises
        for expected failure modes — check `.success`).
        """
        started = time.monotonic()
        try:
            spec = self._registry.get(tool_name)
            self._authorize(spec, ctx)
            _validate_against_schema(args, spec.parameters_schema)

            async with self._semaphore_for(tool_name):
                output = await asyncio.wait_for(
                    spec.handler(args, ctx), timeout=spec.timeout_seconds
                )

            output = self._cap_result_size(output)
            duration_ms = (time.monotonic() - started) * 1000
            logger.info(
                "tool_executed name=%s user_ref=%s duration_ms=%.1f success=True",
                tool_name, ctx.user_ref, duration_ms,
            )
            return ToolResult(tool_name=tool_name, success=True, output=output, duration_ms=duration_ms)

        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - started) * 1000
            logger.warning("tool_timeout name=%s user_ref=%s", tool_name, ctx.user_ref)
            return ToolResult(
                tool_name=tool_name, success=False,
                error="Tool execution timed out.", duration_ms=duration_ms,
            )

        except (ToolNotFoundError, ToolPermissionError, ToolValidationError) as exc:
            # Caller/config errors — safe to surface the message directly.
            duration_ms = (time.monotonic() - started) * 1000
            logger.info("tool_rejected name=%s reason=%s", tool_name, type(exc).__name__)
            return ToolResult(tool_name=tool_name, success=False, error=str(exc), duration_ms=duration_ms)

        except Exception as exc:  # noqa: BLE001 - normalize unexpected tool failures
            duration_ms = (time.monotonic() - started) * 1000
            logger.error(
                "tool_execution_failed name=%s user_ref=%s error_type=%s",
                tool_name, ctx.user_ref, type(exc).__name__,
            )
            return ToolResult(
                tool_name=tool_name, success=False,
                error="Tool execution failed unexpectedly.", duration_ms=duration_ms,
            )

    async def execute_many(
        self, calls: list[tuple[str, dict[str, Any]]], ctx: ToolContext
    ) -> list[ToolResult]:
        """Run several tool calls concurrently (e.g. multiple tool calls an
        AI provider requested in one turn). Failures in one call do not
        cancel the others."""
        results = await asyncio.gather(
            *(self.execute(name, args, ctx) for name, args in calls)
        )
        return list(results)

    def available_tools(self, ctx: ToolContext) -> list[dict[str, Any]]:
        """List tools visible to this context, in a shape suitable for
        exposing to an AI provider's function-calling API."""
        return self._registry.as_llm_function_schema(ctx.permission_level)

    # -- Internal helpers ---------------------------------------------------------

    @staticmethod
    def _authorize(spec: ToolSpec, ctx: ToolContext) -> None:
        if not spec.enabled:
            raise ToolPermissionError(f"Tool '{spec.name}' is currently disabled.")

        rank = {PermissionLevel.PUBLIC: 0, PermissionLevel.PRIVILEGED: 1, PermissionLevel.INTERNAL: 2}
        if rank[ctx.permission_level] < rank[spec.required_permission]:
            raise ToolPermissionError(
                f"Tool '{spec.name}' requires '{spec.required_permission.value}' permission."
            )

    @staticmethod
    def _cap_result_size(output: Any) -> Any:
        """Prevent a misbehaving tool from returning an unbounded payload
        that balloons downstream prompts/responses."""
        if isinstance(output, str) and len(output) > MAX_RESULT_SIZE_CHARS:
            return output[:MAX_RESULT_SIZE_CHARS] + "...[truncated]"
        return output


# -----------------------------------------------------------------------------
# Example tool registration helper (for reference — remove or adapt)
# -----------------------------------------------------------------------------

def make_tool(
    name: str,
    description: str,
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[Any]],
    *,
    parameters_schema: Optional[dict[str, Any]] = None,
    required_permission: PermissionLevel = PermissionLevel.PUBLIC,
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> ToolSpec:
    """Small convenience factory so individual tool modules don't need to
    import ToolSpec's full dataclass signature directly."""
    return ToolSpec(
        name=name,
        description=description,
        handler=handler,
        parameters_schema=parameters_schema or {},
        required_permission=required_permission,
        timeout_seconds=timeout_seconds,
    )