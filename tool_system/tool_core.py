"""Core abstractions shared by every part of the tool system.

Contents
--------
* Exceptions (:class:`ToolError` and subclasses, each with a stable ``code``).
* :class:`ToolParameter` / :class:`ToolSpec` - declarative description of a tool that
  can be rendered as a JSON Schema (what LLM function calling needs).
* :func:`validate_arguments` - strict validation + gentle coercion of model-produced
  arguments against a spec.
* :class:`ToolContext` - per-call execution context (permissions, working dir,
  deadline, cancellation, config).
* :class:`ToolResult` - uniform, serialisable outcome of a tool call.
* :class:`BaseTool` - the abstract base class every tool must inherit from.

This module has no dependency on the rest of the package.
"""
from __future__ import annotations

import abc
import dataclasses
import enum
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "ToolError", "ToolNotFoundError", "ToolRegistrationError", "ToolDiscoveryError",
    "ToolValidationError", "ToolPermissionError", "ToolUnavailableError", "ToolExecutionError",
    "ToolTimeoutError", "ToolCancelledError", "ToolBusyError", "ToolCategory", "ToolParameter",
    "ToolSpec", "ToolContext", "ToolResult", "BaseTool", "validate_arguments",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ToolError(Exception):
    """Base class for tool-system errors. ``code`` is stable and machine readable."""

    code: ClassVar[str] = "tool_error"


class ToolNotFoundError(ToolError):
    code = "not_found"


class ToolRegistrationError(ToolError):
    code = "registration"


class ToolDiscoveryError(ToolError):
    code = "discovery"


class ToolValidationError(ToolError):
    code = "validation"


class ToolPermissionError(ToolError):
    code = "permission"


class ToolUnavailableError(ToolError):
    code = "unavailable"


class ToolExecutionError(ToolError):
    code = "execution"


class ToolTimeoutError(ToolError):
    code = "timeout"


class ToolCancelledError(ToolError):
    code = "cancelled"


class ToolBusyError(ToolError):
    code = "busy"


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #
class ToolCategory(str, enum.Enum):
    """Coarse grouping used for filtering and display."""

    BROWSER = "browser"
    SEARCH = "search"
    CODE = "code"
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    API = "api"
    UTILITY = "utility"


_JSON_TYPES = ("string", "integer", "number", "boolean", "array", "object")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}$")  # also valid for OpenAI/Anthropic


@dataclass(frozen=True)
class ToolParameter:
    """One argument of a tool (maps to a JSON-Schema property)."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[Tuple[Any, ...]] = None
    items_type: Optional[str] = None  # element type for arrays
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    max_length: Optional[int] = None

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"invalid parameter name {self.name!r}")
        if self.type not in _JSON_TYPES:
            raise ValueError(f"parameter {self.name!r}: unsupported type {self.type!r}")
        if self.items_type is not None and self.items_type not in _JSON_TYPES:
            raise ValueError(f"parameter {self.name!r}: unsupported items_type {self.items_type!r}")
        if self.enum is not None:
            object.__setattr__(self, "enum", tuple(self.enum))

    def to_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.type == "array":
            schema["items"] = {"type": self.items_type or "string"}
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if not self.required and self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass(frozen=True)
class ToolSpec:
    """Static description of a tool.

    Attributes:
        name: Unique identifier shown to the model (``[A-Za-z_][A-Za-z0-9_-]*``, <= 64 chars).
        description: What the tool does and when to use it (this is prompt text!).
        parameters: Argument definitions.
        category: Coarse category.
        version: Tool version string.
        tags: Free-form labels for filtering.
        required_permissions: Permissions the caller must hold (see :class:`ToolContext`).
        timeout: Default wall-clock limit in seconds (``None`` -> executor default).
    """

    name: str
    description: str
    parameters: Tuple[ToolParameter, ...] = ()
    category: ToolCategory = ToolCategory.UTILITY
    version: str = "1.0.0"
    tags: Tuple[str, ...] = ()
    required_permissions: FrozenSet[str] = frozenset()
    timeout: Optional[float] = None

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"invalid tool name {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"tool {self.name!r} needs a description")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "required_permissions", frozenset(self.required_permissions))
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"tool {self.name!r} has duplicate parameter names")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def to_json_schema(self) -> Dict[str, Any]:
        """JSON Schema (``type: object``) describing the tool's arguments."""
        return {
            "type": "object",
            "properties": {p.name: p.to_schema() for p in self.parameters},
            "required": [p.name for p in self.parameters if p.required],
            "additionalProperties": False,
        }


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #
def _coerce(param: ToolParameter, value: Any) -> Any:
    """Coerce ``value`` to ``param.type`` or raise ``ValueError``. Models often send strings."""
    t = param.type
    if t == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise ValueError("expected a string")
    if t == "integer":
        if isinstance(value, bool):
            raise ValueError("expected an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value)
        raise ValueError("expected an integer")
    if t == "number":
        if isinstance(value, bool):
            raise ValueError("expected a number")
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ValueError("expected a number")
    if t == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError("expected a boolean")
    if t in ("array", "object"):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError(f"expected a JSON {t}") from None
        if t == "array":
            if isinstance(value, (list, tuple)):
                items = list(value)
                if param.items_type:
                    sub = ToolParameter("item", param.items_type, "")
                    try:
                        items = [_coerce(sub, v) for v in items]
                    except ValueError as exc:
                        raise ValueError(f"array items: {exc}") from None
                return items
            raise ValueError("expected an array")
        if isinstance(value, dict):
            return value
        raise ValueError("expected an object")
    raise ValueError(f"unsupported type {t}")  # pragma: no cover


def validate_arguments(spec: ToolSpec, arguments: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Validate and normalise ``arguments`` for ``spec``.

    Unknown arguments are rejected, required ones enforced, defaults filled in, simple
    string->number/bool/JSON coercions applied, and enum/range/length constraints checked.
    All problems are reported together in one :class:`ToolValidationError`.
    """
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise ToolValidationError("arguments must be an object")
    known = {p.name: p for p in spec.parameters}
    problems: List[str] = [f"unknown argument '{k}'" for k in arguments if k not in known]
    out: Dict[str, Any] = {}
    for p in spec.parameters:
        if p.name not in arguments or arguments[p.name] is None:
            if p.required:
                problems.append(f"missing required argument '{p.name}'")
            else:
                out[p.name] = p.default
            continue
        try:
            value = _coerce(p, arguments[p.name])
        except ValueError as exc:
            problems.append(f"argument '{p.name}': {exc}")
            continue
        if p.enum is not None and value not in p.enum:
            problems.append(f"argument '{p.name}' must be one of {list(p.enum)}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if p.minimum is not None and value < p.minimum:
                problems.append(f"argument '{p.name}' must be >= {p.minimum}")
            if p.maximum is not None and value > p.maximum:
                problems.append(f"argument '{p.name}' must be <= {p.maximum}")
        if isinstance(value, str) and p.max_length is not None and len(value) > p.max_length:
            problems.append(f"argument '{p.name}' exceeds {p.max_length} characters")
        out[p.name] = value
    if problems:
        raise ToolValidationError(f"invalid arguments for '{spec.name}': " + "; ".join(problems))
    return out


# --------------------------------------------------------------------------- #
# Context and result
# --------------------------------------------------------------------------- #
@dataclass
class ToolContext:
    """Per-call execution context handed to every tool.

    Permissions are dotted strings such as ``fs.read``, ``fs.write``, ``net``, ``db.read``,
    ``db.write``, ``code.exec``.  ``"*"`` grants everything and ``"fs.*"`` grants a family.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: Optional[str] = None
    working_dir: Optional[Path] = None
    permissions: Set[str] = field(default_factory=set)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None  # time.monotonic() value
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def has_permission(self, permission: str) -> bool:
        if "*" in self.permissions or permission in self.permissions:
            return True
        parts = permission.split(".")
        return any(".".join(parts[:i]) + ".*" in self.permissions for i in range(1, len(parts)))

    def remaining(self) -> Optional[float]:
        """Seconds left before the deadline (``None`` if no deadline)."""
        return None if self.deadline is None else self.deadline - time.monotonic()

    def check_cancelled(self) -> None:
        """Raise if the call was cancelled or ran out of time. Long tools should poll this."""
        if self.cancel_event.is_set():
            raise ToolCancelledError("tool call was cancelled")
        left = self.remaining()
        if left is not None and left <= 0:
            raise ToolTimeoutError("tool call exceeded its deadline")

    def derive(self, **changes: Any) -> "ToolContext":
        """Copy of this context with ``changes`` applied (mutable members are copied)."""
        base = dataclasses.replace(
            self, permissions=set(self.permissions), config=dict(self.config),
            metadata=dict(self.metadata))
        return dataclasses.replace(base, **changes)


@dataclass
class ToolResult:
    """Uniform outcome of a tool call; the executor never raises for tool failures."""

    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    duration: float = 0.0
    request_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, tool_name: str, output: Any, **kw: Any) -> "ToolResult":
        return cls(tool_name=tool_name, success=True, output=output, **kw)

    @classmethod
    def fail(cls, tool_name: str, exc: BaseException, **kw: Any) -> "ToolResult":
        code = exc.code if isinstance(exc, ToolError) else "execution"
        return cls(tool_name=tool_name, success=False, error=str(exc) or type(exc).__name__,
                   error_type=code, **kw)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_message(self, max_chars: int = 20000) -> str:
        """Text suitable for sending back to a model."""
        if not self.success:
            text = f"Error ({self.error_type}): {self.error}"
        elif isinstance(self.output, str):
            text = self.output
        else:
            text = json.dumps(self.output, default=str, ensure_ascii=False)
        return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BaseTool(abc.ABC):
    """Interface every tool must implement.

    Subclasses define a class-level ``spec`` and implement :meth:`run`.  Tools are
    discovered automatically when they live in ``tool_system/tools/`` (or a plugin
    directory) and are concrete subclasses of this class.

    Lifecycle: ``__init__(config)`` -> :meth:`initialize` (calls :meth:`setup` once) ->
    many :meth:`run` calls (possibly concurrent, so keep tools thread-safe) ->
    :meth:`close` (calls :meth:`teardown`).
    """

    spec: ClassVar[ToolSpec]

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self._initialized = False
        self._init_lock = threading.Lock()

    # -- identity ----------------------------------------------------------- #
    @property
    def name(self) -> str:
        return self.spec.name

    # -- availability / permissions ------------------------------------------ #
    def is_available(self) -> bool:
        """Return False when optional dependencies/config are missing."""
        return True

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "tool is not available"

    def required_permissions_for(self, arguments: Mapping[str, Any]) -> Set[str]:
        """Permissions needed for these (validated) arguments; override for per-action rules."""
        return set(self.spec.required_permissions)

    # -- lifecycle ---------------------------------------------------------- #
    def setup(self) -> None:
        """Acquire resources (connections, clients). Called once."""

    def teardown(self) -> None:
        """Release resources. Called once on close."""

    def initialize(self) -> None:
        with self._init_lock:
            if not self._initialized:
                self.setup()
                self._initialized = True

    def close(self) -> None:
        with self._init_lock:
            if self._initialized:
                self._initialized = False
                self.teardown()

    # -- execution ---------------------------------------------------------- #
    @abc.abstractmethod
    def run(self, context: ToolContext, **arguments: Any) -> Any:
        """Perform the work with validated ``arguments`` and return a JSON-friendly value.

        Raise :class:`ToolError` subclasses for expected failures; any other exception is
        converted into a generic execution error by the executor.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={getattr(self, 'spec', None) and self.spec.name!r}>"
