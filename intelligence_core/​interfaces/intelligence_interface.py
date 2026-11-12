"""Contracts shared by every module in ``core_intelligence``.

This file is the single dependency all other modules import from, which keeps
the package free of circular imports: concrete modules depend on these
abstractions, never on each other.

Contents
--------
* Exceptions            - one hierarchy rooted at ``IntelligenceError``
* Enums                 - priorities, request kinds, statuses, health states
* Data models           - immutable dataclasses passed between modules
* Protocols             - structural contracts (memory, reasoning, handlers, ...)

External systems (memory store, reasoning engine, ...) only need to satisfy the
relevant ``Protocol``; no inheritance is required.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any,
    Awaitable,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

__all__ = [
    # exceptions
    "IntelligenceError",
    "ModuleNotReadyError",
    "RoutingError",
    "DecisionError",
    "ContextError",
    # enums
    "Priority",
    "RequestKind",
    "DecisionStatus",
    "ResponseStatus",
    "HealthState",
    # models
    "IntelligenceRequest",
    "Signal",
    "ContextEvent",
    "ContextSnapshot",
    "Insight",
    "DecisionOption",
    "Decision",
    "IntelligenceResponse",
    "RouteResult",
    "ModuleHealth",
    # protocols / aliases
    "IntelligenceModule",
    "MemoryProvider",
    "ReasoningProvider",
    "ContextProvider",
    "CognitiveProcessor",
    "DecisionMaker",
    "RequestHandler",
    "RequestRouter",
    "SignalHandler",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class IntelligenceError(Exception):
    """Base class for all errors raised by the intelligence package."""


class ModuleNotReadyError(IntelligenceError):
    """Raised when a module is used before ``start()`` or after ``stop()``."""


class RoutingError(IntelligenceError):
    """Raised when no handler can serve a request."""


class DecisionError(IntelligenceError):
    """Raised when the decision engine receives invalid input."""


class ContextError(IntelligenceError):
    """Raised for context-store failures."""


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Priority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class RequestKind(str, Enum):
    QUERY = "query"
    COMMAND = "command"
    SIGNAL = "signal"
    REASONING = "reasoning"
    MEMORY = "memory"
    ANALYSIS = "analysis"
    UNKNOWN = "unknown"


class DecisionStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"  # options existed but all were vetoed by policy
    DEFERRED = "deferred"  # not enough confidence/utility to act


class ResponseStatus(str, Enum):
    OK = "ok"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    ERROR = "error"


class HealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class IntelligenceRequest:
    """A unit of work submitted to the intelligence core."""

    payload: Any
    kind: RequestKind = RequestKind.UNKNOWN
    source: str = "external"
    priority: Priority = Priority.NORMAL
    session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=_new_id)
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class Signal:
    """An asynchronous event emitted by any sub-module toward the core."""

    source: str
    topic: str
    payload: Any = None
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
    signal_id: str = field(default_factory=_new_id)


@dataclass(frozen=True, slots=True)
class ContextEvent:
    """One entry in a session's history."""

    kind: str
    data: Mapping[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Immutable point-in-time view of a session's context."""

    session_id: str
    state: Mapping[str, Any]
    history: tuple[ContextEvent, ...]
    captured_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class Insight:
    """Output of the cognitive engine: raw input turned into structured meaning."""

    request_id: str
    summary: str
    confidence: float
    data: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    produced_by: str = "cognitive_engine"


@dataclass(frozen=True, slots=True)
class DecisionOption:
    """A candidate action. ``score`` is utility (0..1); ``risk`` is 0..1."""

    option_id: str
    description: str
    score: float
    risk: float = 0.0
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Decision:
    status: DecisionStatus
    chosen: DecisionOption | None
    rationale: str
    confidence: float = 0.0
    considered: tuple[DecisionOption, ...] = ()


@dataclass(frozen=True, slots=True)
class IntelligenceResponse:
    request_id: str
    status: ResponseStatus
    result: Any = None
    insight: Insight | None = None
    decision: Decision | None = None
    handled_by: str | None = None
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is ResponseStatus.OK


@dataclass(frozen=True, slots=True)
class RouteResult:
    """Outcome of routing: which handler served the request and what it returned."""

    handler_name: str
    kind: RequestKind
    result: Any


@dataclass(frozen=True, slots=True)
class ModuleHealth:
    name: str
    state: HealthState
    detail: str = ""


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
@runtime_checkable
class IntelligenceModule(Protocol):
    """Lifecycle contract every long-lived module follows."""

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def health(self) -> ModuleHealth: ...


@runtime_checkable
class MemoryProvider(Protocol):
    """External memory system (vector store, database, ...)."""

    async def recall(self, query: str, limit: int = 5) -> Sequence[Mapping[str, Any]]: ...

    async def remember(self, record: Mapping[str, Any]) -> None: ...


@runtime_checkable
class ReasoningProvider(Protocol):
    """External reasoning engine. May return ``summary`` and ``confidence`` keys."""

    async def reason(
        self, request: IntelligenceRequest, context: ContextSnapshot
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class ContextProvider(Protocol):
    async def snapshot(self, session_id: str | None = None) -> ContextSnapshot: ...

    async def update(self, session_id: str | None, **state: Any) -> None: ...

    async def record(
        self, session_id: str | None, kind: str, data: Mapping[str, Any]
    ) -> None: ...


@runtime_checkable
class CognitiveProcessor(Protocol):
    async def process(
        self, request: IntelligenceRequest, context: ContextSnapshot
    ) -> Insight: ...


@runtime_checkable
class DecisionMaker(Protocol):
    async def decide(
        self,
        insight: Insight,
        options: Sequence[DecisionOption],
        context: ContextSnapshot,
    ) -> Decision: ...


@runtime_checkable
class RequestHandler(Protocol):
    """A subsystem that can serve routed requests."""

    async def handle(
        self,
        request: IntelligenceRequest,
        context: ContextSnapshot,
        insight: Insight,
    ) -> Any: ...


@runtime_checkable
class RequestRouter(Protocol):
    async def route(
        self,
        request: IntelligenceRequest,
        context: ContextSnapshot,
        insight: Insight,
    ) -> RouteResult: ...


SignalHandler = Callable[[Signal], Awaitable[None]]
 
