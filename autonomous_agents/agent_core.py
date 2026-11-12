"""Core primitives for the ZentraX-AI autonomous agent ecosystem.

This module defines the shared vocabulary used by every other module in
``autonomous_agents``: enums, exceptions, ``Task`` / ``TaskResult`` /
``Message`` data structures, and the ``BaseAgent`` class that all concrete
agents extend.

Design notes
------------
* Agents are fully asynchronous (``asyncio``) and never block the event loop.
* Agents talk to the outside world through a ``MessageTransport`` (normally the
  ``MessageBus`` from ``agent_communication``); this module does not import it,
  so there are no circular dependencies.
* ``BaseAgent.execute`` never raises for task-level problems: it always returns
  a ``TaskResult``. It only raises ``AgentStateError`` when the agent is not
  able to accept work at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Union,
)

logger = logging.getLogger("zentrax.agents")

__all__ = [
    "AgentConfig",
    "AgentError",
    "AgentMetrics",
    "AgentState",
    "AgentStateError",
    "BaseAgent",
    "CommunicationError",
    "DelegationError",
    "Message",
    "MessageHandler",
    "MessageTransport",
    "MessageType",
    "NoCapableAgentError",
    "Reasoner",
    "RegistryError",
    "Task",
    "TaskExecutionError",
    "TaskPriority",
    "TaskResult",
    "TaskStatus",
    "WorkflowError",
    "call_reasoner",
    "new_id",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: A pluggable language-model callable: ``prompt -> completion`` (sync or async).
Reasoner = Callable[[str], Union[str, Awaitable[str]]]


def new_id(prefix: str = "") -> str:
    """Return a short unique identifier with an optional prefix."""
    return f"{prefix}{uuid.uuid4().hex[:12]}"


async def call_reasoner(reasoner: Reasoner, prompt: str) -> str:
    """Invoke a sync or async reasoner and always return a ``str``."""
    result = reasoner(prompt)
    if inspect.isawaitable(result):
        result = await result
    return str(result)


def _as_frozenset(values: Union[str, Iterable[str], None]) -> FrozenSet[str]:
    if values is None:
        return frozenset()
    if isinstance(values, str):
        return frozenset({values})
    return frozenset(values)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class AgentError(Exception):
    """Base class for all framework errors."""


class AgentStateError(AgentError):
    """The agent is in a state that does not allow the requested operation."""


class TaskExecutionError(AgentError):
    """A task failed to execute."""


class NoCapableAgentError(AgentError):
    """No registered agent can handle the requested task."""


class DelegationError(AgentError):
    """Task delegation failed."""


class CommunicationError(AgentError):
    """A message could not be delivered or a request timed out."""


class WorkflowError(AgentError):
    """A workflow is invalid or could not be executed."""


class RegistryError(AgentError):
    """Registry misuse (duplicate or unknown agent / agent type)."""


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class AgentState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TaskPriority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class TaskResult:
    """Outcome of a single task execution attempt."""

    task_id: str
    agent_id: Optional[str]
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(
        cls,
        task_id: str,
        error: str,
        agent_id: Optional[str] = None,
        duration: float = 0.0,
    ) -> "TaskResult":
        return cls(task_id=task_id, agent_id=agent_id, success=False, error=error, duration=duration)

    def unwrap(self) -> Any:
        """Return ``output`` or raise ``TaskExecutionError`` if the task failed."""
        if not self.success:
            raise TaskExecutionError(self.error or "task failed")
        return self.output


@dataclass
class Task:
    """A unit of work that can be delegated to an agent.

    ``payload`` is interpreted by the receiving agent (by convention it has an
    ``"action"`` key). ``context`` carries framework-injected data such as
    ``dependency_results`` and ``previous_output``.
    """

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: FrozenSet[str] = frozenset()
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    target_agent: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("task-"))
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[TaskResult] = None

    def __post_init__(self) -> None:
        self.required_capabilities = _as_frozenset(self.required_capabilities)
        self.priority = TaskPriority(self.priority)
        self.dependencies = list(self.dependencies)
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

    def reset(self) -> None:
        """Clear runtime state so the task can be run again."""
        self.status = TaskStatus.PENDING
        self.assigned_to = None
        self.attempts = 0
        self.started_at = None
        self.finished_at = None
        self.result = None

    def clone(self, **overrides: Any) -> "Task":
        """Return a fresh copy (new id, reset runtime state) with overrides applied."""
        overrides.setdefault("payload", dict(self.payload))
        overrides.setdefault("context", dict(self.context))
        overrides.setdefault("id", new_id("task-"))
        new = dataclasses.replace(self, **overrides)
        new.reset()
        return new


@dataclass
class Message:
    """An asynchronous message exchanged between agents."""

    sender: str
    recipient: Optional[str]  # ``None`` means broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    type: MessageType = MessageType.EVENT
    topic: Optional[str] = None
    correlation_id: Optional[str] = None
    id: str = field(default_factory=lambda: new_id("msg-"))
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentConfig:
    name: str = "agent"
    capabilities: FrozenSet[str] = frozenset()
    max_concurrent_tasks: int = 1
    default_task_timeout: Optional[float] = 300.0
    inbox_size: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.capabilities = _as_frozenset(self.capabilities)
        if self.max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be >= 1")
        if self.inbox_size < 1:
            raise ValueError("inbox_size must be >= 1")


@dataclass
class AgentMetrics:
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration: float = 0.0
    last_active: Optional[float] = None

    def record(self, success: bool, duration: float) -> None:
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        self.total_duration += duration
        self.last_active = time.time()

    @property
    def avg_duration(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.total_duration / total if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "avg_duration": round(self.avg_duration, 6),
            "last_active": self.last_active,
        }


class MessageTransport(Protocol):
    """What an agent needs from the thing that carries its messages."""

    async def send(self, message: Message) -> bool: ...

    async def request(
        self, sender_id: str, recipient_id: str, payload: Dict[str, Any], timeout: float = 30.0
    ) -> Message: ...

    async def publish(self, sender_id: str, topic: str, payload: Dict[str, Any]) -> int: ...


MessageHandler = Callable[[Message], Awaitable[Any]]


# --------------------------------------------------------------------------- #
# BaseAgent
# --------------------------------------------------------------------------- #


class BaseAgent(ABC):
    """Abstract base class for every autonomous agent.

    Subclasses implement :meth:`handle_task` and may override
    :meth:`on_start`, :meth:`on_stop`, :meth:`handle_message` and
    :meth:`health_check`. Simple message handlers can also be registered by
    ``payload["action"]`` with :meth:`register_message_handler`.
    """

    agent_type: ClassVar[str] = "base"

    def __init__(self, config: AgentConfig, agent_id: Optional[str] = None) -> None:
        self.id: str = agent_id or new_id(f"{self.agent_type}-")
        self.config: AgentConfig = config
        self.state: AgentState = AgentState.CREATED
        self.metrics: AgentMetrics = AgentMetrics()
        self.log: logging.Logger = logging.getLogger(f"zentrax.agents.{self.agent_type}.{self.id}")

        self._transport: Optional[MessageTransport] = None
        self._inbox: Optional[asyncio.Queue[Message]] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._idle: Optional[asyncio.Event] = None
        self._message_task: Optional[asyncio.Task[None]] = None
        self._message_jobs: Set[asyncio.Task[None]] = set()
        self._active: int = 0
        self._handlers: Dict[str, MessageHandler] = {}

        self.register_message_handler("ping", self._on_ping)
        self.register_message_handler("status", self._on_status)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def capabilities(self) -> FrozenSet[str]:
        return self.config.capabilities

    @property
    def active_tasks(self) -> int:
        """Tasks currently running or waiting for a concurrency slot."""
        return self._active

    @property
    def load(self) -> float:
        return self._active / self.config.max_concurrent_tasks

    @property
    def is_running(self) -> bool:
        return self.state in (AgentState.IDLE, AgentState.BUSY)

    @property
    def is_available(self) -> bool:
        return self.is_running and self._active < self.config.max_concurrent_tasks

    def health(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.agent_type,
            "state": self.state.value,
            "capabilities": sorted(self.capabilities),
            "active_tasks": self._active,
            "load": round(self.load, 3),
            "metrics": self.metrics.to_dict(),
        }

    async def health_check(self) -> bool:
        """Override to add custom liveness checks; used by ``AgentManager``."""
        return self.state != AgentState.FAILED

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        if self.state in (AgentState.IDLE, AgentState.BUSY):
            return
        if self.state == AgentState.PAUSED:
            self.resume()
            return
        if self.state in (AgentState.STARTING, AgentState.STOPPING):
            raise AgentStateError(f"agent {self.id} is {self.state.value}")

        self.state = AgentState.STARTING
        try:
            self._inbox = asyncio.Queue(maxsize=self.config.inbox_size)
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
            self._idle = asyncio.Event()
            self._idle.set()
            self._active = 0
            await self.on_start()
        except Exception:
            self.state = AgentState.FAILED
            self.log.exception("agent failed to start")
            raise
        self._message_task = asyncio.create_task(self._message_loop(), name=f"{self.id}-inbox")
        self.state = AgentState.IDLE
        self.log.info("agent started (%s)", ", ".join(sorted(self.capabilities)))

    async def stop(self, timeout: float = 10.0, drain: bool = True) -> None:
        if self.state in (AgentState.CREATED, AgentState.STOPPED):
            self.state = AgentState.STOPPED
            return
        self.state = AgentState.STOPPING
        if drain and self._idle is not None:
            try:
                await asyncio.wait_for(self._idle.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self.log.warning("stop(): %d task(s) still running after %.1fs", self._active, timeout)

        jobs = list(self._message_jobs)
        if self._message_task is not None:
            jobs.append(self._message_task)
        for job in jobs:
            job.cancel()
        for job in jobs:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await job
        self._message_jobs.clear()
        self._message_task = None

        try:
            await self.on_stop()
        except Exception:
            self.log.exception("on_stop() raised")
        self.state = AgentState.STOPPED
        self.log.info("agent stopped")

    def pause(self) -> None:
        """Stop accepting new tasks (running ones finish)."""
        if self.is_running:
            self.state = AgentState.PAUSED

    def resume(self) -> None:
        if self.state == AgentState.PAUSED:
            self.state = AgentState.BUSY if self._active else AgentState.IDLE

    async def on_start(self) -> None:
        """Hook: acquire resources."""

    async def on_stop(self) -> None:
        """Hook: release resources."""

    # ------------------------------------------------------------------ #
    # Task execution
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def handle_task(self, task: Task) -> Any:
        """Do the work for ``task`` and return its output (any JSON-friendly value)."""

    async def execute(self, task: Task) -> TaskResult:
        """Run ``task`` with concurrency limiting, timeout and metrics."""
        if not self.is_running:
            raise AgentStateError(f"agent {self.id} cannot accept tasks while {self.state.value}")
        missing = task.required_capabilities - self.capabilities
        if missing:
            return TaskResult.failure(task.id, f"agent lacks capabilities: {sorted(missing)}", self.id)
        assert self._semaphore is not None and self._idle is not None

        self._active += 1
        self._idle.clear()
        self.state = AgentState.BUSY
        started = time.monotonic()
        try:
            async with self._semaphore:
                timeout = task.timeout if task.timeout is not None else self.config.default_task_timeout
                try:
                    output = await asyncio.wait_for(
                        self.handle_task(task),
                        timeout=timeout if timeout and timeout > 0 else None,
                    )
                    result = TaskResult(task.id, self.id, True, output, duration=time.monotonic() - started)
                except asyncio.TimeoutError:
                    result = TaskResult.failure(
                        task.id, f"task timed out after {timeout}s", self.id, time.monotonic() - started
                    )
                except Exception as exc:  # noqa: BLE001 - task code is untrusted
                    self.log.exception("task %s failed", task.id)
                    result = TaskResult.failure(
                        task.id, f"{type(exc).__name__}: {exc}", self.id, time.monotonic() - started
                    )
        finally:
            self._active -= 1
            if self._active == 0:
                if self.state == AgentState.BUSY:
                    self.state = AgentState.IDLE
                self._idle.set()
        self.metrics.record(result.success, result.duration)
        return result

    # ------------------------------------------------------------------ #
    # Messaging
    # ------------------------------------------------------------------ #

    def attach_transport(self, transport: Optional[MessageTransport]) -> None:
        self._transport = transport

    def register_message_handler(self, action: str, handler: MessageHandler) -> None:
        """Route messages whose ``payload["action"] == action`` to ``handler``."""
        self._handlers[action] = handler

    async def deliver(self, message: Message) -> None:
        """Called by the transport to put a message in this agent's inbox."""
        if self._inbox is None or self.state in (
            AgentState.CREATED,
            AgentState.STOPPING,
            AgentState.STOPPED,
            AgentState.FAILED,
        ):
            raise CommunicationError(f"agent {self.id} is not accepting messages ({self.state.value})")
        try:
            self._inbox.put_nowait(message)
        except asyncio.QueueFull:
            raise CommunicationError(f"inbox of agent {self.id} is full") from None

    def _require_transport(self) -> MessageTransport:
        if self._transport is None:
            raise CommunicationError(f"agent {self.id} is not attached to a message transport")
        return self._transport

    async def send(
        self,
        recipient: Optional[str],
        payload: Dict[str, Any],
        *,
        type: MessageType = MessageType.EVENT,
        topic: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> bool:
        message = Message(self.id, recipient, dict(payload), type, topic, correlation_id)
        return await self._require_transport().send(message)

    async def broadcast(self, payload: Dict[str, Any]) -> bool:
        return await self.send(None, payload, type=MessageType.BROADCAST)

    async def publish(self, topic: str, payload: Dict[str, Any]) -> int:
        return await self._require_transport().publish(self.id, topic, dict(payload))

    async def request(self, recipient: str, payload: Dict[str, Any], timeout: float = 30.0) -> Any:
        """Send a request and await the peer's reply (its handler's return value)."""
        response = await self._require_transport().request(self.id, recipient, dict(payload), timeout)
        return response.payload.get("result")

    async def handle_message(self, message: Message) -> Any:
        """Default dispatch on ``payload["action"]``. Override for custom behaviour."""
        action = message.payload.get("action")
        handler = self._handlers.get(action) if isinstance(action, str) else None
        if handler is None:
            self.log.debug("unhandled message %s from %s", message.id, message.sender)
            return None
        return await handler(message)

    async def _on_ping(self, message: Message) -> Dict[str, Any]:
        return {"pong": True, "agent_id": self.id, "timestamp": time.time()}

    async def _on_status(self, message: Message) -> Dict[str, Any]:
        return self.health()

    async def _message_loop(self) -> None:
        assert self._inbox is not None
        inbox = self._inbox
        while True:
            message = await inbox.get()
            # One job per message so a handler that awaits another agent can't deadlock the inbox.
            job = asyncio.create_task(self._process_message(message))
            self._message_jobs.add(job)
            job.add_done_callback(self._message_jobs.discard)
            inbox.task_done()

    async def _process_message(self, message: Message) -> None:
        try:
            reply = await self.handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.log.exception("message handler failed for %s", message.id)
            if message.type == MessageType.REQUEST:
                await self._reply(message, {"error": f"{type(exc).__name__}: {exc}"}, MessageType.ERROR)
            return
        if message.type == MessageType.REQUEST:
            await self._reply(message, {"result": reply}, MessageType.RESPONSE)

    async def _reply(self, request: Message, payload: Dict[str, Any], type: MessageType) -> None:
        if self._transport is None:
            return
        try:
            await self._transport.send(
                Message(self.id, request.sender, payload, type, request.topic, correlation_id=request.id)
            )
        except Exception:  # noqa: BLE001
            self.log.exception("failed to send reply to %s", request.sender)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id} state={self.state.value}>"
