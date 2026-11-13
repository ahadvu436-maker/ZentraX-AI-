"""Core types, exceptions and the :class:`ExecutionCore` facade.

This module is the hub of the package.  It defines the shared data model
(``TaskSpec``, ``TaskResult``, ``TaskStatus``, ``ExecutionConfig``), the
exception hierarchy, a few blocking-call helpers and ``ExecutionCore`` which
wires every other component together.

To avoid circular imports, this module never imports the executors, managers
or sandbox at module level; ``ExecutionCore.__init__`` imports them lazily.
"""
from __future__ import annotations

import asyncio
import dataclasses
import enum
import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:  # pragma: no cover
    from .retry_manager import RetryPolicy
    from .sandbox.resource_limits import ResourceLimits
    from .workflow_executor import Workflow, WorkflowResult

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionError", "TaskTimeoutError", "TaskFailedError", "TaskCancelledError",
    "RetryExhaustedError", "WorkflowError", "WorkflowValidationError",
    "ActionError", "ActionNotFoundError", "ActionArgumentError", "ProcessError",
    "SandboxError", "ResourceLimitError", "TaskStatus", "TaskSpec", "TaskResult",
    "ExecutionConfig", "ExecutionCore", "build_result", "classify_exception",
    "run_awaitable_blocking", "invoke_callable",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ExecutionError(Exception):
    """Base class for all execution-engine errors."""


class TaskTimeoutError(ExecutionError):
    """A task exceeded its wall-clock timeout."""


class TaskFailedError(ExecutionError):
    """A task failed inside a sandbox; ``remote_type`` holds the original type name."""

    remote_type: str = ""


class TaskCancelledError(ExecutionError):
    """A task was cancelled (e.g. the engine shut down during a retry back-off)."""


class RetryExhaustedError(ExecutionError):
    """Raised by the retry manager when no further attempt will be made."""

    def __init__(self, attempts: int, last_exception: BaseException, label: str = "") -> None:
        super().__init__(
            f"{label or 'task'} gave up after {attempts} attempt(s): "
            f"{type(last_exception).__name__}: {last_exception}"
        )
        self.attempts = attempts
        self.last_exception = last_exception
        self.label = label


class WorkflowError(ExecutionError):
    """Generic workflow failure."""


class WorkflowValidationError(WorkflowError):
    """Workflow definition is invalid (duplicate id, unknown dependency, cycle)."""


class ActionError(ExecutionError):
    """Generic action-manager failure."""


class ActionNotFoundError(ActionError):
    """No action is registered under the requested name."""


class ActionArgumentError(ActionError):
    """Arguments do not match the action's signature."""


class ProcessError(ExecutionError):
    """Process-manager failure."""


class SandboxError(ExecutionError):
    """Sandbox could not run, or isolation could not be enforced."""


class ResourceLimitError(SandboxError):
    """A resource limit (CPU, memory, file size, ...) was exceeded or unenforceable."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
class TaskStatus(str, enum.Enum):
    """Lifecycle states of a task or workflow."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        return self not in (TaskStatus.PENDING, TaskStatus.RUNNING)

    @property
    def is_failure(self) -> bool:
        return self in (TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED)


@dataclass
class TaskSpec:
    """Description of one unit of work.

    Attributes:
        func: Sync or async callable to run.
        args/kwargs: Arguments passed to ``func``.
        name: Human readable name (defaults to ``func.__name__``).
        task_id: Unique id (auto-generated).
        timeout: Wall-clock limit in seconds (``None`` -> engine default).
        retry_policy: Per-task retry policy (``None`` -> engine default).
        sandboxed: ``True``/``False`` to force, ``None`` -> engine default.
        limits: Resource limits used when sandboxed.
        metadata: Free-form data carried through to results/monitoring.
    """

    func: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timeout: Optional[float] = None
    retry_policy: Optional["RetryPolicy"] = None
    sandboxed: Optional[bool] = None
    limits: Optional["ResourceLimits"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise TypeError(f"TaskSpec.func must be callable, got {type(self.func).__name__}")
        self.args = tuple(self.args)
        self.kwargs = dict(self.kwargs or {})
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("TaskSpec.timeout must be positive")
        if not self.name:
            self.name = getattr(self.func, "__name__", type(self.func).__name__)

    def with_(self, **changes: Any) -> "TaskSpec":
        """Return a copy with ``changes`` applied."""
        return dataclasses.replace(self, **changes)


@dataclass
class TaskResult:
    """Outcome of a task execution (executors never raise for task failures)."""

    task_id: str
    name: str
    status: TaskStatus
    value: Any = None
    error: Optional[str] = None
    exception: Optional[BaseException] = None
    attempts: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.SUCCEEDED

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def unwrap(self) -> Any:
        """Return the value or raise the underlying exception."""
        if self.ok:
            return self.value
        if self.exception is not None:
            raise self.exception
        raise ExecutionError(self.error or f"task {self.name} ended with {self.status.value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "name": self.name, "status": self.status.value,
            "value": self.value, "error": self.error, "attempts": self.attempts,
            "duration": self.duration, "started_at": self.started_at,
            "finished_at": self.finished_at, "metadata": dict(self.metadata),
        }


@dataclass
class ExecutionConfig:
    """Engine-wide configuration."""

    max_workers: int = 8
    max_async_concurrency: int = 32
    default_timeout: Optional[float] = None
    default_retry_policy: Optional["RetryPolicy"] = None
    default_limits: Optional["ResourceLimits"] = None
    sandbox_by_default: bool = False
    max_processes: int = 16
    history_size: int = 1000
    stall_threshold: Optional[float] = None  # seconds; enables the stall watchdog
    watchdog_interval: float = 5.0

    def __post_init__(self) -> None:
        if self.max_workers < 1 or self.max_async_concurrency < 1 or self.max_processes < 1:
            raise ValueError("max_workers, max_async_concurrency and max_processes must be >= 1")
        if self.default_timeout is not None and self.default_timeout <= 0:
            raise ValueError("default_timeout must be positive")


# --------------------------------------------------------------------------- #
# Helpers shared by executors
# --------------------------------------------------------------------------- #
async def _await(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def run_awaitable_blocking(awaitable: Awaitable[Any]) -> Any:
    """Run ``awaitable`` to completion from synchronous code.

    Uses ``asyncio.run`` directly when no loop is running in this thread, and a
    helper thread otherwise (so it is safe to call from inside an event loop).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(awaitable))
    box: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(_await(awaitable))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True, name="await-helper")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def invoke_callable(func: Callable[..., Any], args: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
    """Call ``func`` and transparently resolve awaitable results."""
    value = func(*args, **kwargs)
    if inspect.isawaitable(value):
        value = run_awaitable_blocking(value)
    return value


def classify_exception(exc: BaseException) -> TaskStatus:
    """Map an exception to the terminal :class:`TaskStatus` it represents."""
    if isinstance(exc, TaskTimeoutError):
        return TaskStatus.TIMEOUT
    if isinstance(exc, (TaskCancelledError, asyncio.CancelledError)):
        return TaskStatus.CANCELLED
    return TaskStatus.FAILED


def build_result(
    spec: TaskSpec,
    *,
    started_at: float,
    attempts: int,
    value: Any = None,
    exception: Optional[BaseException] = None,
) -> TaskResult:
    """Build a ``TaskResult`` from a success value or a failure exception."""
    if exception is None:
        status, error = TaskStatus.SUCCEEDED, None
    else:
        status = classify_exception(exception)
        error = f"{type(exception).__name__}: {exception}"
    return TaskResult(
        task_id=spec.task_id, name=spec.name, status=status, value=value, error=error,
        exception=exception, attempts=attempts, started_at=started_at,
        finished_at=time.time(), metadata=dict(spec.metadata),
    )


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class ExecutionCore:
    """Facade that owns and wires all engine components.

    Components (all available as attributes): ``monitor``, ``retry_manager``,
    ``sandbox``, ``action_manager``, ``process_manager``, ``task_executor``,
    ``workflow_executor`` and ``async_executor``.

    Example::

        with ExecutionCore() as core:
            result = core.execute(core.create_task(fetch, args=(url,), timeout=5))
            print(result.status, result.value)
    """

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        # Lazy imports break the import cycle (components import this module).
        from .action_manager import ActionManager
        from .async_executor import AsyncExecutor
        from .execution_monitor import ExecutionMonitor
        from .process_manager import ProcessManager
        from .retry_manager import RetryManager, RetryPolicy
        from .sandbox.execution_sandbox import ExecutionSandbox
        from .task_executor import TaskExecutor
        from .workflow_executor import WorkflowExecutor

        self.config = config or ExecutionConfig()
        self._lock = threading.Lock()
        self._closed = False

        self.monitor = ExecutionMonitor(
            history_size=self.config.history_size,
            stall_threshold=self.config.stall_threshold,
        )
        self.retry_manager = RetryManager(self.config.default_retry_policy or RetryPolicy(max_attempts=1))
        self.sandbox = ExecutionSandbox(default_limits=self.config.default_limits)
        self.action_manager = ActionManager()
        self.process_manager = ProcessManager(
            self.sandbox, monitor=self.monitor, max_processes=self.config.max_processes
        )
        self.task_executor = TaskExecutor(self.monitor, self.retry_manager, self.sandbox, self.config)
        self.workflow_executor = WorkflowExecutor(self.task_executor, self.monitor)
        self.async_executor = AsyncExecutor(self.monitor, self.retry_manager, self.sandbox, self.config)

        if self.config.stall_threshold:
            self.monitor.start_watchdog(self.config.watchdog_interval)
        self.process_manager.start_watchdog()
        logger.debug("ExecutionCore started (%s)", self.config)

    # -- lifecycle ---------------------------------------------------------- #
    def _ensure_open(self) -> None:
        if self._closed:
            raise ExecutionError("ExecutionCore has been shut down")

    def __enter__(self) -> "ExecutionCore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.shutdown()

    def shutdown(self, wait: bool = True) -> None:
        """Stop all components; aborts retry back-offs and kills managed processes."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for step in (
            self.retry_manager.shutdown,
            lambda: self.task_executor.shutdown(wait),
            self.async_executor.shutdown,
            self.process_manager.shutdown,
            self.monitor.stop_watchdog,
        ):
            try:
                step()
            except Exception:  # noqa: BLE001 - keep shutting down
                logger.exception("error during ExecutionCore shutdown")

    # -- tasks -------------------------------------------------------------- #
    def create_task(
        self, func: Callable[..., Any], args: Sequence[Any] = (),
        kwargs: Optional[Dict[str, Any]] = None, **options: Any,
    ) -> TaskSpec:
        """Build a :class:`TaskSpec`; ``options`` are TaskSpec fields (timeout, retry_policy, ...)."""
        return TaskSpec(func=func, args=tuple(args), kwargs=dict(kwargs or {}), **options)

    def execute(self, task: Union[TaskSpec, Callable[..., Any]]) -> TaskResult:
        """Run one task synchronously (with retries, monitoring, optional sandbox)."""
        self._ensure_open()
        return self.task_executor.execute(self._coerce(task))

    def submit(self, task: Union[TaskSpec, Callable[..., Any]]) -> "Future[TaskResult]":  # noqa: F821
        """Run one task on the thread pool; returns a ``concurrent.futures.Future``."""
        self._ensure_open()
        return self.task_executor.submit(self._coerce(task))

    def execute_many(self, tasks: Sequence[Union[TaskSpec, Callable[..., Any]]]) -> List[TaskResult]:
        """Run tasks concurrently; results keep input order."""
        self._ensure_open()
        return self.task_executor.execute_many([self._coerce(t) for t in tasks])

    # -- actions ------------------------------------------------------------ #
    def run_action(self, name: str, *args: Any, **kwargs: Any) -> TaskResult:
        """Run a registered action through the full pipeline."""
        self._ensure_open()
        return self.task_executor.execute(self.action_manager.build_spec(name, args, kwargs))

    # -- workflows ---------------------------------------------------------- #
    def run_workflow(self, workflow: "Workflow") -> "WorkflowResult":
        """Execute a DAG workflow."""
        self._ensure_open()
        return self.workflow_executor.execute(workflow)

    # -- async -------------------------------------------------------------- #
    async def execute_async(self, task: Union[TaskSpec, Callable[..., Any]]) -> TaskResult:
        """Awaitable execution (use inside an event loop)."""
        self._ensure_open()
        return await self.async_executor.execute(self._coerce(task))

    def run_async(self, task: Union[TaskSpec, Callable[..., Any]]) -> TaskResult:
        """Run a task on the engine's background event loop from sync code."""
        self._ensure_open()
        return self.async_executor.run_sync(self._coerce(task))

    # -- processes ---------------------------------------------------------- #
    def spawn_process(self, argv: Sequence[str], **kwargs: Any):
        """Start a sandboxed OS process tracked by the process manager."""
        self._ensure_open()
        return self.process_manager.spawn(argv, **kwargs)

    # -- introspection ------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        data = self.monitor.snapshot()
        data["actions"] = len(self.action_manager.list())
        data["processes"] = len(self.process_manager.list())
        return data

    @staticmethod
    def _coerce(task: Union[TaskSpec, Callable[..., Any]]) -> TaskSpec:
        if isinstance(task, TaskSpec):
            return task
        if callable(task):
            return TaskSpec(func=task)
        raise TypeError(f"expected TaskSpec or callable, got {type(task).__name__}")
