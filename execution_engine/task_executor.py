"""Synchronous task execution with timeouts, retries, monitoring and sandboxing."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from .execution_core import (
    ExecutionConfig, ExecutionError, RetryExhaustedError, TaskResult, TaskSpec,
    TaskTimeoutError, build_result, invoke_callable,
)
from .execution_monitor import ExecutionMonitor
from .retry_manager import RetryManager, RetryPolicy
from .sandbox.execution_sandbox import ExecutionSandbox

logger = logging.getLogger(__name__)

__all__ = ["TaskExecutor"]


class TaskExecutor:
    """Runs :class:`TaskSpec` objects.

    ``execute`` never raises for *task* failures; inspect ``TaskResult.status``.
    Non-sandboxed timeouts run the attempt in a daemon thread which cannot be
    killed: on timeout the call is abandoned (use ``sandboxed=True`` for hard
    termination of runaway work).

    Don't call :meth:`execute_many` from inside a task running on this executor's
    pool (all workers could end up waiting on each other).
    """

    def __init__(self, monitor: ExecutionMonitor, retry_manager: RetryManager,
                 sandbox: ExecutionSandbox, config: ExecutionConfig) -> None:
        self.monitor = monitor
        self.retry_manager = retry_manager
        self.sandbox = sandbox
        self.config = config
        self._pool = ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="exec-task")
        self._closed = False

    # -- public API ---------------------------------------------------------- #
    def execute(self, spec: TaskSpec) -> TaskResult:
        """Run ``spec`` in the calling thread (with retries) and return its result."""
        policy = self._policy(spec)
        label = f"{spec.name}[{spec.task_id}]"
        self.monitor.task_started(spec)
        started = time.time()
        try:
            value, attempts = self.retry_manager.run(
                lambda: self._attempt(spec), policy, label=label,
                on_retry=lambda n, exc, delay: self.monitor.task_retry(spec, n, exc, delay))
            result = build_result(spec, started_at=started, attempts=attempts, value=value)
        except RetryExhaustedError as err:
            result = build_result(spec, started_at=started, attempts=err.attempts,
                                  exception=err.last_exception)
        except Exception as exc:  # noqa: BLE001 - defensive: never leak
            logger.exception("unexpected error executing %s", label)
            result = build_result(spec, started_at=started, attempts=0, exception=exc)
        self.monitor.task_finished(result)
        return result

    def submit(self, spec: TaskSpec) -> "Future[TaskResult]":
        """Schedule ``spec`` on the worker pool."""
        if self._closed:
            raise ExecutionError("TaskExecutor has been shut down")
        try:
            return self._pool.submit(self.execute, spec)
        except RuntimeError as exc:
            raise ExecutionError(f"cannot schedule task: {exc}") from exc

    def execute_many(self, specs: Sequence[TaskSpec]) -> List[TaskResult]:
        """Run all specs concurrently; results are returned in input order."""
        futures = [self.submit(s) for s in specs]
        return [f.result() for f in futures]

    def shutdown(self, wait: bool = True) -> None:
        self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=True)

    # -- internals ----------------------------------------------------------- #
    def _policy(self, spec: TaskSpec) -> RetryPolicy:
        return spec.retry_policy or self.config.default_retry_policy or RetryPolicy.none()

    def _attempt(self, spec: TaskSpec) -> Any:
        timeout = spec.timeout if spec.timeout is not None else self.config.default_timeout
        sandboxed = self.config.sandbox_by_default if spec.sandboxed is None else spec.sandboxed
        if sandboxed:
            return self.sandbox.run_callable(
                spec.func, spec.args, spec.kwargs,
                limits=spec.limits or self.config.default_limits, timeout=timeout)
        return self._call_with_timeout(spec, timeout)

    @staticmethod
    def _call_with_timeout(spec: TaskSpec, timeout: Optional[float]) -> Any:
        if timeout is None:
            return invoke_callable(spec.func, spec.args, spec.kwargs)
        box: Dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = invoke_callable(spec.func, spec.args, spec.kwargs)
            except BaseException as exc:  # noqa: BLE001 - propagated to the caller
                box["error"] = exc

        thread = threading.Thread(target=target, daemon=True, name=f"task-{spec.task_id}")
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TaskTimeoutError(f"task '{spec.name}' exceeded {timeout:.2f}s (thread abandoned)")
        if "error" in box:
            raise box["error"]
        return box.get("value")
