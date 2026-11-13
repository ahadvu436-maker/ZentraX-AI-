"""asyncio-based execution with bounded concurrency, timeouts, retries and monitoring."""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
import time
import weakref
from typing import Any, Coroutine, List, Optional, Sequence

from .execution_core import (
    ExecutionConfig, ExecutionError, RetryExhaustedError, TaskResult, TaskSpec, TaskStatus,
    TaskTimeoutError, build_result,
)
from .execution_monitor import ExecutionMonitor
from .retry_manager import RetryManager, RetryPolicy
from .sandbox.execution_sandbox import ExecutionSandbox

logger = logging.getLogger(__name__)

__all__ = ["AsyncExecutor"]


class AsyncExecutor:
    """Runs tasks on an event loop.

    * ``async def`` functions are awaited directly; sync functions run in the loop's
      default thread pool (a timed-out sync function keeps running in its thread -
      use ``sandboxed=True`` for hard kills).
    * Concurrency is bounded by ``config.max_async_concurrency``.
    * Use :meth:`execute`/:meth:`gather` inside a running loop, or :meth:`run_sync`
      / :meth:`run_many_sync` from synchronous code (a background loop is started lazily).
    """

    def __init__(self, monitor: ExecutionMonitor, retry_manager: RetryManager,
                 sandbox: ExecutionSandbox, config: ExecutionConfig) -> None:
        self.monitor = monitor
        self.retry_manager = retry_manager
        self.sandbox = sandbox
        self.config = config
        self._semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
            weakref.WeakKeyDictionary())
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    # -- async API ---------------------------------------------------------- #
    async def execute(self, spec: TaskSpec) -> TaskResult:
        """Run one task; task failures are reported in the result, not raised."""
        if self._closed:
            raise ExecutionError("AsyncExecutor has been shut down")
        policy = spec.retry_policy or self.config.default_retry_policy or RetryPolicy.none()
        label = f"{spec.name}[{spec.task_id}]"
        async with self._semaphore():
            self.monitor.task_started(spec)
            started = time.time()
            try:
                value, attempts = await self.retry_manager.arun(
                    lambda: self._attempt(spec), policy, label=label,
                    on_retry=lambda n, exc, delay: self.monitor.task_retry(spec, n, exc, delay))
                result = build_result(spec, started_at=started, attempts=attempts, value=value)
            except RetryExhaustedError as err:
                result = build_result(spec, started_at=started, attempts=err.attempts,
                                      exception=err.last_exception)
            except asyncio.CancelledError as exc:
                self.monitor.task_finished(
                    build_result(spec, started_at=started, attempts=0, exception=exc))
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("unexpected error executing %s", label)
                result = build_result(spec, started_at=started, attempts=0, exception=exc)
            self.monitor.task_finished(result)
            return result

    async def gather(self, specs: Sequence[TaskSpec]) -> List[TaskResult]:
        """Run tasks concurrently (bounded); results keep input order."""
        return list(await asyncio.gather(*(self.execute(s) for s in specs)))

    # -- sync bridge -------------------------------------------------------- #
    def run(self, coro: Coroutine[Any, Any, Any], timeout: Optional[float] = None) -> Any:
        """Run a coroutine on the background loop and block for its result."""
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)

    def run_sync(self, spec: TaskSpec) -> TaskResult:
        return self.run(self.execute(spec))

    def run_many_sync(self, specs: Sequence[TaskSpec]) -> List[TaskResult]:
        return self.run(self.gather(specs))

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=5)
            if not loop.is_running():
                loop.close()

    # -- internals ----------------------------------------------------------- #
    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._lock:
            sem = self._semaphores.get(loop)
            if sem is None:
                sem = self._semaphores[loop] = asyncio.Semaphore(self.config.max_async_concurrency)
            return sem

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise ExecutionError("AsyncExecutor has been shut down")
            if self._loop is None:
                loop = asyncio.new_event_loop()
                ready = threading.Event()

                def runner() -> None:
                    asyncio.set_event_loop(loop)
                    loop.call_soon(ready.set)
                    loop.run_forever()

                self._thread = threading.Thread(target=runner, daemon=True, name="async-executor-loop")
                self._loop = loop
                self._thread.start()
                ready.wait(5)
            return self._loop

    async def _attempt(self, spec: TaskSpec) -> Any:
        loop = asyncio.get_running_loop()
        timeout = spec.timeout if spec.timeout is not None else self.config.default_timeout
        sandboxed = self.config.sandbox_by_default if spec.sandboxed is None else spec.sandboxed

        if sandboxed:
            call = functools.partial(
                self.sandbox.run_callable, spec.func, spec.args, spec.kwargs,
                limits=spec.limits or self.config.default_limits, timeout=timeout)
            return await loop.run_in_executor(None, call)

        async def run() -> Any:
            if inspect.iscoroutinefunction(spec.func):
                value = await spec.func(*spec.args, **spec.kwargs)
            else:
                value = await loop.run_in_executor(
                    None, functools.partial(spec.func, *spec.args, **spec.kwargs))
            if inspect.isawaitable(value):
                value = await value
            return value

        try:
            return await asyncio.wait_for(run(), timeout)
        except asyncio.TimeoutError:
            if timeout is None:  # the task itself raised TimeoutError
                raise
            raise TaskTimeoutError(f"task '{spec.name}' exceeded {timeout:.2f}s") from None
