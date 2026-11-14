"""Safe execution of tools.

Pipeline for every call: lookup -> availability -> argument validation -> permission check
-> before-hooks -> concurrency slot -> run in a worker thread with timeout/cancellation ->
output limiting -> after-hooks -> :class:`ToolResult`.

``execute`` **never raises** for tool problems; failures come back as ``ToolResult(success=False)``
with a stable ``error_type``.  Tool exceptions' tracebacks are logged, not returned to the model.

A tool that ignores cancellation keeps its (daemon) thread after a timeout; that thread keeps
its concurrency slot until it really finishes, so stuck tools can't cause unbounded growth.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .tool_core import (
    BaseTool, ToolBusyError, ToolContext, ToolError, ToolExecutionError, ToolPermissionError,
    ToolResult, ToolTimeoutError, ToolUnavailableError, validate_arguments,
)
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = ["ToolExecutor"]

BeforeHook = Callable[[str, Dict[str, Any], ToolContext], None]
AfterHook = Callable[[ToolResult], None]
ToolCallSpec = Tuple[str, Mapping[str, Any]]


class ToolExecutor:
    """Runs registered tools safely.

    Args:
        registry: Where tools are looked up.
        max_concurrency: Maximum simultaneously running tool calls.
        default_timeout: Wall-clock limit (seconds) when neither call nor spec sets one.
        max_output_chars: Serialized output larger than this is replaced by a truncated string.
        queue_timeout: How long a call may wait for a free concurrency slot.
        default_context: Factory/instance used when a call passes no context.
    """

    def __init__(self, registry: ToolRegistry, *, max_concurrency: int = 8,
                 default_timeout: float = 30.0, max_output_chars: int = 50_000,
                 queue_timeout: float = 30.0,
                 default_context: Optional[Callable[[], ToolContext]] = None) -> None:
        self.registry = registry
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self.queue_timeout = queue_timeout
        self._default_context = default_context or (lambda: ToolContext())
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._max_concurrency = max_concurrency
        self._before: List[BeforeHook] = []
        self._after: List[AfterHook] = []
        self._stats: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._closed = False

    # -- hooks -------------------------------------------------------------- #
    def add_before_hook(self, hook: BeforeHook) -> None:
        """Hook ``(tool_name, arguments, context)`` run before execution; raise a ToolError to veto."""
        self._before.append(hook)

    def add_after_hook(self, hook: AfterHook) -> None:
        """Hook run with every final :class:`ToolResult` (audit/logging). Errors are swallowed."""
        self._after.append(hook)

    # -- public API ---------------------------------------------------------- #
    def execute(self, name: str, arguments: Optional[Mapping[str, Any]] = None,
                context: Optional[ToolContext] = None, timeout: Optional[float] = None) -> ToolResult:
        """Execute one tool call and return its result (never raises for tool failures)."""
        started = time.monotonic()
        ctx = context or self._default_context()
        result = self._execute(name, arguments, ctx, timeout, started)
        result.duration = time.monotonic() - started
        result.request_id = ctx.request_id
        self._record(result)
        for hook in self._after:
            try:
                hook(result)
            except Exception:  # noqa: BLE001
                logger.exception("after-hook failed")
        return result

    def execute_many(self, calls: Sequence[ToolCallSpec], context: Optional[ToolContext] = None,
                     timeout: Optional[float] = None, max_parallel: Optional[int] = None) -> List[ToolResult]:
        """Run several calls concurrently; results keep input order."""
        if not calls:
            return []
        workers = min(max_parallel or self._max_concurrency, len(calls))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tool-batch")
        try:
            futures = [pool.submit(self.execute, n, a, (context.derive() if context else None), timeout)
                       for n, a in calls]
            return [f.result() for f in futures]
        finally:
            pool.shutdown(wait=False)

    async def aexecute(self, name: str, arguments: Optional[Mapping[str, Any]] = None,
                       context: Optional[ToolContext] = None, timeout: Optional[float] = None) -> ToolResult:
        """asyncio-friendly wrapper around :meth:`execute`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.execute(name, arguments, context, timeout))

    def stats(self) -> Dict[str, Dict[str, float]]:
        """Per-tool counters: calls, errors, total_duration."""
        with self._lock:
            return {k: dict(v) for k, v in self._stats.items()}

    def shutdown(self) -> None:
        self._closed = True

    # -- internals ----------------------------------------------------------- #
    def _execute(self, name: str, arguments: Optional[Mapping[str, Any]], ctx: ToolContext,
                 timeout: Optional[float], started: float) -> ToolResult:
        if self._closed:
            return ToolResult.fail(name, ToolUnavailableError("executor is shut down"))
        try:
            tool = self.registry.get(name)
            if not tool.is_available():
                raise ToolUnavailableError(tool.unavailable_reason() or f"tool '{name}' is unavailable")
            args = validate_arguments(tool.spec, arguments)
            self._check_permissions(tool, args, ctx)
            for hook in self._before:
                hook(name, dict(args), ctx)

            limit = timeout or tool.spec.timeout or self.default_timeout
            run_ctx = ctx.derive(deadline=time.monotonic() + limit,
                                 cancel_event=threading.Event())
            output = self._run_in_thread(tool, run_ctx, args, limit)
            output, truncated = self._limit_output(output)
            result = ToolResult.ok(name, output)
            if truncated:
                result.metadata["truncated"] = True
            return result
        except ToolError as exc:
            return ToolResult.fail(name, exc)
        except Exception as exc:  # noqa: BLE001 - hooks etc.
            logger.exception("unexpected error executing tool %s", name)
            return ToolResult.fail(name, ToolExecutionError(f"{type(exc).__name__}: {exc}"))

    @staticmethod
    def _check_permissions(tool: BaseTool, args: Dict[str, Any], ctx: ToolContext) -> None:
        missing = sorted(p for p in tool.required_permissions_for(args) if not ctx.has_permission(p))
        if missing:
            raise ToolPermissionError(f"missing permission(s) for '{tool.name}': {', '.join(missing)}")

    def _run_in_thread(self, tool: BaseTool, ctx: ToolContext, args: Dict[str, Any], limit: float) -> Any:
        if not self._slots.acquire(timeout=self.queue_timeout):
            raise ToolBusyError("too many concurrent tool calls; try again shortly")
        box: Dict[str, Any] = {}

        def worker() -> None:
            try:
                box["value"] = tool.run(ctx, **args)
            except BaseException as exc:  # noqa: BLE001 - reported to caller
                box["error"] = exc
            finally:
                self._slots.release()

        thread = threading.Thread(target=worker, daemon=True, name=f"tool-{tool.name}")
        try:
            thread.start()
        except BaseException:
            self._slots.release()
            raise
        thread.join(limit)
        if thread.is_alive():
            ctx.cancel_event.set()  # cooperative cancellation for tools that poll it
            thread.join(0.5)
            if thread.is_alive():
                logger.warning("tool %s ignored cancellation after timeout", tool.name)
            raise ToolTimeoutError(f"tool '{tool.name}' timed out after {limit:.1f}s")
        if "error" in box:
            exc = box["error"]
            if isinstance(exc, ToolError):
                raise exc
            if isinstance(exc, Exception):
                logger.error("tool %s raised", tool.name, exc_info=exc)
                raise ToolExecutionError(f"{type(exc).__name__}: {exc}") from exc
            raise exc  # KeyboardInterrupt / SystemExit
        return box.get("value")

    def _limit_output(self, output: Any) -> Tuple[Any, bool]:
        limit = self.max_output_chars
        if isinstance(output, str):
            return (output, False) if len(output) <= limit else (output[:limit] + "\n...[truncated]", True)
        try:
            text = json.dumps(output, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(output)
            return (text, False) if len(text) <= limit else (text[:limit] + "...[truncated]", True)
        return (output, False) if len(text) <= limit else (text[:limit] + "...[truncated]", True)

    def _record(self, result: ToolResult) -> None:
        with self._lock:
            s = self._stats.setdefault(result.tool_name, {"calls": 0, "errors": 0, "total_duration": 0.0})
            s["calls"] += 1
            s["errors"] += 0 if result.success else 1
            s["total_duration"] += result.duration
