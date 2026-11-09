"""
ZentraX-AI  |  super_intelligence/autonomous_executor.py

Real-time decision-execution engine.

High-level directives (plain dicts or `Directive` objects) are validated, queued by
priority, and executed asynchronously by a pool of workers. Only actions that were
explicitly registered can ever run: there is no generic "run this shell command"
path. Elevated actions require an approver callback (fail closed if none is set).

    directive ──► submit() ──► validate ─► policy ─► rate limit ─► priority queue
                                                                        │
         listeners / audit log ◄── result ◄── retry/timeout ◄── worker pool

Design guarantees
    * Allowlist-only: unknown actions are rejected at submit time.
    * Schema-checked params; unknown params rejected.
    * Bounded everything: queue size, retries, timeouts, result cache, output size.
    * Filesystem actions are confined to `workspace_root` (no '..', no symlinks).
    * Handlers never crash the engine; every directive yields exactly one result.
    * Graceful shutdown: drain (with timeout), then cancel and mark leftovers.

Requires Python 3.10+. Standard library only.

Quick start
    registry = build_default_registry()

    @registry.register("graphics.render_preview", schema={"design_id": (str, True)})
    async def render(params, ctx):
        ...

    async with AutonomousExecutor(ExecutorPolicy(workspace_root=Path(".")), registry) as ex:
        result = await ex.run({"action": "system.ping"})
"""
from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, Deque, Mapping, Optional, Union

log = logging.getLogger("zentrax.executor")

# --------------------------------------------------------------------------- #
# Errors  (each carries a stable `code` that the bridge can expose to clients)
# --------------------------------------------------------------------------- #


class ExecutorError(Exception):
    code = "executor_error"


class UnknownAction(ExecutorError):
    code = "unknown_action"


class InvalidDirective(ExecutorError):
    code = "invalid_directive"


class PolicyDenied(ExecutorError):
    code = "policy_denied"


class ExecutorBusy(ExecutorError):
    code = "busy"


class UnknownDirective(ExecutorError):
    code = "not_found"


class RetryableError(ExecutorError):
    """Raise from a handler to request a retry (bounded by max_retries)."""
    code = "retryable"


class FatalActionError(ExecutorError):
    """Raise from a handler for a known, non-retryable failure."""
    code = "action_failed"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class Risk(Enum):
    SAFE = "safe"
    ELEVATED = "elevated"  # needs approver sign-off


class Status(Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    DENIED = "denied"
    CANCELLED = "cancelled"


_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


@dataclass(slots=True)
class Directive:
    action: str
    params: dict = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timeout: Optional[float] = None
    max_retries: int = 0
    idempotency_key: Optional[str] = None
    source: str = "local"
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Directive":
        """Strictly parse an untrusted dict (e.g. arriving over the bridge)."""
        if not isinstance(d, Mapping):
            raise InvalidDirective("directive must be an object")
        action = d.get("action")
        if not isinstance(action, str) or not _ACTION_RE.match(action) or len(action) > 128:
            raise InvalidDirective("invalid 'action'")
        params = d.get("params", {})
        if not isinstance(params, dict):
            raise InvalidDirective("'params' must be an object")
        prio = d.get("priority", Priority.NORMAL)
        try:
            prio = Priority[prio.upper()] if isinstance(prio, str) else Priority(prio)
        except (KeyError, ValueError):
            raise InvalidDirective("invalid 'priority'")
        timeout = d.get("timeout")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or timeout <= 0):
            raise InvalidDirective("'timeout' must be a positive number")
        retries = d.get("max_retries", 0)
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise InvalidDirective("'max_retries' must be a non-negative integer")
        idem = d.get("idempotency_key")
        if idem is not None and (not isinstance(idem, str) or not 0 < len(idem) <= 128):
            raise InvalidDirective("invalid 'idempotency_key'")
        source = d.get("source", "local")
        if not isinstance(source, str) or len(source) > 64:
            raise InvalidDirective("invalid 'source'")
        return cls(action=action, params=params, priority=prio, timeout=timeout,
                   max_retries=retries, idempotency_key=idem, source=source)


@dataclass(slots=True)
class ExecutionResult:
    directive_id: str
    action: str
    status: Status
    output: Any = None
    error: Optional[str] = None
    attempts: int = 0
    queued_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return {"directive_id": self.directive_id, "action": self.action,
                "status": self.status.value, "output": self.output, "error": self.error,
                "attempts": self.attempts, "queued_at": self.queued_at,
                "started_at": self.started_at, "finished_at": self.finished_at}


Handler = Callable[[dict, "ExecutionContext"], Union[Awaitable[Any], Any]]
Schema = Mapping[str, tuple]  # name -> (type or tuple of types, required)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    handler: Handler
    schema: Schema
    risk: Risk
    timeout: Optional[float]
    description: str = ""


def validate_params(schema: Schema, params: Mapping[str, Any]) -> None:
    unknown = set(params) - set(schema)
    if unknown:
        raise InvalidDirective(f"unknown params: {sorted(unknown)}")
    for name, (typ, required) in schema.items():
        if name not in params:
            if required:
                raise InvalidDirective(f"missing param '{name}'")
            continue
        types = typ if isinstance(typ, tuple) else (typ,)
        value = params[name]
        if isinstance(value, bool) and bool not in types:  # bool is an int subclass
            raise InvalidDirective(f"param '{name}' has wrong type")
        if not isinstance(value, types):
            raise InvalidDirective(f"param '{name}' has wrong type")


class ActionRegistry:
    """The allowlist of everything the engine is able to do."""

    def __init__(self) -> None:
        self._actions: dict[str, ActionSpec] = {}

    def register(self, name: str, *, schema: Optional[Schema] = None, risk: Risk = Risk.SAFE,
                 timeout: Optional[float] = None, description: str = ""):
        if not _ACTION_RE.match(name):
            raise ValueError(f"invalid action name: {name!r}")

        def decorator(fn: Handler) -> Handler:
            if name in self._actions:
                raise ValueError(f"action already registered: {name}")
            self._actions[name] = ActionSpec(name, fn, dict(schema or {}), risk, timeout, description)
            return fn
        return decorator

    def get(self, name: str) -> ActionSpec:
        try:
            return self._actions[name]
        except KeyError:
            raise UnknownAction(f"unknown action: {name}") from None

    def names(self) -> list:
        return sorted(self._actions)


# --------------------------------------------------------------------------- #
# Policy and execution context
# --------------------------------------------------------------------------- #

Approver = Callable[[Directive, ActionSpec], Union[Awaitable[bool], bool]]


@dataclass
class ExecutorPolicy:
    workspace_root: Path
    workers: int = 4
    max_queue_size: int = 1000
    default_timeout: float = 30.0
    max_timeout: float = 300.0
    max_retries_cap: int = 5
    retry_base_delay: float = 0.2
    rate_per_sec: float = 50.0          # token bucket refill rate
    rate_burst: int = 100
    approver: Optional[Approver] = None  # required for Risk.ELEVATED actions
    approval_timeout: float = 120.0
    allowed_sources: Optional[frozenset] = None  # None = any source
    audit_path: Optional[Path] = None
    result_cache_size: int = 1000
    idempotency_cache_size: int = 5000
    max_output_bytes: int = 1_000_000


class _TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate, self.burst = rate, float(burst)
        self.tokens, self.stamp = float(burst), time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.stamp) * self.rate)
        self.stamp = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class ExecutionContext:
    """Passed to every handler: confined filesystem access + a scoped logger."""

    def __init__(self, directive: Directive, workspace_root: Path, max_bytes: int) -> None:
        self.directive = directive
        self.workspace_root = workspace_root.resolve()
        self.max_bytes = max_bytes
        self.log = logging.getLogger(f"zentrax.executor.action.{directive.action}")

    def resolve_path(self, rel: str) -> Path:
        """Resolve a relative path strictly inside the workspace."""
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise PolicyDenied("invalid path")
        p = PurePosixPath(rel.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts:
            raise PolicyDenied("path escapes workspace")
        if ".git" in p.parts or any(part.startswith(".env") for part in p.parts):
            raise PolicyDenied("protected path")
        target = (self.workspace_root / Path(*p.parts)).resolve()
        if not target.is_relative_to(self.workspace_root):
            raise PolicyDenied("path escapes workspace")
        cur = self.workspace_root
        for part in p.parts:
            cur = cur / part
            if cur.is_symlink():
                raise PolicyDenied("symlinks are not allowed")
        return target


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #

Listener = Callable[[ExecutionResult], Union[Awaitable[None], None]]


class AutonomousExecutor:
    def __init__(self, policy: ExecutorPolicy, registry: Optional[ActionRegistry] = None) -> None:
        self.policy = policy
        self.registry = registry or build_default_registry()
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=policy.max_queue_size)
        self._seq = itertools.count()
        self._workers: list = []
        self._futures: dict[str, asyncio.Future] = {}
        self._results: "OrderedDict[str, ExecutionResult]" = OrderedDict()
        self._idem: "OrderedDict[str, str]" = OrderedDict()
        self._running: set = set()
        self._listeners: list = []
        self._dead_letters: Deque[ExecutionResult] = deque(maxlen=100)
        self._counters: Counter = Counter()
        self._bucket = _TokenBucket(policy.rate_per_sec, policy.rate_burst)
        self._gate = asyncio.Event()
        self._gate.set()
        self._accepting = False
        self._audit_lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------- #

    async def start(self) -> None:
        if self._workers:
            return
        if self.policy.audit_path:
            self.policy.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._accepting = True
        self._workers = [asyncio.create_task(self._worker(i), name=f"zentrax-exec-{i}")
                         for i in range(self.policy.workers)]
        log.info("executor started (%d workers, actions=%d)", len(self._workers), len(self.registry.names()))

    async def stop(self, *, drain: bool = True, timeout: float = 10.0) -> None:
        if not self._workers:
            return
        self._accepting = False
        self._gate.set()
        if drain:
            try:
                await asyncio.wait_for(self._queue.join(), timeout)
            except asyncio.TimeoutError:
                log.warning("drain timed out; cancelling remaining work")
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        while not self._queue.empty():  # anything never started
            _, _, d = self._queue.get_nowait()
            await self._complete(d, self._result(d, Status.CANCELLED, error="executor stopped"))
        log.info("executor stopped")

    async def __aenter__(self) -> "AutonomousExecutor":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    def pause(self) -> None:
        """Kill-switch: workers finish current items and take no new ones."""
        self._gate.clear()
        log.warning("executor paused")

    def resume(self) -> None:
        self._gate.set()
        log.info("executor resumed")

    def add_listener(self, callback: Listener) -> None:
        self._listeners.append(callback)

    # ---- public API ------------------------------------------------------- #

    async def submit(self, directive: Union[Directive, Mapping[str, Any]], *,
                     source: Optional[str] = None) -> str:
        """Validate and enqueue. Returns the directive id. Raises ExecutorError."""
        if not self._accepting:
            raise ExecutorBusy("executor is not accepting directives")
        d = directive if isinstance(directive, Directive) else Directive.from_dict(directive)
        if source:
            d.source = source
        if d.idempotency_key and d.idempotency_key in self._idem:
            return self._idem[d.idempotency_key]

        spec = self.registry.get(d.action)
        validate_params(spec.schema, d.params)
        pol = self.policy
        if pol.allowed_sources is not None and d.source not in pol.allowed_sources:
            raise PolicyDenied(f"source '{d.source}' is not allowed")
        if spec.risk is Risk.ELEVATED and pol.approver is None:
            raise PolicyDenied(f"'{d.action}' is elevated and no approver is configured")
        if not self._bucket.allow():
            raise ExecutorBusy("rate limit exceeded")

        d.timeout = min(d.timeout or spec.timeout or pol.default_timeout, pol.max_timeout)
        d.max_retries = min(d.max_retries, pol.max_retries_cap)
        try:
            self._queue.put_nowait((int(d.priority), next(self._seq), d))
        except asyncio.QueueFull:
            raise ExecutorBusy("queue is full") from None
        self._futures[d.id] = asyncio.get_running_loop().create_future()
        if d.idempotency_key:
            self._idem[d.idempotency_key] = d.id
            while len(self._idem) > pol.idempotency_cache_size:
                self._idem.popitem(last=False)
        self._counters["submitted"] += 1
        return d.id

    async def result(self, directive_id: str, timeout: Optional[float] = None) -> ExecutionResult:
        if directive_id in self._results:
            return self._results[directive_id]
        fut = self._futures.get(directive_id)
        if fut is None:
            raise UnknownDirective("unknown or expired directive id")
        return await asyncio.wait_for(asyncio.shield(fut), timeout)

    async def run(self, directive: Union[Directive, Mapping[str, Any]], *,
                  source: Optional[str] = None, timeout: Optional[float] = None) -> ExecutionResult:
        return await self.result(await self.submit(directive, source=source), timeout)

    def stats(self) -> dict:
        return {"queued": self._queue.qsize(), "running": len(self._running),
                "paused": not self._gate.is_set(), "counters": dict(self._counters),
                "dead_letters": len(self._dead_letters), "actions": self.registry.names()}

    def dead_letters(self) -> list:
        return [r.to_dict() for r in self._dead_letters]

    # ---- internals -------------------------------------------------------- #

    async def _worker(self, n: int) -> None:
        while True:
            await self._gate.wait()
            _, _, d = await self._queue.get()
            self._running.add(d.id)
            try:
                result = await self._process(d)
                await self._complete(d, result)
            except asyncio.CancelledError:
                await self._complete(d, self._result(d, Status.CANCELLED, error="cancelled"))
                raise
            except Exception:  # last line of defense; the loop must survive
                log.exception("[%s] worker %d internal error", d.id[:8], n)
                await self._complete(d, self._result(d, Status.FAILED, error="internal error"))
            finally:
                self._running.discard(d.id)
                self._queue.task_done()

    def _result(self, d: Directive, status: Status, *, output: Any = None, error: Optional[str] = None,
                attempts: int = 0, started: float = 0.0) -> ExecutionResult:
        return ExecutionResult(d.id, d.action, status, output, error, attempts,
                               d.created_at, started, time.time())

    async def _process(self, d: Directive) -> ExecutionResult:
        spec = self.registry.get(d.action)
        started = time.time()

        if spec.risk is Risk.ELEVATED:
            try:
                ok = self.policy.approver(d, spec)  # type: ignore[misc]
                if inspect.isawaitable(ok):
                    ok = await asyncio.wait_for(ok, self.policy.approval_timeout)
            except Exception as exc:
                log.warning("[%s] approval error: %s", d.id[:8], type(exc).__name__)
                ok = False
            if not ok:
                return self._result(d, Status.DENIED, error="approval denied", started=started)

        ctx = ExecutionContext(d, self.policy.workspace_root, self.policy.max_output_bytes)
        attempts = 0
        while True:
            attempts += 1
            try:
                out = await asyncio.wait_for(self._invoke(spec, d, ctx), d.timeout)
                return self._result(d, Status.SUCCEEDED, output=self._jsonable(out),
                                    attempts=attempts, started=started)
            except asyncio.TimeoutError:
                log.warning("[%s] %s timed out after %.1fs", d.id[:8], d.action, d.timeout)
                return self._result(d, Status.TIMED_OUT, error=f"timed out after {d.timeout}s",
                                    attempts=attempts, started=started)
            except RetryableError as exc:
                if attempts <= d.max_retries:
                    delay = min(self.policy.retry_base_delay * 2 ** (attempts - 1), 10.0)
                    log.info("[%s] retry %d/%d in %.2fs: %s", d.id[:8], attempts, d.max_retries, delay, exc)
                    await asyncio.sleep(delay * (0.5 + random.random()))
                    continue
                return self._result(d, Status.FAILED, error=f"retries exhausted: {exc}"[:300],
                                    attempts=attempts, started=started)
            except ExecutorError as exc:  # known failures carry safe messages
                return self._result(d, Status.FAILED, error=f"{exc.code}: {exc}"[:300],
                                    attempts=attempts, started=started)
            except Exception as exc:  # unknown: log details, expose only the type
                log.exception("[%s] %s crashed", d.id[:8], d.action)
                return self._result(d, Status.FAILED, error=type(exc).__name__,
                                    attempts=attempts, started=started)

    @staticmethod
    async def _invoke(spec: ActionSpec, d: Directive, ctx: ExecutionContext) -> Any:
        if inspect.iscoroutinefunction(spec.handler):
            return await spec.handler(d.params, ctx)
        # Sync handlers run in a thread. NB: a timeout cannot kill the thread itself.
        return await asyncio.to_thread(spec.handler, d.params, ctx)

    def _jsonable(self, value: Any) -> Any:
        try:
            raw = json.dumps(value)
        except (TypeError, ValueError):
            value = {"repr": repr(value)[:2000]}
            raw = json.dumps(value)
        if len(raw) > self.policy.max_output_bytes:
            raise FatalActionError("output too large")
        return value

    async def _complete(self, d: Directive, result: ExecutionResult) -> None:
        self._counters[result.status.value] += 1
        self._results[d.id] = result
        while len(self._results) > self.policy.result_cache_size:
            self._results.popitem(last=False)
        if result.status in (Status.FAILED, Status.TIMED_OUT):
            self._dead_letters.append(result)
        fut = self._futures.pop(d.id, None)
        if fut and not fut.done():
            fut.set_result(result)
        log.info("[%s] %s -> %s (%d attempt(s))", d.id[:8], d.action, result.status.value, result.attempts)
        await self._audit({"ts": result.finished_at, "id": d.id, "action": d.action, "source": d.source,
                           "param_keys": sorted(d.params), "status": result.status.value,
                           "attempts": result.attempts, "error": result.error})
        for cb in list(self._listeners):
            try:
                r = cb(result)
                if inspect.isawaitable(r):
                    await r
            except Exception:
                log.exception("listener failed")

    async def _audit(self, record: dict) -> None:
        if self.policy.audit_path:
            line = json.dumps(record, sort_keys=True)  # param VALUES are never logged
            await asyncio.to_thread(self._append_audit, line)

    def _append_audit(self, line: str) -> None:
        with self._audit_lock, open(self.policy.audit_path, "a", encoding="utf-8") as f:  # type: ignore[arg-type]
            f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Built-in actions (small, safe, confined to the workspace)
# --------------------------------------------------------------------------- #


def build_default_registry() -> ActionRegistry:
    reg = ActionRegistry()

    @reg.register("system.ping", description="Liveness check")
    async def ping(params: dict, ctx: ExecutionContext) -> dict:
        return {"pong": time.time()}

    @reg.register("system.echo", schema={"message": (str, True)})
    async def echo(params: dict, ctx: ExecutionContext) -> dict:
        return {"message": params["message"][:10_000]}

    @reg.register("fs.read_text", schema={"path": (str, True)}, timeout=10)
    async def read_text(params: dict, ctx: ExecutionContext) -> dict:
        p = ctx.resolve_path(params["path"])

        def _read() -> str:
            if p.stat().st_size > ctx.max_bytes:
                raise FatalActionError("file too large")
            return p.read_text(encoding="utf-8")
        return {"path": params["path"], "content": await asyncio.to_thread(_read)}

    @reg.register("fs.list_dir", schema={"path": (str, False)}, timeout=10)
    async def list_dir(params: dict, ctx: ExecutionContext) -> dict:
        p = ctx.resolve_path(params.get("path") or ".")
        names = await asyncio.to_thread(lambda: sorted(x.name + ("/" if x.is_dir() else "")
                                                       for x in p.iterdir())[:500])
        return {"entries": names}

    @reg.register("fs.write_text", risk=Risk.ELEVATED, timeout=10,
                  schema={"path": (str, True), "content": (str, True), "overwrite": (bool, False)})
    async def write_text(params: dict, ctx: ExecutionContext) -> dict:
        p = ctx.resolve_path(params["path"])
        data = params["content"]
        if len(data.encode("utf-8")) > ctx.max_bytes:
            raise FatalActionError("content too large")
        if p.exists() and not params.get("overwrite", False):
            raise FatalActionError("file exists (set overwrite=true)")

        def _write() -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
            try:
                tmp.write_text(data, encoding="utf-8", newline="")
                os.replace(tmp, p)  # atomic
            finally:
                tmp.unlink(missing_ok=True)
        await asyncio.to_thread(_write)
        return {"path": params["path"], "bytes": len(data.encode("utf-8"))}

    return reg


# --------------------------------------------------------------------------- #
# Demo:  python -m super_intelligence.autonomous_executor
# --------------------------------------------------------------------------- #


async def _demo() -> None:
    import tempfile
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory() as tmp:
        policy = ExecutorPolicy(workspace_root=Path(tmp), approver=lambda d, s: True,
                                audit_path=Path(tmp) / ".zentrax" / "executor_audit.jsonl")
        async with AutonomousExecutor(policy) as ex:
            print((await ex.run({"action": "system.ping"})).to_dict())
            print((await ex.run({"action": "fs.write_text",
                                 "params": {"path": "notes/a.txt", "content": "hello"}})).to_dict())
            print((await ex.run({"action": "fs.read_text", "params": {"path": "notes/a.txt"}})).to_dict())
            try:
                await ex.run({"action": "fs.read_text", "params": {"path": "../etc/passwd"}})
            except ExecutorError as e:  # confinement violations surface as FAILED results
                print("error:", e)
            print(ex.stats())


if __name__ == "__main__":
    asyncio.run(_demo())
