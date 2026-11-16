"""Core execution engine for the ``Data_pipeline`` package.

This module contains everything that is *domain agnostic*:

* :class:`PipelineStep` - abstract unit of work with retry / failure policies.
* :class:`Pipeline` - sequential engine with structured logging, per-step
  snapshots (rollback on failure), hooks and an execution report.
* :class:`PipelineContext` - the object handed from step to step.
* :class:`ArtifactBundle` - integrity-checked (SHA-256, optional HMAC)
  persistence of fitted pipeline state and models.
* Small shared helpers (atomic file writes, JSON conversion, column-kind
  detection) used by the other modules.

Contract for steps
------------------
A step receives a :class:`PipelineContext`, must treat ``context.data`` as
*immutable* (assign a new frame instead of mutating in place) and returns the
context.  That contract is what makes snapshot-based recovery cheap and safe.
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import hashlib
import hmac
import json
import logging
import math
import os
import pickle
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd

__all__ = [
    "LOGGER_NAME",
    "configure_logging",
    "utc_now_iso",
    "PipelineError",
    "PipelineConfigurationError",
    "StepExecutionError",
    "ArtifactError",
    "StepStatus",
    "FailurePolicy",
    "RetryPolicy",
    "StepResult",
    "PipelineReport",
    "PipelineContext",
    "PipelineStep",
    "FunctionStep",
    "Pipeline",
    "LoadedBundle",
    "ArtifactBundle",
    "column_kind",
    "columns_of_kind",
    "to_jsonable",
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "sha256_bytes",
    "sha256_file",
]

LOGGER_NAME = "data_pipeline"
logger = logging.getLogger(LOGGER_NAME)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def configure_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """Attach a single stream handler to the package logger (idempotent)."""
    pkg_logger = logging.getLogger(LOGGER_NAME)
    pkg_logger.setLevel(level)
    if not any(getattr(h, "_data_pipeline_handler", False) for h in pkg_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        handler._data_pipeline_handler = True  # type: ignore[attr-defined]
        pkg_logger.addHandler(handler)
    return pkg_logger


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def column_kind(series: pd.Series) -> str:
    """Classify a column as ``numeric``, ``bool``, ``datetime`` or ``categorical``.

    Works across pandas 1.x - 3.x (including the default ``str`` dtype that
    pandas 3 uses for text) because it only inspects the dtype object.
    """
    dtype = series.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    return "categorical"


def columns_of_kind(
    df: pd.DataFrame, kind: str, exclude: Iterable[str] = ()
) -> List[str]:
    """Return the columns of ``df`` whose :func:`column_kind` equals ``kind``."""
    skip = set(exclude)
    return [c for c in df.columns if c not in skip and column_kind(df[c]) == kind]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into something ``json.dumps`` accepts.

    Numpy / pandas scalars become Python scalars, timestamps become ISO
    strings, non-finite floats become ``None`` and sets/tuples become lists.
    """
    if obj is None or obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        return [to_jsonable(v) for v in list(obj)]
    return str(obj)


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Atomically write UTF-8 text."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Union[str, Path], payload: Any) -> None:
    """Atomically write ``payload`` as pretty, key-sorted JSON."""
    atomic_write_text(path, json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class PipelineError(Exception):
    """Base class for every error raised by this package."""


class PipelineConfigurationError(PipelineError):
    """The pipeline or one of its steps is mis-configured."""


class ArtifactError(PipelineError):
    """Artifact bundle could not be written, read or verified."""


class StepExecutionError(PipelineError):
    """A step failed after exhausting its retry / recovery policy.

    Attributes:
        step_name: Name of the failing step.
        original: The underlying exception (also set as ``__cause__``).
        report: Partial :class:`PipelineReport` up to and including the failure.
    """

    def __init__(
        self,
        step_name: str,
        original: BaseException,
        report: Optional["PipelineReport"] = None,
    ) -> None:
        super().__init__(
            f"Step '{step_name}' failed: {type(original).__name__}: {original}"
        )
        self.step_name = step_name
        self.original = original
        self.report = report


# --------------------------------------------------------------------------- #
# Policies, results and context
# --------------------------------------------------------------------------- #
class StepStatus(str, Enum):
    """Lifecycle state of a step inside one pipeline run."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RECOVERED = "recovered"


class FailurePolicy(str, Enum):
    """What the engine does when a step still fails after its retries."""

    RAISE = "raise"  # abort the pipeline (default)
    SKIP = "skip"  # roll back the step's changes and continue
    FALLBACK = "fallback"  # call ``step.fallback(context, exc)`` and continue


@dataclass(frozen=True)
class RetryPolicy:
    """Retry behaviour for a step.

    Attributes:
        max_attempts: Total attempts (``1`` means no retry).
        backoff_s: Delay before the second attempt.
        backoff_factor: Multiplier applied to the delay after each retry.
        retry_on: Only exceptions of these types are retried.
    """

    max_attempts: int = 1
    backoff_s: float = 0.0
    backoff_factor: float = 2.0
    retry_on: Tuple[Type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise PipelineConfigurationError("RetryPolicy.max_attempts must be >= 1")
        if self.backoff_s < 0 or self.backoff_factor < 1:
            raise PipelineConfigurationError("Invalid RetryPolicy backoff settings")


@dataclass
class StepResult:
    """Execution record of one step."""

    name: str
    status: StepStatus = StepStatus.PENDING
    duration_s: float = 0.0
    attempts: int = 0
    error: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["status"] = self.status.value
        return out


@dataclass
class PipelineReport:
    """Structured summary of a pipeline run."""

    pipeline: str
    run_id: str
    started_at: str
    finished_at: Optional[str] = None
    results: List[StepResult] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return sum(r.duration_s for r in self.results)

    @property
    def success(self) -> bool:
        return self.finished_at is not None and not any(
            r.status in (StepStatus.FAILED, StepStatus.PENDING) for r in self.results
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success": self.success,
            "duration_s": round(self.duration_s, 4),
            "steps": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        head = (
            f"Pipeline '{self.pipeline}' run {self.run_id}: "
            f"{'OK' if self.success else 'FAILED'} in {self.duration_s:.2f}s"
        )
        lines = [head]
        for r in self.results:
            detail = r.error or r.note or ""
            lines.append(
                f"  [{r.status.value:<9}] {r.name:<28} {r.duration_s:7.3f}s "
                f"attempts={r.attempts} {detail}".rstrip()
            )
        return "\n".join(lines)


@dataclass
class PipelineContext:
    """State passed between steps.

    Attributes:
        data: The working DataFrame.
        artifacts: Fitted objects and other in-memory outputs, keyed by name.
        metadata: JSON-friendly facts about the run (row counts, versions ...).
        run_id: Unique id of this run.
        mode: ``"training"`` (fit on data) or ``"inference"`` (apply fitted state).
        report: Filled in by :meth:`Pipeline.run`.
    """

    data: Optional[pd.DataFrame] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    mode: str = "training"
    report: Optional[PipelineReport] = None

    def require_data(self) -> pd.DataFrame:
        """Return ``data`` or raise if no step has produced it yet."""
        if self.data is None:
            raise PipelineError("No data in context; run an ingestion step first.")
        return self.data


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
FallbackFn = Callable[[PipelineContext, Exception], PipelineContext]


class PipelineStep(abc.ABC):
    """Abstract base class for a pipeline step.

    Args:
        name: Unique name inside a pipeline (defaults to the class name).
        retry: :class:`RetryPolicy`; default is a single attempt.
        on_failure: :class:`FailurePolicy` applied after retries are exhausted.
        fallback: Required when ``on_failure`` is ``FALLBACK``.
        modes: If given, the step only runs in these context modes
            (e.g. ``("training",)``); otherwise it is reported as skipped.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        retry: Optional[RetryPolicy] = None,
        on_failure: FailurePolicy = FailurePolicy.RAISE,
        fallback: Optional[FallbackFn] = None,
        modes: Optional[Sequence[str]] = None,
    ) -> None:
        self.name = name or type(self).__name__
        self.retry = retry or RetryPolicy()
        self.on_failure = FailurePolicy(on_failure)
        self.fallback = fallback
        self.modes = tuple(modes) if modes is not None else None
        if self.on_failure is FailurePolicy.FALLBACK and fallback is None:
            raise PipelineConfigurationError(
                f"Step '{self.name}' uses FailurePolicy.FALLBACK but no fallback was given"
            )

    @abc.abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the step and return the (possibly updated) context."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"


class FunctionStep(PipelineStep):
    """Wrap a plain function ``fn(context)`` as a step.

    The function may return ``None`` (context unchanged), a ``DataFrame``
    (becomes ``context.data``) or a ``PipelineContext``.
    """

    def __init__(
        self,
        func: Callable[[PipelineContext], Any],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name or getattr(func, "__name__", "function_step"), **kwargs)
        self._func = func

    def run(self, context: PipelineContext) -> PipelineContext:
        result = self._func(context)
        if result is None:
            return context
        if isinstance(result, pd.DataFrame):
            context.data = result
            return context
        if isinstance(result, PipelineContext):
            return result
        raise PipelineConfigurationError(
            f"FunctionStep '{self.name}' returned unsupported type {type(result).__name__}"
        )


@dataclass
class _Snapshot:
    """Cheap rollback point taken before each step attempt."""

    data: Optional[pd.DataFrame]
    artifacts: Dict[str, Any]
    metadata: Dict[str, Any]

    @classmethod
    def take(cls, ctx: PipelineContext) -> "_Snapshot":
        try:
            metadata = copy.deepcopy(ctx.metadata)
        except Exception:  # un-copyable metadata: fall back to a shallow copy
            metadata = dict(ctx.metadata)
        return cls(ctx.data, dict(ctx.artifacts), metadata)

    def restore(self, ctx: PipelineContext) -> None:
        ctx.data = self.data
        ctx.artifacts.clear()
        ctx.artifacts.update(self.artifacts)
        ctx.metadata.clear()
        ctx.metadata.update(self.metadata)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
_HOOK_EVENTS = ("before_step", "after_step", "on_error")


class Pipeline:
    """Sequential pipeline engine.

    Features: ordered steps, per-step retry with exponential backoff, rollback
    to the pre-step snapshot on failure, ``SKIP`` / ``FALLBACK`` recovery,
    lifecycle hooks and a :class:`PipelineReport`.

    Example:
        >>> pipe = Pipeline("demo", [FunctionStep(lambda c: c.require_data().head(3))])
        >>> ctx = pipe.run(data=pd.DataFrame({"a": range(10)}))
        >>> len(ctx.data)
        3
    """

    def __init__(
        self, name: str, steps: Optional[Iterable[PipelineStep]] = None
    ) -> None:
        self.name = name
        self._steps: List[PipelineStep] = []
        self._hooks: Dict[str, List[Callable[..., None]]] = {e: [] for e in _HOOK_EVENTS}
        for step in steps or ():
            self.add_step(step)

    # -- composition ------------------------------------------------------ #
    @property
    def steps(self) -> Tuple[PipelineStep, ...]:
        return tuple(self._steps)

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Append a step (names must be unique). Returns ``self`` for chaining."""
        return self.insert_step(len(self._steps), step)

    def insert_step(self, index: int, step: PipelineStep) -> "Pipeline":
        if not isinstance(step, PipelineStep):
            raise PipelineConfigurationError(
                f"Expected PipelineStep, got {type(step).__name__}"
            )
        if any(s.name == step.name for s in self._steps):
            raise PipelineConfigurationError(f"Duplicate step name '{step.name}'")
        self._steps.insert(index, step)
        return self

    def remove_step(self, name: str) -> "Pipeline":
        before = len(self._steps)
        self._steps = [s for s in self._steps if s.name != name]
        if len(self._steps) == before:
            raise PipelineConfigurationError(f"No step named '{name}'")
        return self

    def add_hook(self, event: str, callback: Callable[..., None]) -> "Pipeline":
        """Register a hook. Events and signatures:

        * ``before_step(step, context)``
        * ``after_step(step, context, result)``
        * ``on_error(step, context, exception)``

        Hook failures are logged and never break the run.
        """
        if event not in self._hooks:
            raise PipelineConfigurationError(
                f"Unknown hook '{event}'. Choose from {_HOOK_EVENTS}"
            )
        self._hooks[event].append(callback)
        return self

    def _fire(self, event: str, *args: Any) -> None:
        for callback in self._hooks[event]:
            try:
                callback(*args)
            except Exception:  # noqa: BLE001 - hooks must never break a run
                logger.exception("Hook '%s' raised; ignoring", event)

    # -- execution -------------------------------------------------------- #
    def run(
        self,
        context: Optional[PipelineContext] = None,
        *,
        data: Optional[pd.DataFrame] = None,
        mode: Optional[str] = None,
    ) -> PipelineContext:
        """Run all steps in order.

        Raises:
            StepExecutionError: A step failed and its policy is ``RAISE``.
                ``exc.report`` holds the partial report.
        """
        if not self._steps:
            raise PipelineConfigurationError(f"Pipeline '{self.name}' has no steps")
        ctx = context if context is not None else PipelineContext()
        if data is not None:
            ctx.data = data
        if mode is not None:
            ctx.mode = mode
        report = PipelineReport(self.name, ctx.run_id, utc_now_iso())
        ctx.report = report
        logger.info(
            "Pipeline '%s' started (run=%s, mode=%s, steps=%d)",
            self.name, ctx.run_id, ctx.mode, len(self._steps),
        )

        for step in self._steps:
            result = StepResult(name=step.name)
            report.results.append(result)

            if step.modes is not None and ctx.mode not in step.modes:
                result.status = StepStatus.SKIPPED
                result.note = f"not applicable in mode '{ctx.mode}'"
                logger.debug("Step '%s' skipped (mode=%s)", step.name, ctx.mode)
                continue

            self._fire("before_step", step, ctx)
            started = time.perf_counter()
            try:
                ctx = self._execute_step(step, ctx, result)
            except StepExecutionError as exc:
                result.duration_s = time.perf_counter() - started
                report.finished_at = utc_now_iso()
                exc.report = report
                ctx.report = report
                self._fire("on_error", step, ctx, exc)
                logger.error("Pipeline '%s' aborted at step '%s'", self.name, step.name)
                raise
            result.duration_s = time.perf_counter() - started
            ctx.report = report
            self._fire("after_step", step, ctx, result)
            logger.info(
                "Step '%s' %s in %.3fs (attempts=%d)",
                step.name, result.status.value, result.duration_s, result.attempts,
            )

        report.finished_at = utc_now_iso()
        logger.info("Pipeline '%s' finished in %.2fs", self.name, report.duration_s)
        return ctx

    def _execute_step(
        self, step: PipelineStep, ctx: PipelineContext, result: StepResult
    ) -> PipelineContext:
        policy = step.retry
        delay = policy.backoff_s
        last_exc: Optional[Exception] = None

        for attempt in range(1, policy.max_attempts + 1):
            result.attempts = attempt
            snapshot = _Snapshot.take(ctx)
            try:
                out = step.run(ctx)
            except Exception as exc:  # noqa: BLE001 - engine boundary
                last_exc = exc
                snapshot.restore(ctx)  # roll back partial changes
                retryable = isinstance(exc, policy.retry_on) and attempt < policy.max_attempts
                logger.warning(
                    "Step '%s' attempt %d/%d failed: %s: %s%s",
                    step.name, attempt, policy.max_attempts,
                    type(exc).__name__, exc, " - retrying" if retryable else "",
                )
                if not retryable:
                    break
                if delay > 0:
                    time.sleep(delay)
                delay *= policy.backoff_factor
            else:
                if not isinstance(out, PipelineContext):
                    raise PipelineConfigurationError(
                        f"Step '{step.name}' must return a PipelineContext, "
                        f"got {type(out).__name__}"
                    )
                result.status = StepStatus.SUCCESS
                return out

        assert last_exc is not None
        return self._recover(step, ctx, result, last_exc)

    def _recover(
        self,
        step: PipelineStep,
        ctx: PipelineContext,
        result: StepResult,
        exc: Exception,
    ) -> PipelineContext:
        result.error = f"{type(exc).__name__}: {exc}"

        if step.on_failure is FailurePolicy.SKIP:
            result.status = StepStatus.SKIPPED
            result.note = "failure ignored (SKIP policy); changes rolled back"
            logger.warning("Step '%s' skipped after failure: %s", step.name, result.error)
            return ctx

        if step.on_failure is FailurePolicy.FALLBACK:
            assert step.fallback is not None
            try:
                recovered = step.fallback(ctx, exc)
                if not isinstance(recovered, PipelineContext):
                    raise PipelineConfigurationError(
                        f"Fallback of '{step.name}' must return a PipelineContext"
                    )
            except Exception as fb_exc:  # noqa: BLE001
                result.status = StepStatus.FAILED
                result.error = f"fallback failed: {type(fb_exc).__name__}: {fb_exc}"
                raise StepExecutionError(step.name, fb_exc) from fb_exc
            result.status = StepStatus.RECOVERED
            result.note = "recovered via fallback"
            logger.warning("Step '%s' recovered via fallback", step.name)
            return recovered

        result.status = StepStatus.FAILED
        raise StepExecutionError(step.name, exc) from exc


# --------------------------------------------------------------------------- #
# Artifact bundle (fitted state + model persistence)
# --------------------------------------------------------------------------- #
_FILE_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def _sign_manifest(manifest: Mapping[str, Any], key: bytes) -> str:
    payload = json.dumps(
        {k: v for k, v in manifest.items() if k != "signature"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


@dataclass
class LoadedBundle:
    """Verified contents of an artifact directory."""

    directory: Path
    manifest: Dict[str, Any]
    json_artifacts: Dict[str, Dict[str, Any]]
    model: Any


class ArtifactBundle:
    """Immutable, integrity-checked persistence of a training run's outputs.

    Layout of one run directory::

        <run>/manifest.json   # metadata + SHA-256 of every file (+ optional HMAC)
        <run>/<name>.json     # one file per JSON artifact (cleaner, schema ...)
        <run>/model.pkl       # pickled model (optional)

    Security notes:
        * Every file is hashed; hashes are verified **before** anything is
          unpickled.
        * Hashes alone do not prove *origin*. Pass ``signing_key`` to sign the
          manifest with HMAC-SHA256 and require a valid signature on load.
        * Only load bundles from sources you trust: pickle can execute code.
    """

    MANIFEST = "manifest.json"
    MODEL_FILE = "model.pkl"
    LATEST = "latest.json"
    FORMAT_VERSION = 1

    @classmethod
    def save(
        cls,
        directory: Union[str, Path],
        *,
        json_artifacts: Mapping[str, Mapping[str, Any]],
        model: Any = None,
        manifest: Optional[Mapping[str, Any]] = None,
        signing_key: Optional[bytes] = None,
    ) -> Path:
        """Write a new run directory atomically; refuses to overwrite."""
        target = Path(directory)
        if target.exists():
            raise ArtifactError(f"Refusing to overwrite existing artifacts: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(dir=str(target.parent), prefix=".stage-"))
        try:
            files: Dict[str, str] = {}
            for name, payload in json_artifacts.items():
                if not _FILE_NAME_RE.match(name):
                    raise ArtifactError(f"Illegal artifact name: {name!r}")
                blob = json.dumps(to_jsonable(payload), indent=2, sort_keys=True).encode()
                (stage / f"{name}.json").write_bytes(blob)
                files[f"{name}.json"] = sha256_bytes(blob)
            if model is not None:
                blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                (stage / cls.MODEL_FILE).write_bytes(blob)
                files[cls.MODEL_FILE] = sha256_bytes(blob)

            body: Dict[str, Any] = to_jsonable(dict(manifest or {}))
            body.update(
                format_version=cls.FORMAT_VERSION, created_at=utc_now_iso(), files=files
            )
            if signing_key:
                body["signature"] = _sign_manifest(body, signing_key)
            (stage / cls.MANIFEST).write_text(
                json.dumps(body, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.replace(stage, target)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        logger.info("Artifacts written to %s (%d files)", target, len(files) + 1)
        return target

    @classmethod
    def write_latest_pointer(cls, root: Union[str, Path], run_dir: Union[str, Path]) -> None:
        """Record ``run_dir`` (a child of ``root``) as the latest run."""
        run_name = Path(run_dir).name
        atomic_write_json(Path(root) / cls.LATEST, {"run": run_name, "updated_at": utc_now_iso()})

    @classmethod
    def resolve(cls, directory: Union[str, Path]) -> Path:
        """Return a run directory, following ``latest.json`` if given a root."""
        path = Path(directory)
        if (path / cls.MANIFEST).is_file():
            return path
        pointer = path / cls.LATEST
        if pointer.is_file():
            run = json.loads(pointer.read_text(encoding="utf-8")).get("run", "")
            if not _FILE_NAME_RE.match(str(run)):
                raise ArtifactError(f"Invalid run name in {pointer}")
            return path / run
        raise ArtifactError(f"No artifact bundle found at {path}")

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        *,
        signing_key: Optional[bytes] = None,
    ) -> LoadedBundle:
        """Load and verify a bundle (hashes, then optional HMAC, then content)."""
        run_dir = cls.resolve(directory)
        try:
            manifest = json.loads((run_dir / cls.MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"Unreadable manifest in {run_dir}: {exc}") from exc

        if signing_key is not None:
            signature = manifest.get("signature")
            if not signature or not hmac.compare_digest(
                str(signature), _sign_manifest(manifest, signing_key)
            ):
                raise ArtifactError("Manifest signature missing or invalid")
        elif manifest.get("signature"):
            logger.warning("Bundle is signed but no signing_key was supplied; not verifying")

        files: Mapping[str, str] = manifest.get("files", {})
        blobs: Dict[str, bytes] = {}
        for file_name, expected in files.items():
            if not _FILE_NAME_RE.match(file_name):
                raise ArtifactError(f"Illegal file name in manifest: {file_name!r}")
            try:
                blob = (run_dir / file_name).read_bytes()
            except OSError as exc:
                raise ArtifactError(f"Missing artifact file {file_name}: {exc}") from exc
            if not hmac.compare_digest(sha256_bytes(blob), str(expected)):
                raise ArtifactError(f"Integrity check failed for {file_name}")
            blobs[file_name] = blob

        json_artifacts = {
            name[: -len(".json")]: json.loads(blob.decode("utf-8"))
            for name, blob in blobs.items()
            if name.endswith(".json")
        }
        model = pickle.loads(blobs[cls.MODEL_FILE]) if cls.MODEL_FILE in blobs else None  # noqa: S301
        return LoadedBundle(run_dir, manifest, json_artifacts, model)
