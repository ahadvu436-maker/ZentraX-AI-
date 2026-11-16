"""Benchmark engine: run a model over a task suite, evaluate outputs, compare runs."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .evaluation_core import (
    CaseReport,
    EvaluationCase,
    EvaluationError,
    EvaluationManager,
    InvalidCaseError,
    tokenize,
)

logger = logging.getLogger("evaluation_system.benchmark")


class BenchmarkError(EvaluationError):
    """Raised for invalid suites or unrecoverable benchmark failures."""


@dataclass
class ModelOutput:
    """Structured model output (a plain ``str`` return value is also accepted)."""

    text: str
    reasoning_trace: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    ttft_s: Optional[float] = None
    cost_usd: Optional[float] = None


@dataclass
class BenchmarkTask:
    """One benchmark prompt plus the ground truth needed to evaluate it."""

    task_id: str
    prompt: str
    reference: Optional[str] = None
    category: str = "general"
    context: Sequence[str] = field(default_factory=tuple)
    expected_steps: Sequence[str] = field(default_factory=tuple)
    required_keywords: Sequence[str] = field(default_factory=tuple)
    constraints: Dict[str, Any] = field(default_factory=dict)
    should_refuse: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    _KNOWN = ("task_id", "prompt", "reference", "category", "context", "expected_steps",
              "required_keywords", "constraints", "should_refuse", "metadata")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkTask":
        if "task_id" not in data or "prompt" not in data:
            raise BenchmarkError("Each task requires 'task_id' and 'prompt'")
        known = {k: data[k] for k in cls._KNOWN if k in data}
        extra = {k: v for k, v in data.items() if k not in cls._KNOWN}
        known["metadata"] = {**dict(known.get("metadata", {})), **extra}
        return cls(**known)

    def to_case(self, output: ModelOutput, latency_s: float) -> EvaluationCase:
        tokens_estimated = output.output_tokens is None
        out_tokens = output.output_tokens if output.output_tokens is not None else round(len(tokenize(output.text)) * 1.3)
        metrics: Dict[str, float] = {"latency_s": latency_s, "output_tokens": float(out_tokens)}
        for key, value in (("input_tokens", output.input_tokens), ("ttft_s", output.ttft_s),
                           ("cost_usd", output.cost_usd)):
            if value is not None:
                metrics[key] = float(value)
        return EvaluationCase(
            prompt=self.prompt, response=output.text, case_id=self.task_id, reference=self.reference,
            context=self.context, expected_steps=self.expected_steps, reasoning_trace=output.reasoning_trace,
            required_keywords=self.required_keywords, constraints=dict(self.constraints),
            should_refuse=self.should_refuse, category=self.category, metrics=metrics,
            metadata={**self.metadata, "tokens_estimated": tokens_estimated},
        )


ModelFn = Callable[[BenchmarkTask], Union[str, ModelOutput]]


@dataclass
class BenchmarkSuite:
    """A named, versioned collection of tasks."""

    name: str
    tasks: List[BenchmarkTask]
    version: str = "1.0"
    description: str = ""

    def __post_init__(self) -> None:
        ids = [t.task_id for t in self.tasks]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise BenchmarkError(f"Duplicate task ids in suite '{self.name}': {sorted(dupes)}")

    @property
    def categories(self) -> List[str]:
        return sorted({t.category for t in self.tasks})

    def filter(self, categories: Optional[Sequence[str]] = None) -> List[BenchmarkTask]:
        return [t for t in self.tasks if not categories or t.category in categories]

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BenchmarkSuite":
        """Load ``{"name":..., "version":..., "tasks":[...]}`` or a bare list of tasks."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BenchmarkError(f"Cannot load suite from {path}: {exc}") from exc
        if isinstance(raw, list):
            raw = {"name": Path(path).stem, "tasks": raw}
        return cls(raw.get("name", Path(path).stem),
                   [BenchmarkTask.from_dict(t) for t in raw.get("tasks", [])],
                   str(raw.get("version", "1.0")), raw.get("description", ""))

    @classmethod
    def from_jsonl(cls, path: Union[str, Path], name: Optional[str] = None) -> "BenchmarkSuite":
        try:
            lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
            tasks = [BenchmarkTask.from_dict(json.loads(ln)) for ln in lines]
        except (OSError, ValueError) as exc:
            raise BenchmarkError(f"Cannot load suite from {path}: {exc}") from exc
        return cls(name or Path(path).stem, tasks)


@dataclass
class BenchmarkResult:
    """Outcome of one benchmark run."""

    suite: str
    suite_version: str
    model_name: str
    started_at: str
    finished_at: str
    duration_s: float
    reports: List[CaseReport]
    task_errors: Dict[str, str] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        return sum(r.overall_score for r in self.reports) / len(self.reports) if self.reports else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.reports) / len(self.reports) if self.reports else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite, "suite_version": self.suite_version, "model_name": self.model_name,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3), "mean_score": round(self.mean_score, 4),
            "pass_rate": round(self.pass_rate, 4), "task_errors": self.task_errors,
            "reports": [r.to_dict() for r in self.reports],
        }


class BenchmarkEngine:
    """Executes a model over a suite and evaluates the results via an ``EvaluationManager``.

    Args:
        manager: configured evaluation manager.
        max_workers: parallel model calls (use 1 for rate-limited backends).
        timeout_s: per-call timeout (``None`` disables). A timed-out call's thread
            cannot be killed, so make ``model_fn`` honour its own client timeouts too.
        retries: extra attempts after a failed call (exponential backoff).
    """

    def __init__(self, manager: EvaluationManager, max_workers: int = 4,
                 timeout_s: Optional[float] = None, retries: int = 0, backoff_s: float = 0.5) -> None:
        if max_workers < 1 or retries < 0:
            raise ValueError("max_workers must be >= 1 and retries >= 0")
        self.manager = manager
        self.max_workers = max_workers
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s

    # -- generation -------------------------------------------------------- #

    def _call(self, model_fn: ModelFn, task: BenchmarkTask) -> Union[str, ModelOutput]:
        if self.timeout_s is None:
            return model_fn(task)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(model_fn, task).result(timeout=self.timeout_s)
        finally:
            pool.shutdown(wait=False)

    @staticmethod
    def _coerce(output: Union[str, ModelOutput]) -> ModelOutput:
        if isinstance(output, ModelOutput):
            return output
        if isinstance(output, str):
            return ModelOutput(text=output)
        raise TypeError(f"model_fn must return str or ModelOutput, got {type(output).__name__}")

    def _generate(self, model_fn: ModelFn, task: BenchmarkTask) -> Tuple[EvaluationCase, Optional[str]]:
        last_error = ""
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                output = self._coerce(self._call(model_fn, task))
                return task.to_case(output, time.perf_counter() - start), None
            except Exception as exc:  # noqa: BLE001 - model backends fail in many ways
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Task %s attempt %d/%d failed: %s", task.task_id, attempt + 1,
                               self.retries + 1, last_error)
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2 ** attempt))
        failed = task.to_case(ModelOutput(text=""), 0.0)
        failed.metrics = {}  # no meaningful performance data for a failed call
        failed.metadata["generation_error"] = last_error
        return failed, last_error

    # -- public API -------------------------------------------------------- #

    def run(
        self,
        suite: BenchmarkSuite,
        model_fn: ModelFn,
        *,
        evaluators: Optional[Sequence[str]] = None,
        model_name: str = "unknown",
        categories: Optional[Sequence[str]] = None,
    ) -> BenchmarkResult:
        tasks = suite.filter(categories)
        if not tasks:
            raise BenchmarkError(f"No tasks to run in suite '{suite.name}' (categories={categories})")
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        logger.info("Benchmark '%s' v%s: %d tasks on %s", suite.name, suite.version, len(tasks), model_name)

        if self.max_workers == 1 or len(tasks) == 1:
            generated = [self._generate(model_fn, t) for t in tasks]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="bench") as pool:
                generated = list(pool.map(lambda t: self._generate(model_fn, t), tasks))

        cases = [case for case, _ in generated]
        reports = self.manager.evaluate_batch(cases, evaluators)
        task_errors: Dict[str, str] = {}
        for report, (case, error) in zip(reports, generated):
            if error:
                report.passed = False
                report.metadata["generation_error"] = error
                task_errors[case.case_id] = error

        finished = datetime.now(timezone.utc)
        return BenchmarkResult(suite.name, suite.version, model_name, started.isoformat(),
                               finished.isoformat(), time.perf_counter() - t0, reports, task_errors)

    @staticmethod
    def compare(current: BenchmarkResult, baseline: BenchmarkResult, tolerance: float = 0.02) -> Dict[str, Any]:
        """Detect regressions of ``current`` relative to ``baseline`` (matched by task id)."""
        base = {r.case_id: r for r in baseline.reports}
        regressions, improvements, newly_failing = [], [], []
        by_cat: Dict[str, List[float]] = {}
        for rep in current.reports:
            prev = base.get(rep.case_id)
            if prev is None:
                continue
            delta = rep.overall_score - prev.overall_score
            by_cat.setdefault(rep.category, []).append(delta)
            entry = {"case_id": rep.case_id, "category": rep.category,
                     "baseline": round(prev.overall_score, 4), "current": round(rep.overall_score, 4),
                     "delta": round(delta, 4)}
            if delta < -tolerance:
                regressions.append(entry)
            elif delta > tolerance:
                improvements.append(entry)
            if prev.passed and not rep.passed:
                newly_failing.append(rep.case_id)
        compared = sum(len(v) for v in by_cat.values())
        mean_delta = sum(sum(v) for v in by_cat.values()) / compared if compared else 0.0
        return {
            "compared_tasks": compared,
            "mean_score_delta": round(mean_delta, 4),
            "category_deltas": {c: round(sum(v) / len(v), 4) for c, v in by_cat.items()},
            "regressions": sorted(regressions, key=lambda e: e["delta"]),
            "improvements": sorted(improvements, key=lambda e: -e["delta"]),
            "newly_failing": newly_failing,
            "has_regression": bool(regressions) or bool(newly_failing),
        }
