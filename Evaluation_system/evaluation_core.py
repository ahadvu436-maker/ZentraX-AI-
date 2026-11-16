"""Core types, utilities and orchestration for the evaluation system.

This module is the dependency root of the package. It defines:

* the data model (``EvaluationCase``, ``MetricScore``, ``EvaluationResult``,
  ``CaseReport``),
* small dependency-free text utilities shared by all evaluators,
* ``BaseEvaluator`` - the contract every evaluator implements, and
* ``EvaluationManager`` - the central coordinator that runs evaluators over
  cases (optionally in parallel), aggregates results and delegates to the
  benchmark engine and report generator.

All scores are floats in the closed interval ``[0.0, 1.0]`` (higher is better).
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
)

if TYPE_CHECKING:  # pragma: no cover - imported lazily at runtime (avoids cycles)
    from .benchmark_engine import BenchmarkResult, BenchmarkSuite, ModelFn

logger = logging.getLogger("evaluation_system")

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class EvaluationError(Exception):
    """Base class for all evaluation-system errors."""


class InvalidCaseError(EvaluationError):
    """Raised when an ``EvaluationCase`` is malformed."""


class EvaluatorNotFoundError(EvaluationError):
    """Raised when an evaluator name is not registered."""


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ARTICLES_RE = re.compile(r"\b(?:a|an|the)\b")

STOPWORDS: frozenset = frozenset(
    """a an the and or but if then else of to in on at by for with about as into
    from is are was were be been being it its this that these those i you he she
    we they me my your our their them do does did have has had will would can
    could should may might what which who whom how why when where not no yes so
    than too very just also""".split()
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``; NaN becomes ``low``."""
    if math.isnan(value):
        return low
    return max(low, min(high, value))


def tokenize(text: str, *, remove_stopwords: bool = False) -> List[str]:
    """Lower-case word tokenisation."""
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into non-empty sentences / lines."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split((text or "").strip()) if s.strip()]


def normalize_text(text: str) -> str:
    """Normalise for comparison: lower-case, strip punctuation and articles."""
    lowered = (text or "").lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = _ARTICLES_RE.sub(" ", lowered)
    return " ".join(lowered.split())


def token_f1(pred: Sequence[str], ref: Sequence[str]) -> float:
    """Bag-of-token F1 between two token sequences."""
    if not pred or not ref:
        return 1.0 if not pred and not ref else 0.0
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """Term-frequency cosine similarity between two token sequences."""
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    norm = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values()))
    return dot / norm if norm else 0.0


def linear_score(value: float, target: float, worst: float) -> float:
    """Score 1.0 at/below ``target`` decaying linearly to 0.0 at ``worst`` (lower is better)."""
    if value <= target:
        return 1.0
    if value >= worst or worst <= target:
        return 0.0
    return 1.0 - (value - target) / (worst - target)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in 0-100). Returns 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


class Status(str, Enum):
    """Outcome of an evaluator run on a single case."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class EvaluationCase:
    """A single model interaction to be evaluated.

    Evaluators only look at the fields relevant to them and skip themselves
    when required inputs are missing (e.g. accuracy needs a reference, context
    or key facts).
    """

    prompt: str = ""
    response: str = ""
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    reference: Optional[str] = None
    context: Sequence[str] = field(default_factory=tuple)
    expected_steps: Sequence[str] = field(default_factory=tuple)
    reasoning_trace: Optional[str] = None
    required_keywords: Sequence[str] = field(default_factory=tuple)
    constraints: Dict[str, Any] = field(default_factory=dict)
    should_refuse: bool = False
    category: str = "general"
    metrics: Dict[str, float] = field(default_factory=dict)  # perf: latency_s, ttft_s, ...
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ``InvalidCaseError`` if the case is structurally invalid."""
        if not isinstance(self.prompt, str) or not isinstance(self.response, str):
            raise InvalidCaseError(f"case {self.case_id}: prompt and response must be strings")
        for key, value in self.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
                raise InvalidCaseError(f"case {self.case_id}: metric '{key}' must be a finite number")


@dataclass
class MetricScore:
    """One named sub-score produced by an evaluator."""

    name: str
    score: float
    weight: float = 1.0
    passed: Optional[bool] = None
    critical: bool = False  # a failed critical metric fails the whole evaluator
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = clamp(float(self.score))
        if self.weight < 0:
            raise ValueError(f"metric '{self.name}': weight must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "weight": self.weight,
            "passed": self.passed,
            "critical": self.critical,
            "details": self.details,
        }


@dataclass
class EvaluationResult:
    """Result of one evaluator on one case."""

    case_id: str
    evaluator: str
    status: Status
    score: Optional[float] = None
    metrics: List[MetricScore] = field(default_factory=list)
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status is Status.PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "evaluator": self.evaluator,
            "status": self.status.value,
            "score": None if self.score is None else round(self.score, 4),
            "metrics": [m.to_dict() for m in self.metrics],
            "duration_s": round(self.duration_s, 6),
            "error": self.error,
        }


@dataclass
class CaseReport:
    """Aggregated outcome of all evaluators for one case."""

    case_id: str
    category: str
    overall_score: float
    passed: bool
    results: Dict[str, EvaluationResult] = field(default_factory=dict)
    duration_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "duration_s": round(self.duration_s, 6),
            "metadata": self.metadata,
            "results": {name: r.to_dict() for name, r in self.results.items()},
        }


# --------------------------------------------------------------------------- #
# Evaluator base class
# --------------------------------------------------------------------------- #


class BaseEvaluator(ABC):
    """Contract for all evaluators.

    Subclasses implement :meth:`_compute_metrics` (and optionally
    :meth:`is_applicable`). :meth:`evaluate` handles timing, aggregation,
    critical-metric gating and error isolation so a faulty evaluator can never
    crash a whole evaluation run.
    """

    name: str = "base"

    def __init__(self, pass_threshold: float = 0.6, enabled_metrics: Optional[Iterable[str]] = None) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be within [0, 1]")
        self.pass_threshold = pass_threshold
        self.enabled_metrics: Optional[Set[str]] = set(enabled_metrics) if enabled_metrics else None

    # -- hooks ------------------------------------------------------------- #

    def is_applicable(self, case: EvaluationCase) -> bool:  # noqa: D401
        """Return False to skip this evaluator for ``case``."""
        return True

    @abstractmethod
    def _compute_metrics(self, case: EvaluationCase) -> List[Optional[MetricScore]]:
        """Return metric scores for ``case`` (``None`` entries are ignored)."""

    # -- helpers ----------------------------------------------------------- #

    def _metric(
        self,
        name: str,
        score: float,
        weight: float = 1.0,
        *,
        critical: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> MetricScore:
        """Build a ``MetricScore`` whose ``passed`` flag uses this evaluator's threshold."""
        score = clamp(float(score))
        return MetricScore(
            name=name,
            score=score,
            weight=weight,
            passed=score >= self.pass_threshold,
            critical=critical,
            details=details or {},
        )

    @staticmethod
    def weighted_average(metrics: Sequence[MetricScore]) -> float:
        total = sum(m.weight for m in metrics)
        if total <= 0:
            return sum(m.score for m in metrics) / len(metrics) if metrics else 0.0
        return sum(m.score * m.weight for m in metrics) / total

    # -- public API -------------------------------------------------------- #

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluate ``case``; never raises (errors become ``Status.ERROR``)."""
        start = time.perf_counter()
        try:
            case.validate()
            if not self.is_applicable(case):
                return EvaluationResult(case.case_id, self.name, Status.SKIPPED,
                                        duration_s=time.perf_counter() - start)
            metrics = [m for m in self._compute_metrics(case) if m is not None]
            if self.enabled_metrics is not None:
                metrics = [m for m in metrics if m.name in self.enabled_metrics]
            if not metrics:
                return EvaluationResult(case.case_id, self.name, Status.SKIPPED,
                                        duration_s=time.perf_counter() - start)
            score = self.weighted_average(metrics)
            critical_fail = any(m.critical and m.passed is False for m in metrics)
            status = Status.PASSED if score >= self.pass_threshold and not critical_fail else Status.FAILED
            return EvaluationResult(case.case_id, self.name, status, score, metrics,
                                    time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 - isolation boundary
            logger.exception("Evaluator '%s' failed on case %s", self.name, case.case_id)
            return EvaluationResult(case.case_id, self.name, Status.ERROR,
                                    duration_s=time.perf_counter() - start,
                                    error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #

DEFAULT_WEIGHTS: Dict[str, float] = {
    "response": 0.25,
    "reasoning": 0.15,
    "accuracy": 0.30,
    "safety": 0.20,
    "performance": 0.10,
}


class EvaluationManager:
    """Central coordinator for evaluators, benchmarks and reports.

    Args:
        pass_threshold: minimum overall score for a case to pass.
        max_workers: thread-pool size used by ``evaluate_batch``.
        gate_evaluators: evaluators whose failure caps the overall score at
            their own score (default: ``{"safety"}``).
    """

    def __init__(
        self,
        pass_threshold: float = 0.6,
        max_workers: int = 4,
        gate_evaluators: Optional[Iterable[str]] = ("safety",),
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.pass_threshold = pass_threshold
        self.max_workers = max_workers
        self.gate_evaluators: Set[str] = set(gate_evaluators or ())
        self._evaluators: Dict[str, BaseEvaluator] = {}
        self._weights: Dict[str, float] = {}

    # -- registry ---------------------------------------------------------- #

    @classmethod
    def with_default_evaluators(cls, **kwargs: Any) -> "EvaluationManager":
        """Create a manager with all built-in evaluators registered."""
        from .accuracy_evaluator import AccuracyEvaluator
        from .performance_evaluator import PerformanceEvaluator
        from .reasoning_evaluator import ReasoningEvaluator
        from .response_evaluator import ResponseEvaluator
        from .safety_evaluator import SafetyEvaluator

        manager = cls(**kwargs)
        for evaluator in (ResponseEvaluator(), ReasoningEvaluator(), AccuracyEvaluator(),
                          SafetyEvaluator(), PerformanceEvaluator()):
            manager.register(evaluator, DEFAULT_WEIGHTS.get(evaluator.name, 1.0))
        return manager

    def register(self, evaluator: BaseEvaluator, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError("weight must be >= 0")
        if evaluator.name in self._evaluators:
            logger.warning("Replacing already-registered evaluator '%s'", evaluator.name)
        self._evaluators[evaluator.name] = evaluator
        self._weights[evaluator.name] = weight

    def unregister(self, name: str) -> None:
        self.get(name)
        del self._evaluators[name]
        del self._weights[name]

    def get(self, name: str) -> BaseEvaluator:
        try:
            return self._evaluators[name]
        except KeyError:
            raise EvaluatorNotFoundError(
                f"Evaluator '{name}' is not registered (available: {sorted(self._evaluators)})"
            ) from None

    @property
    def evaluator_names(self) -> List[str]:
        return list(self._evaluators)

    def _select(self, names: Optional[Sequence[str]]) -> List[BaseEvaluator]:
        if not self._evaluators:
            raise EvaluationError("No evaluators registered")
        return [self.get(n) for n in names] if names else list(self._evaluators.values())

    # -- evaluation -------------------------------------------------------- #

    def evaluate_case(self, case: EvaluationCase, evaluators: Optional[Sequence[str]] = None) -> CaseReport:
        """Run the selected evaluators on one case and aggregate the outcome."""
        selected = self._select(evaluators)
        start = time.perf_counter()
        results = {ev.name: ev.evaluate(case) for ev in selected}
        return self._aggregate(case, results, time.perf_counter() - start)

    def evaluate_batch(
        self,
        cases: Sequence[EvaluationCase],
        evaluators: Optional[Sequence[str]] = None,
        parallel: bool = True,
    ) -> List[CaseReport]:
        """Evaluate many cases, preserving input order."""
        self._select(evaluators)  # fail fast on unknown names
        if not parallel or self.max_workers == 1 or len(cases) < 2:
            return [self.evaluate_case(c, evaluators) for c in cases]
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="eval") as pool:
            return list(pool.map(lambda c: self.evaluate_case(c, evaluators), cases))

    def _aggregate(self, case: EvaluationCase, results: Dict[str, EvaluationResult], duration: float) -> CaseReport:
        scored = [(r, self._weights.get(r.evaluator, 1.0)) for r in results.values()
                  if r.score is not None and r.status in (Status.PASSED, Status.FAILED)]
        total_weight = sum(w for _, w in scored)
        overall = (sum(r.score * w for r, w in scored) / total_weight) if total_weight > 0 else 0.0  # type: ignore[operator]
        for r, _ in scored:
            if r.evaluator in self.gate_evaluators and r.status is Status.FAILED:
                overall = min(overall, r.score or 0.0)
        has_bad = any(r.status in (Status.FAILED, Status.ERROR) for r in results.values())
        passed = bool(scored) and not has_bad and overall >= self.pass_threshold
        return CaseReport(case.case_id, case.category, overall, passed, results, duration, dict(case.metadata))

    # -- benchmarks & reports --------------------------------------------- #

    def run_benchmark(
        self,
        suite: "BenchmarkSuite",
        model_fn: "ModelFn",
        *,
        evaluators: Optional[Sequence[str]] = None,
        model_name: str = "unknown",
        **engine_kwargs: Any,
    ) -> "BenchmarkResult":
        """Run ``model_fn`` over ``suite`` and evaluate the outputs."""
        from .benchmark_engine import BenchmarkEngine

        engine = BenchmarkEngine(self, **engine_kwargs)
        return engine.run(suite, model_fn, evaluators=evaluators, model_name=model_name)

    def build_report(
        self,
        reports: Sequence[CaseReport],
        *,
        benchmark: Optional["BenchmarkResult"] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **generator_kwargs: Any,
    ) -> Dict[str, Any]:
        """Aggregate case reports into a structured report dictionary."""
        from .reports.report_generator import ReportGenerator

        return ReportGenerator(**generator_kwargs).build(reports, benchmark=benchmark, metadata=metadata)
