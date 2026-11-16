"""Performance evaluator: latency, time-to-first-token, throughput, memory, cost."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from types import TracebackType
from typing import Dict, List, Optional, Sequence, Type

from .evaluation_core import BaseEvaluator, EvaluationCase, MetricScore, clamp, linear_score, percentile


@dataclass(frozen=True)
class PerformanceThresholds:
    """Targets; scores decay linearly to 0 at ``degrade_factor`` x the target."""

    max_latency_s: float = 5.0
    max_ttft_s: float = 1.5
    min_tokens_per_s: float = 10.0
    max_memory_mb: Optional[float] = None
    max_cost_usd: Optional[float] = None
    degrade_factor: float = 3.0


class PerformanceTracker:
    """Context manager measuring wall time (and optionally peak Python memory).

    Example::

        with PerformanceTracker(track_memory=True) as t:
            output = model(prompt)
        case.metrics.update(t.to_metrics(output_tokens=120))
    """

    def __init__(self, track_memory: bool = False) -> None:
        self.track_memory = track_memory
        self.elapsed_s: float = 0.0
        self.peak_memory_mb: Optional[float] = None
        self._start: float = 0.0
        self._started_tracing = False

    def __enter__(self) -> "PerformanceTracker":
        if self.track_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._started_tracing = True
            tracemalloc.reset_peak()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        self.elapsed_s = time.perf_counter() - self._start
        if self.track_memory:
            self.peak_memory_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
            if self._started_tracing:
                tracemalloc.stop()

    def to_metrics(self, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None) -> Dict[str, float]:
        metrics: Dict[str, float] = {"latency_s": self.elapsed_s}
        if input_tokens is not None:
            metrics["input_tokens"] = float(input_tokens)
        if output_tokens is not None:
            metrics["output_tokens"] = float(output_tokens)
        if self.peak_memory_mb is not None:
            metrics["memory_mb"] = self.peak_memory_mb
        return metrics


class PerformanceEvaluator(BaseEvaluator):
    """Scores per-case performance from ``case.metrics``.

    Recognised keys: ``latency_s``, ``ttft_s``, ``input_tokens``, ``output_tokens``,
    ``memory_mb``, ``cost_usd``.
    """

    name = "performance"

    def __init__(
        self,
        thresholds: Optional[PerformanceThresholds] = None,
        pass_threshold: float = 0.5,
        enabled_metrics: Optional[List[str]] = None,
    ) -> None:
        super().__init__(pass_threshold, enabled_metrics)
        self.thresholds = thresholds or PerformanceThresholds()

    def is_applicable(self, case: EvaluationCase) -> bool:
        return bool(case.metrics)

    def _compute_metrics(self, case: EvaluationCase) -> List[Optional[MetricScore]]:
        t, m = self.thresholds, case.metrics
        k = t.degrade_factor
        out: List[Optional[MetricScore]] = []
        if "latency_s" in m:
            out.append(self._metric("latency", linear_score(m["latency_s"], t.max_latency_s, t.max_latency_s * k),
                                    2.0, details={"value": m["latency_s"], "target": t.max_latency_s}))
        if "ttft_s" in m:
            out.append(self._metric("ttft", linear_score(m["ttft_s"], t.max_ttft_s, t.max_ttft_s * k),
                                    1.0, details={"value": m["ttft_s"], "target": t.max_ttft_s}))
        latency, out_tokens = m.get("latency_s", 0.0), m.get("output_tokens", 0.0)
        if latency > 0 and out_tokens > 0:
            tps = out_tokens / latency
            out.append(self._metric("throughput", clamp(tps / t.min_tokens_per_s), 1.0,
                                    details={"tokens_per_s": round(tps, 2), "target": t.min_tokens_per_s}))
        if "memory_mb" in m and t.max_memory_mb:
            out.append(self._metric("memory", linear_score(m["memory_mb"], t.max_memory_mb, t.max_memory_mb * k),
                                    1.0, details={"value": m["memory_mb"], "target": t.max_memory_mb}))
        if "cost_usd" in m and t.max_cost_usd:
            out.append(self._metric("cost", linear_score(m["cost_usd"], t.max_cost_usd, t.max_cost_usd * k),
                                    1.0, details={"value": m["cost_usd"], "target": t.max_cost_usd}))
        return out

    @staticmethod
    def summarize(values: Sequence[float]) -> Dict[str, float]:
        """Latency-style summary statistics (mean, p50, p95, p99, max)."""
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "max": max(values),
        }
