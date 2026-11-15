"""Scores and ranks candidate models by cost, latency and availability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .model_core import ModelSpec, RoutingStrategy
from .model_registry import ModelMetrics

_DEFAULT_WEIGHTS: Dict[RoutingStrategy, Dict[str, float]] = {
    RoutingStrategy.COST: {"cost": 0.8, "latency": 0.1, "availability": 0.1},
    RoutingStrategy.PERFORMANCE: {"cost": 0.05, "latency": 0.8, "availability": 0.15},
    RoutingStrategy.AVAILABILITY: {"cost": 0.05, "latency": 0.15, "availability": 0.8},
    RoutingStrategy.BALANCED: {"cost": 0.3, "latency": 0.35, "availability": 0.35},
}


@dataclass(frozen=True)
class ModelScore:
    model_id: str
    score: float
    components: Mapping[str, float]


class ModelSelector:
    """Pure ranking logic; holds no state about models."""

    def __init__(self, weights: Optional[Mapping[RoutingStrategy, Mapping[str, float]]] = None,
                 default_latency_ms: float = 1000.0) -> None:
        self._weights: Dict[RoutingStrategy, Dict[str, float]] = {
            k: dict(v) for k, v in _DEFAULT_WEIGHTS.items()
        }
        if weights:
            for strategy, w in weights.items():
                total = sum(w.values())
                if total <= 0 or any(v < 0 for v in w.values()):
                    raise ValueError("weights must be non-negative with a positive sum")
                self._weights[strategy] = {k: v / total for k, v in w.items()}
        self._default_latency = default_latency_ms

    def rank(self, candidates: Sequence[ModelSpec], metrics: Mapping[str, ModelMetrics],
             strategy: RoutingStrategy = RoutingStrategy.BALANCED) -> List[ModelScore]:
        if not candidates:
            return []
        w = self._weights[strategy]
        latencies, costs, avail = [], [], []
        for s in candidates:
            m = metrics.get(s.model_id) or ModelMetrics()
            latencies.append(m.avg_latency_ms or s.expected_latency_ms or self._default_latency)
            costs.append(s.cost_per_1k_tokens)
            avail.append(m.success_rate)
        lat_n = self._lower_is_better(latencies)
        cost_n = self._lower_is_better(costs)
        av_n = self._higher_is_better(avail)
        scores = []
        for i, s in enumerate(candidates):
            comp = {"cost": cost_n[i], "latency": lat_n[i], "availability": av_n[i]}
            total = sum(w.get(k, 0.0) * v for k, v in comp.items())
            scores.append(ModelScore(s.model_id, total, comp))
        return sorted(scores, key=lambda x: (-x.score, x.model_id))

    @staticmethod
    def _lower_is_better(values: Sequence[float]) -> List[float]:
        lo, hi = min(values), max(values)
        return [1.0 if hi == lo else 1.0 - (v - lo) / (hi - lo) for v in values]

    @staticmethod
    def _higher_is_better(values: Sequence[float]) -> List[float]:
        lo, hi = min(values), max(values)
        return [1.0 if hi == lo else (v - lo) / (hi - lo) for v in values]
