"""Uncertainty estimation from probabilities, sampled answers, or numeric ensembles."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Hashable, Optional, Sequence


class UncertaintyLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass(frozen=True)
class UncertaintyEstimate:
    score: float                 # 0 (certain) .. 1 (maximally uncertain)
    level: UncertaintyLevel
    entropy: float = 0.0         # bits
    disagreement: float = 0.0    # 1 - share of the modal answer
    variance: float = 0.0
    n_samples: int = 0
    method: str = ""


def _level(score: float) -> UncertaintyLevel:
    if score < 0.2:
        return UncertaintyLevel.LOW
    if score < 0.5:
        return UncertaintyLevel.MODERATE
    if score < 0.8:
        return UncertaintyLevel.HIGH
    return UncertaintyLevel.EXTREME


def _entropy_bits(probs: Sequence[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


class UncertaintyEstimator:
    """Epistemic-style estimates via predictive entropy and self-consistency."""

    def from_probabilities(self, probs: Sequence[float]) -> UncertaintyEstimate:
        if not probs or any(p < 0 for p in probs):
            raise ValueError("probs must be non-empty and non-negative")
        total = sum(probs)
        if total <= 0:
            raise ValueError("probs must sum to > 0")
        p = [x / total for x in probs]
        h = _entropy_bits(p)
        norm = h / math.log2(len(p)) if len(p) > 1 else 0.0
        dis = 1.0 - max(p)
        score = min(1.0, 0.6 * norm + 0.4 * dis)
        return UncertaintyEstimate(score, _level(score), h, dis, 0.0, len(p), "probabilities")

    def from_samples(self, answers: Sequence[Hashable]) -> UncertaintyEstimate:
        """Self-consistency: how much do repeated answers to the same question disagree?"""
        if not answers:
            raise ValueError("answers must be non-empty")
        counts = Counter(answers)
        n = len(answers)
        probs = [c / n for c in counts.values()]
        h = _entropy_bits(probs)
        norm = h / math.log2(n) if n > 1 else 0.0
        dis = 1.0 - max(probs)
        score = min(1.0, 0.5 * norm + 0.5 * dis)
        return UncertaintyEstimate(score, _level(score), h, dis, 0.0, n, "self_consistency")

    def from_numeric(self, values: Sequence[float], scale: Optional[float] = None) -> UncertaintyEstimate:
        """Ensemble spread: coefficient-of-variation (or spread over `scale`) squashed into [0, 1]."""
        if len(values) < 2:
            raise ValueError("need at least 2 values")
        mean, sd = statistics.fmean(values), statistics.pstdev(values)
        denom = scale if scale else (abs(mean) or 1.0)
        rel = sd / denom
        score = rel / (1.0 + rel)
        return UncertaintyEstimate(score, _level(score), 0.0, 0.0, sd ** 2, len(values), "ensemble_spread")

    def combine(self, *estimates: UncertaintyEstimate) -> UncertaintyEstimate:
        """Conservative fusion: weighted toward the worst signal."""
        if not estimates:
            raise ValueError("no estimates to combine")
        scores = [e.score for e in estimates]
        s = 0.5 * max(scores) + 0.5 * statistics.fmean(scores)
        return UncertaintyEstimate(s, _level(s), max(e.entropy for e in estimates),
                                   max(e.disagreement for e in estimates),
                                   max(e.variance for e in estimates),
                                   sum(e.n_samples for e in estimates), "combined")

    def introspect(self) -> Dict[str, Any]:
        return {"methods": ["probabilities", "self_consistency", "ensemble_spread", "combined"]}
