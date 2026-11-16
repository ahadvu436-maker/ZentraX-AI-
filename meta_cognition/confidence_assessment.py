"""Confidence fusion and calibration (binned reliability mapping, ECE, Brier)."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ConfidenceSignals:
    model_confidence: Optional[float] = None   # self-reported, 0..1
    consistency: Optional[float] = None        # 1 - uncertainty score
    reasoning_soundness: Optional[float] = None
    recent_error_rate: float = 0.0             # 0..1, penalizes confidence

    def __post_init__(self) -> None:
        for n in ("model_confidence", "consistency", "reasoning_soundness"):
            v = getattr(self, n)
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"{n} must be in [0, 1]")
        if not 0.0 <= self.recent_error_rate <= 1.0:
            raise ValueError("recent_error_rate must be in [0, 1]")


@dataclass(frozen=True)
class ConfidenceReport:
    raw: float
    calibrated: float
    label: str
    components: Dict[str, float]
    calibrated_with_samples: int


class Calibrator:
    """Histogram-binning calibration with Laplace smoothing; identity until enough data."""

    def __init__(self, bins: int = 10, max_history: int = 5000, min_samples: int = 30) -> None:
        self.bins, self.min_samples = bins, min_samples
        self._hist: Deque[Tuple[float, bool]] = deque(maxlen=max_history)
        self._lock = threading.Lock()

    def add(self, confidence: float, correct: bool) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        with self._lock:
            self._hist.append((confidence, bool(correct)))

    def _bin(self, c: float) -> int:
        return min(self.bins - 1, int(c * self.bins))

    def _tables(self) -> Tuple[List[int], List[int]]:
        n, k = [0] * self.bins, [0] * self.bins
        for c, ok in self._hist:
            b = self._bin(c)
            n[b] += 1
            k[b] += ok
        return n, k

    def calibrate(self, c: float) -> float:
        with self._lock:
            if len(self._hist) < self.min_samples:
                return c
            n, k = self._tables()
        b = self._bin(c)
        mid = (b + 0.5) / self.bins
        return (k[b] + 2 * mid) / (n[b] + 2)  # prior of strength 2 at bin midpoint

    def ece(self) -> Optional[float]:
        with self._lock:
            total = len(self._hist)
            if total == 0:
                return None
            n, k = self._tables()
            sums = [0.0] * self.bins
            for c, _ in self._hist:
                sums[self._bin(c)] += c
        return sum(abs(k[b] / n[b] - sums[b] / n[b]) * n[b] for b in range(self.bins) if n[b]) / total

    def brier(self) -> Optional[float]:
        with self._lock:
            if not self._hist:
                return None
            return sum((c - ok) ** 2 for c, ok in self._hist) / len(self._hist)

    def __len__(self) -> int:
        return len(self._hist)


class ConfidenceAssessor:
    DEFAULT_WEIGHTS = {"model_confidence": 0.3, "consistency": 0.4, "reasoning_soundness": 0.3}

    def __init__(self, calibrator: Optional[Calibrator] = None,
                 weights: Optional[Dict[str, float]] = None) -> None:
        self.calibrator = calibrator or Calibrator()
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)

    @staticmethod
    def label(c: float) -> str:
        return ("very_low" if c < 0.2 else "low" if c < 0.4 else "medium" if c < 0.6
                else "high" if c < 0.85 else "very_high")

    def assess(self, s: ConfidenceSignals) -> ConfidenceReport:
        comps = {k: getattr(s, k) for k in self.weights if getattr(s, k) is not None}
        if not comps:
            raise ValueError("at least one confidence signal is required")
        wsum = sum(self.weights[k] for k in comps)
        base = sum(self.weights[k] * v for k, v in comps.items()) / wsum
        # A weak signal drags the total below the mean: blend in the minimum.
        raw = (0.7 * base + 0.3 * min(comps.values())) * (1.0 - 0.5 * s.recent_error_rate)
        raw = max(0.0, min(1.0, raw))
        cal = self.calibrator.calibrate(raw)
        return ConfidenceReport(raw, cal, self.label(cal), comps, len(self.calibrator))

    def record_outcome(self, predicted_confidence: float, correct: bool) -> None:
        self.calibrator.add(predicted_confidence, correct)

    def introspect(self) -> Dict[str, Any]:
        return {"weights": self.weights, "calibration_samples": len(self.calibrator),
                "ece": self.calibrator.ece(), "brier": self.calibrator.brier()}
