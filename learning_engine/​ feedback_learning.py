"""Feedback learning: turns human/system feedback into weighted training signals."""
from __future__ import annotations

import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .learning_core import (
    TOPIC_SAMPLE,
    BaseLearner,
    EventBus,
    LearnerMetrics,
    LearningMode,
    Sample,
)

logger = logging.getLogger("zentrax.learning.feedback")


class FeedbackKind(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CORRECTION = "correction"


@dataclass
class Feedback:
    target_id: str
    score: float  # -1.0 (bad) .. +1.0 (good)
    kind: FeedbackKind = FeedbackKind.EXPLICIT
    source: str = "user"
    category: Optional[str] = None
    comment: str = ""
    features: Optional[np.ndarray] = None
    predicted_label: Any = None
    corrected_label: Any = None
    action: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not np.isfinite(self.score):
            raise ValueError("Feedback score must be finite")
        self.score = float(np.clip(self.score, -1.0, 1.0))
        if self.features is not None:
            self.features = np.asarray(self.features, dtype=np.float64).ravel()


class FeedbackLearner(BaseLearner):
    """Tracks source reliability and aggregate sentiment, and derives learning samples."""

    def __init__(
        self,
        name: str = "feedback",
        reward_scale: float = 1.0,
        reliability_prior: float = 0.7,
        reliability_alpha: float = 0.1,
        score_alpha: float = 0.2,
        max_targets: int = 10_000,
        history: int = 5_000,
        bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(name, LearningMode.FEEDBACK, bus)
        self.routes = {LearningMode.FEEDBACK}
        self.reward_scale = reward_scale
        self.reliability_prior = reliability_prior
        self.reliability_alpha = reliability_alpha
        self.score_alpha = score_alpha
        self.max_targets = max_targets
        self.reliability: Dict[str, float] = {}
        self._by_target: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
        self._aggregate: Dict[str, List[float]] = {}  # key -> [ema_score, count]
        self._log: Deque[Feedback] = deque(maxlen=history)

    # -- ingestion ------------------------------------------------------- #
    def _reliability_of(self, source: str) -> float:
        return self.reliability.setdefault(source, self.reliability_prior)

    def _ingest(self, fb: Feedback) -> List[Sample]:
        self._log.append(fb)
        # Reliability: agreement with other sources on the same target.
        peers = self._by_target.setdefault(fb.target_id, {})
        self._by_target.move_to_end(fb.target_id)
        for other_source, other_score in peers.items():
            if other_source == fb.source or abs(other_score) < 1e-6 or abs(fb.score) < 1e-6:
                continue
            agree = 1.0 if np.sign(other_score) == np.sign(fb.score) else 0.0
            for src in (fb.source, other_source):
                r = self._reliability_of(src)
                self.reliability[src] = (1 - self.reliability_alpha) * r + self.reliability_alpha * agree
        peers[fb.source] = fb.score
        while len(self._by_target) > self.max_targets:
            self._by_target.popitem(last=False)

        key = fb.category or fb.target_id
        agg = self._aggregate.setdefault(key, [0.0, 0.0])
        agg[1] += 1
        alpha = max(self.score_alpha, 1.0 / agg[1])
        agg[0] += alpha * (fb.score - agg[0])

        return self._derive_samples(fb)

    def _derive_samples(self, fb: Feedback) -> List[Sample]:
        samples: List[Sample] = []
        if fb.features is None:
            return samples
        rel = self._reliability_of(fb.source)
        meta = {"feedback_target": fb.target_id, "feedback_source": fb.source, "feedback_kind": fb.kind.value}
        if fb.corrected_label is not None:
            samples.append(
                Sample(fb.features, label=fb.corrected_label, mode=LearningMode.SUPERVISED, weight=rel, metadata=meta)
            )
        elif fb.predicted_label is not None and fb.score > 0:
            samples.append(
                Sample(
                    fb.features,
                    label=fb.predicted_label,
                    mode=LearningMode.SUPERVISED,
                    weight=rel * 0.5 * fb.score,
                    metadata=meta,
                )
            )
        if fb.action is not None:
            samples.append(
                Sample(
                    fb.features,
                    reward=fb.score * self.reward_scale * rel,
                    mode=LearningMode.REINFORCEMENT,
                    metadata={**meta, "action": int(fb.action), "done": True},
                )
            )
        return [s for s in samples if s.weight > 0]

    async def submit_feedback(self, fb: Feedback) -> List[Sample]:
        """Primary API: ingest feedback and publish derived samples on the bus."""
        async with self._lock:
            samples = self._ingest(fb)
            self.updates += 1
        if self.bus is not None:
            await self.bus.publish("feedback.received", fb, self.name)
            if samples:
                await self.bus.publish(TOPIC_SAMPLE, samples, self.name)
        return samples

    # -- BaseLearner ------------------------------------------------------ #
    async def partial_fit(self, samples: Sequence[Sample]) -> LearnerMetrics:
        items: List[Feedback] = []
        for s in samples:
            raw = s.metadata.get("feedback")
            if isinstance(raw, Feedback):
                items.append(raw)
            elif isinstance(raw, Mapping):
                try:
                    items.append(Feedback(**raw))
                except (TypeError, ValueError):
                    logger.warning("Dropping malformed feedback payload")
        if not items:
            return self._metrics(0)
        derived: List[Sample] = []
        async with self._lock:
            for fb in items:
                derived.extend(self._ingest(fb))
        if derived and self.bus is not None:
            await self.bus.publish(TOPIC_SAMPLE, derived, self.name)
        mean_score = float(np.mean([fb.score for fb in items]))
        return self._metrics(
            len(items),
            reward=mean_score,
            extra={"derived_samples": len(derived), "sources": len(self.reliability)},
        )

    async def predict(self, features: Any) -> Dict[str, Any]:
        """``features`` is a target/category key; returns the aggregate expected score."""
        key = str(features)
        async with self._lock:
            agg = self._aggregate.get(key)
        if agg is None:
            return {"score": 0.0, "count": 0, "confidence": 0.0}
        return {"score": agg[0], "count": int(agg[1]), "confidence": float(1.0 - 1.0 / (1.0 + agg[1]))}

    def get_hparams(self) -> Dict[str, float]:
        return {"reward_scale": self.reward_scale, "score_alpha": self.score_alpha}

    def set_hparams(self, params: Mapping[str, float]) -> None:
        if "reward_scale" in params:
            self.reward_scale = float(max(params["reward_scale"], 0.0))
        if "score_alpha" in params:
            self.score_alpha = float(np.clip(params["score_alpha"], 0.01, 1.0))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "reliability": dict(self.reliability),
            "aggregate": {k: list(v) for k, v in self._aggregate.items()},
            "hparams": self.get_hparams(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.reliability = {str(k): float(v) for k, v in state.get("reliability", {}).items()}
        self._aggregate = {str(k): [float(v[0]), float(v[1])] for k, v in state.get("aggregate", {}).items()}
        self.set_hparams(state.get("hparams", {}))
