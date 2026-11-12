"""Continual learning: reservoir replay, online EWC consolidation and forgetting reports."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .adaptation_engine import PageHinkley
from .learning_core import (
    BaseLearner,
    EventBus,
    LearnerMetrics,
    LearningMode,
    Sample,
)
from .supervised_learning import SupervisedLearner

logger = logging.getLogger("zentrax.learning.continual")

TOPIC_TASK_BOUNDARY = "continual.task_boundary"


class ReservoirBuffer:
    """Uniform reservoir sample over the whole stream."""

    def __init__(self, capacity: int = 2_000, seed: int = 0) -> None:
        self.capacity = capacity
        self._items: List[Sample] = []
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def add(self, sample: Sample) -> None:
        self._seen += 1
        if len(self._items) < self.capacity:
            self._items.append(sample)
            return
        j = int(self._rng.integers(0, self._seen))
        if j < self.capacity:
            self._items[j] = sample

    def sample(self, k: int) -> List[Sample]:
        k = min(k, len(self._items))
        if k <= 0:
            return []
        idx = self._rng.choice(len(self._items), size=k, replace=False)
        return [self._items[int(i)] for i in idx]

    def items(self) -> List[Sample]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


class ContinualLearner(BaseLearner):
    """Wraps a :class:`SupervisedLearner` to mitigate catastrophic forgetting."""

    def __init__(
        self,
        base: SupervisedLearner,
        name: str = "continual",
        buffer_size: int = 2_000,
        replay_ratio: float = 0.5,
        ewc_lambda: float = 50.0,
        min_buffer_for_consolidation: int = 50,
        novelty_task_fraction: float = 0.5,
        bus: Optional[EventBus] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(name, LearningMode.CONTINUAL, bus)
        self.routes = {LearningMode.CONTINUAL, LearningMode.SUPERVISED, None}
        self.base = base
        self.base.routes = set()  # the wrapper owns routing for supervised samples
        self.buffer = ReservoirBuffer(buffer_size, seed)
        self.replay_ratio = replay_ratio
        self.ewc_lambda = ewc_lambda
        self.min_buffer = min_buffer_for_consolidation
        self.novelty_task_fraction = novelty_task_fraction
        self.detector = PageHinkley(delta=0.01, threshold=1.0, min_samples=5)
        self.task_id = 0
        self._task_end_accuracy: Dict[int, float] = {}

    # -- task management -------------------------------------------------- #
    async def mark_task_boundary(self, reason: str = "manual") -> None:
        """Consolidate knowledge of the finished task and start a new one."""
        items = self.buffer.items()
        if len(items) >= self.min_buffer:
            await self.base.consolidate(items, self.ewc_lambda)
            current = [s for s in items if s.metadata.get("task_id") == self.task_id]
            if current:
                metrics = await self.base.evaluate(current)
                if metrics.accuracy is not None:
                    self._task_end_accuracy[self.task_id] = metrics.accuracy
        self.task_id += 1
        self.detector.reset()
        if self.bus is not None:
            await self.bus.publish(TOPIC_TASK_BOUNDARY, {"task_id": self.task_id, "reason": reason}, self.name)

    async def _detect_boundary(self, new: Sequence[Sample]) -> Optional[str]:
        known = [self.base.is_known(s.label) for s in new]
        if self.base.classes and known and (1.0 - float(np.mean(known))) >= self.novelty_task_fraction:
            return "novel_classes"
        if not self.base.classes:
            return None
        pre = await self.base.evaluate(new)
        if pre.loss is not None and self.detector.update(pre.loss):
            return "loss_drift"
        return None

    # -- BaseLearner ------------------------------------------------------ #
    async def partial_fit(self, samples: Sequence[Sample]) -> LearnerMetrics:
        new = [s for s in samples if s.label is not None]
        if not new:
            return self._metrics(0)
        async with self._lock:
            reason = await self._detect_boundary(new)
            if reason:
                await self.mark_task_boundary(reason)
            replay = self.buffer.sample(int(len(new) * self.replay_ratio))
            metrics = await self.base.partial_fit(list(new) + replay)
            for s in new:
                s.metadata.setdefault("task_id", self.task_id)
                self.buffer.add(s)
        return self._metrics(
            len(new),
            loss=metrics.loss,
            accuracy=metrics.accuracy,
            extra={
                "task_id": self.task_id,
                "replayed": len(replay),
                "buffer": len(self.buffer),
                "boundary": reason,
            },
        )

    async def predict(self, features: Any) -> Any:
        return await self.base.predict(features)

    async def evaluate(self, samples: Sequence[Sample]) -> LearnerMetrics:
        m = await self.base.evaluate(samples)
        m.learner, m.mode = self.name, self.mode
        return m

    async def forgetting_report(self) -> Dict[str, Any]:
        """Current accuracy per past task versus accuracy when that task was consolidated."""
        by_task: Dict[int, List[Sample]] = defaultdict(list)
        for s in self.buffer.items():
            by_task[int(s.metadata.get("task_id", 0))].append(s)
        report: Dict[str, Any] = {}
        for tid, items in sorted(by_task.items()):
            m = await self.base.evaluate(items)
            then = self._task_end_accuracy.get(tid)
            report[str(tid)] = {
                "samples": len(items),
                "accuracy_now": m.accuracy,
                "accuracy_at_consolidation": then,
                "forgetting": None if then is None or m.accuracy is None else max(0.0, then - m.accuracy),
            }
        return report

    # -- hyper-parameters ------------------------------------------------- #
    def get_hparams(self) -> Dict[str, float]:
        return {**self.base.get_hparams(), "replay_ratio": self.replay_ratio, "ewc_lambda": self.ewc_lambda}

    def set_hparams(self, params: Mapping[str, float]) -> None:
        own = {"replay_ratio", "ewc_lambda"}
        if "replay_ratio" in params:
            self.replay_ratio = float(np.clip(params["replay_ratio"], 0.0, 4.0))
        if "ewc_lambda" in params:
            self.ewc_lambda = float(max(params["ewc_lambda"], 0.0))
            self.base.set_hparams({"ewc_lambda": self.ewc_lambda})
        self.base.set_hparams({k: v for k, v in params.items() if k not in own})

    # -- persistence ------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        items = self.buffer.items()
        return {
            "base": self.base.state_dict(),
            "task_id": self.task_id,
            "task_end_accuracy": {str(k): v for k, v in self._task_end_accuracy.items()},
            "hparams": {"replay_ratio": self.replay_ratio, "ewc_lambda": self.ewc_lambda},
            "buffer": {
                "features": np.vstack([s.features for s in items]) if items else np.zeros((0, 0)),
                "labels": [s.label for s in items],
                "tasks": [int(s.metadata.get("task_id", 0)) for s in items],
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.base.load_state_dict(state["base"])
        self.task_id = int(state.get("task_id", 0))
        self._task_end_accuracy = {int(k): float(v) for k, v in state.get("task_end_accuracy", {}).items()}
        hp = state.get("hparams", {})
        self.replay_ratio = float(hp.get("replay_ratio", self.replay_ratio))
        self.ewc_lambda = float(hp.get("ewc_lambda", self.ewc_lambda))
        buf = state.get("buffer", {})
        feats = np.asarray(buf.get("features", np.zeros((0, 0))))
        self.buffer = ReservoirBuffer(self.buffer.capacity)
        for row, label, tid in zip(feats, buf.get("labels", []), buf.get("tasks", [])):
            self.buffer.add(Sample(row, label=label, mode=LearningMode.SUPERVISED, metadata={"task_id": int(tid)}))
