"""Trainer: async batch training with LR scheduling, early stopping and checkpointing."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..learning_core import (
    BaseLearner,
    EventBus,
    LearnerMetrics,
    LearningMode,
    atomic_write_json,
)
from .dataset_manager import Dataset, DatasetManager
from .model_optimizer import LRScheduler

logger = logging.getLogger("zentrax.learning.trainer")

TOPIC_EPOCH = "training.epoch"
TOPIC_COMPLETED = "training.completed"


@dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 64
    shuffle: bool = True
    seed: int = 0
    patience: int = 3
    min_delta: float = 1e-4
    restore_best: bool = True
    checkpoint_dir: Optional[str] = None
    scheduler: Optional[LRScheduler] = None
    sample_mode: LearningMode = LearningMode.SUPERVISED


@dataclass
class EpochRecord:
    epoch: int
    train_loss: Optional[float]
    train_accuracy: Optional[float]
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    lr: Optional[float]
    duration: float


@dataclass
class TrainingReport:
    learner: str
    history: List[EpochRecord] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_metric: Optional[float] = None
    stopped_early: bool = False
    cancelled: bool = False
    duration: float = 0.0
    checkpoint_path: Optional[str] = None


class TrainerCallback:
    """Override any hook. ``on_epoch_end`` may return True to request a stop."""

    async def on_train_begin(self, trainer: "Trainer", learner: BaseLearner) -> None:
        return None

    async def on_epoch_end(self, trainer: "Trainer", learner: BaseLearner, record: EpochRecord) -> Optional[bool]:
        return None

    async def on_train_end(self, trainer: "Trainer", learner: BaseLearner, report: TrainingReport) -> None:
        return None


def _weighted(values: Sequence[Optional[float]], counts: Sequence[int]) -> Optional[float]:
    pairs = [(v, c) for v, c in zip(values, counts) if v is not None and c > 0]
    if not pairs:
        return None
    return float(np.average([p[0] for p in pairs], weights=[p[1] for p in pairs]))


class Trainer:
    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        bus: Optional[EventBus] = None,
        callbacks: Optional[Sequence[TrainerCallback]] = None,
        datasets: Optional[DatasetManager] = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.bus = bus
        self.callbacks = list(callbacks or [])
        self.datasets = datasets or DatasetManager()
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    @staticmethod
    def _monitor_value(loss: Optional[float], accuracy: Optional[float]) -> Optional[float]:
        if loss is not None:
            return loss
        return None if accuracy is None else 1.0 - accuracy

    async def _evaluate(self, learner: BaseLearner, ds: Dataset) -> Optional[LearnerMetrics]:
        try:
            m = await learner.evaluate(ds.to_samples(self.config.sample_mode))
        except NotImplementedError:
            return None
        return m if m.n_samples > 0 else None

    async def fit(self, learner: BaseLearner, train: Dataset, val: Optional[Dataset] = None) -> TrainingReport:
        cfg = self.config
        report = TrainingReport(learner=learner.name)
        self._stop_requested = False
        started = time.monotonic()
        best_snapshot: Optional[Dict[str, Any]] = None
        stale = 0
        last_monitor: Optional[float] = None

        for cb in self.callbacks:
            await cb.on_train_begin(self, learner)

        try:
            for epoch in range(cfg.epochs):
                if self._stop_requested:
                    report.cancelled = True
                    break
                epoch_start = time.monotonic()

                lr: Optional[float] = None
                if cfg.scheduler is not None and "lr" in learner.get_hparams():
                    lr = cfg.scheduler.value(epoch, last_monitor)
                    learner.set_hparams({"lr": lr})

                losses: List[Optional[float]] = []
                accs: List[Optional[float]] = []
                counts: List[int] = []
                async for batch in self.datasets.batches(
                    train, cfg.batch_size, cfg.shuffle, cfg.seed, epoch
                ):
                    metrics = await learner.partial_fit(batch.to_samples(cfg.sample_mode))
                    losses.append(metrics.loss)
                    accs.append(metrics.accuracy)
                    counts.append(metrics.n_samples)
                    if self._stop_requested:
                        break

                train_loss, train_acc = _weighted(losses, counts), _weighted(accs, counts)
                val_metrics = await self._evaluate(learner, val) if val is not None else None
                val_loss = val_metrics.loss if val_metrics else None
                val_acc = val_metrics.accuracy if val_metrics else None

                record = EpochRecord(
                    epoch, train_loss, train_acc, val_loss, val_acc, lr, time.monotonic() - epoch_start
                )
                report.history.append(record)

                monitor = (
                    self._monitor_value(val_loss, val_acc)
                    if val_metrics is not None
                    else self._monitor_value(train_loss, train_acc)
                )
                last_monitor = monitor

                if monitor is not None and (
                    report.best_metric is None or monitor < report.best_metric - cfg.min_delta
                ):
                    report.best_metric, report.best_epoch, stale = monitor, epoch, 0
                    if cfg.restore_best or cfg.checkpoint_dir:
                        best_snapshot = await learner.snapshot()
                    if cfg.checkpoint_dir and best_snapshot is not None:
                        path = Path(cfg.checkpoint_dir) / f"{learner.name}_best.json"
                        await asyncio.to_thread(
                            atomic_write_json, path, {"epoch": epoch, "metric": monitor, "state": best_snapshot}
                        )
                        report.checkpoint_path = str(path)
                elif monitor is not None:
                    stale += 1

                if self.bus is not None:
                    await self.bus.publish(TOPIC_EPOCH, {"learner": learner.name, "record": record}, "trainer")

                stop_requested = False
                for cb in self.callbacks:
                    if await cb.on_epoch_end(self, learner, record):
                        stop_requested = True
                if stop_requested:
                    report.cancelled = True
                    break
                if monitor is not None and stale >= cfg.patience:
                    report.stopped_early = True
                    break
        except asyncio.CancelledError:
            report.cancelled = True
            raise
        finally:
            if cfg.restore_best and best_snapshot is not None and report.best_epoch != (
                report.history[-1].epoch if report.history else None
            ):
                await learner.restore(best_snapshot)
            report.duration = time.monotonic() - started
            for cb in self.callbacks:
                await cb.on_train_end(self, learner, report)
            if self.bus is not None:
                await self.bus.publish(TOPIC_COMPLETED, report, "trainer")
        return report
