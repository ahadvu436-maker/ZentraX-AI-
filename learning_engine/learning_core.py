"""ZentraX-AI learning core: shared types, event bus, base learner and orchestrator."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union,
)

import numpy as np

logger = logging.getLogger("zentrax.learning")

TOPIC_METRICS = "learner.metrics"
TOPIC_SAMPLE = "sample.submit"
TOPIC_STATUS = "learner.status"


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def to_python(value: Any) -> Any:
    """Convert numpy scalars to plain Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def encode_state(obj: Any) -> Any:
    """Make a state structure JSON-serialisable (numpy arrays are tagged)."""
    if isinstance(obj, np.ndarray):
        return {"__nd__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): encode_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [encode_state(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def decode_state(obj: Any) -> Any:
    """Inverse of :func:`encode_state`."""
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return np.asarray(obj["__nd__"], dtype=obj.get("dtype", "float64"))
        return {k: decode_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_state(v) for v in obj]
    return obj


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


class RunningStats:
    """Streaming per-feature mean/variance (Welford, batch-parallel form)."""

    def __init__(self, dim: int = 0) -> None:
        self.count = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)

    def update(self, X: np.ndarray) -> None:
        n_b = X.shape[0]
        if n_b == 0:
            return
        b_mean = X.mean(axis=0)
        b_m2 = ((X - b_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = n_b, b_mean, b_m2
            return
        delta = b_mean - self.mean
        total = self.count + n_b
        self.mean = self.mean + delta * n_b / total
        self.m2 = self.m2 + b_m2 + delta ** 2 * self.count * n_b / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        std = np.sqrt(self.m2 / max(self.count, 1))
        return np.where(std < 1e-8, 1.0, std)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        if self.count == 0:
            return X
        return np.clip((X - self.mean) / self.std, -10.0, 10.0)

    def state(self) -> Dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load(self, state: Mapping[str, Any]) -> None:
        self.count = int(state["count"])
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.m2 = np.asarray(state["m2"], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
class LearningMode(str, Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    FEEDBACK = "feedback"
    CONTINUAL = "continual"
    ADAPTATION = "adaptation"
    SKILL = "skill"


class LearnerStatus(str, Enum):
    IDLE = "idle"
    TRAINING = "training"
    DEGRADED = "degraded"
    DISABLED = "disabled"


@dataclass
class Sample:
    """A single learning experience."""

    features: np.ndarray
    label: Any = None
    reward: Optional[float] = None
    mode: Optional[LearningMode] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float64).ravel()
        if not np.all(np.isfinite(self.features)):
            raise ValueError("Sample features must be finite")
        if self.reward is not None and not np.isfinite(self.reward):
            raise ValueError("Sample reward must be finite")
        self.label = to_python(self.label)
        self.weight = float(max(self.weight, 0.0))


@dataclass
class LearnerMetrics:
    learner: str
    mode: LearningMode
    n_samples: int = 0
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    reward: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learner": self.learner,
            "mode": self.mode.value,
            "n_samples": self.n_samples,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "reward": self.reward,
            "extra": self.extra,
            "timestamp": self.timestamp,
        }


@dataclass
class LearningEvent:
    topic: str
    payload: Any = None
    source: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class LearningConfig:
    queue_size: int = 10_000
    batch_size: int = 64
    flush_interval: float = 0.5
    train_timeout: Optional[float] = 120.0
    max_failures: int = 3
    cooldown_seconds: float = 30.0
    history_size: int = 1_000
    checkpoint_dir: Optional[str] = None
    autosave_interval: Optional[float] = None


# --------------------------------------------------------------------------- #
# Event bus
# --------------------------------------------------------------------------- #
Handler = Callable[[LearningEvent], Union[None, Awaitable[None]]]


class EventBus:
    """Minimal async pub/sub with error isolation. Topic "*" receives everything."""

    def __init__(self) -> None:
        self._subs: Dict[str, List[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        if handler not in self._subs[topic]:
            self._subs[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        if handler in self._subs.get(topic, []):
            self._subs[topic].remove(handler)

    async def publish(self, topic: str, payload: Any = None, source: str = "") -> LearningEvent:
        event = LearningEvent(topic=topic, payload=payload, source=source)
        for handler in list(self._subs.get(topic, [])) + list(self._subs.get("*", [])):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - handlers must never break the publisher
                logger.exception("Event handler failed for topic %s", topic)
        return event


# --------------------------------------------------------------------------- #
# Base learner
# --------------------------------------------------------------------------- #
class BaseLearner(ABC):
    """Contract every learner implements so the core can orchestrate it."""

    def __init__(self, name: str, mode: LearningMode, bus: Optional[EventBus] = None) -> None:
        self.name = name
        self.mode = mode
        self.bus = bus
        self.status = LearnerStatus.IDLE
        self.updates = 0
        # Which sample modes the core routes here (None = unlabelled/unmoded samples).
        self.routes: Set[Optional[LearningMode]] = {mode, None}
        self._lock = asyncio.Lock()

    def attach_bus(self, bus: EventBus) -> None:
        self.bus = bus

    def accepts(self, sample: Sample) -> bool:
        return sample.mode in self.routes

    @abstractmethod
    async def partial_fit(self, samples: Sequence[Sample]) -> LearnerMetrics:
        """Learn from a batch. Must return metrics with n_samples == 0 if nothing was usable."""

    @abstractmethod
    async def predict(self, features: Any) -> Any:
        """Inference entry point."""

    async def evaluate(self, samples: Sequence[Sample]) -> LearnerMetrics:
        raise NotImplementedError(f"{type(self).__name__} does not support evaluation")

    def get_hparams(self) -> Dict[str, float]:
        return {}

    def set_hparams(self, params: Mapping[str, float]) -> None:
        return None

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return encode_state(self.state_dict())

    async def restore(self, state: Mapping[str, Any]) -> None:
        async with self._lock:
            self.load_state_dict(decode_state(dict(state)))

    def _metrics(self, n: int, **kwargs: Any) -> LearnerMetrics:
        return LearnerMetrics(learner=self.name, mode=self.mode, n_samples=n, **kwargs)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class LearningCore:
    """Routes samples to learners and runs the adaptive learning loop."""

    def __init__(self, config: Optional[LearningConfig] = None, bus: Optional[EventBus] = None) -> None:
        self.config = config or LearningConfig()
        self.bus = bus or EventBus()
        self._learners: Dict[str, BaseLearner] = {}
        self._pending: Dict[str, List[Sample]] = {}
        self._failures: Dict[str, int] = defaultdict(int)
        self._cooldown_until: Dict[str, float] = {}
        self._queue: Optional["asyncio.Queue[Sample]"] = None
        self._task: Optional["asyncio.Task[None]"] = None
        self._stop = asyncio.Event()
        self._history: Deque[LearnerMetrics] = deque(maxlen=self.config.history_size)
        self._dropped = 0
        self._last_save = time.monotonic()
        self.bus.subscribe(TOPIC_SAMPLE, self._on_sample_event)

    # -- registry -------------------------------------------------------- #
    def register(self, learner: BaseLearner) -> BaseLearner:
        if learner.name in self._learners:
            raise ValueError(f"Learner '{learner.name}' already registered")
        learner.attach_bus(self.bus)
        self._learners[learner.name] = learner
        self._pending[learner.name] = []
        return learner

    def unregister(self, name: str) -> None:
        self._learners.pop(name, None)
        self._pending.pop(name, None)

    def get(self, name: str) -> Optional[BaseLearner]:
        return self._learners.get(name)

    @property
    def learners(self) -> Dict[str, BaseLearner]:
        return dict(self._learners)

    # -- ingestion ------------------------------------------------------- #
    def _ensure_queue(self) -> "asyncio.Queue[Sample]":
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.config.queue_size)
        return self._queue

    async def submit(self, sample: Sample) -> None:
        await self._ensure_queue().put(sample)

    def submit_nowait(self, sample: Sample) -> bool:
        try:
            self._ensure_queue().put_nowait(sample)
            return True
        except asyncio.QueueFull:
            self._dropped += 1
            return False

    async def _on_sample_event(self, event: LearningEvent) -> None:
        payload = event.payload
        items: Iterable[Any] = [payload] if isinstance(payload, Sample) else (payload or [])
        for item in items:
            if isinstance(item, Sample):
                self.submit_nowait(item)

    def _route(self, sample: Sample) -> None:
        now = time.time()
        for name, learner in self._learners.items():
            if not learner.accepts(sample):
                continue
            if self._cooldown_until.get(name, 0.0) > now:
                continue
            self._pending[name].append(sample)

    # -- training -------------------------------------------------------- #
    async def learn(self, samples: Iterable[Sample]) -> Dict[str, LearnerMetrics]:
        """Synchronously route and train on samples (bypasses the queue)."""
        for sample in samples:
            self._route(sample)
        return await self.flush()

    async def flush(self) -> Dict[str, LearnerMetrics]:
        work = {n: b for n, b in self._pending.items() if b}
        for name in work:
            self._pending[name] = []
        results = await asyncio.gather(
            *(self._train(self._learners[n], b) for n, b in work.items() if n in self._learners)
        )
        return {m.learner: m for m in results if m is not None}

    async def _train(self, learner: BaseLearner, batch: List[Sample]) -> Optional[LearnerMetrics]:
        learner.status = LearnerStatus.TRAINING
        try:
            coro = learner.partial_fit(batch)
            metrics = await (
                asyncio.wait_for(coro, self.config.train_timeout) if self.config.train_timeout else coro
            )
        except Exception:  # noqa: BLE001
            self._failures[learner.name] += 1
            logger.exception("Learner %s failed (%d consecutive)", learner.name, self._failures[learner.name])
            if self._failures[learner.name] >= self.config.max_failures:
                learner.status = LearnerStatus.DEGRADED
                self._cooldown_until[learner.name] = time.time() + self.config.cooldown_seconds
                self._failures[learner.name] = 0
                await self.bus.publish(TOPIC_STATUS, {"learner": learner.name, "status": "degraded"}, "core")
            else:
                learner.status = LearnerStatus.IDLE
            return None
        self._failures[learner.name] = 0
        learner.status = LearnerStatus.IDLE
        if metrics.n_samples > 0:
            learner.updates += 1
            self._history.append(metrics)
            await self.bus.publish(TOPIC_METRICS, metrics, learner.name)
        return metrics

    async def predict(self, learner_name: str, features: Any) -> Any:
        learner = self._learners.get(learner_name)
        if learner is None:
            raise KeyError(f"Unknown learner '{learner_name}'")
        return await learner.predict(features)

    # -- loop ------------------------------------------------------------ #
    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._ensure_queue()
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="zentrax-learning-loop")

    async def stop(self, drain: bool = True) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None
        if drain:
            queue = self._ensure_queue()
            while not queue.empty():
                self._route(queue.get_nowait())
            await self.flush()

    async def _run(self) -> None:
        queue = self._ensure_queue()
        while not self._stop.is_set():
            try:
                first = await asyncio.wait_for(queue.get(), timeout=self.config.flush_interval)
                self._route(first)
                for _ in range(self.config.batch_size - 1):
                    try:
                        self._route(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                pass
            try:
                await self.flush()
                await self._maybe_autosave()
            except Exception:  # noqa: BLE001
                logger.exception("Learning loop iteration failed")

    async def _maybe_autosave(self) -> None:
        interval = self.config.autosave_interval
        if interval and self.config.checkpoint_dir and time.monotonic() - self._last_save >= interval:
            self._last_save = time.monotonic()
            await self.save()

    # -- persistence ----------------------------------------------------- #
    async def save(self, directory: Optional[str] = None) -> Path:
        target = directory or self.config.checkpoint_dir
        if not target:
            raise ValueError("No checkpoint directory configured")
        root = Path(target)
        payloads: Dict[str, Dict[str, Any]] = {}
        for name, learner in self._learners.items():
            payloads[name] = {
                "learner": name,
                "class": type(learner).__name__,
                "updates": learner.updates,
                "saved_at": time.time(),
                "state": await learner.snapshot(),
            }

        def _write() -> None:
            for name, payload in payloads.items():
                atomic_write_json(root / f"{name}.json", payload)

        await asyncio.to_thread(_write)
        return root

    async def load(self, directory: Optional[str] = None) -> List[str]:
        target = directory or self.config.checkpoint_dir
        if not target:
            return []
        root = Path(target)
        loaded: List[str] = []
        for name, learner in self._learners.items():
            path = root / f"{name}.json"
            if not path.exists():
                continue
            try:
                payload = await asyncio.to_thread(lambda p=path: json.loads(p.read_text(encoding="utf-8")))
                await learner.restore(payload["state"])
                learner.updates = int(payload.get("updates", 0))
                loaded.append(name)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to restore learner %s from %s", name, path)
        return loaded

    # -- introspection --------------------------------------------------- #
    def history(self, learner: Optional[str] = None, limit: int = 100) -> List[LearnerMetrics]:
        items = [m for m in self._history if learner is None or m.learner == learner]
        return items[-limit:]

    def health(self) -> Dict[str, Any]:
        return {
            "running": bool(self._task and not self._task.done()),
            "queue_size": self._queue.qsize() if self._queue else 0,
            "dropped_samples": self._dropped,
            "learners": {
                n: {"status": l.status.value, "updates": l.updates} for n, l in self._learners.items()
            },
        }
