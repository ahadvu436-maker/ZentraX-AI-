"""Model optimisation: gradient optimisers, LR schedules and async hyper-parameter search."""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger("zentrax.learning.optimizer")

Params = Dict[str, np.ndarray]


# --------------------------------------------------------------------------- #
# Gradient optimisers
# --------------------------------------------------------------------------- #
def clip_by_global_norm(grads: Params, max_norm: float) -> float:
    """Scale gradients in place so their global L2 norm is <= max_norm. Returns the original norm."""
    norm = float(math.sqrt(sum(float((g ** 2).sum()) for g in grads.values())))
    if norm > max_norm > 0:
        scale = max_norm / (norm + 1e-12)
        for g in grads.values():
            g *= scale
    return norm


class Optimizer(ABC):
    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.t = 0

    @abstractmethod
    def step(self, params: Params, grads: Params) -> None:
        """Update ``params`` in place."""


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, nesterov: bool = True, weight_decay: float = 0.0) -> None:
        super().__init__(lr)
        self.momentum, self.nesterov, self.weight_decay = momentum, nesterov, weight_decay
        self._v: Params = {}

    def step(self, params: Params, grads: Params) -> None:
        self.t += 1
        for k, p in params.items():
            g = grads[k] + self.weight_decay * p
            v = self._v.get(k, np.zeros_like(p))
            v = self.momentum * v + g
            self._v[k] = v
            p -= self.lr * ((g + self.momentum * v) if self.nesterov else v)


class Adam(Optimizer):
    """Adam with decoupled weight decay (AdamW)."""

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr)
        self.b1, self.b2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self._m: Params = {}
        self._v: Params = {}

    def step(self, params: Params, grads: Params) -> None:
        self.t += 1
        for k, p in params.items():
            g = grads[k]
            m = self.b1 * self._m.get(k, np.zeros_like(p)) + (1 - self.b1) * g
            v = self.b2 * self._v.get(k, np.zeros_like(p)) + (1 - self.b2) * g * g
            self._m[k], self._v[k] = m, v
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p)


# --------------------------------------------------------------------------- #
# Learning-rate schedules
# --------------------------------------------------------------------------- #
class LRScheduler(ABC):
    def __init__(self, base_lr: float) -> None:
        self.base_lr = base_lr

    @abstractmethod
    def value(self, step: int, metric: Optional[float] = None) -> float:
        ...


class ConstantLR(LRScheduler):
    def value(self, step: int, metric: Optional[float] = None) -> float:
        return self.base_lr


class StepLR(LRScheduler):
    def __init__(self, base_lr: float, step_size: int = 5, gamma: float = 0.5) -> None:
        super().__init__(base_lr)
        self.step_size, self.gamma = step_size, gamma

    def value(self, step: int, metric: Optional[float] = None) -> float:
        return self.base_lr * (self.gamma ** (step // max(1, self.step_size)))


class CosineAnnealingLR(LRScheduler):
    def __init__(self, base_lr: float, total_steps: int, min_lr: float = 0.0, warmup_steps: int = 0) -> None:
        super().__init__(base_lr)
        self.total_steps, self.min_lr, self.warmup_steps = max(1, total_steps), min_lr, warmup_steps

    def value(self, step: int, metric: Optional[float] = None) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / (self.warmup_steps + 1)
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


class ReduceOnPlateau(LRScheduler):
    """Stateful: cuts the LR when the monitored metric (lower is better) stops improving."""

    def __init__(self, base_lr: float, factor: float = 0.5, patience: int = 2, min_lr: float = 1e-6, min_delta: float = 1e-4) -> None:
        super().__init__(base_lr)
        self.factor, self.patience, self.min_lr, self.min_delta = factor, patience, min_lr, min_delta
        self.current = base_lr
        self.best = float("inf")
        self.bad = 0

    def value(self, step: int, metric: Optional[float] = None) -> float:
        if metric is not None:
            if metric < self.best - self.min_delta:
                self.best, self.bad = metric, 0
            else:
                self.bad += 1
                if self.bad > self.patience:
                    self.current = max(self.min_lr, self.current * self.factor)
                    self.bad = 0
        return self.current


# --------------------------------------------------------------------------- #
# Hyper-parameter search
# --------------------------------------------------------------------------- #
@dataclass
class ParamSpec:
    kind: str  # uniform | log_uniform | int | choice
    low: float = 0.0
    high: float = 1.0
    choices: Sequence[Any] = ()

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "uniform":
            return float(rng.uniform(self.low, self.high))
        if self.kind == "log_uniform":
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        if self.kind == "int":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.kind == "choice":
            return self.choices[int(rng.integers(len(self.choices)))]
        raise ValueError(f"Unknown ParamSpec kind '{self.kind}'")

    def perturb(self, value: Any, rng: np.random.Generator, scale: float = 0.25) -> Any:
        if self.kind == "choice":
            return value if rng.random() > 0.3 else self.sample(rng)
        if self.kind == "log_uniform":
            span = math.log(self.high) - math.log(self.low)
            new = math.exp(math.log(value) + rng.normal(0.0, scale * span))
            return float(min(max(new, self.low), self.high))
        span = self.high - self.low
        new = value + rng.normal(0.0, scale * span)
        new = min(max(new, self.low), self.high)
        return int(round(new)) if self.kind == "int" else float(new)


class SearchSpace:
    def __init__(self, specs: Mapping[str, ParamSpec]) -> None:
        self.specs = dict(specs)

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        return {k: s.sample(rng) for k, s in self.specs.items()}

    def perturb(self, cfg: Mapping[str, Any], rng: np.random.Generator, scale: float = 0.25) -> Dict[str, Any]:
        return {k: s.perturb(cfg[k], rng, scale) for k, s in self.specs.items()}


@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, Any]
    score: float
    duration: float
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


Objective = Callable[[Dict[str, Any]], Awaitable[float]]


class ModelOptimizer:
    """Random search followed by local (evolutionary) refinement around the best configuration."""

    def __init__(self, space: SearchSpace, maximize: bool = False, concurrency: int = 2, seed: int = 0) -> None:
        self.space = space
        self.maximize = maximize
        self.concurrency = max(1, concurrency)
        self.rng = np.random.default_rng(seed)
        self.results: List[TrialResult] = []

    @property
    def best(self) -> Optional[TrialResult]:
        valid = [r for r in self.results if r.error is None and math.isfinite(r.score)]
        if not valid:
            return None
        return max(valid, key=lambda r: r.score) if self.maximize else min(valid, key=lambda r: r.score)

    async def _run_trial(self, trial_id: int, params: Dict[str, Any], objective: Objective, timeout: Optional[float], sem: asyncio.Semaphore) -> TrialResult:
        async with sem:
            start = time.monotonic()
            try:
                score = float(await asyncio.wait_for(objective(params), timeout))
                if not math.isfinite(score):
                    raise ValueError("objective returned a non-finite score")
                return TrialResult(trial_id, params, score, time.monotonic() - start)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Trial %d failed: %r", trial_id, exc)
                bad = float("-inf") if self.maximize else float("inf")
                return TrialResult(trial_id, params, bad, time.monotonic() - start, error=repr(exc))

    async def search(
        self,
        objective: Objective,
        n_trials: int = 20,
        refine_fraction: float = 0.4,
        trial_timeout: Optional[float] = None,
    ) -> List[TrialResult]:
        sem = asyncio.Semaphore(self.concurrency)
        n_refine = int(n_trials * refine_fraction)
        n_random = max(1, n_trials - n_refine)
        trial_id = len(self.results)

        batch = []
        for _ in range(n_random):
            batch.append(self._run_trial(trial_id, self.space.sample(self.rng), objective, trial_timeout, sem))
            trial_id += 1
        self.results.extend(await asyncio.gather(*batch))

        remaining = n_refine
        while remaining > 0:
            best = self.best
            if best is None:
                break
            round_size = min(remaining, self.concurrency)
            scale = 0.25 * (remaining / max(n_refine, 1)) + 0.05
            round_trials = []
            for _ in range(round_size):
                round_trials.append(
                    self._run_trial(trial_id, self.space.perturb(best.params, self.rng, scale), objective, trial_timeout, sem)
                )
                trial_id += 1
            self.results.extend(await asyncio.gather(*round_trials))
            remaining -= round_size
        return sorted(self.results, key=lambda r: r.score, reverse=self.maximize)

    async def tune_learner(
        self,
        factory: Callable[..., Any],
        train_ds: Any,
        val_ds: Any,
        n_trials: int = 10,
        epochs: int = 3,
        batch_size: int = 64,
        trial_timeout: Optional[float] = None,
    ) -> Optional[TrialResult]:
        """Tune a learner built by ``factory(**params)``; the score is the best validation metric (lower is better)."""
        from .trainer import Trainer, TrainerConfig  # local import avoids a module cycle

        async def objective(params: Dict[str, Any]) -> float:
            learner = factory(**params)
            if inspect.isawaitable(learner):
                learner = await learner
            report = await Trainer(TrainerConfig(epochs=epochs, batch_size=batch_size, patience=2)).fit(learner, train_ds, val_ds)
            if report.best_metric is None:
                raise ValueError("trainer produced no monitor metric")
            return report.best_metric

        self.maximize = False
        await self.search(objective, n_trials=n_trials, trial_timeout=trial_timeout)
        return self.best
