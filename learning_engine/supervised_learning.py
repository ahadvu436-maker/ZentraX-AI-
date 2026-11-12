"""Online supervised learning: incremental softmax classifier with EWC hooks."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .learning_core import (
    BaseLearner,
    EventBus,
    LearnerMetrics,
    LearningMode,
    RunningStats,
    Sample,
    to_python,
)

logger = logging.getLogger("zentrax.learning.supervised")


class SoftmaxClassifier:
    """Multinomial logistic regression with growable classes and EWC regularisation."""

    def __init__(self, n_features: int, lr: float = 0.05, l2: float = 1e-4, seed: int = 0) -> None:
        self.n_features = n_features
        self.lr = lr
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((0, n_features))
        self.b = np.zeros(0)
        self.fisher: Optional[np.ndarray] = None
        self.anchor: Optional[np.ndarray] = None
        self.ewc_lambda = 0.0

    @property
    def n_classes(self) -> int:
        return self.W.shape[0]

    def ensure_classes(self, n: int) -> None:
        if n <= self.n_classes:
            return
        add = n - self.n_classes
        self.W = np.vstack([self.W, self.rng.normal(0.0, 0.01, (add, self.n_features))])
        self.b = np.concatenate([self.b, np.zeros(add)])
        if self.fisher is not None and self.anchor is not None:
            pad = np.zeros((add, self.n_features))
            self.fisher = np.vstack([self.fisher, pad])
            self.anchor = np.vstack([self.anchor, pad])

    def probs(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W.T + self.b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def step(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        n = X.shape[0]
        P = self.probs(X)
        sw = np.ones(n) if sample_weight is None else sample_weight / max(sample_weight.mean(), 1e-12)
        loss = float(-(sw * np.log(P[np.arange(n), y] + 1e-12)).mean())
        Y = np.zeros_like(P)
        Y[np.arange(n), y] = 1.0
        G = (P - Y) * sw[:, None] / n
        gW = G.T @ X + self.l2 * self.W
        gb = G.sum(axis=0)
        if self.fisher is not None and self.anchor is not None and self.ewc_lambda > 0:
            gW = gW + self.ewc_lambda * self.fisher * (self.W - self.anchor)
        norm = float(np.sqrt((gW ** 2).sum() + (gb ** 2).sum()))
        if norm > 5.0:
            gW, gb = gW * (5.0 / norm), gb * (5.0 / norm)
        self.W -= self.lr * gW
        self.b -= self.lr * gb
        return loss

    def fisher_diagonal(self, X: np.ndarray) -> np.ndarray:
        """Empirical Fisher diagonal under the model's own predictive distribution."""
        P = self.probs(X)
        n = X.shape[0]
        fisher = np.zeros_like(self.W)
        for c in range(self.n_classes):
            Y = np.zeros_like(P)
            Y[:, c] = 1.0
            fisher += ((P[:, [c]]) * ((P - Y) ** 2)).T @ (X ** 2) / n
        return fisher


class SupervisedLearner(BaseLearner):
    """Streaming classifier compatible with the LearningCore contract."""

    def __init__(
        self,
        name: str = "supervised",
        n_features: Optional[int] = None,
        lr: float = 0.05,
        l2: float = 1e-4,
        batch_size: int = 32,
        epochs_per_call: int = 2,
        standardize: bool = True,
        bus: Optional[EventBus] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(name, LearningMode.SUPERVISED, bus)
        self.n_features = n_features
        self.lr, self.l2 = lr, l2
        self.batch_size = batch_size
        self.epochs_per_call = epochs_per_call
        self.standardize = standardize
        self.seed = seed
        self.classes: List[Any] = []
        self._index: Dict[Any, int] = {}
        self.stats = RunningStats()
        self.model: Optional[SoftmaxClassifier] = None
        if n_features:
            self._init_model(n_features)

    # -- setup ----------------------------------------------------------- #
    def _init_model(self, n_features: int) -> None:
        self.n_features = n_features
        self.model = SoftmaxClassifier(n_features, self.lr, self.l2, self.seed)
        self.stats = RunningStats(n_features)

    def is_known(self, label: Any) -> bool:
        return to_python(label) in self._index

    def _label_index(self, label: Any) -> int:
        label = to_python(label)
        if label not in self._index:
            self._index[label] = len(self.classes)
            self.classes.append(label)
        return self._index[label]

    def _prepare(self, samples: Sequence[Sample]) -> Tuple[List[Sample], int]:
        usable: List[Sample] = []
        skipped = 0
        for s in samples:
            if s.label is None:
                continue
            if self.n_features is None:
                self._init_model(s.features.size)
            if s.features.size != self.n_features:
                skipped += 1
                continue
            usable.append(s)
        return usable, skipped

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.stats.normalize(X) if self.standardize else X

    # -- training -------------------------------------------------------- #
    def _fit_sync(self, samples: List[Sample]) -> Tuple[float, float]:
        assert self.model is not None
        X_raw = np.vstack([s.features for s in samples])
        y = np.array([self._label_index(s.label) for s in samples], dtype=np.int64)
        w = np.array([s.weight for s in samples], dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones_like(w)
        self.model.ensure_classes(len(self.classes))
        if self.standardize:
            self.stats.update(X_raw)
        X = self._transform(X_raw)
        rng = np.random.default_rng(self.seed + self.updates)
        loss = 0.0
        for _ in range(max(1, self.epochs_per_call)):
            order = rng.permutation(len(X))
            losses, counts = [], []
            for start in range(0, len(X), self.batch_size):
                idx = order[start:start + self.batch_size]
                losses.append(self.model.step(X[idx], y[idx], w[idx]))
                counts.append(len(idx))
            loss = float(np.average(losses, weights=counts))
        acc = float((self.model.probs(X).argmax(axis=1) == y).mean())
        return loss, acc

    async def partial_fit(self, samples: Sequence[Sample]) -> LearnerMetrics:
        usable, skipped = self._prepare(samples)
        if not usable:
            return self._metrics(0, extra={"skipped": skipped})
        async with self._lock:
            loss, acc = await asyncio.to_thread(self._fit_sync, usable)
        return self._metrics(
            len(usable),
            loss=loss,
            accuracy=acc,
            extra={"skipped": skipped, "n_classes": len(self.classes)},
        )

    # -- inference ------------------------------------------------------- #
    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.probs(self._transform(X))

    async def predict(self, features: Any) -> Dict[str, Any]:
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        async with self._lock:
            if self.model is None or not self.classes or x.shape[1] != self.n_features:
                return {"label": None, "confidence": 0.0, "probabilities": {}}
            p = self.predict_proba_batch(x)[0]
        order = np.argsort(p)[::-1]
        top = int(order[0])
        return {
            "label": self.classes[top],
            "confidence": float(p[top]),
            "probabilities": {str(self.classes[int(i)]): float(p[int(i)]) for i in order[:5]},
        }

    async def evaluate(self, samples: Sequence[Sample]) -> LearnerMetrics:
        usable, _ = self._prepare(samples)
        if not usable or self.model is None or not self.classes:
            return self._metrics(0)
        async with self._lock:
            X = np.vstack([s.features for s in usable])
            P = self.predict_proba_batch(X)
            correct, losses = 0, []
            for i, s in enumerate(usable):
                if s.label in self._index:
                    idx = self._index[s.label]
                    losses.append(-np.log(P[i, idx] + 1e-12))
                    correct += int(P[i].argmax() == idx)
        return self._metrics(
            len(usable),
            loss=float(np.mean(losses)) if losses else None,
            accuracy=correct / len(usable),
        )

    # -- continual-learning hook ---------------------------------------- #
    async def consolidate(self, samples: Sequence[Sample], ewc_lambda: float, decay: float = 0.9) -> None:
        """Anchor current parameters and accumulate Fisher information (online EWC)."""
        usable, _ = self._prepare(samples)
        usable = [s for s in usable if s.label in self._index]
        if not usable or self.model is None:
            return

        def _compute() -> None:
            assert self.model is not None
            X = self._transform(np.vstack([s.features for s in usable]))
            fisher = self.model.fisher_diagonal(X)
            if self.model.fisher is not None and self.model.fisher.shape == fisher.shape:
                fisher = decay * self.model.fisher + fisher
            self.model.fisher = fisher
            self.model.anchor = self.model.W.copy()
            self.model.ewc_lambda = ewc_lambda

        async with self._lock:
            await asyncio.to_thread(_compute)

    # -- hyper-parameters ------------------------------------------------ #
    def get_hparams(self) -> Dict[str, float]:
        return {
            "lr": self.lr,
            "l2": self.l2,
            "batch_size": float(self.batch_size),
            "epochs_per_call": float(self.epochs_per_call),
        }

    def set_hparams(self, params: Mapping[str, float]) -> None:
        if "lr" in params:
            self.lr = float(np.clip(params["lr"], 1e-6, 5.0))
        if "l2" in params:
            self.l2 = float(max(params["l2"], 0.0))
        if "batch_size" in params:
            self.batch_size = int(max(1, params["batch_size"]))
        if "epochs_per_call" in params:
            self.epochs_per_call = int(max(1, params["epochs_per_call"]))
        if "ewc_lambda" in params and self.model is not None:
            self.model.ewc_lambda = float(max(params["ewc_lambda"], 0.0))
        if self.model is not None:
            self.model.lr, self.model.l2 = self.lr, self.l2

    # -- persistence ----------------------------------------------------- #
    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "n_features": self.n_features,
            "classes": list(self.classes),
            "hparams": self.get_hparams(),
            "stats": self.stats.state() if self.n_features else None,
        }
        if self.model is not None:
            state.update(
                W=self.model.W,
                b=self.model.b,
                fisher=self.model.fisher,
                anchor=self.model.anchor,
                ewc_lambda=self.model.ewc_lambda,
            )
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state.get("n_features"):
            return
        self._init_model(int(state["n_features"]))
        assert self.model is not None
        self.classes = list(state.get("classes", []))
        self._index = {c: i for i, c in enumerate(self.classes)}
        if state.get("stats"):
            self.stats.load(state["stats"])
        if state.get("W") is not None:
            self.model.W = np.asarray(state["W"], dtype=np.float64)
            self.model.b = np.asarray(state["b"], dtype=np.float64)
        self.model.fisher = None if state.get("fisher") is None else np.asarray(state["fisher"])
        self.model.anchor = None if state.get("anchor") is None else np.asarray(state["anchor"])
        self.model.ewc_lambda = float(state.get("ewc_lambda", 0.0))
        self.set_hparams(state.get("hparams", {}))
