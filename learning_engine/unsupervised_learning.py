"""Unsupervised learning: online k-means with dynamic cluster birth/pruning and anomaly scoring."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .learning_core import (
    TOPIC_SAMPLE,
    BaseLearner,
    EventBus,
    LearnerMetrics,
    LearningMode,
    RunningStats,
    Sample,
)

logger = logging.getLogger("zentrax.learning.unsupervised")


class OnlineKMeans:
    """Sequential k-means whose cluster count adapts between ``min_k`` and ``max_k``."""

    def __init__(
        self,
        n_features: int,
        min_k: int = 2,
        max_k: int = 16,
        novelty_sigma: float = 3.0,
        warmup: int = 50,
    ) -> None:
        self.n_features = n_features
        self.min_k, self.max_k = min_k, max_k
        self.novelty_sigma = novelty_sigma
        self.warmup = warmup
        self.centroids: List[np.ndarray] = []
        self.counts: List[int] = []
        self.born_at: List[int] = []
        self.seen = 0
        self.d_mean = 0.0
        self.d_var = 0.0

    def _spawn(self, x: np.ndarray) -> int:
        self.centroids.append(x.copy())
        self.counts.append(1)
        self.born_at.append(self.seen)
        return len(self.centroids) - 1

    def _track_distance(self, dist: float) -> None:
        alpha = max(0.02, 1.0 / max(self.seen, 1))
        delta = dist - self.d_mean
        self.d_mean += alpha * delta
        self.d_var = (1 - alpha) * (self.d_var + alpha * delta ** 2)

    def std(self) -> float:
        return float(max(np.sqrt(self.d_var), 1e-8))

    def update(self, x: np.ndarray) -> Tuple[int, float]:
        self.seen += 1
        if not self.centroids:
            return self._spawn(x), 0.0
        C = np.asarray(self.centroids)
        d = np.linalg.norm(C - x, axis=1)
        j = int(d.argmin())
        dist = float(d[j])
        if len(self.centroids) < self.min_k and dist > 1e-9:
            return self._spawn(x), 0.0
        is_novel = (
            self.seen >= self.warmup
            and len(self.centroids) < self.max_k
            and self.d_var > 0
            and dist > self.d_mean + self.novelty_sigma * self.std()
        )
        if is_novel:
            return self._spawn(x), 0.0
        self.counts[j] += 1
        self.centroids[j] += (x - self.centroids[j]) / self.counts[j]
        self._track_distance(dist)
        return j, dist

    def assign(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.centroids:
            return np.zeros(len(X), dtype=int), np.zeros(len(X))
        C = np.asarray(self.centroids)
        D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        return D.argmin(axis=1), D.min(axis=1)

    def anomaly_score(self, dist: np.ndarray) -> np.ndarray:
        return (dist - self.d_mean) / self.std()

    def prune(self, min_count: int = 2, min_age: int = 200) -> int:
        keep = [
            i
            for i in range(len(self.centroids))
            if self.counts[i] >= min_count or self.seen - self.born_at[i] < min_age
        ]
        if len(keep) == len(self.centroids) or len(keep) < self.min_k:
            return 0
        removed = len(self.centroids) - len(keep)
        self.centroids = [self.centroids[i] for i in keep]
        self.counts = [self.counts[i] for i in keep]
        self.born_at = [self.born_at[i] for i in keep]
        return removed


class UnsupervisedLearner(BaseLearner):
    """Clusters the experience stream, scores novelty, and can emit pseudo-labels."""

    def __init__(
        self,
        name: str = "unsupervised",
        n_features: Optional[int] = None,
        min_k: int = 2,
        max_k: int = 16,
        novelty_sigma: float = 3.0,
        anomaly_threshold: float = 3.0,
        pseudo_label_max_anomaly: Optional[float] = None,
        standardize: bool = True,
        bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(name, LearningMode.UNSUPERVISED, bus)
        # Clusters all raw experience (features only), including supervised-mode samples.
        self.routes = {LearningMode.UNSUPERVISED, LearningMode.SUPERVISED, None}
        self.min_k, self.max_k = min_k, max_k
        self.novelty_sigma = novelty_sigma
        self.anomaly_threshold = anomaly_threshold
        self.pseudo_label_max_anomaly = pseudo_label_max_anomaly
        self.standardize = standardize
        self.n_features = n_features
        self.stats = RunningStats()
        self.model: Optional[OnlineKMeans] = None
        if n_features:
            self._init_model(n_features)

    def _init_model(self, n_features: int) -> None:
        self.n_features = n_features
        self.model = OnlineKMeans(n_features, self.min_k, self.max_k, self.novelty_sigma)
        self.stats = RunningStats(n_features)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.stats.normalize(X) if self.standardize else X

    def _fit_sync(self, X_raw: np.ndarray) -> Tuple[float, List[int], List[float]]:
        assert self.model is not None
        if self.standardize:
            self.stats.update(X_raw)
        X = self._transform(X_raw)
        ids, dists = [], []
        for x in X:
            j, d = self.model.update(x)
            ids.append(j)
            dists.append(d)
        if self.updates % 20 == 19:
            self.model.prune()
        return float(np.mean(dists)), ids, dists

    async def partial_fit(self, samples: Sequence[Sample]) -> LearnerMetrics:
        usable = [s for s in samples if not s.metadata.get("pseudo")]  # never re-ingest our own pseudo-labels
        if usable and self.n_features is None:
            self._init_model(usable[0].features.size)
        usable = [s for s in usable if s.features.size == self.n_features]
        if not usable:
            return self._metrics(0)
        X = np.vstack([s.features for s in usable])
        async with self._lock:
            loss, ids, dists = await asyncio.to_thread(self._fit_sync, X)
            n_clusters = len(self.model.centroids) if self.model else 0
        if self.pseudo_label_max_anomaly is not None and self.bus is not None:
            pseudo = self._pseudo_labels(usable, ids, dists)
            if pseudo:
                await self.bus.publish(TOPIC_SAMPLE, pseudo, self.name)
        return self._metrics(len(usable), loss=loss, extra={"n_clusters": n_clusters})

    def _pseudo_labels(self, samples: Sequence[Sample], ids: Sequence[int], dists: Sequence[float]) -> List[Sample]:
        assert self.model is not None and self.pseudo_label_max_anomaly is not None
        out: List[Sample] = []
        for s, j, d in zip(samples, ids, dists):
            if float(self.model.anomaly_score(np.array([d]))[0]) <= self.pseudo_label_max_anomaly:
                out.append(
                    Sample(
                        features=s.features,
                        label=f"cluster_{j}",
                        mode=LearningMode.SUPERVISED,
                        weight=0.5,
                        metadata={**s.metadata, "pseudo": True, "source": self.name},
                    )
                )
        return out

    async def predict(self, features: Any) -> Dict[str, Any]:
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        async with self._lock:
            if self.model is None or not self.model.centroids or x.shape[1] != self.n_features:
                return {"cluster": None, "distance": None, "anomaly_score": 0.0, "is_anomaly": False}
            ids, dists = self.model.assign(self._transform(x))
            score = float(self.model.anomaly_score(dists)[0])
        return {
            "cluster": int(ids[0]),
            "distance": float(dists[0]),
            "anomaly_score": score,
            "is_anomaly": score > self.anomaly_threshold,
        }

    async def evaluate(self, samples: Sequence[Sample]) -> LearnerMetrics:
        usable = [s for s in samples if s.features.size == self.n_features]
        if not usable or self.model is None or not self.model.centroids:
            return self._metrics(0)
        async with self._lock:
            _, dists = self.model.assign(self._transform(np.vstack([s.features for s in usable])))
        return self._metrics(len(usable), loss=float(dists.mean()))

    def cluster_summary(self) -> List[Dict[str, Any]]:
        if self.model is None:
            return []
        return [
            {"cluster": i, "count": c, "centroid": cen.tolist()}
            for i, (c, cen) in enumerate(zip(self.model.counts, self.model.centroids))
        ]

    def get_hparams(self) -> Dict[str, float]:
        return {"novelty_sigma": self.novelty_sigma, "anomaly_threshold": self.anomaly_threshold}

    def set_hparams(self, params: Mapping[str, float]) -> None:
        if "novelty_sigma" in params:
            self.novelty_sigma = float(max(params["novelty_sigma"], 0.5))
            if self.model:
                self.model.novelty_sigma = self.novelty_sigma
        if "anomaly_threshold" in params:
            self.anomaly_threshold = float(max(params["anomaly_threshold"], 0.5))

    def state_dict(self) -> Dict[str, Any]:
        if self.model is None:
            return {"n_features": None}
        m = self.model
        return {
            "n_features": self.n_features,
            "stats": self.stats.state(),
            "centroids": np.asarray(m.centroids) if m.centroids else np.zeros((0, self.n_features)),
            "counts": list(m.counts),
            "born_at": list(m.born_at),
            "seen": m.seen,
            "d_mean": m.d_mean,
            "d_var": m.d_var,
            "hparams": self.get_hparams(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state.get("n_features"):
            return
        self._init_model(int(state["n_features"]))
        assert self.model is not None
        self.stats.load(state["stats"])
        self.model.centroids = [row.copy() for row in np.asarray(state["centroids"], dtype=np.float64)]
        self.model.counts = [int(c) for c in state["counts"]]
        self.model.born_at = [int(c) for c in state["born_at"]]
        self.model.seen = int(state["seen"])
        self.model.d_mean = float(state["d_mean"])
        self.model.d_var = float(state["d_var"])
        self.set_hparams(state.get("hparams", {}))
