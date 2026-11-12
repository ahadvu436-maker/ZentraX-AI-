"""Dataset management: registration, stratified splits, async batching and persistence."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..learning_core import LearningMode, Sample, to_python

logger = logging.getLogger("zentrax.learning.datasets")


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError("Dataset.X must be 2-D")
        if not np.all(np.isfinite(self.X)):
            raise ValueError("Dataset.X contains non-finite values")
        if self.y is not None:
            self.y = np.asarray(self.y)
            if len(self.y) != len(self.X):
                raise ValueError("X and y length mismatch")
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=np.float64)
            if len(self.weights) != len(self.X):
                raise ValueError("X and weights length mismatch")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256(np.ascontiguousarray(self.X).tobytes())
        if self.y is not None:
            h.update(str(self.y.tolist()).encode("utf-8"))
        return h.hexdigest()[:16]

    def subset(self, idx: np.ndarray, name: Optional[str] = None) -> "Dataset":
        return Dataset(
            name or self.name,
            self.X[idx],
            None if self.y is None else self.y[idx],
            None if self.weights is None else self.weights[idx],
            dict(self.metadata),
        )

    def to_samples(self, mode: Optional[LearningMode] = LearningMode.SUPERVISED) -> List[Sample]:
        return [
            Sample(
                self.X[i],
                label=None if self.y is None else to_python(self.y[i]),
                mode=mode,
                weight=1.0 if self.weights is None else float(self.weights[i]),
            )
            for i in range(len(self))
        ]


@dataclass
class Batch:
    X: np.ndarray
    y: Optional[np.ndarray]
    weights: Optional[np.ndarray]
    indices: np.ndarray

    def to_samples(self, mode: Optional[LearningMode] = LearningMode.SUPERVISED) -> List[Sample]:
        return [
            Sample(
                self.X[i],
                label=None if self.y is None else to_python(self.y[i]),
                mode=mode,
                weight=1.0 if self.weights is None else float(self.weights[i]),
            )
            for i in range(len(self.X))
        ]


class DatasetManager:
    def __init__(self) -> None:
        self._datasets: Dict[str, Dataset] = {}

    # -- registry --------------------------------------------------------- #
    def register(self, ds: Dataset) -> Dataset:
        self._datasets[ds.name] = ds
        return ds

    def get(self, name: str) -> Dataset:
        try:
            return self._datasets[name]
        except KeyError:
            raise KeyError(f"Unknown dataset '{name}'") from None

    def names(self) -> List[str]:
        return sorted(self._datasets)

    def remove(self, name: str) -> None:
        self._datasets.pop(name, None)

    # -- construction ------------------------------------------------------ #
    def from_arrays(self, name: str, X: Any, y: Any = None, weights: Any = None, register: bool = True) -> Dataset:
        ds = Dataset(name, X, y, weights)
        return self.register(ds) if register else ds

    def from_samples(self, name: str, samples: Sequence[Sample], register: bool = True) -> Dataset:
        if not samples:
            raise ValueError("No samples provided")
        size = samples[0].features.size
        kept = [s for s in samples if s.features.size == size]
        X = np.vstack([s.features for s in kept])
        labels = [s.label for s in kept]
        y = None if any(l is None for l in labels) else np.array(labels)
        return self.from_arrays(name, X, y, np.array([s.weight for s in kept]), register)

    async def from_jsonl(
        self,
        name: str,
        path: str,
        feature_key: str = "features",
        label_key: Optional[str] = "label",
        max_rows: Optional[int] = None,
        register: bool = True,
    ) -> Dataset:
        def _load() -> Tuple[np.ndarray, Optional[np.ndarray], int]:
            rows, labels, bad = [], [], 0
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if max_rows is not None and len(rows) >= max_rows:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        vec = np.asarray(rec[feature_key], dtype=np.float64).ravel()
                        if rows and vec.size != rows[0].size or not np.all(np.isfinite(vec)):
                            raise ValueError("bad vector")
                        lab = rec.get(label_key) if label_key else None
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                        bad += 1
                        continue
                    rows.append(vec)
                    labels.append(lab)
            if not rows:
                raise ValueError(f"No valid rows in {path}")
            y = None if label_key is None or any(l is None for l in labels) else np.array(labels)
            return np.vstack(rows), y, bad

        X, y, bad = await asyncio.to_thread(_load)
        if bad:
            logger.warning("Skipped %d malformed rows while loading %s", bad, path)
        return self.from_arrays(name, X, y, register=register)

    # -- splitting --------------------------------------------------------- #
    def split(
        self,
        name: str,
        val_fraction: float = 0.2,
        stratify: bool = True,
        seed: int = 0,
        register: bool = True,
    ) -> Tuple[Dataset, Dataset]:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        ds = self.get(name)
        rng = np.random.default_rng(seed)
        n = len(ds)
        if stratify and ds.y is not None:
            val_idx: List[int] = []
            for label in np.unique(ds.y):
                members = np.flatnonzero(ds.y == label)
                rng.shuffle(members)
                k = int(round(len(members) * val_fraction))
                if len(members) > 1:
                    k = min(max(k, 1), len(members) - 1)
                val_idx.extend(members[:k].tolist())
            val = np.array(sorted(val_idx), dtype=np.int64)
        else:
            order = rng.permutation(n)
            val = np.sort(order[: max(1, int(round(n * val_fraction)))])
        mask = np.ones(n, dtype=bool)
        mask[val] = False
        train, valid = ds.subset(np.flatnonzero(mask), f"{name}:train"), ds.subset(val, f"{name}:val")
        if register:
            self.register(train)
            self.register(valid)
        return train, valid

    # -- iteration --------------------------------------------------------- #
    async def batches(
        self,
        ds: Dataset,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: Optional[int] = None,
        epoch: int = 0,
        drop_last: bool = False,
    ) -> AsyncIterator[Batch]:
        n = len(ds)
        rng = np.random.default_rng(None if seed is None else seed + epoch)
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            if drop_last and len(idx) < batch_size:
                break
            yield Batch(
                ds.X[idx],
                None if ds.y is None else ds.y[idx],
                None if ds.weights is None else ds.weights[idx],
                idx,
            )
            await asyncio.sleep(0)

    # -- analysis & transforms --------------------------------------------- #
    def class_distribution(self, ds: Dataset) -> Dict[Any, int]:
        if ds.y is None:
            return {}
        return {to_python(k): v for k, v in Counter(ds.y.tolist()).items()}

    def apply_class_weights(self, ds: Dataset) -> Dataset:
        """Inverse-frequency sample weights (mean-normalised) for imbalanced data."""
        if ds.y is None:
            return ds
        dist = self.class_distribution(ds)
        total, k = len(ds), max(len(dist), 1)
        cw = {label: total / (k * count) for label, count in dist.items()}
        ds.weights = np.array([cw[to_python(label)] for label in ds.y], dtype=np.float64)
        return ds

    def add_noise_augmentation(self, ds: Dataset, scale: float = 0.01, copies: int = 1, seed: int = 0) -> Dataset:
        rng = np.random.default_rng(seed)
        std = ds.X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        Xs = [ds.X] + [ds.X + rng.normal(0.0, scale, ds.X.shape) * std for _ in range(copies)]
        ys = None if ds.y is None else np.concatenate([ds.y] * (copies + 1))
        ws = None if ds.weights is None else np.concatenate([ds.weights] * (copies + 1))
        return Dataset(f"{ds.name}:aug", np.vstack(Xs), ys, ws, dict(ds.metadata))

    def merge(self, name: str, parts: Sequence[Dataset], register: bool = True) -> Dataset:
        if not parts:
            raise ValueError("Nothing to merge")
        labelled = all(p.y is not None for p in parts)
        merged = Dataset(
            name,
            np.vstack([p.X for p in parts]),
            np.concatenate([p.y for p in parts]) if labelled else None,  # type: ignore[misc]
        )
        return self.register(merged) if register else merged

    def describe(self, ds: Dataset) -> Dict[str, Any]:
        return {
            "name": ds.name,
            "rows": len(ds),
            "features": ds.X.shape[1],
            "fingerprint": ds.fingerprint,
            "classes": self.class_distribution(ds),
            "feature_mean": ds.X.mean(axis=0).tolist(),
            "feature_std": ds.X.std(axis=0).tolist(),
        }

    # -- persistence -------------------------------------------------------- #
    async def save(self, name: str, path: str) -> Path:
        ds = self.get(name)
        target = Path(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            arrays: Dict[str, Any] = {"X": ds.X}
            if ds.y is not None:
                arrays["y"] = ds.y.astype(str) if ds.y.dtype == object else ds.y
            if ds.weights is not None:
                arrays["w"] = ds.weights
            tmp = target.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            tmp.replace(target)

        await asyncio.to_thread(_write)
        return target

    async def load(self, name: str, path: str, register: bool = True) -> Dataset:
        def _read() -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
            with np.load(path, allow_pickle=False) as data:
                return data["X"], data["y"] if "y" in data else None, data["w"] if "w" in data else None

        X, y, w = await asyncio.to_thread(_read)
        return self.from_arrays(name, X, y, w, register)
