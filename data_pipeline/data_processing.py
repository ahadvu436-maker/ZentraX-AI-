"""Final dataset assembly: feature selection, splitting and balancing.

:class:`DataProcessor` takes the *numeric* output of the transformer and

1. removes constant and (optionally) highly correlated features (learned once
   in ``fit`` and re-applied by column name at inference),
2. produces a reproducible train / (validation) / test :class:`Dataset`
   - random, stratified (classification) or chronological,
3. optionally balances the *training* split by random oversampling.

In inference mode only :meth:`DataProcessor.select_features` is used: it
returns the exact training feature columns in the exact training order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .pipeline_core import (
    PipelineContext,
    PipelineError,
    PipelineStep,
    column_kind,
    to_jsonable,
)

__all__ = [
    "ProcessingError",
    "ProcessingConfig",
    "Dataset",
    "DataProcessor",
    "ProcessingStep",
    "infer_task",
    "batch_frames",
]

logger = logging.getLogger(__name__)

_MAX_CORRELATION_FEATURES = 1500


class ProcessingError(PipelineError):
    """Processing could not be fitted or applied."""


def infer_task(y: pd.Series, max_classes: int = 20) -> str:
    """Guess ``"classification"`` or ``"regression"`` from the target values."""
    kind = column_kind(y)
    if kind in ("categorical", "bool"):
        return "classification"
    values = y.dropna()
    if not len(values):
        raise ProcessingError("Target has no non-missing values")
    integral = bool(np.all(np.mod(values.to_numpy(dtype="float64"), 1) == 0))
    return "classification" if integral and values.nunique() <= max_classes else "regression"


def batch_frames(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield ``df`` in consecutive batches of at most ``batch_size`` rows."""
    if batch_size < 1:
        raise ProcessingError("batch_size must be >= 1")
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


@dataclass
class ProcessingConfig:
    """Processing policy.

    Attributes:
        target: Target column (required for ``split``).
        task: ``classification`` / ``regression``; inferred when ``None``.
        drop_columns: Columns that are never features (ids ...).
        drop_constant: Remove features with a single distinct value.
        correlation_threshold: Drop one of each feature pair with
            ``|corr| > threshold`` (``None`` disables).
        test_size / val_size: Fractions of the data.
        shuffle: Shuffle before splitting (ignored with ``time_column``).
        stratify: Stratify by class for classification.
        time_column: Sort by this column and use the *latest* rows as
            test/validation (no leakage from the future).
        balance: ``None`` or ``"oversample"`` (training split only).
        random_state: Seed for every random decision.
        max_classes: Max distinct integer targets still treated as classes.
    """

    target: Optional[str] = None
    task: Optional[str] = None
    drop_columns: Tuple[str, ...] = ()
    drop_constant: bool = True
    correlation_threshold: Optional[float] = 0.98
    test_size: float = 0.2
    val_size: float = 0.0
    shuffle: bool = True
    stratify: bool = True
    time_column: Optional[str] = None
    balance: Optional[str] = None
    random_state: int = 42
    max_classes: int = 20

    def __post_init__(self) -> None:
        self.drop_columns = tuple(self.drop_columns)
        if self.task not in (None, "classification", "regression"):
            raise ProcessingError("task must be None, 'classification' or 'regression'")
        if self.balance not in (None, "oversample"):
            raise ProcessingError("balance must be None or 'oversample'")
        if not 0.0 < self.test_size < 1.0:
            raise ProcessingError("test_size must be within (0, 1)")
        if not 0.0 <= self.val_size < 1.0 or self.test_size + self.val_size >= 1.0:
            raise ProcessingError("val_size must be >= 0 and test_size + val_size < 1")
        if self.correlation_threshold is not None and not 0.0 < self.correlation_threshold <= 1.0:
            raise ProcessingError("correlation_threshold must be within (0, 1]")


@dataclass
class Dataset:
    """Split dataset produced by :meth:`DataProcessor.split`."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    task: str
    target: str
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None

    def summary(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "task": self.task,
            "target": self.target,
            "n_features": len(self.feature_names),
            "rows": {"train": len(self.X_train), "test": len(self.X_test)},
        }
        if self.X_val is not None:
            info["rows"]["validation"] = len(self.X_val)
        if self.task == "classification":
            info["train_class_counts"] = {
                str(k): int(v) for k, v in self.y_train.value_counts().items()
            }
        return info


@dataclass
class _ProcessingState:
    feature_columns: List[str] = field(default_factory=list)
    dropped_constant: List[str] = field(default_factory=list)
    dropped_correlated: List[str] = field(default_factory=list)
    task: Optional[str] = None


class DataProcessor:
    """Feature selection + dataset splitting."""

    def __init__(self, config: Optional[ProcessingConfig] = None) -> None:
        self.config = config or ProcessingConfig()
        self._state: Optional[_ProcessingState] = None

    # -- state ------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    def _require_state(self) -> _ProcessingState:
        if self._state is None:
            raise ProcessingError("DataProcessor is not fitted; call fit() first")
        return self._state

    @property
    def feature_columns(self) -> List[str]:
        return list(self._require_state().feature_columns)

    @property
    def task(self) -> Optional[str]:
        return self._require_state().task

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": to_jsonable(self.config),
            "state": to_jsonable(self._state) if self._state else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataProcessor":
        processor = cls(ProcessingConfig(**payload["config"]))
        if payload.get("state") is not None:
            processor._state = _ProcessingState(**payload["state"])
        return processor

    # -- fit --------------------------------------------------------------- #
    def _non_feature_columns(self) -> set:
        cfg = self.config
        extras = {cfg.target, cfg.time_column, *cfg.drop_columns}
        extras.discard(None)
        return extras

    def fit(self, df: pd.DataFrame) -> "DataProcessor":
        """Learn the final feature columns (and the task) from training data."""
        cfg = self.config
        if df.empty:
            raise ProcessingError("Cannot fit on an empty DataFrame")
        if cfg.target is not None and cfg.target not in df.columns:
            raise ProcessingError(f"Target column '{cfg.target}' not found")
        if cfg.time_column is not None and cfg.time_column not in df.columns:
            raise ProcessingError(f"time_column '{cfg.time_column}' not found")

        features = df.drop(columns=[c for c in self._non_feature_columns() if c in df.columns])
        non_numeric = [c for c in features.columns if column_kind(features[c]) not in ("numeric", "bool")]
        if non_numeric:
            raise ProcessingError(
                f"Non-numeric feature columns {non_numeric}; run FeatureTransformer first "
                "or list them in drop_columns"
            )
        features = features.astype("float64")

        constant: List[str] = []
        if cfg.drop_constant:
            constant = [c for c in features.columns if features[c].nunique(dropna=False) <= 1]
            features = features.drop(columns=constant)

        correlated: List[str] = []
        threshold = cfg.correlation_threshold
        if threshold is not None and features.shape[1] > 1:
            if features.shape[1] > _MAX_CORRELATION_FEATURES:
                logger.warning(
                    "Skipping correlation filter: %d features exceeds %d",
                    features.shape[1], _MAX_CORRELATION_FEATURES,
                )
            else:
                corr = features.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
                correlated = [c for c in upper.columns if (upper[c] > threshold).any()]
                features = features.drop(columns=correlated)

        if features.shape[1] == 0:
            raise ProcessingError("No usable feature columns remain after processing")

        task = cfg.task
        if task is None and cfg.target is not None:
            task = infer_task(df[cfg.target], cfg.max_classes)

        self._state = _ProcessingState(list(features.columns), constant, correlated, task)
        logger.info(
            "Processor fitted: %d features (dropped %d constant, %d correlated), task=%s",
            len(features.columns), len(constant), len(correlated), task,
        )
        return self

    # -- feature selection (training + inference) -------------------------- #
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the training feature columns, in training order, as float64."""
        state = self._require_state()
        missing = [c for c in state.feature_columns if c not in df.columns]
        if missing:
            raise ProcessingError(f"Input is missing {len(missing)} feature column(s): {missing[:10]}")
        return df.loc[:, state.feature_columns].astype("float64")

    # -- splitting --------------------------------------------------------- #
    @staticmethod
    def _partition(
        positions: np.ndarray, test_size: float, val_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cut an ordered array into (train, val, test); test comes first."""
        n = len(positions)
        n_test = min(int(round(n * test_size)), max(n - 1, 0))
        n_val = min(int(round(n * val_size)), max(n - 1 - n_test, 0))
        test = positions[:n_test]
        val = positions[n_test : n_test + n_val]
        train = positions[n_test + n_val :]
        return train, val, test

    def split(self, df: pd.DataFrame) -> Dataset:
        """Create the train/(val)/test :class:`Dataset` (training mode only)."""
        cfg = self.config
        state = self._require_state()
        if cfg.target is None:
            raise ProcessingError("ProcessingConfig.target is required for split()")
        if df[cfg.target].isna().any():
            raise ProcessingError(
                f"Target '{cfg.target}' contains missing values; drop them during cleaning "
                "(CleaningConfig.drop_rows_missing)"
            )
        task = state.task or infer_task(df[cfg.target], cfg.max_classes)
        X = self.select_features(df)
        y = df[cfg.target]
        n = len(df)
        if n < 3:
            raise ProcessingError(f"Need at least 3 rows to split, got {n}")
        rng = np.random.default_rng(cfg.random_state)

        if cfg.time_column is not None:
            order = np.argsort(df[cfg.time_column].to_numpy(), kind="stable")
            train, val, test = self._partition(order[::-1], cfg.test_size, cfg.val_size)
            train, val, test = train[::-1], val[::-1], test[::-1]  # back to chronological order
        elif cfg.stratify and task == "classification":
            parts: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]] = ([], [], [])
            codes, _ = pd.factorize(y)
            for code in np.unique(codes):
                members = np.flatnonzero(codes == code)
                members = rng.permutation(members) if cfg.shuffle else members
                for bucket, piece in zip(parts, self._partition(members, cfg.test_size, cfg.val_size)):
                    bucket.append(piece)
            train, val, test = (
                np.concatenate(p) if p else np.array([], dtype=int) for p in parts
            )
            if cfg.shuffle:
                train, val, test = rng.permutation(train), rng.permutation(val), rng.permutation(test)
        else:
            positions = rng.permutation(n) if cfg.shuffle else np.arange(n)
            train, val, test = self._partition(positions, cfg.test_size, cfg.val_size)

        if cfg.balance == "oversample" and task == "classification":
            train = self._oversample(train, y.to_numpy(), rng)
        if len(test) == 0 or len(train) == 0:
            raise ProcessingError("Split produced an empty train or test set; provide more data")

        dataset = Dataset(
            X_train=X.iloc[train], X_test=X.iloc[test],
            y_train=y.iloc[train], y_test=y.iloc[test],
            feature_names=list(X.columns), task=task, target=cfg.target,
            X_val=X.iloc[val] if len(val) else None,
            y_val=y.iloc[val] if len(val) else None,
        )
        logger.info("Split: %s", dataset.summary()["rows"])
        return dataset

    @staticmethod
    def _oversample(train: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        codes, _ = pd.factorize(labels[train])
        counts = np.bincount(codes)
        target = counts.max()
        extra: List[np.ndarray] = []
        for code, count in enumerate(counts):
            if count < target:
                members = train[codes == code]
                extra.append(rng.choice(members, size=target - count, replace=True))
        return rng.permutation(np.concatenate([train, *extra])) if extra else train


class ProcessingStep(PipelineStep):
    """Training: fit + split into a :class:`Dataset`. Inference: select features.

    In ``training`` mode ``context.data`` is left untouched and the dataset is
    stored in ``context.artifacts["dataset"]``; in ``inference`` mode
    ``context.data`` becomes the feature matrix ``X``.
    """

    def __init__(
        self,
        processor: Optional[DataProcessor] = None,
        config: Optional[ProcessingConfig] = None,
        *,
        name: str = "processing",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.processor = processor or DataProcessor(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        frame = context.require_data()
        if context.mode == "training":
            self.processor.fit(frame)
            dataset = self.processor.split(frame)
            context.artifacts["dataset"] = dataset
            context.metadata["processing"] = dataset.summary()
            context.metadata["task"] = dataset.task
        else:
            context.data = self.processor.select_features(frame)
        context.artifacts["processor"] = self.processor
        return context
