"""End-to-end training workflow.

Stages (each one a :class:`~Data_pipeline.pipeline_core.PipelineStep`)::

    ingest -> [validate raw] -> [version raw] -> clean -> validate
           -> [version clean] -> transform -> [version features]
           -> process (fit + split) -> train model -> evaluate -> persist

Everything fitted along the way (cleaner, transformer, processor, schema,
model) is written to an immutable, hash-checked artifact directory that
:class:`~Data_pipeline.pipelines.inference_pipeline.InferencePipeline` loads.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .. import __version__
from ..data_cleaning import CleaningConfig, CleaningStep, DataCleaner
from ..data_ingestion import DataIngestor, DataSource, IngestionStep
from ..data_processing import DataProcessor, Dataset, ProcessingConfig, ProcessingStep
from ..data_transformation import FeatureTransformer, TransformConfig, TransformationStep
from ..data_validation import Schema, ValidationStep
from ..data_versioning import DataVersionStore, VersioningStep
from ..pipeline_core import (
    ArtifactBundle,
    Pipeline,
    PipelineConfigurationError,
    PipelineContext,
    PipelineError,
    PipelineReport,
    PipelineStep,
)

__all__ = [
    "ModelFactory",
    "TrainingConfig",
    "TrainingResult",
    "TrainingPipeline",
    "TrainModelStep",
    "EvaluateStep",
    "PersistArtifactsStep",
    "compute_metrics",
    "default_model_factory",
]

logger = logging.getLogger(__name__)

ModelFactory = Callable[[str, int], Any]
"""``factory(task, random_state) -> estimator`` with ``fit(X, y)`` / ``predict(X)``."""


# --------------------------------------------------------------------------- #
# Models (sklearn when installed, small numpy fallbacks otherwise)
# --------------------------------------------------------------------------- #
class _RidgeRegressor:
    """Closed-form ridge regression (numpy only)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.x_mean_: Optional[np.ndarray] = None
        self.y_mean_: float = 0.0

    def fit(self, X: Any, y: Any) -> "_RidgeRegressor":
        x = np.asarray(X, dtype="float64")
        t = np.asarray(y, dtype="float64")
        self.x_mean_, self.y_mean_ = x.mean(axis=0), float(t.mean())
        xc, tc = x - self.x_mean_, t - self.y_mean_
        gram = xc.T @ xc + self.alpha * np.eye(x.shape[1])
        self.coef_ = np.linalg.solve(gram, xc.T @ tc)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise PipelineError("Model is not fitted")
        return (np.asarray(X, dtype="float64") - self.x_mean_) @ self.coef_ + self.y_mean_


class _NearestCentroidClassifier:
    """Nearest-centroid classifier (numpy only)."""

    def __init__(self) -> None:
        self.classes_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "_NearestCentroidClassifier":
        x = np.asarray(X, dtype="float64")
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        self.centroids_ = np.vstack([x[labels == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.centroids_ is None or self.classes_ is None:
            raise PipelineError("Model is not fitted")
        x = np.asarray(X, dtype="float64")
        dist = (
            (x**2).sum(axis=1, keepdims=True)
            - 2 * x @ self.centroids_.T
            + (self.centroids_**2).sum(axis=1)
        )
        return self.classes_[np.argmin(dist, axis=1)]


def default_model_factory(task: str, random_state: int = 42) -> Any:
    """RandomForest from scikit-learn when available, else a numpy baseline."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError:
        logger.warning("scikit-learn not installed; using a simple numpy baseline model")
        return _NearestCentroidClassifier() if task == "classification" else _RidgeRegressor()
    if task == "classification":
        return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    return RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)


def compute_metrics(task: str, y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Numpy-only metrics. Classification: accuracy, macro F1. Regression: RMSE, MAE, R2."""
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if truth.shape != pred.shape:
        raise PipelineError(f"Shape mismatch: {truth.shape} vs {pred.shape}")
    if task == "classification":
        f1_scores: List[float] = []
        for label in np.unique(np.concatenate([truth, pred])):
            tp = float(np.sum((pred == label) & (truth == label)))
            fp = float(np.sum((pred == label) & (truth != label)))
            fn = float(np.sum((pred != label) & (truth == label)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {"accuracy": float(np.mean(truth == pred)), "macro_f1": float(np.mean(f1_scores))}
    err = truth.astype("float64") - pred.astype("float64")
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((truth.astype("float64") - truth.astype("float64").mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Configuration / result
# --------------------------------------------------------------------------- #
def _merge(*groups: Sequence[str]) -> Tuple[str, ...]:
    """Order-preserving union of string sequences."""
    seen: Dict[str, None] = {}
    for group in groups:
        for item in group:
            seen.setdefault(item, None)
    return tuple(seen)


@dataclass
class TrainingConfig:
    """Everything needed to run a training pipeline.

    Attributes:
        source: File path, :class:`DataSource` or :class:`DataIngestor`.
        target: Name of the (normalised, snake_case) target column.
        id_columns: Identifier columns - kept out of the features and echoed
            back by the inference pipeline.
        task: ``classification`` / ``regression`` (inferred when ``None``).
        raw_schema: Optional schema checked on the raw data (before cleaning).
        schema: Optional schema checked after cleaning; inferred when ``None``.
            Declare ``datetime`` columns here to get calendar features
            (or use ``parse_dates`` in ``CSVSource.read_kwargs``).
        cleaning / transformation / processing: Optional sub-configs. The
            target, ids and ``test_size`` / ``random_state`` are wired in
            automatically.
        test_size / random_state: Defaults forwarded to the processor.
        model_factory: ``factory(task, random_state) -> estimator``.
        artifacts_dir: Parent directory of the per-run artifact folders.
        versions_dir: Root of the :class:`DataVersionStore` (``None`` disables
            dataset versioning).
        dataset_name: Base name of the versioned datasets.
        run_name: Artifact folder name (defaults to the run id).
        min_rows: Minimum number of ingested rows.
    """

    source: Union[str, Path, DataSource, DataIngestor]
    target: str
    id_columns: Tuple[str, ...] = ()
    task: Optional[str] = None
    raw_schema: Optional[Schema] = None
    schema: Optional[Schema] = None
    cleaning: Optional[CleaningConfig] = None
    transformation: Optional[TransformConfig] = None
    processing: Optional[ProcessingConfig] = None
    test_size: float = 0.2
    random_state: int = 42
    model_factory: ModelFactory = default_model_factory
    artifacts_dir: Union[str, Path] = "artifacts"
    versions_dir: Optional[Union[str, Path]] = "data_versions"
    dataset_name: str = "training_data"
    run_name: Optional[str] = None
    min_rows: int = 10

    def __post_init__(self) -> None:
        self.id_columns = tuple(self.id_columns)
        if not self.target:
            raise PipelineConfigurationError("TrainingConfig.target is required")
        if self.target in self.id_columns:
            raise PipelineConfigurationError("target cannot also be an id column")


@dataclass
class TrainingResult:
    """Outcome of :meth:`TrainingPipeline.run`."""

    run_id: str
    artifact_dir: Path
    metrics: Dict[str, Dict[str, float]]
    task: str
    feature_names: List[str]
    versions: Dict[str, str]
    report: PipelineReport
    context: PipelineContext = field(repr=False)

    def summary(self) -> str:
        lines = [self.report.summary(), f"Task: {self.task}   Artifacts: {self.artifact_dir}"]
        for split, values in self.metrics.items():
            lines.append(f"  {split:<10} " + "  ".join(f"{k}={v:.4f}" for k, v in values.items()))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Steps specific to training
# --------------------------------------------------------------------------- #
class TrainModelStep(PipelineStep):
    """Fit a model on ``dataset.X_train`` / ``y_train``."""

    def __init__(self, model_factory: ModelFactory = default_model_factory, *,
                 random_state: int = 42, name: str = "train_model", **kwargs: Any) -> None:
        super().__init__(name, modes=("training",), **kwargs)
        self.model_factory = model_factory
        self.random_state = random_state

    def run(self, context: PipelineContext) -> PipelineContext:
        dataset: Dataset = context.artifacts["dataset"]
        model = self.model_factory(dataset.task, self.random_state)
        model.fit(dataset.X_train, dataset.y_train)
        context.artifacts["model"] = model
        context.metadata["model"] = {"class": f"{type(model).__module__}.{type(model).__qualname__}"}
        return context


class EvaluateStep(PipelineStep):
    """Compute metrics on the train / validation / test splits."""

    def __init__(self, *, name: str = "evaluate", **kwargs: Any) -> None:
        super().__init__(name, modes=("training",), **kwargs)

    def run(self, context: PipelineContext) -> PipelineContext:
        dataset: Dataset = context.artifacts["dataset"]
        model = context.artifacts["model"]
        splits = {"train": (dataset.X_train, dataset.y_train), "test": (dataset.X_test, dataset.y_test)}
        if dataset.X_val is not None and dataset.y_val is not None:
            splits["validation"] = (dataset.X_val, dataset.y_val)
        metrics = {
            split: compute_metrics(dataset.task, y, model.predict(X)) for split, (X, y) in splits.items()
        }
        context.metadata["metrics"] = metrics
        logger.info("Test metrics: %s", metrics["test"])
        return context


class PersistArtifactsStep(PipelineStep):
    """Write the fitted state and model as an :class:`ArtifactBundle`."""

    def __init__(self, config: TrainingConfig, *, signing_key: Optional[bytes] = None,
                 name: str = "persist_artifacts", **kwargs: Any) -> None:
        super().__init__(name, modes=("training",), **kwargs)
        self.config = config
        self.signing_key = signing_key

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg = self.config
        art = context.artifacts
        dataset: Dataset = art["dataset"]
        schema: Schema = art["schema"].without(cfg.target)
        ingestion = context.metadata.get("ingestion", {})
        manifest = {
            "package_version": __version__,
            "run_id": context.run_id,
            "task": dataset.task,
            "target": cfg.target,
            "id_columns": list(cfg.id_columns),
            "feature_names": dataset.feature_names,
            "metrics": context.metadata.get("metrics", {}),
            "versions": context.metadata.get("versions", {}),
            "ingestion": {
                "source": ingestion.get("source"),
                "normalize_columns": ingestion.get("normalize_columns", True),
                "rows": ingestion.get("rows"),
            },
            "model": context.metadata.get("model", {}),
            "environment": {
                "python": sys.version.split()[0],
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        }
        root = Path(cfg.artifacts_dir)
        run_dir = root / (cfg.run_name or context.run_id)
        ArtifactBundle.save(
            run_dir,
            json_artifacts={
                "cleaner": art["cleaner"].to_dict(),
                "transformer": art["transformer"].to_dict(),
                "processor": art["processor"].to_dict(),
                "schema": schema.to_dict(),
            },
            model=art["model"],
            manifest=manifest,
            signing_key=self.signing_key,
        )
        ArtifactBundle.write_latest_pointer(root, run_dir)
        context.artifacts["artifact_dir"] = run_dir
        return context


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class TrainingPipeline:
    """Build and run the full training workflow described by a :class:`TrainingConfig`.

    Example:
        >>> cfg = TrainingConfig(source="customers.csv", target="churned",
        ...                      id_columns=("customer_id",))
        >>> result = TrainingPipeline(cfg).run()          # doctest: +SKIP
        >>> print(result.summary())                       # doctest: +SKIP
    """

    def __init__(self, config: TrainingConfig, *, signing_key: Optional[bytes] = None) -> None:
        self.config = config
        self.signing_key = signing_key

    # -- sub-config wiring -------------------------------------------------- #
    def _cleaning_config(self) -> CleaningConfig:
        cfg = self.config
        base = cfg.cleaning or CleaningConfig()
        return dataclasses.replace(
            base,
            protected_columns=_merge(base.protected_columns, (cfg.target,)),
            drop_rows_missing=_merge(base.drop_rows_missing, (cfg.target,)),
            no_infer_columns=_merge(base.no_infer_columns, cfg.id_columns),
        )

    def _transform_config(self) -> TransformConfig:
        cfg = self.config
        base = cfg.transformation or TransformConfig()
        extra: Tuple[str, ...] = ()
        if cfg.processing is not None and cfg.processing.time_column:
            extra = (cfg.processing.time_column,)
        return dataclasses.replace(
            base, exclude_columns=_merge(base.exclude_columns, (cfg.target,), cfg.id_columns, extra)
        )

    def _processing_config(self) -> ProcessingConfig:
        cfg = self.config
        if cfg.processing is None:
            return ProcessingConfig(
                target=cfg.target, task=cfg.task, drop_columns=cfg.id_columns,
                test_size=cfg.test_size, random_state=cfg.random_state,
            )
        return dataclasses.replace(
            cfg.processing, target=cfg.target, task=cfg.task or cfg.processing.task,
            drop_columns=_merge(cfg.processing.drop_columns, cfg.id_columns),
        )

    def _ingestor(self) -> DataIngestor:
        src = self.config.source
        if isinstance(src, DataIngestor):
            return src
        if isinstance(src, DataSource):
            return DataIngestor(src, min_rows=self.config.min_rows)
        return DataIngestor.from_path(src, ingestor_kwargs={"min_rows": self.config.min_rows})

    # -- build / run -------------------------------------------------------- #
    def build(self) -> Pipeline:
        """Assemble the :class:`Pipeline` (one fresh set of stateful objects per call)."""
        cfg = self.config
        store = DataVersionStore(cfg.versions_dir) if cfg.versions_dir is not None else None
        name = cfg.dataset_name

        def versioning(stage: str) -> List[PipelineStep]:
            if store is None:
                return []
            return [VersioningStep(store, f"{name}_{stage}", metadata={"stage": stage})]

        steps: List[PipelineStep] = [IngestionStep(self._ingestor())]
        if cfg.raw_schema is not None:
            steps.append(ValidationStep(cfg.raw_schema, stage="raw", coerce=False))
        steps += versioning("raw")
        steps.append(CleaningStep(DataCleaner(self._cleaning_config())))
        steps.append(ValidationStep(cfg.schema, stage="clean", infer_schema=cfg.schema is None))
        steps += versioning("clean")
        steps.append(TransformationStep(FeatureTransformer(self._transform_config())))
        steps += versioning("features")
        steps.append(ProcessingStep(DataProcessor(self._processing_config())))
        steps.append(TrainModelStep(cfg.model_factory, random_state=cfg.random_state))
        steps.append(EvaluateStep())
        steps.append(PersistArtifactsStep(cfg, signing_key=self.signing_key))
        return Pipeline("training", steps)

    def run(self) -> TrainingResult:
        """Execute the pipeline and return a :class:`TrainingResult`.

        Raises:
            StepExecutionError: A step failed (``exc.report`` has the partial report).
        """
        ctx = self.build().run(PipelineContext(mode="training"))
        assert ctx.report is not None
        return TrainingResult(
            run_id=ctx.run_id,
            artifact_dir=Path(ctx.artifacts["artifact_dir"]),
            metrics=ctx.metadata.get("metrics", {}),
            task=ctx.metadata.get("task", ""),
            feature_names=list(ctx.artifacts["dataset"].feature_names),
            versions=dict(ctx.metadata.get("versions", {})),
            report=ctx.report,
            context=ctx,
        )
