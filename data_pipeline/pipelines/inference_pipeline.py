"""Inference workflow: load verified training artifacts and score new data.

Stages::

    (ingest) -> clean (no row loss) -> validate/coerce -> transform
             -> select features -> predict

The pipeline never drops, reorders or duplicates rows: the output has exactly
one prediction row per input row, in the same order and with the same index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

from ..data_cleaning import CleaningStep, DataCleaner
from ..data_ingestion import DataIngestor, DataSource
from ..data_processing import DataProcessor, ProcessingStep
from ..data_transformation import FeatureTransformer, TransformationStep
from ..data_validation import Schema, ValidationStep
from ..pipeline_core import (
    ArtifactBundle,
    LoadedBundle,
    Pipeline,
    PipelineContext,
    PipelineError,
    PipelineReport,
    PipelineStep,
    StepExecutionError,
)

__all__ = ["InferenceError", "PredictStep", "InferencePipeline"]

logger = logging.getLogger(__name__)

InferenceInput = Union[pd.DataFrame, DataSource, DataIngestor, str, Path]


class InferenceError(PipelineError):
    """Scoring failed for a reason that is not a data-quality problem."""


class PredictStep(PipelineStep):
    """Run the model on the feature matrix in ``context.data``.

    Stores a ``DataFrame`` of predictions (and optional class probabilities)
    in ``context.artifacts["predictions"]``.
    """

    def __init__(self, model: Any, *, return_proba: bool = False,
                 name: str = "predict", **kwargs: Any) -> None:
        super().__init__(name, modes=("inference",), **kwargs)
        self.model = model
        self.return_proba = return_proba

    def run(self, context: PipelineContext) -> PipelineContext:
        features = context.require_data()
        result = pd.DataFrame({"prediction": np.asarray(self.model.predict(features))}, index=features.index)
        if self.return_proba:
            if not hasattr(self.model, "predict_proba") or not hasattr(self.model, "classes_"):
                raise InferenceError(f"{type(self.model).__name__} does not provide class probabilities")
            proba = np.asarray(self.model.predict_proba(features))
            for position, label in enumerate(self.model.classes_):
                result[f"proba_{label}"] = proba[:, position]
        context.artifacts["predictions"] = result
        return context


class InferencePipeline:
    """Score new data with the artifacts produced by ``TrainingPipeline``.

    Args:
        bundle: A verified :class:`LoadedBundle`.
        return_proba: Add ``proba_<class>`` columns (classification models
            that support ``predict_proba``).

    Example:
        >>> pipe = InferencePipeline.from_artifacts("artifacts")   # doctest: +SKIP
        >>> pipe.predict(new_df)                                    # doctest: +SKIP
    """

    def __init__(self, bundle: LoadedBundle, *, return_proba: bool = False) -> None:
        if bundle.model is None:
            raise InferenceError("Artifact bundle contains no model")
        manifest = bundle.manifest
        self.manifest = manifest
        self.task: str = manifest.get("task", "")
        self.target: str = manifest.get("target", "")
        self.id_columns: List[str] = list(manifest.get("id_columns", []))
        self.feature_names: List[str] = list(manifest.get("feature_names", []))
        self.normalize_columns: bool = bool(
            manifest.get("ingestion", {}).get("normalize_columns", True)
        )
        self.model = bundle.model
        self.last_report: Optional[PipelineReport] = None

        arts = bundle.json_artifacts
        try:
            self._cleaner = DataCleaner.from_dict(arts["cleaner"])
            self._transformer = FeatureTransformer.from_dict(arts["transformer"])
            self._processor = DataProcessor.from_dict(arts["processor"])
            self._schema = Schema.from_dict(arts["schema"])
        except KeyError as exc:
            raise InferenceError(f"Artifact bundle is incomplete: missing {exc}") from exc
        self.pipeline = self._build(return_proba)

    @classmethod
    def from_artifacts(
        cls,
        path: Union[str, Path],
        *,
        signing_key: Optional[bytes] = None,
        return_proba: bool = False,
    ) -> "InferencePipeline":
        """Load and verify artifacts from a run directory (or an artifacts root -> latest run)."""
        return cls(ArtifactBundle.load(path, signing_key=signing_key), return_proba=return_proba)

    def _build(self, return_proba: bool) -> Pipeline:
        return Pipeline(
            "inference",
            [
                CleaningStep(self._cleaner),
                ValidationStep(self._schema, stage="inference_input", coerce=True),
                TransformationStep(self._transformer),
                ProcessingStep(self._processor),
                PredictStep(self.model, return_proba=return_proba),
            ],
        )

    # -- input handling ------------------------------------------------------ #
    def _to_frame(self, data: InferenceInput) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = DataIngestor.normalize_frame(data) if self.normalize_columns else data.copy()
            if frame.empty:
                raise InferenceError("Input DataFrame is empty")
            return frame
        if isinstance(data, DataIngestor):
            return data.ingest()
        if isinstance(data, DataSource):
            return DataIngestor(data, normalize_columns=self.normalize_columns).ingest()
        return DataIngestor.from_path(
            data, ingestor_kwargs={"normalize_columns": self.normalize_columns}
        ).ingest()

    # -- public API ----------------------------------------------------------- #
    def predict(self, data: InferenceInput) -> pd.DataFrame:
        """Return one prediction row per input row.

        Columns: ``id_columns`` (if present in the input), ``prediction`` and,
        when enabled, ``proba_<class>``. The index equals the input index.

        Raises:
            DataValidationError: Input violates the training schema.
            InferenceError: Model/serving problems.
        """
        frame = self._to_frame(data)
        try:
            ctx = self.pipeline.run(PipelineContext(data=frame, mode="inference"))
        except StepExecutionError as exc:
            self.last_report = exc.report
            if isinstance(exc.original, PipelineError):
                raise exc.original from exc
            raise InferenceError(str(exc)) from exc
        self.last_report = ctx.report

        result: pd.DataFrame = ctx.artifacts["predictions"]
        if len(result) != len(frame):  # pragma: no cover - safety invariant
            raise InferenceError(f"Row count changed: {len(frame)} in, {len(result)} out")
        present_ids = [c for c in self.id_columns if c in frame.columns]
        for offset, col in enumerate(present_ids):
            result.insert(offset, col, frame[col].to_numpy())  # positional: safe with duplicate indexes
        return result

    def predict_batches(
        self, data: Union[DataSource, DataIngestor, str, Path], batch_size: int = 10_000
    ) -> Iterator[pd.DataFrame]:
        """Stream predictions for a large file/source without loading it fully."""
        if isinstance(data, DataIngestor):
            ingestor = data
        elif isinstance(data, DataSource):
            ingestor = DataIngestor(data, normalize_columns=self.normalize_columns, min_rows=0)
        else:
            ingestor = DataIngestor.from_path(
                data, ingestor_kwargs={"normalize_columns": self.normalize_columns, "min_rows": 0}
            )
        for batch in ingestor.iter_batches(batch_size):
            if len(batch):
                yield self.predict(batch)

    def info(self) -> Dict[str, Any]:
        """Key facts about the loaded model (handy for health/status endpoints)."""
        return {
            "run_id": self.manifest.get("run_id"),
            "task": self.task,
            "target": self.target,
            "n_features": len(self.feature_names),
            "metrics": self.manifest.get("metrics", {}),
            "data_versions": self.manifest.get("versions", {}),
            "created_at": self.manifest.get("created_at"),
        }
