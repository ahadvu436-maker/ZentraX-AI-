"""Ready-made training and inference pipelines wired from ``Settings``."""
from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel

from configuration.settings import Settings

from .cleaning import CleaningStage
from .core import Pipeline, Record
from .ingestion import IngestionStage, Source
from .processing import BatchFn, BatchProcessingStage
from .transformation import RecordFn, TransformationStage
from .validation import SchemaValidationStage
from .versioning import DatasetVersionStore, VersioningStage


def build_training_pipeline(
    settings: Settings,
    *,
    source: Source,
    schema: type[BaseModel],
    dataset_name: str,
    transforms: Sequence[RecordFn] = (),
    dedupe_on: Sequence[str] = (),
) -> Pipeline[None, list[Record]]:
    p = settings.pipeline
    return Pipeline(
        "training",
        [
            IngestionStage(source),
            CleaningStage(dedupe_on=dedupe_on),
            SchemaValidationStage(schema, max_reject_ratio=p.max_reject_ratio),
            TransformationStage(transforms),
            VersioningStage(DatasetVersionStore(p.dataset_root), dataset_name),
        ],
    )


def build_inference_pipeline(
    settings: Settings,
    *,
    schema: type[BaseModel],
    predict: BatchFn,
    transforms: Sequence[RecordFn] = (),
) -> Pipeline[list[Record], list[Record]]:
    p = settings.pipeline
    return Pipeline(
        "inference",
        [
            CleaningStage(),
            SchemaValidationStage(schema, max_reject_ratio=1.0),  # rejects are reported, not fatal
            TransformationStage(transforms),
            BatchProcessingStage(predict, batch_size=settings.model.max_batch_size, max_workers=p.max_workers),
        ],
    )
