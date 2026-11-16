"""Validation: Pydantic-schema enforcement with a dead-letter list and reject threshold."""
from __future__ import annotations

from pydantic import BaseModel, ValidationError

from .core import PipelineContext, PipelineError, Record, RejectedRecord, Stage


class RejectThresholdExceeded(PipelineError):
    pass


class SchemaValidationStage(Stage[list[Record], list[Record]]):
    name = "validation"

    def __init__(self, schema: type[BaseModel], *, max_reject_ratio: float = 0.2) -> None:
        super().__init__()
        self.schema = schema
        self.max_reject_ratio = max_reject_ratio

    def process(self, data: list[Record], ctx: PipelineContext) -> list[Record]:
        valid: list[Record] = []
        for idx, rec in enumerate(data):
            try:
                valid.append(self.schema.model_validate(rec).model_dump())
            except ValidationError as exc:
                reasons = tuple(
                    f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors()
                )
                ctx.rejects.append(RejectedRecord(self.name, idx, rec, reasons))
        ratio = (len(data) - len(valid)) / len(data) if data else 0.0
        ctx.metrics["validation.reject_ratio"] = ratio
        if ratio > self.max_reject_ratio:
            raise RejectThresholdExceeded(
                f"{ratio:.1%} of records rejected (limit {self.max_reject_ratio:.1%})"
            )
        return valid
