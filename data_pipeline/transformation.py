"""Transformation: composable record-level functions."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .core import PipelineContext, Record, Stage

RecordFn = Callable[[Record], Record]


def rename_fields(mapping: Mapping[str, str]) -> RecordFn:
    return lambda r: {mapping.get(k, k): v for k, v in r.items()}


def derive_field(name: str, fn: Callable[[Record], Any]) -> RecordFn:
    return lambda r: {**r, name: fn(r)}


def select_fields(fields: Sequence[str]) -> RecordFn:
    return lambda r: {k: r[k] for k in fields if k in r}


class TransformationStage(Stage[list[Record], list[Record]]):
    name = "transformation"

    def __init__(self, transforms: Sequence[RecordFn]) -> None:
        super().__init__()
        self.transforms = list(transforms)

    def process(self, data: list[Record], ctx: PipelineContext) -> list[Record]:
        out: list[Record] = []
        for rec in data:
            for fn in self.transforms:
                rec = fn(rec)
            out.append(rec)
        return out
