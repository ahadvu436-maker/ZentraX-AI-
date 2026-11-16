"""Cleaning: normalisation, empty-row removal, de-duplication."""
from __future__ import annotations

import re
from typing import Sequence

from .core import PipelineContext, Record, Stage

_WS = re.compile(r"\s+")


class CleaningStage(Stage[list[Record], list[Record]]):
    name = "cleaning"

    def __init__(
        self,
        *,
        lowercase_keys: bool = True,
        strip_strings: bool = True,
        empty_as_none: bool = True,
        drop_empty_rows: bool = True,
        dedupe_on: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.lowercase_keys = lowercase_keys
        self.strip_strings = strip_strings
        self.empty_as_none = empty_as_none
        self.drop_empty_rows = drop_empty_rows
        self.dedupe_on = tuple(dedupe_on)

    def _clean_value(self, value: object) -> object:
        if isinstance(value, str) and self.strip_strings:
            value = _WS.sub(" ", value).strip()
        if self.empty_as_none and value == "":
            return None
        return value

    def process(self, data: list[Record], ctx: PipelineContext) -> list[Record]:
        out: list[Record] = []
        seen: set[tuple[object, ...]] = set()
        for rec in data:
            cleaned = {
                (str(k).strip().lower() if self.lowercase_keys else str(k).strip()): self._clean_value(v)
                for k, v in rec.items()
            }
            if self.drop_empty_rows and all(v is None for v in cleaned.values()):
                continue
            if self.dedupe_on:
                key = tuple(cleaned.get(k) for k in self.dedupe_on)
                if key in seen:
                    continue
                seen.add(key)
            out.append(cleaned)
        ctx.metrics["cleaning.dropped"] = float(len(data) - len(out))
        return out
