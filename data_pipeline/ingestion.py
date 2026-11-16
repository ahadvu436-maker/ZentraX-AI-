"""Ingestion: pluggable sources feeding an ``IngestionStage``."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from .core import PipelineContext, Record, Stage


class Source(Protocol):
    def read(self) -> Iterator[Record]: ...


class CsvSource:
    def __init__(self, path: Path, *, delimiter: str = ",", encoding: str = "utf-8") -> None:
        self.path, self.delimiter, self.encoding = Path(path), delimiter, encoding

    def read(self) -> Iterator[Record]:
        with self.path.open(newline="", encoding=self.encoding) as fh:
            yield from (dict(row) for row in csv.DictReader(fh, delimiter=self.delimiter))


class JsonlSource:
    def __init__(self, path: Path, *, encoding: str = "utf-8") -> None:
        self.path, self.encoding = Path(path), encoding

    def read(self) -> Iterator[Record]:
        with self.path.open(encoding=self.encoding) as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{line_no}: invalid JSON ({exc.msg})") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"{self.path}:{line_no}: expected an object")
                yield obj


class InMemorySource:
    def __init__(self, records: Iterable[Record]) -> None:
        self._records = list(records)

    def read(self) -> Iterator[Record]:
        return iter(self._records)


class IngestionStage(Stage[None, list[Record]]):
    name = "ingestion"
    retries = 2

    def __init__(self, source: Source, *, limit: int | None = None) -> None:
        super().__init__()
        self.source, self.limit = source, limit

    def process(self, data: None, ctx: PipelineContext) -> list[Record]:
        records: list[Record] = []
        for rec in self.source.read():
            records.append(rec)
            if self.limit is not None and len(records) >= self.limit:
                break
        ctx.metrics["ingested_records"] = float(len(records))
        return records
