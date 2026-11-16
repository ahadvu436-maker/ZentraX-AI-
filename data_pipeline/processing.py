"""Processing: ordered, bounded-concurrency batch execution."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .core import PipelineContext, Record, Stage

BatchFn = Callable[[list[Record]], list[Record]]


class BatchProcessingStage(Stage[list[Record], list[Record]]):
    name = "processing"

    def __init__(self, fn: BatchFn, *, batch_size: int = 500, max_workers: int = 4) -> None:
        super().__init__()
        if batch_size < 1 or max_workers < 1:
            raise ValueError("batch_size and max_workers must be >= 1")
        self.fn, self.batch_size, self.max_workers = fn, batch_size, max_workers

    def process(self, data: list[Record], ctx: PipelineContext) -> list[Record]:
        batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        ctx.metrics["processing.batches"] = float(len(batches))
        if len(batches) <= 1 or self.max_workers == 1:
            results = [self.fn(b) for b in batches]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(self.fn, batches))  # preserves order
        return [rec for batch in results for rec in batch]
