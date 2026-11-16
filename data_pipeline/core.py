"""Pipeline core: typed stages, run context, and the orchestrator."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Sequence, TypeVar
from uuid import uuid4

Record = dict[str, Any]
In = TypeVar("In")
Out = TypeVar("Out")

log = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base error for the pipeline package."""


class StageError(PipelineError):
    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"stage '{stage}' failed: {cause}")
        self.stage = stage
        self.__cause__ = cause


@dataclass(frozen=True)
class RejectedRecord:
    stage: str
    index: int
    record: Record
    reasons: tuple[str, ...]


@dataclass
class PipelineContext:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)
    rejects: list[RejectedRecord] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class Stage(ABC, Generic[In, Out]):
    """A single, side-effect-explicit transformation step."""

    name: str = ""
    retries: int = 0
    retry_backoff_s: float = 0.5

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.name or type(self).__name__

    @abstractmethod
    def process(self, data: In, ctx: PipelineContext) -> Out: ...


@dataclass(frozen=True)
class PipelineResult(Generic[Out]):
    output: Out
    context: PipelineContext
    durations_s: dict[str, float]


class Pipeline(Generic[In, Out]):
    def __init__(self, name: str, stages: Sequence[Stage[Any, Any]]) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique: {names}")
        self.name = name
        self.stages = list(stages)

    def run(self, data: In, ctx: PipelineContext | None = None) -> PipelineResult[Out]:
        ctx = ctx or PipelineContext()
        durations: dict[str, float] = {}
        current: Any = data
        log.info("pipeline started", extra={"pipeline": self.name, "run_id": ctx.run_id})
        for stage in self.stages:
            start = time.perf_counter()
            current = self._run_stage(stage, current, ctx)
            durations[stage.name] = time.perf_counter() - start
            ctx.metrics[f"{stage.name}.duration_s"] = durations[stage.name]
            log.info(
                "stage finished",
                extra={"pipeline": self.name, "stage": stage.name, "duration_s": round(durations[stage.name], 4)},
            )
        return PipelineResult(output=current, context=ctx, durations_s=durations)

    @staticmethod
    def _run_stage(stage: Stage[Any, Any], data: Any, ctx: PipelineContext) -> Any:
        attempt = 0
        while True:
            try:
                return stage.process(data, ctx)
            except PipelineError:
                raise
            except Exception as exc:  # noqa: BLE001 - wrapped and re-raised
                if attempt >= stage.retries:
                    raise StageError(stage.name, exc) from exc
                attempt += 1
                delay = stage.retry_backoff_s * 2 ** (attempt - 1)
                log.warning("stage retry", extra={"stage": stage.name, "attempt": attempt, "delay_s": delay})
                time.sleep(delay)
