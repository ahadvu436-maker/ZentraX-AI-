"""DAG workflow execution: dependency ordering, parallel steps, conditions, fail-fast."""
from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .execution_core import (
    TaskResult, TaskSpec, TaskStatus, WorkflowError, WorkflowValidationError, build_result,
)
from .execution_monitor import ExecutionMonitor
from .task_executor import TaskExecutor

logger = logging.getLogger(__name__)

__all__ = ["WorkflowStep", "Workflow", "WorkflowResult", "WorkflowExecutor"]

Results = Mapping[str, TaskResult]
Condition = Callable[[Results], bool]
InputMapper = Callable[[Results], Tuple[Sequence[Any], Dict[str, Any]]]


@dataclass
class WorkflowStep:
    """One node of a workflow.

    Attributes:
        step_id: Unique id within the workflow.
        task: The work to run (retries/timeouts/sandbox come from the spec).
        depends_on: Step ids that must finish first.
        condition: Optional predicate over finished results; False -> step is SKIPPED.
        input_mapper: Optional ``results -> (args, kwargs)`` to feed upstream outputs in.
        continue_on_failure: If True, this step's failure neither aborts the workflow nor
            blocks dependents, and doesn't make the workflow fail.
    """

    step_id: str
    task: TaskSpec
    depends_on: Tuple[str, ...] = ()
    condition: Optional[Condition] = None
    input_mapper: Optional[InputMapper] = None
    continue_on_failure: bool = False

    def __post_init__(self) -> None:
        if not self.step_id:
            raise WorkflowValidationError("step_id must be non-empty")
        self.depends_on = tuple(self.depends_on)


@dataclass
class Workflow:
    """A named collection of steps forming a DAG."""

    name: str
    steps: List[WorkflowStep] = field(default_factory=list)
    fail_fast: bool = True

    def add_step(
        self, step_id: str, func: Callable[..., Any], *, args: Sequence[Any] = (),
        kwargs: Optional[Dict[str, Any]] = None, depends_on: Sequence[str] = (),
        condition: Optional[Condition] = None, input_mapper: Optional[InputMapper] = None,
        continue_on_failure: bool = False, **task_options: Any,
    ) -> "Workflow":
        """Append a step (chainable). ``task_options`` are TaskSpec fields."""
        spec = TaskSpec(func=func, args=tuple(args), kwargs=dict(kwargs or {}),
                        name=task_options.pop("name", step_id), **task_options)
        self.steps.append(WorkflowStep(step_id, spec, tuple(depends_on), condition,
                                       input_mapper, continue_on_failure))
        return self

    def validate(self) -> List[List[str]]:
        """Check ids, dependencies and cycles; return execution layers (Kahn's algorithm)."""
        ids = [s.step_id for s in self.steps]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise WorkflowValidationError(f"duplicate step ids: {dup}")
        known = set(ids)
        for s in self.steps:
            missing = [d for d in s.depends_on if d not in known]
            if missing:
                raise WorkflowValidationError(f"step '{s.step_id}' depends on unknown steps {missing}")
            if s.step_id in s.depends_on:
                raise WorkflowValidationError(f"step '{s.step_id}' depends on itself")
        indegree = {s.step_id: len(set(s.depends_on)) for s in self.steps}
        layers: List[List[str]] = []
        ready = [i for i in ids if indegree[i] == 0]
        placed = 0
        while ready:
            layers.append(ready)
            placed += len(ready)
            nxt: List[str] = []
            for s in self.steps:
                if s.step_id in indegree and indegree[s.step_id] > 0:
                    indegree[s.step_id] -= len(set(s.depends_on) & set(ready))
                    if indegree[s.step_id] == 0:
                        nxt.append(s.step_id)
            ready = nxt
        if placed != len(ids):
            stuck = sorted(i for i in ids if i not in {x for l in layers for x in l})
            raise WorkflowValidationError(f"dependency cycle among steps {stuck}")
        return layers


@dataclass
class WorkflowResult:
    """Outcome of a workflow run."""

    workflow_id: str
    name: str
    status: TaskStatus
    step_results: Dict[str, TaskResult]
    started_at: float
    finished_at: float

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.SUCCEEDED

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def failed_steps(self) -> List[str]:
        return [k for k, r in self.step_results.items() if r.status.is_failure]

    @property
    def skipped_steps(self) -> List[str]:
        return [k for k, r in self.step_results.items() if r.status is TaskStatus.SKIPPED]

    def value(self, step_id: str) -> Any:
        return self.step_results[step_id].unwrap()


class WorkflowExecutor:
    """Executes a :class:`Workflow`, running independent steps in parallel.

    Steps are submitted to the :class:`TaskExecutor` pool (so retries, timeouts,
    sandboxing and monitoring all apply).  A step whose dependency failed (and
    didn't set ``continue_on_failure``) is SKIPPED.  With ``fail_fast`` the first
    hard failure stops new steps from starting; running steps finish.
    """

    def __init__(self, task_executor: TaskExecutor, monitor: ExecutionMonitor) -> None:
        self.tasks = task_executor
        self.monitor = monitor

    def execute(self, workflow: Workflow) -> WorkflowResult:
        workflow.validate()
        wf_id = uuid.uuid4().hex[:12]
        started = time.time()
        self.monitor.emit("workflow_started", wf_id, workflow.name, steps=len(workflow.steps))

        steps = {s.step_id: s for s in workflow.steps}
        remaining = dict(steps)
        results: Dict[str, TaskResult] = {}
        running: Dict["Future[TaskResult]", str] = {}
        abort = False

        while remaining or running:
            progressed = False
            for sid in list(remaining):
                step = remaining[sid]
                if not all(d in results for d in step.depends_on):
                    continue
                del remaining[sid]
                progressed = True
                reason = self._skip_reason(step, steps, results, abort, workflow.name)
                if reason:
                    results[sid] = self._skipped(step, reason)
                    continue
                try:
                    running[self.tasks.submit(self._materialize(workflow, step, results))] = sid
                except Exception as exc:  # noqa: BLE001 - mapper/submit failure
                    results[sid] = build_result(step.task, started_at=time.time(), attempts=0,
                                                exception=exc)
                    abort = abort or (workflow.fail_fast and self._is_hard_failure(step, results[sid]))
            if running:
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in done:
                    sid = running.pop(fut)
                    res = fut.result()
                    results[sid] = res
                    if workflow.fail_fast and self._is_hard_failure(steps[sid], res):
                        abort = True
                        logger.warning("workflow %s: step %s failed; aborting", workflow.name, sid)
            elif remaining and not progressed:
                raise WorkflowError("workflow scheduler stalled (internal error)")

        failed = any(self._is_hard_failure(steps[k], r) or (
            r.status is TaskStatus.SKIPPED and not steps[k].continue_on_failure
            and any(results[d].status.is_failure for d in steps[k].depends_on))
            for k, r in results.items())
        status = TaskStatus.FAILED if failed else TaskStatus.SUCCEEDED
        result = WorkflowResult(wf_id, workflow.name, status,
                                {s.step_id: results[s.step_id] for s in workflow.steps},
                                started, time.time())
        self.monitor.emit("workflow_finished", wf_id, workflow.name, status=status.value,
                          duration=round(result.duration, 4), failed=result.failed_steps,
                          skipped=result.skipped_steps)
        return result

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _is_hard_failure(step: WorkflowStep, res: TaskResult, wf: Workflow = None,
                         ignore_fail_fast: bool = False) -> bool:
        """A failure that counts against the workflow (``continue_on_failure`` exempts it)."""
        return res.status.is_failure and not step.continue_on_failure

    def _skip_reason(self, step: WorkflowStep, steps: Dict[str, WorkflowStep],
                     results: Results, abort: bool, wf_name: str) -> Optional[str]:
        if abort:
            return "workflow aborted after an earlier failure"
        for dep in step.depends_on:
            res = results[dep]
            if res.status is TaskStatus.SUCCEEDED:
                continue
            if res.status.is_failure and steps[dep].continue_on_failure:
                continue
            return f"dependency '{dep}' did not succeed ({res.status.value})"
        if step.condition is not None:
            try:
                if not step.condition(results):
                    return "condition evaluated to False"
            except Exception as exc:  # noqa: BLE001
                logger.exception("condition of step %s raised", step.step_id)
                return f"condition raised {type(exc).__name__}: {exc}"
        return None

    @staticmethod
    def _skipped(step: WorkflowStep, reason: str) -> TaskResult:
        now = time.time()
        return TaskResult(task_id=step.task.task_id, name=step.step_id, status=TaskStatus.SKIPPED,
                          error=reason, started_at=now, finished_at=now,
                          metadata=dict(step.task.metadata))

    @staticmethod
    def _materialize(wf: Workflow, step: WorkflowStep, results: Results) -> TaskSpec:
        changes: Dict[str, Any] = {
            "task_id": uuid.uuid4().hex[:12],
            "metadata": {**step.task.metadata, "workflow": wf.name, "step": step.step_id},
        }
        if step.input_mapper is not None:
            args, kwargs = step.input_mapper(results)
            changes["args"], changes["kwargs"] = tuple(args), dict(kwargs)
        return step.task.with_(**changes)
