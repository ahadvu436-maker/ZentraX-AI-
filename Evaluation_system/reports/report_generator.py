"""Aggregate case reports into structured JSON, Markdown and plain-text summary logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

from ..evaluation_core import CaseReport, EvaluationError, Status

if TYPE_CHECKING:  # pragma: no cover
    from ..benchmark_engine import BenchmarkResult

ReportDict = Dict[str, Any]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class ReportGenerator:
    """Builds and serialises evaluation reports.

    Args:
        title: report title.
        include_case_details: embed full per-case results under ``"cases"``.
        failure_limit: number of worst failing cases listed in ``"failures"``.
    """

    def __init__(self, title: str = "Evaluation Report", include_case_details: bool = False,
                 failure_limit: int = 20) -> None:
        self.title = title
        self.include_case_details = include_case_details
        self.failure_limit = failure_limit

    # -- aggregation ------------------------------------------------------- #

    def build(self, reports: Sequence[CaseReport], *, benchmark: Optional["BenchmarkResult"] = None,
              metadata: Optional[Mapping[str, Any]] = None) -> ReportDict:
        from ..performance_evaluator import PerformanceEvaluator  # local import: avoids cycles

        total = len(reports)
        report: ReportDict = {
            "title": self.title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata or {}),
            "summary": {
                "total_cases": total,
                "passed": sum(r.passed for r in reports),
                "failed": sum(not r.passed for r in reports),
                "pass_rate": round(sum(r.passed for r in reports) / total, 4) if total else 0.0,
                "mean_overall_score": round(_mean([r.overall_score for r in reports]), 4),
            },
            "evaluators": self._per_evaluator(reports),
            "metrics": self._per_metric(reports),
            "categories": self._per_category(reports),
            "failures": self._failures(reports),
            "safety_incidents": [r.case_id for r in reports
                                 if "safety" in r.results and r.results["safety"].status is Status.FAILED],
        }
        latencies = self._latencies(reports)
        if latencies:
            report["latency"] = {k: round(v, 4) for k, v in PerformanceEvaluator.summarize(latencies).items()}
        if benchmark is not None:
            report["benchmark"] = {
                "suite": benchmark.suite, "version": benchmark.suite_version, "model": benchmark.model_name,
                "started_at": benchmark.started_at, "finished_at": benchmark.finished_at,
                "duration_s": round(benchmark.duration_s, 3), "task_errors": benchmark.task_errors,
            }
        if self.include_case_details:
            report["cases"] = [r.to_dict() for r in reports]
        return report

    @staticmethod
    def _per_evaluator(reports: Sequence[CaseReport]) -> Dict[str, Any]:
        names = sorted({n for r in reports for n in r.results})
        out: Dict[str, Any] = {}
        for name in names:
            results = [r.results[name] for r in reports if name in r.results]
            scored = [x for x in results if x.status in (Status.PASSED, Status.FAILED) and x.score is not None]
            scores = [x.score for x in scored]  # type: ignore[misc]
            out[name] = {
                "evaluated": len(scored),
                "skipped": sum(x.status is Status.SKIPPED for x in results),
                "errors": sum(x.status is Status.ERROR for x in results),
                "pass_rate": round(sum(x.passed for x in scored) / len(scored), 4) if scored else None,
                "mean_score": round(_mean(scores), 4) if scores else None,
                "min_score": round(min(scores), 4) if scores else None,
                "max_score": round(max(scores), 4) if scores else None,
            }
        return out

    @staticmethod
    def _per_metric(reports: Sequence[CaseReport]) -> Dict[str, Any]:
        buckets: Dict[str, List[float]] = {}
        for r in reports:
            for name, result in r.results.items():
                for m in result.metrics:
                    buckets.setdefault(f"{name}.{m.name}", []).append(m.score)
        return {k: {"mean": round(_mean(v), 4), "count": len(v)} for k, v in sorted(buckets.items())}

    @staticmethod
    def _per_category(reports: Sequence[CaseReport]) -> Dict[str, Any]:
        buckets: Dict[str, List[CaseReport]] = {}
        for r in reports:
            buckets.setdefault(r.category, []).append(r)
        return {
            cat: {
                "count": len(items),
                "pass_rate": round(sum(i.passed for i in items) / len(items), 4),
                "mean_score": round(_mean([i.overall_score for i in items]), 4),
            }
            for cat, items in sorted(buckets.items())
        }

    def _failures(self, reports: Sequence[CaseReport]) -> List[Dict[str, Any]]:
        failing = sorted((r for r in reports if not r.passed), key=lambda r: r.overall_score)
        out = []
        for r in failing[: self.failure_limit]:
            out.append({
                "case_id": r.case_id,
                "category": r.category,
                "overall_score": round(r.overall_score, 4),
                "failed_evaluators": [n for n, x in r.results.items() if x.status in (Status.FAILED, Status.ERROR)],
                "failed_metrics": [f"{n}.{m.name}" for n, x in r.results.items() for m in x.metrics
                                   if m.passed is False],
                "errors": {n: x.error for n, x in r.results.items() if x.error},
                "generation_error": r.metadata.get("generation_error"),
            })
        return out

    @staticmethod
    def _latencies(reports: Sequence[CaseReport]) -> List[float]:
        values: List[float] = []
        for r in reports:
            perf = r.results.get("performance")
            if perf:
                values.extend(float(m.details["value"]) for m in perf.metrics
                              if m.name == "latency" and "value" in m.details)
        return values

    # -- serialisation ----------------------------------------------------- #

    @staticmethod
    def to_json(report: ReportDict, indent: int = 2) -> str:
        return json.dumps(report, indent=indent, default=str, ensure_ascii=False)

    @staticmethod
    def to_summary_log(report: ReportDict) -> str:
        s = report["summary"]
        lines = [
            f"[{report['generated_at']}] {report['title']}",
            f"cases={s['total_cases']} passed={s['passed']} failed={s['failed']} "
            f"pass_rate={s['pass_rate']:.1%} mean_score={s['mean_overall_score']:.3f}",
        ]
        for name, e in report["evaluators"].items():
            lines.append(f"  evaluator={name} evaluated={e['evaluated']} skipped={e['skipped']} "
                         f"errors={e['errors']} mean={e['mean_score']} pass_rate={e['pass_rate']}")
        if "latency" in report:
            lat = report["latency"]
            lines.append(f"  latency p50={lat['p50']}s p95={lat['p95']}s max={lat['max']}s")
        if report["safety_incidents"]:
            lines.append(f"  SAFETY INCIDENTS: {', '.join(report['safety_incidents'])}")
        for f in report["failures"][:5]:
            lines.append(f"  FAIL {f['case_id']} score={f['overall_score']} metrics={f['failed_metrics']}")
        return "\n".join(lines)

    @staticmethod
    def to_markdown(report: ReportDict) -> str:
        s = report["summary"]
        out = [
            f"# {report['title']}", f"_Generated {report['generated_at']}_", "",
            "## Summary",
            f"- **Cases:** {s['total_cases']} (passed {s['passed']}, failed {s['failed']})",
            f"- **Pass rate:** {s['pass_rate']:.1%}",
            f"- **Mean score:** {s['mean_overall_score']:.3f}", "",
            "## Evaluators", "| Evaluator | Evaluated | Skipped | Errors | Mean | Pass rate |", "|---|---|---|---|---|---|",
        ]
        for name, e in report["evaluators"].items():
            out.append(f"| {name} | {e['evaluated']} | {e['skipped']} | {e['errors']} | {e['mean_score']} | {e['pass_rate']} |")
        if report["categories"]:
            out += ["", "## Categories", "| Category | Cases | Pass rate | Mean |", "|---|---|---|---|"]
            out += [f"| {c} | {v['count']} | {v['pass_rate']:.1%} | {v['mean_score']} |"
                    for c, v in report["categories"].items()]
        if "latency" in report:
            lat = report["latency"]
            out += ["", "## Latency (s)", f"p50 {lat['p50']} · p95 {lat['p95']} · p99 {lat['p99']} · max {lat['max']}"]
        if report["failures"]:
            out += ["", "## Worst failures"]
            out += [f"- `{f['case_id']}` ({f['category']}) score {f['overall_score']} - {', '.join(f['failed_metrics']) or 'n/a'}"
                    for f in report["failures"]]
        return "\n".join(out)

    def save(self, report: ReportDict, path: Union[str, Path], fmt: Optional[str] = None) -> Path:
        """Write the report; ``fmt`` is ``json``/``md``/``log`` (inferred from the suffix if omitted)."""
        target = Path(path)
        fmt = (fmt or target.suffix.lstrip(".") or "json").lower()
        renderers = {"json": self.to_json, "md": self.to_markdown, "markdown": self.to_markdown,
                     "log": self.to_summary_log, "txt": self.to_summary_log}
        if fmt not in renderers:
            raise EvaluationError(f"Unsupported report format '{fmt}' (use json, md or log)")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(renderers[fmt](report), encoding="utf-8")
        except OSError as exc:
            raise EvaluationError(f"Could not write report to {target}: {exc}") from exc
        return target
