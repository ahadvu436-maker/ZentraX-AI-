"""Central orchestrator coordinating all metacognitive subsystems."""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, TypeVar

from .cognitive_reflection import CognitiveReflector, Reflection, Verdict
from .confidence_assessment import Calibrator, ConfidenceAssessor, ConfidenceReport, ConfidenceSignals
from .error_awareness import ErrorAwareness, ErrorRecord, StageFailure
from .introspection.introspection_engine import IntrospectionEngine, Snapshot, to_plain
from .reasoning_inspector import InspectionReport, ReasoningInspector, ReasoningTrace
from .self_monitoring import Severity, SelfMonitor, Threshold
from .uncertainty_estimator import UncertaintyEstimate, UncertaintyEstimator

log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class MetaCognitiveReport:
    id: str
    verdict: Verdict
    inspection: Optional[InspectionReport]
    uncertainty: Optional[UncertaintyEstimate]
    confidence: Optional[ConfidenceReport]
    errors: List[ErrorRecord]
    reflection: Optional[Reflection]
    snapshot: Optional[Snapshot]
    degraded_stages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_plain({k: v for k, v in self.__dict__.items()})


class MetaCognitionCore:
    """Runs: inspect -> estimate uncertainty -> assess confidence -> detect errors
    -> reflect -> introspect. Each stage is isolated: a failing stage degrades the
    report (and is recorded as an error) instead of crashing the agent."""

    def __init__(self, *, monitor: Optional[SelfMonitor] = None,
                 inspector: Optional[ReasoningInspector] = None,
                 estimator: Optional[UncertaintyEstimator] = None,
                 assessor: Optional[ConfidenceAssessor] = None,
                 errors: Optional[ErrorAwareness] = None,
                 reflector: Optional[CognitiveReflector] = None,
                 introspection: Optional[IntrospectionEngine] = None) -> None:
        self.monitor = monitor or SelfMonitor()
        self.inspector = inspector or ReasoningInspector()
        self.estimator = estimator or UncertaintyEstimator()
        self.assessor = assessor or ConfidenceAssessor(Calibrator())
        self.errors = errors or ErrorAwareness()
        self.reflector = reflector or CognitiveReflector()
        self.introspection = introspection or IntrospectionEngine()
        self._issued: Dict[str, float] = {}  # report id -> raw/calibrated confidence for outcome feedback
        self._lock = threading.Lock()

        self.monitor.add_threshold(Threshold("evaluate.latency_s", max_value=2.0, severity=Severity.WARNING))
        for name, comp in (("self_monitor", self.monitor), ("reasoning_inspector", self.inspector),
                           ("uncertainty_estimator", self.estimator), ("confidence", self.assessor),
                           ("error_awareness", self.errors), ("reflection", self.reflector)):
            self.introspection.register(name, comp.introspect)

    def _stage(self, name: str, fn: Callable[[], T], degraded: List[str]) -> Optional[T]:
        try:
            with self.monitor.track(f"stage.{name}"):
                return fn()
        except Exception as e:
            degraded.append(name)
            self.errors.record_exception(StageFailure(name, e), source=name)
            log.warning("metacognition stage '%s' degraded: %r", name, e)
            return None

    def evaluate(self, trace: ReasoningTrace, *, samples: Optional[Sequence[Hashable]] = None,
                 probabilities: Optional[Sequence[float]] = None,
                 model_confidence: Optional[float] = None,
                 tags: Sequence[str] = ()) -> MetaCognitiveReport:
        degraded: List[str] = []
        with self.monitor.track("evaluate"):
            inspection = self._stage("inspection", lambda: self.inspector.inspect(trace), degraded)

            def estimate() -> Optional[UncertaintyEstimate]:
                ests = []
                if samples:
                    ests.append(self.estimator.from_samples(samples))
                if probabilities:
                    ests.append(self.estimator.from_probabilities(probabilities))
                return self.estimator.combine(*ests) if ests else None
            uncertainty = self._stage("uncertainty", estimate, degraded)

            def assess() -> Optional[ConfidenceReport]:
                sig = ConfidenceSignals(
                    model_confidence=model_confidence,
                    consistency=None if uncertainty is None else 1.0 - uncertainty.score,
                    reasoning_soundness=None if inspection is None else inspection.soundness,
                    recent_error_rate=self.errors.error_rate())
                return self.assessor.assess(sig)
            confidence = self._stage("confidence", assess, degraded)

            new_errors = self._stage("error_awareness", lambda: self.errors.scan({
                "flaws": inspection.flaws if inspection else [],
                "confidence": confidence.calibrated if confidence else None,
                "uncertainty": uncertainty.score if uncertainty else None}), degraded) or []

            reflection: Optional[Reflection] = None
            if confidence is not None:
                reflection = self._stage("reflection", lambda: self.reflector.reflect(
                    flaws=inspection.flaws if inspection else [], uncertainty=uncertainty,
                    confidence=confidence, errors=new_errors,
                    recurring_errors=self.errors.patterns(min_count=2, window_s=600), tags=tags), degraded)

            snapshot = self._stage("introspection", self.introspection.snapshot, degraded)

        # Fail-safe: with no usable confidence/reflection the agent must not proceed unchecked.
        verdict = reflection.verdict if reflection else Verdict.ESCALATE
        rid = uuid.uuid4().hex[:12]
        if confidence is not None:
            with self._lock:
                self._issued[rid] = confidence.raw
                if len(self._issued) > 10_000:
                    self._issued.pop(next(iter(self._issued)))
        return MetaCognitiveReport(rid, verdict, inspection, uncertainty, confidence,
                                   new_errors, reflection, snapshot, degraded)

    def record_outcome(self, report_id: str, correct: bool) -> None:
        """Feed ground truth back so confidence calibration improves over time."""
        with self._lock:
            conf = self._issued.pop(report_id, None)
        if conf is None:
            raise KeyError(f"unknown or already-resolved report id: {report_id}")
        self.assessor.record_outcome(conf, correct)
        if not correct:
            self.monitor.record("outcome.incorrect", 1.0)

    def introspect(self) -> Snapshot:
        return self.introspection.snapshot()
