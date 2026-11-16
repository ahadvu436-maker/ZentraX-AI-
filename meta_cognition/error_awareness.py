"""Error awareness: classification, detection, pattern tracking, recovery guidance."""
from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Mapping, Optional

from .self_monitoring import Severity


class MetaCognitionError(Exception):
    """Base class for framework-level failures."""


class StageFailure(MetaCognitionError):
    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"stage '{stage}' failed: {cause!r}")
        self.stage, self.cause = stage, cause


class ErrorCategory(str, Enum):
    LOGICAL = "logical"
    FACTUAL = "factual"
    TOOL = "tool"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    VALIDATION = "validation"
    OVERCONFIDENCE = "overconfidence"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorRecord:
    category: ErrorCategory
    message: str
    severity: Severity
    source: str = ""
    context: Mapping[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


class ErrorDetector(ABC):
    name: str = "detector"

    @abstractmethod
    def detect(self, context: Mapping[str, Any]) -> List[ErrorRecord]: ...


class InspectionFlawDetector(ErrorDetector):
    """Turns reasoning flaws (ERROR+) into logical error records."""
    name = "inspection_flaws"

    def __init__(self, min_severity: Severity = Severity.ERROR) -> None:
        self.min_severity = min_severity

    def detect(self, context: Mapping[str, Any]) -> List[ErrorRecord]:
        return [ErrorRecord(ErrorCategory.LOGICAL, f"[{f.kind}] {f.message}", f.severity,
                            self.name, {"step_id": f.step_id})
                for f in context.get("flaws", ()) if f.severity >= self.min_severity]


class OverconfidenceDetector(ErrorDetector):
    """High stated confidence while uncertainty is high is a calibration error."""
    name = "overconfidence"

    def __init__(self, conf_floor: float = 0.8, uncertainty_ceiling: float = 0.5) -> None:
        self.conf_floor, self.uncertainty_ceiling = conf_floor, uncertainty_ceiling

    def detect(self, context: Mapping[str, Any]) -> List[ErrorRecord]:
        c, u = context.get("confidence"), context.get("uncertainty")
        if c is not None and u is not None and c >= self.conf_floor and u >= self.uncertainty_ceiling:
            return [ErrorRecord(ErrorCategory.OVERCONFIDENCE,
                                f"confidence {c:.2f} despite uncertainty {u:.2f}",
                                Severity.ERROR, self.name)]
        return []


_EXC_MAP = ((TimeoutError, ErrorCategory.TIMEOUT), (MemoryError, ErrorCategory.RESOURCE),
            ((ValueError, TypeError, KeyError), ErrorCategory.VALIDATION),
            ((ConnectionError, OSError), ErrorCategory.TOOL))

_RECOVERY = {
    ErrorCategory.LOGICAL: "Re-derive the flagged steps from their premises; add evidence or drop unsupported claims.",
    ErrorCategory.FACTUAL: "Verify against an authoritative source before asserting; lower stated confidence.",
    ErrorCategory.TOOL: "Retry with backoff, then fall back to an alternative tool or degrade gracefully.",
    ErrorCategory.TIMEOUT: "Reduce scope or budget the task into smaller steps; retry with a longer deadline.",
    ErrorCategory.RESOURCE: "Shed load, free caches, and retry with a smaller working set.",
    ErrorCategory.VALIDATION: "Validate and normalize inputs before processing; fail fast with a clear message.",
    ErrorCategory.OVERCONFIDENCE: "Recalibrate confidence, gather more samples, or abstain / escalate.",
    ErrorCategory.UNKNOWN: "Capture full context and escalate for human review.",
}


class ErrorAwareness:
    def __init__(self, detectors: Optional[List[ErrorDetector]] = None, max_records: int = 2000) -> None:
        self.detectors: List[ErrorDetector] = detectors if detectors is not None else [
            InspectionFlawDetector(), OverconfidenceDetector()]
        self._records: Deque[ErrorRecord] = deque(maxlen=max_records)
        self._lock = threading.RLock()

    def add_detector(self, d: ErrorDetector) -> None:
        self.detectors.append(d)

    @staticmethod
    def classify(exc: BaseException) -> ErrorCategory:
        for types, cat in _EXC_MAP:
            if isinstance(exc, types):
                return cat
        return ErrorCategory.UNKNOWN

    def record(self, rec: ErrorRecord) -> ErrorRecord:
        with self._lock:
            self._records.append(rec)
        return rec

    def record_exception(self, exc: BaseException, source: str = "",
                         severity: Severity = Severity.ERROR) -> ErrorRecord:
        return self.record(ErrorRecord(self.classify(exc), repr(exc), severity, source))

    def scan(self, context: Mapping[str, Any]) -> List[ErrorRecord]:
        found: List[ErrorRecord] = []
        for d in self.detectors:
            try:
                found.extend(d.detect(context))
            except Exception as e:
                found.append(ErrorRecord(ErrorCategory.UNKNOWN, f"detector '{d.name}' crashed: {e!r}",
                                         Severity.WARNING, "error_awareness"))
        for r in found:
            self.record(r)
        return found

    def recent(self, window_s: Optional[float] = None) -> List[ErrorRecord]:
        with self._lock:
            recs = list(self._records)
        if window_s is None:
            return recs
        cutoff = time.time() - window_s
        return [r for r in recs if r.ts >= cutoff]

    def error_rate(self, window_s: float = 300.0, cap: int = 20) -> float:
        """Recent-error pressure in [0, 1] (saturates at `cap` errors)."""
        return min(1.0, len(self.recent(window_s)) / cap)

    def patterns(self, min_count: int = 3, window_s: Optional[float] = None) -> Dict[ErrorCategory, int]:
        c = Counter(r.category for r in self.recent(window_s))
        return {k: v for k, v in c.items() if v >= min_count}

    @staticmethod
    def suggest_recovery(category: ErrorCategory) -> str:
        return _RECOVERY[category]

    def introspect(self) -> Dict[str, Any]:
        recs = self.recent()
        return {"total_errors": len(recs),
                "by_category": dict(Counter(r.category.value for r in recs)),
                "detectors": [d.name for d in self.detectors]}
