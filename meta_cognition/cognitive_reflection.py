"""Cognitive reflection: turn diagnostics into a verdict, insights and reusable lessons."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

from .confidence_assessment import ConfidenceReport
from .error_awareness import ErrorRecord
from .reasoning_inspector import Flaw
from .self_monitoring import Severity
from .uncertainty_estimator import UncertaintyEstimate


class Verdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Insight:
    kind: str
    message: str
    severity: Severity


@dataclass
class Lesson:
    text: str
    tags: frozenset
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ts: float = field(default_factory=time.time)
    hits: int = 0


@dataclass(frozen=True)
class ReflectionPolicy:
    abstain_below: float = 0.25
    accept_above: float = 0.6
    high_uncertainty: float = 0.5
    overconfidence_gap: float = 0.25
    escalate_error_count: int = 3


@dataclass
class Reflection:
    verdict: Verdict
    insights: List[Insight]
    revision_hints: List[str]
    lessons_applied: List[Lesson]
    new_lessons: List[Lesson]


class LessonMemory:
    def __init__(self, capacity: int = 500) -> None:
        self._items: Deque[Lesson] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, text: str, tags: Iterable[str]) -> Lesson:
        tagset = frozenset(tags)
        with self._lock:
            for l in self._items:  # dedupe
                if l.text == text:
                    l.hits += 1
                    return l
            l = Lesson(text, tagset)
            self._items.append(l)
            return l

    def retrieve(self, tags: Iterable[str], limit: int = 5) -> List[Lesson]:
        want = set(tags)
        with self._lock:
            scored = [(len(want & l.tags), l.hits, l) for l in self._items if want & l.tags]
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        out = [l for _, _, l in scored[:limit]]
        for l in out:
            l.hits += 1
        return out

    def __len__(self) -> int:
        return len(self._items)


class CognitiveReflector:
    def __init__(self, policy: Optional[ReflectionPolicy] = None,
                 memory: Optional[LessonMemory] = None) -> None:
        self.policy = policy or ReflectionPolicy()
        self.memory = memory or LessonMemory()
        self._last: Optional[Reflection] = None

    def reflect(self, *, flaws: Sequence[Flaw], uncertainty: Optional[UncertaintyEstimate],
                confidence: ConfidenceReport, errors: Sequence[ErrorRecord] = (),
                recurring_errors: Optional[Dict[Any, int]] = None,
                tags: Iterable[str] = ()) -> Reflection:
        p = self.policy
        insights: List[Insight] = []
        hints: List[str] = []
        new: List[Lesson] = []
        tagset = set(tags)

        critical = [f for f in flaws if f.severity is Severity.CRITICAL]
        serious = [f for f in flaws if f.severity is Severity.ERROR]
        for f in critical + serious:
            insights.append(Insight("logical_flaw", f"{f.kind}: {f.message}", f.severity))
            hints.append(f"Fix step {f.step_id or '(trace)'}: {f.message}")
            tagset.add(f.kind)
        if uncertainty is not None and uncertainty.score >= p.high_uncertainty:
            insights.append(Insight("high_uncertainty",
                                    f"uncertainty {uncertainty.score:.2f} ({uncertainty.level.value})",
                                    Severity.WARNING))
            hints.append("Sample more answers or gather evidence to reduce disagreement.")
        gap = confidence.raw - confidence.calibrated
        if gap >= p.overconfidence_gap:
            insights.append(Insight("overconfidence",
                                    f"raw {confidence.raw:.2f} vs calibrated {confidence.calibrated:.2f}",
                                    Severity.WARNING))
            new.append(self.memory.add("Self-assessed confidence ran above calibrated reliability; "
                                       "discount it.", {"overconfidence"}))
        for f in critical:
            new.append(self.memory.add(f"Avoid '{f.kind}' in reasoning: {f.message.split(':')[0]}.", {f.kind}))
        recurring = recurring_errors or {}
        for cat, n in recurring.items():
            insights.append(Insight("recurring_error", f"{getattr(cat, 'value', cat)} x{n}", Severity.ERROR))

        # --- verdict (most conservative rule wins) ---
        if critical or confidence.calibrated < p.abstain_below:
            verdict = Verdict.ABSTAIN
        elif sum(recurring.values()) >= p.escalate_error_count:
            verdict = Verdict.ESCALATE
        elif serious or (uncertainty is not None and uncertainty.score >= p.high_uncertainty) \
                or confidence.calibrated < p.accept_above:
            verdict = Verdict.REVISE
        else:
            verdict = Verdict.ACCEPT

        applied = self.memory.retrieve(tagset) if tagset else []
        self._last = Reflection(verdict, insights, hints, applied, new)
        return self._last

    def introspect(self) -> Dict[str, Any]:
        return {"lessons": len(self.memory),
                "last_verdict": self._last.verdict.value if self._last else None}
