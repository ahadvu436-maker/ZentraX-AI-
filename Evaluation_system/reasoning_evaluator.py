"""Reasoning evaluator: step structure, coverage, logical flow, arithmetic, consistency."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

from .evaluation_core import (
    BaseEvaluator,
    EvaluationCase,
    MetricScore,
    clamp,
    normalize_text,
    split_sentences,
    token_f1,
    tokenize,
)

_STEP_RE = re.compile(r"^\s*(?:step\s*\d+\s*[:.)\-]?|\d+\s*[.):]|[-*\u2022])\s*(.+?)\s*$", re.IGNORECASE)
_ARITH_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x\u00d7\u00f7])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_CONNECTIVES = frozenset(
    "therefore because so thus hence since then if consequently given next finally first second third "
    "thereby implies".split()
)


def parse_steps(text: str) -> List[str]:
    """Extract reasoning steps from numbered/bulleted lines, falling back to sentences."""
    steps = [m.group(1) for line in (text or "").splitlines() if (m := _STEP_RE.match(line))]
    return steps if len(steps) >= 2 else split_sentences(text)


def extract_final_answer(text: str) -> Optional[str]:
    """Best-effort extraction of the final answer from a reasoning trace."""
    last = None
    for last in re.finditer(r"(?:final answer|answer)\s*(?:is|:|=)\s*([^\n]+)", text or "", re.IGNORECASE):
        pass
    if last:
        return last.group(1).strip(" .")
    steps = parse_steps(text)
    return steps[-1] if steps else None


def _numbers(text: str) -> List[float]:
    return [float(n.replace(",", "")) for n in _NUMBER_RE.findall(text or "")]


def _compute(a: float, op: str, b: float) -> Optional[float]:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op in ("*", "x", "\u00d7"):
        return a * b
    if op in ("/", "\u00f7"):
        return None if b == 0 else a / b
    return None


class ReasoningEvaluator(BaseEvaluator):
    """Checks the structure and validity of step-by-step reasoning.

    Uses ``case.reasoning_trace`` when present, otherwise the response itself.
    """

    name = "reasoning"

    def __init__(
        self,
        min_steps: int = 2,
        step_match_threshold: float = 0.5,
        pass_threshold: float = 0.6,
        enabled_metrics: Optional[List[str]] = None,
    ) -> None:
        super().__init__(pass_threshold, enabled_metrics)
        self.min_steps = min_steps
        self.step_match_threshold = step_match_threshold

    def is_applicable(self, case: EvaluationCase) -> bool:
        return bool(case.reasoning_trace or case.expected_steps or _STEP_RE.search(case.response or ""))

    def _compute_metrics(self, case: EvaluationCase) -> List[Optional[MetricScore]]:
        trace = (case.reasoning_trace or case.response or "").strip()
        if not trace:
            return [self._metric("non_empty_trace", 0.0, critical=True, details={"reason": "empty trace"})]
        steps = parse_steps(trace)
        return [
            self._structure(steps),
            self._coverage(case, steps),
            self._flow(steps),
            self._arithmetic(trace),
            self._redundancy(steps),
            self._final_answer(case, trace),
        ]

    def _structure(self, steps: Sequence[str]) -> MetricScore:
        return self._metric("step_structure", clamp(len(steps) / max(1, self.min_steps)), 1.0,
                            details={"step_count": len(steps)})

    def _coverage(self, case: EvaluationCase, steps: Sequence[str]) -> Optional[MetricScore]:
        if not case.expected_steps:
            return None
        step_tokens = [tokenize(s, remove_stopwords=True) for s in steps]
        missing: List[str] = []
        for expected in case.expected_steps:
            exp_tokens = tokenize(expected, remove_stopwords=True)
            best = max((token_f1(t, exp_tokens) for t in step_tokens), default=0.0)
            if best < self.step_match_threshold:
                missing.append(expected)
        score = 1 - len(missing) / len(case.expected_steps)
        return self._metric("step_coverage", score, 3.0, details={"missing_steps": missing})

    def _flow(self, steps: Sequence[str]) -> Optional[MetricScore]:
        if len(steps) < 2:
            return None
        linked = 0
        for prev, cur in zip(steps, steps[1:]):
            cur_tokens = tokenize(cur)
            has_connective = any(t in _CONNECTIVES for t in cur_tokens)
            shares_terms = bool(set(tokenize(prev, remove_stopwords=True)) & set(tokenize(cur, remove_stopwords=True)))
            if has_connective or shares_terms:
                linked += 1
        return self._metric("logical_flow", linked / (len(steps) - 1), 1.5,
                            details={"linked_transitions": linked, "transitions": len(steps) - 1})

    def _arithmetic(self, trace: str) -> Optional[MetricScore]:
        checked, invalid = 0, []
        for a, op, b, claimed in _ARITH_RE.findall(trace):
            expected = _compute(float(a), op, float(b))
            if expected is None:
                continue
            checked += 1
            if not math.isclose(expected, float(claimed), rel_tol=1e-3, abs_tol=0.01):
                invalid.append(f"{a} {op} {b} = {claimed} (expected {expected:g})")
        if checked == 0:
            return None
        return self._metric("arithmetic_validity", 1 - len(invalid) / checked, 3.0,
                            critical=bool(invalid), details={"checked": checked, "invalid": invalid})

    def _redundancy(self, steps: Sequence[str]) -> Optional[MetricScore]:
        if len(steps) < 2:
            return None
        normed = [normalize_text(s) for s in steps]
        dup_ratio = 1 - len(set(normed)) / len(normed)
        return self._metric("step_uniqueness", 1 - dup_ratio, 1.0, details={"duplicate_ratio": round(dup_ratio, 3)})

    def _final_answer(self, case: EvaluationCase, trace: str) -> Optional[MetricScore]:
        if not case.reference:
            return None
        final = extract_final_answer(trace) or ""
        ref_norm, fin_norm = normalize_text(case.reference), normalize_text(final)
        ref_nums, fin_nums = _numbers(case.reference), _numbers(final)
        if ref_nums:
            ok = any(math.isclose(r, f, rel_tol=1e-6, abs_tol=1e-6) for r in ref_nums for f in fin_nums)
        else:
            ok = bool(ref_norm) and (ref_norm in fin_norm or fin_norm in ref_norm) and bool(fin_norm)
        details: Dict[str, Any] = {"final_answer": final, "reference": case.reference}
        return self._metric("final_answer_consistency", 1.0 if ok else 0.0, 3.0, details=details)
