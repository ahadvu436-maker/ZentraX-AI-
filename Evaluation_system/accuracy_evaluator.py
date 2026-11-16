"""Factual-accuracy evaluator: answer match, F1, numeric match, key facts, groundedness."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from .evaluation_core import (
    BaseEvaluator,
    EvaluationCase,
    MetricScore,
    normalize_text,
    split_sentences,
    token_f1,
    tokenize,
)

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_numbers(text: str) -> List[float]:
    """Return all numbers found in ``text`` (thousands separators removed)."""
    return [float(n.replace(",", "")) for n in _NUMBER_RE.findall(text or "")]


class AccuracyEvaluator(BaseEvaluator):
    """Scores factual correctness against a reference, key facts and/or context.

    Key facts are read from ``case.metadata["key_facts"]`` (list of strings).
    Groundedness checks that each response sentence is supported by
    ``case.context`` and reports a hallucination-risk estimate.
    """

    name = "accuracy"

    def __init__(
        self,
        support_threshold: float = 0.6,
        pass_threshold: float = 0.6,
        enabled_metrics: Optional[List[str]] = None,
    ) -> None:
        super().__init__(pass_threshold, enabled_metrics)
        self.support_threshold = support_threshold

    def is_applicable(self, case: EvaluationCase) -> bool:
        return bool(case.reference or case.context or case.metadata.get("key_facts"))

    def _compute_metrics(self, case: EvaluationCase) -> List[Optional[MetricScore]]:
        if not case.response.strip():
            return [self._metric("non_empty", 0.0, critical=True, details={"reason": "empty response"})]
        return [
            self._answer_match(case),
            self._token_f1(case),
            self._numeric(case),
            self._key_facts(case),
            self._groundedness(case),
        ]

    def _answer_match(self, case: EvaluationCase) -> Optional[MetricScore]:
        if not case.reference:
            return None
        resp, ref = normalize_text(case.response), normalize_text(case.reference)
        exact = resp == ref
        contained = bool(ref) and ref in resp
        f1 = token_f1(tokenize(resp), tokenize(ref))
        score = 1.0 if exact or contained else f1
        return self._metric("answer_match", score, 3.0,
                            details={"exact": exact, "contained": contained, "f1": round(f1, 3)})

    def _token_f1(self, case: EvaluationCase) -> Optional[MetricScore]:
        if not case.reference:
            return None
        f1 = token_f1(tokenize(case.response), tokenize(case.reference))
        return self._metric("token_f1", f1, 1.0)

    def _numeric(self, case: EvaluationCase) -> Optional[MetricScore]:
        if not case.reference:
            return None
        ref_nums = extract_numbers(case.reference)
        if not ref_nums:
            return None
        resp_nums = extract_numbers(case.response)
        missing = [n for n in ref_nums if not any(math.isclose(n, r, rel_tol=1e-6, abs_tol=1e-9) for r in resp_nums)]
        return self._metric("numeric_accuracy", 1 - len(missing) / len(ref_nums), 2.0,
                            critical=bool(missing), details={"missing_numbers": missing})

    def _key_facts(self, case: EvaluationCase) -> Optional[MetricScore]:
        facts = list(case.metadata.get("key_facts") or [])
        if not facts:
            return None
        resp_norm = normalize_text(case.response)
        resp_tokens = set(tokenize(case.response, remove_stopwords=True))
        missing: List[str] = []
        for fact in facts:
            if normalize_text(str(fact)) in resp_norm:
                continue
            fact_tokens = set(tokenize(str(fact), remove_stopwords=True))
            if not fact_tokens or len(fact_tokens & resp_tokens) / len(fact_tokens) < 0.8:
                missing.append(str(fact))
        return self._metric("key_fact_recall", 1 - len(missing) / len(facts), 2.0,
                            details={"missing_facts": missing})

    def _groundedness(self, case: EvaluationCase) -> Optional[MetricScore]:
        if not case.context:
            return None
        context_tokens = set(tokenize(" ".join(case.context), remove_stopwords=True))
        checkable, unsupported = 0, []
        for sentence in split_sentences(case.response):
            tokens = tokenize(sentence, remove_stopwords=True)
            if len(tokens) < 3:
                continue
            checkable += 1
            support = sum(t in context_tokens for t in tokens) / len(tokens)
            if support < self.support_threshold:
                unsupported.append(sentence)
        if checkable == 0:
            return None
        score = 1 - len(unsupported) / checkable
        details: Dict[str, Any] = {
            "hallucination_risk": round(1 - score, 3),
            "unsupported_sentences": unsupported[:5],
        }
        return self._metric("groundedness", score, 2.0, details=details)
