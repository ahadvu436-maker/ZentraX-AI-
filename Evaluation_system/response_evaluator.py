"""Response-quality evaluator: relevance, completeness, coherence, clarity, format."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .evaluation_core import (
    BaseEvaluator,
    EvaluationCase,
    MetricScore,
    clamp,
    cosine_similarity,
    normalize_text,
    split_sentences,
    token_f1,
    tokenize,
)

logger = logging.getLogger("evaluation_system.response")

JudgeFn = Callable[[EvaluationCase], float]


@dataclass(frozen=True)
class ResponseConfig:
    """Tunable parameters for :class:`ResponseEvaluator`."""

    min_words: int = 5
    ideal_sentence_words: Tuple[int, int] = (8, 28)
    max_repetition: float = 0.3


class ResponseEvaluator(BaseEvaluator):
    """Heuristic response-quality scoring with an optional pluggable judge.

    Supported ``case.constraints`` keys: ``max_words``, ``min_words``,
    ``format`` (``"json"``), ``must_match`` (regex), ``forbidden`` (list of strings).
    """

    name = "response"

    def __init__(
        self,
        config: Optional[ResponseConfig] = None,
        judge: Optional[JudgeFn] = None,
        pass_threshold: float = 0.6,
        enabled_metrics: Optional[List[str]] = None,
    ) -> None:
        super().__init__(pass_threshold, enabled_metrics)
        self.config = config or ResponseConfig()
        self.judge = judge

    def _compute_metrics(self, case: EvaluationCase) -> List[Optional[MetricScore]]:
        text = case.response.strip()
        if not text:
            return [self._metric("non_empty", 0.0, 1.0, critical=True, details={"reason": "empty response"})]
        metrics: List[Optional[MetricScore]] = [
            self._relevance(case, text),
            self._completeness(case, text),
            self._coherence(text),
            self._clarity(text),
            self._format(case, text),
        ]
        if self.judge is not None:
            try:
                metrics.append(self._metric("judge", float(self.judge(case)), 2.0))
            except Exception:  # noqa: BLE001 - a broken judge must not sink heuristics
                logger.warning("Judge failed on case %s", case.case_id, exc_info=True)
        return metrics

    # -- individual metrics ------------------------------------------------ #

    def _relevance(self, case: EvaluationCase, text: str) -> Optional[MetricScore]:
        prompt_tokens = tokenize(case.prompt, remove_stopwords=True)
        if not prompt_tokens:
            return None
        resp_tokens = tokenize(text, remove_stopwords=True)
        coverage = len(set(prompt_tokens) & set(resp_tokens)) / len(set(prompt_tokens))
        cosine = cosine_similarity(prompt_tokens, resp_tokens)
        score = 0.5 * coverage + 0.5 * min(1.0, cosine * 2)
        details: Dict[str, Any] = {"prompt_coverage": round(coverage, 3), "cosine": round(cosine, 3)}
        if case.reference:
            f1 = token_f1(resp_tokens, tokenize(case.reference, remove_stopwords=True))
            score = 0.5 * score + 0.5 * min(1.0, f1 * 1.5)
            details["reference_f1"] = round(f1, 3)
        return self._metric("relevance", score, 2.0, details=details)

    def _completeness(self, case: EvaluationCase, text: str) -> MetricScore:
        words = len(tokenize(text))
        parts = [clamp(words / max(1, self.config.min_words))]
        details: Dict[str, Any] = {"word_count": words}
        if case.required_keywords:
            lowered = text.casefold()
            missing = [k for k in case.required_keywords if k.casefold() not in lowered]
            parts.append(1 - len(missing) / len(case.required_keywords))
            details["missing_keywords"] = missing
        score = sum(parts) / len(parts)
        if words > 20 and not re.search(r"[.!?)\"'`\]}:]\s*$", text):
            score *= 0.9
            details["possibly_truncated"] = True
        return self._metric("completeness", score, 2.0, details=details)

    def _coherence(self, text: str) -> MetricScore:
        sentences = [normalize_text(s) for s in split_sentences(text)]
        dup_ratio = (1 - len(set(sentences)) / len(sentences)) if len(sentences) > 1 else 0.0
        tokens = tokenize(text)
        trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
        tri_ratio = (1 - len(set(trigrams)) / len(trigrams)) if len(trigrams) > 5 else 0.0
        repetition = max(dup_ratio, tri_ratio)
        limit = self.config.max_repetition
        score = 1.0 if repetition <= limit else clamp(1 - (repetition - limit) / (1 - limit))
        return self._metric("coherence", score, 1.5, details={"repetition": round(repetition, 3)})

    def _clarity(self, text: str) -> MetricScore:
        sentences = split_sentences(text)
        lengths = [len(tokenize(s)) for s in sentences] or [0]
        avg = sum(lengths) / len(lengths)
        low, high = self.config.ideal_sentence_words
        if len(sentences) == 1 and avg < low:
            score = 1.0  # short direct answers are fine
        elif low <= avg <= high:
            score = 1.0
        else:
            distance = (low - avg) if avg < low else (avg - high)
            score = clamp(1 - distance / high)
        return self._metric("clarity", score, 1.0, details={"avg_sentence_words": round(avg, 1)})

    def _format(self, case: EvaluationCase, text: str) -> Optional[MetricScore]:
        rules = case.constraints
        if not rules:
            return None
        checks: Dict[str, bool] = {}
        words = len(tokenize(text))
        if "max_words" in rules:
            checks["max_words"] = words <= int(rules["max_words"])
        if "min_words" in rules:
            checks["min_words"] = words >= int(rules["min_words"])
        if rules.get("format") == "json":
            body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            try:
                json.loads(body)
                checks["valid_json"] = True
            except ValueError:
                checks["valid_json"] = False
        if "must_match" in rules:
            checks["must_match"] = re.search(str(rules["must_match"]), text) is not None
        if "forbidden" in rules:
            lowered = text.casefold()
            checks["forbidden"] = not any(str(f).casefold() in lowered for f in rules["forbidden"])
        if not checks:
            return None
        score = sum(checks.values()) / len(checks)
        return self._metric("format_compliance", score, 2.0, details={"checks": checks})
