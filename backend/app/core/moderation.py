"""
backend/app/core/moderation.py

Content moderation layer for ZentraX AI.

Provides an async-first `ContentModeration` service that screens both
inbound user prompts and outbound AI-generated text against a set of
safety categories, using a fast local heuristic pass plus an optional
pluggable external classifier (e.g. a hosted moderation API or local model).

Design goals:
- Non-blocking: all checks are async so they can run concurrently with
  other request-handling work (e.g. via asyncio.gather).
- Extensible: category rules and the external classifier are both
  swappable without touching call sites.
- Cheap by default: local regex/keyword pass runs first and short-circuits
  obvious violations before any network call is made.
- Clean integration surface: a single `moderate()` entry point plus a
  FastAPI dependency and a decorator for router use.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("zentrax.moderation")


# --------------------------------------------------------------------------- #
# Categories & results
# --------------------------------------------------------------------------- #

class ModerationCategory(str, Enum):
    SAFE = "safe"
    HATE = "hate"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    SEXUAL = "sexual"
    SEXUAL_MINORS = "sexual_minors"
    VIOLENCE = "violence"
    WEAPONS = "weapons"
    ILLEGAL_ACTIVITY = "illegal_activity"
    MALWARE = "malware"
    PII_LEAK = "pii_leak"
    SPAM = "spam"


# Categories that should always hard-block regardless of confidence/config.
HARD_BLOCK_CATEGORIES = {
    ModerationCategory.SEXUAL_MINORS,
}

# Default per-category confidence threshold above which content is flagged.
DEFAULT_THRESHOLDS: dict[ModerationCategory, float] = {
    ModerationCategory.HATE: 0.6,
    ModerationCategory.HARASSMENT: 0.6,
    ModerationCategory.SELF_HARM: 0.5,
    ModerationCategory.SEXUAL: 0.6,
    ModerationCategory.SEXUAL_MINORS: 0.0,  # any hit blocks
    ModerationCategory.VIOLENCE: 0.6,
    ModerationCategory.WEAPONS: 0.6,
    ModerationCategory.ILLEGAL_ACTIVITY: 0.6,
    ModerationCategory.MALWARE: 0.6,
    ModerationCategory.PII_LEAK: 0.7,
    ModerationCategory.SPAM: 0.8,
}


@dataclass(frozen=True)
class CategoryScore:
    category: ModerationCategory
    score: float  # 0.0-1.0 confidence
    matched_terms: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModerationResult:
    allowed: bool
    flagged_categories: tuple[CategoryScore, ...]
    source: str  # "local" | "classifier" | "local+classifier"
    original_text: str

    @property
    def top_category(self) -> Optional[ModerationCategory]:
        if not self.flagged_categories:
            return None
        return max(self.flagged_categories, key=lambda c: c.score).category

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "source": self.source,
            "flagged_categories": [
                {
                    "category": c.category.value,
                    "score": round(c.score, 3),
                    "matched_terms": list(c.matched_terms),
                }
                for c in self.flagged_categories
            ],
        }


class ModerationBlockedError(Exception):
    """Raised when content is blocked and the caller wants exception-style flow control."""

    def __init__(self, result: ModerationResult):
        self.result = result
        cats = ", ".join(c.category.value for c in result.flagged_categories) or "unknown"
        super().__init__(f"Content blocked by moderation (categories: {cats})")


# --------------------------------------------------------------------------- #
# Local heuristic rules (fast path, no network required)
# --------------------------------------------------------------------------- #
# NOTE: These are intentionally lightweight placeholder patterns. Swap in a
# maintained wordlist / classifier for production-grade coverage; this layer
# exists mainly to catch obvious cases cheaply and to short-circuit before an
# external call.

_RULES: dict[ModerationCategory, list[re.Pattern]] = {
    ModerationCategory.HATE: [
        re.compile(r"\b(kill all|subhuman|racial slur placeholder)\b", re.I),
    ],
    ModerationCategory.SELF_HARM: [
        re.compile(r"\b(how to (kill|hurt) myself|suicide method|end my life)\b", re.I),
    ],
    ModerationCategory.VIOLENCE: [
        re.compile(r"\b(how to (build|make) a bomb|mass shooting plan)\b", re.I),
    ],
    ModerationCategory.WEAPONS: [
        re.compile(r"\b(untraceable firearm|3d print(ed)? gun)\b", re.I),
    ],
    ModerationCategory.MALWARE: [
        re.compile(r"\b(ransomware source|write me a virus|keylogger code)\b", re.I),
    ],
    ModerationCategory.SEXUAL_MINORS: [
        re.compile(r"\b(child sexual|csam)\b", re.I),
    ],
    ModerationCategory.PII_LEAK: [
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),  # card-number-like
    ],
    ModerationCategory.SPAM: [
        re.compile(r"\b(click here now|buy followers|guaranteed income)\b", re.I),
    ],
}


def _run_local_rules(text: str) -> list[CategoryScore]:
    hits: list[CategoryScore] = []
    for category, patterns in _RULES.items():
        matched: list[str] = []
        for pattern in patterns:
            m = pattern.search(text)
            if m:
                matched.append(m.group(0))
        if matched:
            # Local regex hits are treated as high-confidence for their category.
            hits.append(CategoryScore(category=category, score=0.9, matched_terms=tuple(matched)))
    return hits


# --------------------------------------------------------------------------- #
# External classifier protocol (pluggable)
# --------------------------------------------------------------------------- #

ClassifierFn = Callable[[str], Awaitable[list[CategoryScore]]]


async def _noop_classifier(text: str) -> list[CategoryScore]:
    """Default classifier: no-op. Replace via ContentModeration(classifier=...)."""
    return []


# --------------------------------------------------------------------------- #
# Core service
# --------------------------------------------------------------------------- #

class ContentModeration:
    """
    Async content moderation service.

    Usage:
        moderation = ContentModeration()
        result = await moderation.moderate(user_prompt)
        if not result.allowed:
            raise ModerationBlockedError(result)
    """

    def __init__(
        self,
        classifier: Optional[ClassifierFn] = None,
        thresholds: Optional[dict[ModerationCategory, float]] = None,
        run_local_and_classifier_concurrently: bool = True,
    ):
        self._classifier = classifier or _noop_classifier
        self._thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._concurrent = run_local_and_classifier_concurrently

    async def moderate(self, text: str, *, context: str = "input") -> ModerationResult:
        """
        Screen a piece of text (user prompt or model output).

        `context` is a free-form label ("input" | "output") used only for logging.
        """
        if not text or not text.strip():
            return ModerationResult(True, (), source="local", original_text=text)

        local_hits = _run_local_rules(text)

        # Hard-block categories short-circuit immediately; no need to call
        # out to an external classifier for e.g. CSAM-adjacent content.
        if any(h.category in HARD_BLOCK_CATEGORIES for h in local_hits):
            result = ModerationResult(False, tuple(local_hits), source="local", original_text=text)
            self._log(result, context)
            return result

        classifier_hits: list[CategoryScore] = []
        try:
            if self._concurrent:
                classifier_hits = await self._classifier(text)
            else:
                classifier_hits = await self._classifier(text)
        except Exception:  # classifier failures must never take down the request path
            logger.exception("Moderation classifier failed; falling back to local rules only")
            classifier_hits = []

        all_hits = self._merge_scores(local_hits, classifier_hits)
        flagged = tuple(h for h in all_hits if self._is_flagged(h))
        allowed = not flagged

        source = "local"
        if classifier_hits and local_hits:
            source = "local+classifier"
        elif classifier_hits:
            source = "classifier"

        result = ModerationResult(allowed, flagged, source=source, original_text=text)
        self._log(result, context)
        return result

    async def moderate_many(self, texts: list[str], *, context: str = "input") -> list[ModerationResult]:
        """Screen multiple texts concurrently (e.g. a batch of chat messages)."""
        return await asyncio.gather(*(self.moderate(t, context=context) for t in texts))

    def _is_flagged(self, score: CategoryScore) -> bool:
        threshold = self._thresholds.get(score.category, 0.6)
        return score.score >= threshold

    @staticmethod
    def _merge_scores(
        local_hits: list[CategoryScore], classifier_hits: list[CategoryScore]
    ) -> list[CategoryScore]:
        merged: dict[ModerationCategory, CategoryScore] = {}
        for hit in [*local_hits, *classifier_hits]:
            existing = merged.get(hit.category)
            if existing is None or hit.score > existing.score:
                merged[hit.category] = hit
        return list(merged.values())

    @staticmethod
    def _log(result: ModerationResult, context: str) -> None:
        if not result.allowed:
            logger.warning(
                "Moderation blocked content in %s: categories=%s",
                context,
                [c.category.value for c in result.flagged_categories],
            )
        elif result.flagged_categories:
            logger.info(
                "Moderation flagged but allowed content in %s: categories=%s",
                context,
                [c.category.value for c in result.flagged_categories],
            )


# --------------------------------------------------------------------------- #
# Integration helpers
# --------------------------------------------------------------------------- #

_default_moderation = ContentModeration()


def get_content_moderation() -> ContentModeration:
    """
    FastAPI dependency provider.

    Example:
        from fastapi import Depends
        from backend.app.core.moderation import get_content_moderation, ContentModeration

        @router.post("/chat")
        async def chat(
            payload: ChatRequest,
            moderation: ContentModeration = Depends(get_content_moderation),
        ):
            result = await moderation.moderate(payload.message, context="input")
            if not result.allowed:
                raise HTTPException(status_code=422, detail=result.to_dict())
            ...
    """
    return _default_moderation


def moderate_io(
    *,
    input_arg: str = "message",
    check_output: bool = True,
):
    """
    Decorator for router/service functions that both consumes a text kwarg and
    (optionally) screens the wrapped function's return value if it's a string.

    Example:
        @moderate_io(input_arg="prompt")
        async def generate_reply(prompt: str) -> str:
            ...
    """

    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            moderation = get_content_moderation()
            text = kwargs.get(input_arg)
            if isinstance(text, str):
                result = await moderation.moderate(text, context="input")
                if not result.allowed:
                    raise ModerationBlockedError(result)

            output = await func(*args, **kwargs)

            if check_output and isinstance(output, str):
                out_result = await moderation.moderate(output, context="output")
                if not out_result.allowed:
                    raise ModerationBlockedError(out_result)

            return output

        return wrapper

    return decorator