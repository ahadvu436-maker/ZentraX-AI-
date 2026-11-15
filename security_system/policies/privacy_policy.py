"""Privacy policy: data classification, retention, purpose limitation, redaction.

The policy answers four questions about any piece of data:

* How sensitive is it?                       (:class:`DataClassification`)
* How long may it be kept?                   (:meth:`PrivacyPolicy.retention_for`)
* Must it be encrypted / may it leave?       (:meth:`requires_encryption`, :meth:`can_export`)
* Which personal data must be masked in logs? (:meth:`PrivacyPolicy.redact`)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .security_policy import SecurityError

__all__ = [
    "PrivacyViolation",
    "DataClassification",
    "PIIRule",
    "PrivacyPolicy",
]

_SENSITIVE_KEY_FRAGMENTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "private_key",
    "credential",
    "cvv",
    "ssn",
)
_MAX_REDACTION_DEPTH = 10


class PrivacyViolation(SecurityError):
    """Raised when data handling would violate the privacy policy."""


class DataClassification(IntEnum):
    """Ordered sensitivity levels. Higher value means more sensitive."""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def parse(cls, value: str | int | DataClassification) -> DataClassification:
        """Parse a name (case-insensitive), integer or member."""
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            raise PrivacyViolation(f"unknown data classification: {value!r}") from exc


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def _card_ok(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


def _ipv4_ok(match: str) -> bool:
    return all(0 <= int(part) <= 255 for part in match.split("."))


def _phone_ok(match: str) -> bool:
    return len(re.sub(r"\D", "", match)) >= 10


@dataclass(frozen=True)
class PIIRule:
    """A named pattern with a replacement and an optional false-positive filter."""

    name: str
    pattern: re.Pattern[str]
    replacement: str
    validator: Callable[[str], bool] | None = None

    def apply(self, text: str) -> str:
        def _sub(match: re.Match[str]) -> str:
            value = match.group(0)
            if self.validator is not None and not self.validator(value):
                return value
            return self.replacement

        return self.pattern.sub(_sub, text)


def _default_rules() -> tuple[PIIRule, ...]:
    # Order matters: more specific patterns run first so that, for example,
    # a card number is not half-consumed by the phone pattern.
    return (
        PIIRule("card", re.compile(r"(?<![\w-])(?:\d[ -]?){13,19}(?![\w-])"), "[REDACTED:CARD]", _card_ok),
        PIIRule("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"), "[REDACTED:SSN]"),
        PIIRule(
            "email",
            re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
            "[REDACTED:EMAIL]",
        ),
        PIIRule("ipv4", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"), "[REDACTED:IP]", _ipv4_ok),
        PIIRule(
            "phone",
            re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\d)"),
            "[REDACTED:PHONE]",
            _phone_ok,
        ),
    )


def _default_retention() -> Mapping[DataClassification, int]:
    return MappingProxyType(
        {
            DataClassification.PUBLIC: 3650,
            DataClassification.INTERNAL: 1825,
            DataClassification.CONFIDENTIAL: 730,
            DataClassification.RESTRICTED: 365,
        }
    )


def _default_purposes() -> Mapping[DataClassification, frozenset[str]]:
    return MappingProxyType(
        {
            DataClassification.PUBLIC: frozenset({"*"}),
            DataClassification.INTERNAL: frozenset({"operations", "analytics", "security", "support"}),
            DataClassification.CONFIDENTIAL: frozenset({"operations", "security", "support"}),
            DataClassification.RESTRICTED: frozenset({"security", "legal"}),
        }
    )


@dataclass(frozen=True)
class PrivacyPolicy:
    """Data-handling rules. Extend by subclassing or by passing custom rules."""

    retention_days: Mapping[DataClassification, int] = field(default_factory=_default_retention)
    allowed_purposes: Mapping[DataClassification, frozenset[str]] = field(default_factory=_default_purposes)
    encrypt_at_or_above: DataClassification = DataClassification.CONFIDENTIAL
    max_classification_to_untrusted: DataClassification = DataClassification.PUBLIC
    redact_pii_in_logs: bool = True
    pii_rules: tuple[PIIRule, ...] = field(default_factory=_default_rules)

    # ------------------------------------------------------------------ rules
    def retention_for(self, classification: DataClassification) -> int:
        """Maximum retention in days. Unknown levels fall back to the strictest."""
        if classification in self.retention_days:
            return self.retention_days[classification]
        return min(self.retention_days.values())

    def is_expired(
        self,
        classification: DataClassification,
        created_at: float,
        *,
        now: float | None = None,
    ) -> bool:
        """True once data of this class has outlived its retention period."""
        current = time.time() if now is None else now
        return (current - created_at) > self.retention_for(classification) * 86400

    def requires_encryption(self, classification: DataClassification) -> bool:
        """True if data of this class must be encrypted at rest."""
        return classification >= self.encrypt_at_or_above

    def is_purpose_allowed(self, classification: DataClassification, purpose: str) -> bool:
        """Purpose limitation: only listed purposes may touch this class of data."""
        allowed = self.allowed_purposes.get(classification, frozenset())
        return "*" in allowed or purpose in allowed

    def enforce_purpose(self, classification: DataClassification, purpose: str) -> None:
        if not self.is_purpose_allowed(classification, purpose):
            raise PrivacyViolation(f"purpose '{purpose}' not permitted for {classification.name} data")

    def can_export(self, classification: DataClassification, *, destination_trusted: bool) -> bool:
        """Whether data may leave the trust boundary to the given destination."""
        if destination_trusted:
            return True
        return classification <= self.max_classification_to_untrusted

    def enforce_export(self, classification: DataClassification, *, destination_trusted: bool) -> None:
        if not self.can_export(classification, destination_trusted=destination_trusted):
            raise PrivacyViolation(f"{classification.name} data may not be exported to an untrusted destination")

    # -------------------------------------------------------------- redaction
    def redact(self, text: str) -> str:
        """Mask personal data patterns in free text."""
        if not self.redact_pii_in_logs:
            return text
        for rule in self.pii_rules:
            text = rule.apply(text)
        return text

    def redact_value(self, value: Any, *, key: str | None = None, _depth: int = 0) -> Any:
        """Recursively redact a structure; sensitive-looking keys are fully masked."""
        if _depth > _MAX_REDACTION_DEPTH:
            return "[TRUNCATED]"
        if key is not None and any(fragment in key.lower() for fragment in _SENSITIVE_KEY_FRAGMENTS):
            return "[REDACTED]"
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, (bytes, bytearray)):
            return f"[{len(value)} BYTES]"
        if isinstance(value, Mapping):
            return {str(k): self.redact_value(v, key=str(k), _depth=_depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self.redact_value(v, _depth=_depth + 1) for v in value]
        return value
