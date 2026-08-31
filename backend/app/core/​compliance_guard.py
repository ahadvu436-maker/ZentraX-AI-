"""
compliance_guard.py — Payload-level privacy compliance helper.

Responsibilities:
    1. Scan incoming JSON payloads for likely PII (emails, phone numbers,
       SSNs, credit card numbers, IP addresses) using pattern matching.
    2. Flag GDPR Article 9 / CCPA "sensitive" category data via keyword
       heuristics (health, biometric, religious, political, union, sexual
       orientation terms) so it can be routed for extra handling.
    3. Anonymize/pseudonymize matched fields (masking or salted hashing)
       before payloads are logged, persisted, or forwarded downstream.
    4. Enforce a basic consent-and-legal-basis check: block processing of
       a payload that contains PII categories the caller hasn't declared
       a legal basis for.
    5. Emit compliance audit events that record *what category* of data
       was found and *where* (field path) — never the actual value.

IMPORTANT — scope and limits (read before relying on this in production):
    This module is a technical control that reduces risk and creates an
    audit trail. It is NOT a substitute for legal review. Regex-based PII
    detection has false negatives and false positives; "special category"
    detection here is a keyword heuristic, not a legal determination.
    GDPR/CCPA compliance also requires organizational measures (DPAs,
    retention policies, breach procedures, documented legal basis, etc.)
    that no single code module can provide. Have counsel review your
    actual data flows.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("app.compliance")


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class PIICategory(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"


class SpecialCategory(str, Enum):
    """GDPR Art. 9 / CCPA-sensitive category flags. Heuristic keyword
    match only — treat a hit as "review this field", not as certainty."""

    HEALTH = "health"
    BIOMETRIC = "biometric"
    RELIGIOUS_BELIEF = "religious_belief"
    POLITICAL_OPINION = "political_opinion"
    UNION_MEMBERSHIP = "union_membership"
    SEXUAL_ORIENTATION = "sexual_orientation"


class LegalBasis(str, Enum):
    """GDPR Art. 6 legal bases for processing. CCPA doesn't require a
    named "legal basis" the way GDPR does, but tracking one here still
    documents *why* a given payload's PII is being processed."""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    LEGITIMATE_INTEREST = "legitimate_interest"
    VITAL_INTEREST = "vital_interest"


class DataSubjectRightType(str, Enum):
    ACCESS = "access"
    ERASURE = "erasure"
    PORTABILITY = "portability"
    RESTRICTION = "restriction"
    OBJECTION = "objection"


class AnonymizationMode(str, Enum):
    MASK = "mask"       # human-readable partial redaction, e.g. j***@example.com
    HASH = "hash"        # irreversible salted hash, for correlation without exposure
    REMOVE = "remove"    # drop the field entirely


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class ComplianceConfig(BaseModel):
    # Which PII categories require an explicit declared legal basis before
    # a payload is allowed through. Categories not listed are logged but
    # not blocking.
    categories_requiring_basis: tuple[PIICategory, ...] = (
        PIICategory.SSN,
        PIICategory.CREDIT_CARD,
    )

    # Special categories always block without an explicit legal basis,
    # regardless of categories_requiring_basis — GDPR Art. 9 processing
    # generally needs a distinct, stronger basis (e.g. explicit consent).
    block_special_categories_without_basis: bool = True

    default_anonymization_mode: AnonymizationMode = AnonymizationMode.MASK
    hash_salt: str = Field(default="change-me-in-deployment")

    # Maximum nesting depth walked when scanning a payload, to bound
    # cost on adversarially deep/large JSON.
    max_scan_depth: int = 8


# --------------------------------------------------------------------------
# Detection patterns
# --------------------------------------------------------------------------

_PATTERNS: dict[PIICategory, re.Pattern[str]] = {
    PIICategory.EMAIL: re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    PIICategory.PHONE: re.compile(r"(?<!\d)(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    PIICategory.SSN: re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    PIICategory.CREDIT_CARD: re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    PIICategory.IP_ADDRESS: re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
}

# Keyword heuristics for Art. 9 / CCPA-sensitive categories. Deliberately
# coarse — false positives are cheap (extra review), false negatives are
# not, so these lean broad.
_SPECIAL_CATEGORY_KEYWORDS: dict[SpecialCategory, tuple[str, ...]] = {
    SpecialCategory.HEALTH: ("diagnosis", "medication", "health condition", "medical", "therapy", "disability"),
    SpecialCategory.BIOMETRIC: ("fingerprint", "face scan", "iris scan", "biometric", "retina"),
    SpecialCategory.RELIGIOUS_BELIEF: ("religion", "religious belief", "faith tradition"),
    SpecialCategory.POLITICAL_OPINION: ("political party", "political affiliation", "voting record"),
    SpecialCategory.UNION_MEMBERSHIP: ("union member", "trade union", "labor union"),
    SpecialCategory.SEXUAL_ORIENTATION: ("sexual orientation",),
}


# --------------------------------------------------------------------------
# Result models
# --------------------------------------------------------------------------

class PIIMatch(BaseModel):
    field_path: str
    category: PIICategory


class SpecialCategoryMatch(BaseModel):
    field_path: str
    category: SpecialCategory


class ComplianceReport(BaseModel):
    allowed: bool
    pii_matches: list[PIIMatch] = Field(default_factory=list)
    special_category_matches: list[SpecialCategoryMatch] = Field(default_factory=list)
    blocking_reason: str | None = None
    anonymized_payload: dict[str, Any] | None = None


class AuditEvent(BaseModel):
    """Safe to log/persist — contains categories and field paths only,
    never the underlying values."""

    timestamp: float
    allowed: bool
    pii_categories: list[PIICategory]
    special_categories: list[SpecialCategory]
    legal_basis: LegalBasis | None
    blocking_reason: str | None


class DataSubjectRequest(BaseModel):
    """Record of an incoming data-subject rights request (GDPR Art.
    15-22 / CCPA equivalent). This module only models the request —
    actually fulfilling access/erasure/portability requires wiring into
    your storage layer."""

    subject_ref: str  # pseudonymous reference to the user, not raw PII
    right_type: DataSubjectRightType
    requested_at: float
    notes: str | None = None


# --------------------------------------------------------------------------
# Core guard
# --------------------------------------------------------------------------

class ComplianceGuard:
    """Stateless (aside from config) — safe to instantiate once at
    startup and share across requests."""

    def __init__(self, config: ComplianceConfig | None = None):
        self.config = config or ComplianceConfig()

    # -- scanning -----------------------------------------------------------

    def scan(self, payload: dict[str, Any]) -> tuple[list[PIIMatch], list[SpecialCategoryMatch]]:
        pii_matches: list[PIIMatch] = []
        special_matches: list[SpecialCategoryMatch] = []
        self._walk(payload, path="", depth=0, pii_out=pii_matches, special_out=special_matches)
        return pii_matches, special_matches

    def _walk(
        self,
        node: Any,
        path: str,
        depth: int,
        pii_out: list[PIIMatch],
        special_out: list[SpecialCategoryMatch],
    ) -> None:
        if depth > self.config.max_scan_depth:
            return

        if isinstance(node, dict):
            for key, value in node.items():
                self._walk(value, f"{path}.{key}" if path else str(key), depth + 1, pii_out, special_out)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                self._walk(value, f"{path}[{i}]", depth + 1, pii_out, special_out)
        elif isinstance(node, str):
            self._scan_string(node, path, pii_out, special_out)

    def _scan_string(
        self,
        value: str,
        path: str,
        pii_out: list[PIIMatch],
        special_out: list[SpecialCategoryMatch],
    ) -> None:
        for category, pattern in _PATTERNS.items():
            if pattern.search(value):
                pii_out.append(PIIMatch(field_path=path, category=category))

        lowered = value.lower()
        for category, keywords in _SPECIAL_CATEGORY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                special_out.append(SpecialCategoryMatch(field_path=path, category=category))

    # -- anonymization -----------------------------------------------------

    def anonymize(
        self,
        payload: dict[str, Any],
        matches: list[PIIMatch],
        mode: AnonymizationMode | None = None,
    ) -> dict[str, Any]:
        """Return a deep-copied payload with matched fields anonymized.
        Original payload is left untouched."""
        import copy

        mode = mode or self.config.default_anonymization_mode
        result = copy.deepcopy(payload)
        for match in matches:
            self._apply_anonymization(result, match.field_path, mode)
        return result

    def _apply_anonymization(self, root: dict[str, Any], field_path: str, mode: AnonymizationMode) -> None:
        parts = re.split(r"\.|\[|\]", field_path)
        parts = [p for p in parts if p != ""]
        target: Any = root
        for part in parts[:-1]:
            key: int | str = int(part) if part.isdigit() else part
            try:
                target = target[key]
            except (KeyError, IndexError, TypeError):
                return  # path no longer valid against this copy; skip

        last = parts[-1]
        last_key: int | str = int(last) if last.isdigit() else last
        try:
            original = target[last_key]
        except (KeyError, IndexError, TypeError):
            return

        if not isinstance(original, str):
            return

        target[last_key] = self._anonymize_value(original, mode)

    def _anonymize_value(self, value: str, mode: AnonymizationMode) -> str:
        if mode is AnonymizationMode.REMOVE:
            return ""
        if mode is AnonymizationMode.HASH:
            digest = hashlib.sha256(f"{self.config.hash_salt}:{value}".encode()).hexdigest()
            return f"anon:{digest[:16]}"
        # MASK — keep a small human-readable hint, redact the rest.
        if len(value) <= 2:
            return "*" * len(value)
        return value[0] + "*" * (len(value) - 2) + value[-1]

    # -- consent / legal-basis enforcement ----------------------------------

    def evaluate(
        self,
        payload: dict[str, Any],
        legal_basis: LegalBasis | None = None,
    ) -> ComplianceReport:
        """Scan a payload and decide whether it's allowed to proceed,
        given the declared legal basis (if any) for processing it."""
        pii_matches, special_matches = self.scan(payload)

        blocking_reason: str | None = None

        found_categories = {m.category for m in pii_matches}
        requires_basis = found_categories & set(self.config.categories_requiring_basis)
        if requires_basis and legal_basis is None:
            blocking_reason = (
                f"Payload contains {sorted(c.value for c in requires_basis)} "
                "but no legal basis was declared"
            )

        if (
            special_matches
            and self.config.block_special_categories_without_basis
            and legal_basis is None
        ):
            blocking_reason = blocking_reason or (
                "Payload contains special-category data "
                f"({sorted({m.category.value for m in special_matches})}) "
                "but no legal basis was declared"
            )

        allowed = blocking_reason is None
        anonymized = self.anonymize(payload, pii_matches) if pii_matches else None

        self._audit(pii_matches, special_matches, legal_basis, allowed, blocking_reason)

        return ComplianceReport(
            allowed=allowed,
            pii_matches=pii_matches,
            special_category_matches=special_matches,
            blocking_reason=blocking_reason,
            anonymized_payload=anonymized,
        )

    # -- audit logging -------------------------------------------------------

    def _audit(
        self,
        pii_matches: list[PIIMatch],
        special_matches: list[SpecialCategoryMatch],
        legal_basis: LegalBasis | None,
        allowed: bool,
        blocking_reason: str | None,
    ) -> None:
        event = AuditEvent(
            timestamp=time.time(),
            allowed=allowed,
            pii_categories=sorted({m.category for m in pii_matches}, key=lambda c: c.value),
            special_categories=sorted({m.category for m in special_matches}, key=lambda c: c.value),
            legal_basis=legal_basis,
            blocking_reason=blocking_reason,
        )
        log_fn = logger.info if allowed else logger.warning
        log_fn("compliance_event", extra=event.model_dump(mode="json"))
