"""Shared, deliberately conservative primitives for the ZentraX security baseline."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
import hashlib, json, logging, os, secrets

logger = logging.getLogger("zentrax.zero_trust")

class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"

@dataclass(frozen=True)
class SecurityContext:
    subject: str
    action: str
    resource: str
    roles: tuple[str, ...] = ()
    device_id: str | None = None
    session_id: str | None = None
    mfa_verified: bool = False
    risk_score: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    reason: str
    policy_id: str = "default-deny"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AuditLogger:
    """Append-only JSONL audit writer for local development; harden storage in deployment."""
    def __init__(self, path: str = "var/audit.jsonl"):
        self.path = path
    def record(self, event: Mapping[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line = json.dumps(payload, sort_keys=True, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\\n")

class PolicyEngine:
    """Default-deny policy evaluator. Unknown actions/resources are never implicitly allowed."""
    def __init__(self, audit: AuditLogger | None = None):
        self.audit = audit or AuditLogger()
    def evaluate(self, ctx: SecurityContext) -> DecisionResult:
        if not ctx.subject or not ctx.action or not ctx.resource:
            result = DecisionResult(Decision.DENY, "missing security context")
        elif ctx.risk_score < 0 or ctx.risk_score > 1:
            result = DecisionResult(Decision.DENY, "risk score must be between 0 and 1")
        elif ctx.risk_score >= 0.8:
            result = DecisionResult(Decision.DENY, "high risk: denied by default")
        elif ctx.risk_score >= 0.5 or ctx.action in {"delete", "admin", "export_all", "rotate_root_key"}:
            result = DecisionResult(Decision.REQUIRE_APPROVAL, "sensitive action or elevated risk")
        elif ctx.action in {"read", "write", "execute", "invoke"} and "user" in ctx.roles:
            result = DecisionResult(Decision.ALLOW, "recognized user role and low risk", "baseline-user-policy")
        else:
            result = DecisionResult(Decision.DENY, "no matching allow policy")
        self.audit.record({"subject":ctx.subject, "action":ctx.action, "resource":ctx.resource,
                           "decision":result.decision.value, "reason":result.reason})
        return result

def stable_fingerprint(value: str) -> str:
    """Non-secret identifier fingerprint; do not use as an authentication credential."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def new_correlation_id() -> str:
    return secrets.token_urlsafe(16)
