"""Contextual access control: the Zero-Trust policy enforcement point.

:class:`AccessController` sits in front of the :class:`AuthorizationEngine` and
adds everything that is *not* about identity alone:

* **Data classification** - resources carry a sensitivity level.
* **Clearance** - a subject may not access data above their clearance
  ("no read up", Bell-LaPadula style).
* **Device trust** - devices are registered per principal server-side. The
  caller cannot assert that a device is trusted.
* **Network trust** - sensitive data only from approved CIDR ranges.
* **Strong authentication** - sensitive data requires a session that passed MFA.
* **Risk score** - high risk is denied; medium risk demands step-up auth.
* **ACLs** - per-resource grants that complement role permissions.

Every control that applies is evaluated and *all* failure reasons are returned,
which makes audit records and incident review far more useful.
"""

from __future__ import annotations

import ipaddress
import threading
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Mapping

from .authorization import AuthorizationEngine, AuthorizationRequest
from .policies.privacy_policy import DataClassification
from .policies.security_policy import SecurityError, SecurityPolicy

__all__ = [
    "AccessDeniedError",
    "AccessContext",
    "AccessRequest",
    "AccessDecision",
    "ACLEntry",
    "AccessController",
]


class AccessDeniedError(SecurityError):
    """Raised when access is denied. Messages stay generic; details go to the audit log."""


@dataclass(frozen=True)
class AccessContext:
    """Untrusted-by-default facts about the current request."""

    ip: str | None = None
    device_id: str | None = None
    mfa_verified: bool = False
    risk_score: float = 0.0
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccessRequest:
    principal_id: str
    roles: frozenset[str]
    action: str
    resource: str
    context: AccessContext = field(default_factory=AccessContext)


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reasons: tuple[str, ...]
    classification: DataClassification
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ACLEntry:
    resource: str  # fnmatch pattern
    principal_id: str
    action: str  # fnmatch pattern


class AccessController:
    """Thread-safe policy enforcement point combining context checks and authorisation."""

    def __init__(self, policy: SecurityPolicy, authorization: AuthorizationEngine) -> None:
        self._policy = policy
        self._authz = authorization
        self._lock = threading.RLock()
        self._clearances: dict[str, DataClassification] = {}
        self._resource_classes: list[tuple[str, DataClassification]] = []
        self._acl: list[ACLEntry] = []
        self._devices: dict[str, set[str]] = {}
        self._networks = tuple(ipaddress.ip_network(n, strict=False) for n in policy.zero_trust.trusted_networks)

    # ---------------------------------------------------------- configuration
    def set_clearance(self, principal_id: str, level: DataClassification) -> None:
        with self._lock:
            self._clearances[principal_id] = level

    def clearance_of(self, principal_id: str) -> DataClassification:
        with self._lock:
            return self._clearances.get(principal_id, DataClassification.PUBLIC)

    def classify_resource(self, pattern: str, level: DataClassification) -> None:
        with self._lock:
            self._resource_classes.append((pattern, level))

    def classification_of(self, resource: str) -> DataClassification:
        """Highest level among matching patterns; unknown resources get the policy default."""
        with self._lock:
            matches = [lvl for pat, lvl in self._resource_classes if fnmatchcase(resource, pat)]
        if matches:
            return max(matches)
        return DataClassification(self._policy.zero_trust.default_resource_level)

    def register_device(self, principal_id: str, device_id: str) -> None:
        with self._lock:
            self._devices.setdefault(principal_id, set()).add(device_id)

    def revoke_device(self, principal_id: str, device_id: str) -> None:
        with self._lock:
            self._devices.get(principal_id, set()).discard(device_id)

    def is_device_trusted(self, principal_id: str, device_id: str | None) -> bool:
        if not device_id:
            return False
        with self._lock:
            return device_id in self._devices.get(principal_id, set())

    def grant(self, resource: str, principal_id: str, action: str = "read") -> None:
        with self._lock:
            self._acl.append(ACLEntry(resource, principal_id, action))

    def revoke(self, resource: str, principal_id: str, action: str = "read") -> None:
        with self._lock:
            self._acl = [e for e in self._acl if (e.resource, e.principal_id, e.action) != (resource, principal_id, action)]

    # --------------------------------------------------------------- decisions
    def evaluate(self, request: AccessRequest) -> AccessDecision:
        """Evaluate every applicable control; allowed only if none objects."""
        classification = self.classification_of(request.resource)
        level = int(classification)
        zt = self._policy.zero_trust
        ctx = request.context
        reasons: list[str] = []
        obligations: list[str] = []

        if self._networks and level >= zt.network_restricted_from_level and not self._ip_trusted(ctx.ip):
            reasons.append("source network is not trusted for this data classification")
        if level >= zt.trusted_device_required_from_level and not self.is_device_trusted(
            request.principal_id, ctx.device_id
        ):
            reasons.append("device is not registered as trusted")
        if level >= zt.mfa_required_from_level and not ctx.mfa_verified:
            reasons.append("multi-factor authentication required")
            obligations.append("step_up_authentication")
        if self.clearance_of(request.principal_id) < classification:
            reasons.append("insufficient clearance for this data classification")
        if ctx.risk_score >= zt.risk_deny_threshold:
            reasons.append("request risk score exceeds the deny threshold")
        elif ctx.risk_score >= zt.risk_step_up_threshold:
            reasons.append("elevated risk requires step-up authentication")
            if "step_up_authentication" not in obligations:
                obligations.append("step_up_authentication")

        authz = self._authz.evaluate(
            AuthorizationRequest(
                subject_id=request.principal_id,
                roles=request.roles,
                action=request.action,
                resource=request.resource,
                attributes={
                    **ctx.attributes,
                    "classification": level,
                    "ip": ctx.ip,
                    "device_id": ctx.device_id,
                    "mfa_verified": ctx.mfa_verified,
                    "risk_score": ctx.risk_score,
                },
            )
        )
        if not authz.allowed and not self._acl_allows(request):
            reasons.append(authz.reason)

        return AccessDecision(not reasons, tuple(reasons) or ("all controls satisfied",), classification, tuple(obligations))

    def assert_allowed(self, request: AccessRequest) -> AccessDecision:
        decision = self.evaluate(request)
        if not decision.allowed:
            raise AccessDeniedError("Access denied")
        return decision

    # --------------------------------------------------------------- internals
    def _acl_allows(self, request: AccessRequest) -> bool:
        with self._lock:
            entries = tuple(self._acl)
        return any(
            e.principal_id == request.principal_id
            and fnmatchcase(request.resource, e.resource)
            and fnmatchcase(request.action, e.action)
            for e in entries
        )

    def _ip_trusted(self, ip: str | None) -> bool:
        if not ip:
            return False
        try:
            address = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(address in network for network in self._networks)
