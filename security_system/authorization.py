"""Authorisation: RBAC with role inheritance plus attribute-based (ABAC) rules.

Evaluation order (default deny, deny overrides):

1. Any matching ``DENY`` rule  -> denied.
2. Any matching role permission or ``ALLOW`` rule -> allowed.
3. Otherwise -> denied.

Fail-closed behaviour: if a rule's condition raises, an ``ALLOW`` rule is
treated as *not matching* and a ``DENY`` rule is treated as *matching*. The same
applies to any unexpected error during evaluation.

Actions and resources use shell-style wildcards (``fnmatch``), for example
``Permission("read", "reports/*")``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from typing import Any, Callable, Iterable, Mapping

from .policies.security_policy import SecurityError

__all__ = [
    "AuthorizationError",
    "Effect",
    "Permission",
    "Role",
    "AuthorizationRequest",
    "AuthorizationDecision",
    "PolicyRule",
    "AuthorizationEngine",
]


class AuthorizationError(SecurityError):
    """Raised by :meth:`AuthorizationEngine.assert_allowed` when access is denied."""


class Effect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class Permission:
    action: str
    resource: str

    def matches(self, action: str, resource: str) -> bool:
        return fnmatchcase(action, self.action) and fnmatchcase(resource, self.resource)


@dataclass(frozen=True)
class Role:
    name: str
    permissions: frozenset[Permission] = frozenset()
    inherits: frozenset[str] = frozenset()


@dataclass(frozen=True)
class AuthorizationRequest:
    subject_id: str
    roles: frozenset[str]
    action: str
    resource: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str
    matched_rule: str | None = None


Condition = Callable[[AuthorizationRequest], bool]


@dataclass(frozen=True)
class PolicyRule:
    """An attribute-based rule. ``condition`` receives the full request."""

    rule_id: str
    effect: Effect
    action: str = "*"
    resource: str = "*"
    condition: Condition | None = None
    description: str = ""

    def targets(self, request: AuthorizationRequest) -> bool:
        return fnmatchcase(request.action, self.action) and fnmatchcase(request.resource, self.resource)


class AuthorizationEngine:
    """Thread-safe RBAC + ABAC policy decision point."""

    def __init__(self) -> None:
        self._roles: dict[str, Role] = {}
        self._rules: list[PolicyRule] = []
        self._lock = threading.RLock()

    # ---------------------------------------------------------- configuration
    def define_role(
        self,
        name: str,
        permissions: Iterable[Permission] = (),
        *,
        inherits: Iterable[str] = (),
    ) -> Role:
        role = Role(name, frozenset(permissions), frozenset(inherits))
        with self._lock:
            self._roles[name] = role
        return role

    def add_rule(self, rule: PolicyRule) -> None:
        with self._lock:
            if any(existing.rule_id == rule.rule_id for existing in self._rules):
                raise AuthorizationError(f"rule '{rule.rule_id}' already exists")
            self._rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.rule_id != rule_id]
            return len(self._rules) != before

    # --------------------------------------------------------------- decisions
    def evaluate(self, request: AuthorizationRequest) -> AuthorizationDecision:
        """Return a decision. Never raises; any internal error yields a denial."""
        try:
            with self._lock:
                rules = tuple(self._rules)
                permissions = self._effective_permissions(request.roles)

            for rule in rules:
                if rule.effect is Effect.DENY and rule.targets(request) and self._holds(rule, request, on_error=True):
                    return AuthorizationDecision(False, f"denied by rule '{rule.rule_id}'", rule.rule_id)

            for permission in permissions:
                if permission.matches(request.action, request.resource):
                    return AuthorizationDecision(True, "allowed by role permission", f"{permission.action}:{permission.resource}")

            for rule in rules:
                if rule.effect is Effect.ALLOW and rule.targets(request) and self._holds(rule, request, on_error=False):
                    return AuthorizationDecision(True, f"allowed by rule '{rule.rule_id}'", rule.rule_id)

            return AuthorizationDecision(False, "no matching permission (default deny)")
        except Exception:  # noqa: BLE001 - a security gate must fail closed on *any* error
            return AuthorizationDecision(False, "evaluation error (fail closed)")

    def is_allowed(self, request: AuthorizationRequest) -> bool:
        return self.evaluate(request).allowed

    def assert_allowed(self, request: AuthorizationRequest) -> AuthorizationDecision:
        decision = self.evaluate(request)
        if not decision.allowed:
            raise AuthorizationError(decision.reason)
        return decision

    # --------------------------------------------------------------- internals
    @staticmethod
    def _holds(rule: PolicyRule, request: AuthorizationRequest, *, on_error: bool) -> bool:
        if rule.condition is None:
            return True
        try:
            return bool(rule.condition(request))
        except Exception:  # noqa: BLE001 - faulty conditions must not open access
            return on_error

    def _effective_permissions(self, role_names: Iterable[str]) -> frozenset[Permission]:
        """Resolve inheritance iteratively; cycles are harmless because of the ``seen`` set."""
        seen: set[str] = set()
        stack = list(role_names)
        collected: set[Permission] = set()
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            role = self._roles.get(name)
            if role is None:  # unknown roles grant nothing
                continue
            collected |= role.permissions
            stack.extend(role.inherits)
        return frozenset(collected)
