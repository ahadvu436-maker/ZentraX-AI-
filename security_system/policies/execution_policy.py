"""Execution policy: default-deny gating for actions run by users or autonomous agents.

Designed for systems where software (for example AI agents) takes actions on
behalf of a principal. Every request is evaluated against:

* an explicit allow-list of action names (default deny),
* an explicit deny-list that always wins,
* a risk ceiling with human approval above the autonomy threshold,
* sandbox limits (time, memory, network, filesystem, subprocess),
* a delegation-depth cap (prevents runaway agent-to-agent chains),
* argument screening for dangerous shell/SQL/traversal patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from fnmatch import fnmatchcase
from functools import cached_property
from typing import Any, Mapping

from .security_policy import SecurityError

__all__ = [
    "ExecutionDenied",
    "RiskLevel",
    "ExecutionRequest",
    "SandboxProfile",
    "ExecutionDecision",
    "ExecutionPolicy",
]

_ACTION_RE = re.compile(r"[a-z][a-z0-9_]{0,63}(?:\.[a-z0-9_]{1,64}){0,8}")
_MAX_ARG_DEPTH = 8


class ExecutionDenied(SecurityError):
    """Raised by :meth:`ExecutionPolicy.enforce` when a request is not allowed."""


class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True)
class ExecutionRequest:
    """Description of an action someone wants to execute."""

    action: str
    actor: str
    risk: RiskLevel = RiskLevel.LOW
    args: Mapping[str, Any] = field(default_factory=dict)
    network: bool = False
    filesystem_write: bool = False
    subprocess: bool = False
    estimated_seconds: float = 0.0
    estimated_memory_mb: int = 0
    delegation_depth: int = 0
    approved_by: str | None = None


@dataclass(frozen=True)
class SandboxProfile:
    """Resource limits the executor must enforce for an approved request."""

    max_seconds: float
    max_memory_mb: int
    allow_network: bool
    allow_filesystem_write: bool
    allow_subprocess: bool


@dataclass(frozen=True)
class ExecutionDecision:
    allowed: bool
    reasons: tuple[str, ...]
    requires_approval: bool = False
    sandbox: SandboxProfile | None = None


_DEFAULT_FORBIDDEN_ARGS: tuple[str, ...] = (
    r"(?i)\brm\s+-[a-z]*r[a-z]*f",
    r"(?i)\b(?:drop|truncate)\s+table\b",
    r"\$\(",
    r"`",
    r"(?i)\b(?:curl|wget)\b[^|\n]*\|\s*(?:ba|z)?sh\b",
    r"(?:^|[\\/])\.\.(?:[\\/]|$)",
    r"(?i)/etc/(?:passwd|shadow)",
    r"\x00",
)


@dataclass(frozen=True)
class ExecutionPolicy:
    """Immutable, extensible execution rules. See :meth:`default` for a usable baseline."""

    allowed_actions: tuple[str, ...] = ()
    denied_actions: tuple[str, ...] = ("system.*", "shell.*", "*.self_modify", "*.disable_security*")
    max_autonomous_risk: RiskLevel = RiskLevel.MEDIUM
    max_risk: RiskLevel = RiskLevel.HIGH
    max_seconds: float = 300.0
    max_memory_mb: int = 1024
    max_delegation_depth: int = 3
    allow_network: bool = False
    allow_filesystem_write: bool = False
    allow_subprocess: bool = False
    forbidden_arg_patterns: tuple[str, ...] = _DEFAULT_FORBIDDEN_ARGS
    max_arg_bytes: int = 65536

    @classmethod
    def default(cls) -> ExecutionPolicy:
        """Baseline for agent frameworks: agent/tool/read actions only, no ambient authority."""
        return cls(allowed_actions=("agent.*", "tool.*", "data.read", "data.query"))

    @cached_property
    def _compiled_args(self) -> tuple[re.Pattern[str], ...]:
        return tuple(re.compile(p) for p in self.forbidden_arg_patterns)

    # ------------------------------------------------------------ evaluation
    def evaluate(self, request: ExecutionRequest) -> ExecutionDecision:
        """Evaluate a request. Never raises for policy reasons; see :meth:`enforce`."""
        reasons: list[str] = []
        action = request.action

        if not _ACTION_RE.fullmatch(action):
            reasons.append("malformed action name")
        if any(fnmatchcase(action, pattern) for pattern in self.denied_actions):
            reasons.append(f"action '{action}' is explicitly denied")
        if not any(fnmatchcase(action, pattern) for pattern in self.allowed_actions):
            reasons.append(f"action '{action}' is not on the allow-list (default deny)")

        if request.risk > self.max_risk:
            reasons.append(f"risk {request.risk.name} exceeds the maximum permitted ({self.max_risk.name})")
        if request.network and not self.allow_network:
            reasons.append("network access is not permitted")
        if request.filesystem_write and not self.allow_filesystem_write:
            reasons.append("filesystem writes are not permitted")
        if request.subprocess and not self.allow_subprocess:
            reasons.append("subprocess execution is not permitted")
        if request.estimated_seconds > self.max_seconds:
            reasons.append("estimated runtime exceeds the limit")
        if request.estimated_memory_mb > self.max_memory_mb:
            reasons.append("estimated memory exceeds the limit")
        if request.delegation_depth > self.max_delegation_depth:
            reasons.append("delegation depth exceeds the limit")

        reasons.extend(self._screen_args(request.args))

        requires_approval = request.risk > self.max_autonomous_risk
        if requires_approval:
            if not request.approved_by:
                reasons.append("human approval required for this risk level")
            elif request.approved_by == request.actor:
                reasons.append("separation of duties: approver must differ from the actor")

        allowed = not reasons
        sandbox = (
            SandboxProfile(
                max_seconds=self.max_seconds,
                max_memory_mb=self.max_memory_mb,
                allow_network=self.allow_network and request.network,
                allow_filesystem_write=self.allow_filesystem_write and request.filesystem_write,
                allow_subprocess=self.allow_subprocess and request.subprocess,
            )
            if allowed
            else None
        )
        return ExecutionDecision(allowed, tuple(reasons), requires_approval, sandbox)

    def enforce(self, request: ExecutionRequest) -> ExecutionDecision:
        """Like :meth:`evaluate` but raises :class:`ExecutionDenied` on rejection."""
        decision = self.evaluate(request)
        if not decision.allowed:
            raise ExecutionDenied("; ".join(decision.reasons))
        return decision

    # --------------------------------------------------------------- helpers
    def _screen_args(self, args: Mapping[str, Any]) -> list[str]:
        findings: list[str] = []
        budget = [self.max_arg_bytes]

        def walk(value: Any, depth: int) -> None:
            if depth > _MAX_ARG_DEPTH:
                findings.append("arguments are nested too deeply")
                return
            if isinstance(value, str):
                budget[0] -= len(value.encode("utf-8", "replace"))
                if budget[0] < 0:
                    if "arguments exceed the size limit" not in findings:
                        findings.append("arguments exceed the size limit")
                    return
                for pattern in self._compiled_args:
                    if pattern.search(value):
                        findings.append("arguments contain a forbidden pattern")
                        return
            elif isinstance(value, Mapping):
                for item in value.values():
                    walk(item, depth + 1)
            elif isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    walk(item, depth + 1)

        walk(args, 0)
        return findings
