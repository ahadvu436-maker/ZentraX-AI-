"""Security policy definitions.

This module is the dependency-free base layer of ``Security_system``. Every
other module may import from :mod:`Security_system.policies`, but the policies
never import from the rest of the package. That keeps the dependency graph
acyclic and lets policies be loaded, validated and audited on their own.

All policy objects are immutable (``frozen=True``) and validate themselves on
construction, so a misconfigured policy fails at start-up rather than silently
weakening protection at run time.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

__all__ = [
    "SecurityError",
    "PolicyViolation",
    "PasswordPolicy",
    "LockoutPolicy",
    "SessionPolicy",
    "CryptoPolicy",
    "RateLimitPolicy",
    "ZeroTrustPolicy",
    "SecurityPolicy",
]


class SecurityError(Exception):
    """Base class for every error raised by ``Security_system``."""


class PolicyViolation(SecurityError):
    """Raised when a policy is malformed or an action violates a policy."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyViolation(message)


@dataclass(frozen=True)
class PasswordPolicy:
    """Password complexity and ageing rules (aligned with NIST SP 800-63B).

    Length is favoured over composition rules, but composition checks remain
    available because many enterprise standards still mandate them.
    """

    min_length: int = 12
    max_length: int = 128
    require_upper: bool = True
    require_lower: bool = True
    require_digit: bool = True
    require_symbol: bool = True
    max_repeated_chars: int = 3
    forbidden_substrings: tuple[str, ...] = (
        "password",
        "qwerty",
        "123456",
        "letmein",
        "admin",
        "welcome",
    )
    max_age_days: int = 90

    def __post_init__(self) -> None:
        _require(self.min_length >= 8, "password min_length must be >= 8")
        _require(self.max_length >= self.min_length, "password max_length < min_length")
        _require(self.max_repeated_chars >= 1, "max_repeated_chars must be >= 1")
        _require(self.max_age_days >= 1, "max_age_days must be >= 1")

    def validate(self, password: str, *, username: str | None = None) -> list[str]:
        """Return a list of human-readable problems (empty list means valid)."""
        if not isinstance(password, str):
            return ["password must be a string"]
        problems: list[str] = []
        if len(password) < self.min_length:
            problems.append(f"must be at least {self.min_length} characters")
        if len(password) > self.max_length:
            problems.append(f"must be at most {self.max_length} characters")
        if self.require_upper and not any(c.isupper() for c in password):
            problems.append("must contain an uppercase letter")
        if self.require_lower and not any(c.islower() for c in password):
            problems.append("must contain a lowercase letter")
        if self.require_digit and not any(c.isdigit() for c in password):
            problems.append("must contain a digit")
        if self.require_symbol and not any(not c.isalnum() and not c.isspace() for c in password):
            problems.append("must contain a symbol")
        if re.search(rf"(.)\1{{{self.max_repeated_chars},}}", password):
            problems.append(f"must not repeat a character more than {self.max_repeated_chars} times in a row")
        lowered = password.lower()
        if any(bad and bad.lower() in lowered for bad in self.forbidden_substrings):
            problems.append("contains a commonly used pattern")
        if username and len(username) >= 3 and username.lower() in lowered:
            problems.append("must not contain the username")
        return problems

    def assert_valid(self, password: str, *, username: str | None = None) -> None:
        """Raise :class:`PolicyViolation` if ``password`` breaks the policy."""
        problems = self.validate(password, username=username)
        if problems:
            raise PolicyViolation("Password policy violation: " + "; ".join(problems))


@dataclass(frozen=True)
class LockoutPolicy:
    """Account lockout (online brute-force protection)."""

    max_failed_attempts: int = 5
    failure_window_seconds: int = 900
    lockout_seconds: int = 900

    def __post_init__(self) -> None:
        _require(self.max_failed_attempts >= 1, "max_failed_attempts must be >= 1")
        _require(self.failure_window_seconds >= 1, "failure_window_seconds must be >= 1")
        _require(self.lockout_seconds >= 1, "lockout_seconds must be >= 1")


@dataclass(frozen=True)
class SessionPolicy:
    """Session and token lifetime rules. Short-lived, revocable, device-bound."""

    access_token_ttl_seconds: int = 900
    idle_timeout_seconds: int = 1800
    absolute_lifetime_seconds: int = 28800
    max_concurrent_sessions: int = 5
    require_mfa: bool = True
    max_clock_skew_seconds: int = 30

    def __post_init__(self) -> None:
        _require(self.access_token_ttl_seconds >= 1, "access_token_ttl_seconds must be >= 1")
        _require(
            self.access_token_ttl_seconds <= self.absolute_lifetime_seconds,
            "access_token_ttl_seconds must not exceed absolute_lifetime_seconds",
        )
        _require(self.idle_timeout_seconds >= 1, "idle_timeout_seconds must be >= 1")
        _require(self.max_concurrent_sessions >= 1, "max_concurrent_sessions must be >= 1")
        _require(self.max_clock_skew_seconds >= 0, "max_clock_skew_seconds must be >= 0")


@dataclass(frozen=True)
class CryptoPolicy:
    """Approved algorithms, key sizes and KDF cost parameters."""

    allowed_ciphers: tuple[str, ...] = ("AES-256-GCM", "ChaCha20-Poly1305")
    key_size_bits: int = 256
    key_rotation_days: int = 90
    scrypt_n: int = 2**15
    scrypt_r: int = 8
    scrypt_p: int = 1

    def __post_init__(self) -> None:
        _require(self.key_size_bits == 256, "only 256-bit symmetric keys are permitted")
        _require(len(self.allowed_ciphers) > 0, "allowed_ciphers must not be empty")
        _require(self.key_rotation_days >= 1, "key_rotation_days must be >= 1")
        _require(
            self.scrypt_n >= 2**14 and (self.scrypt_n & (self.scrypt_n - 1)) == 0,
            "scrypt_n must be a power of two >= 2**14",
        )
        _require(self.scrypt_r >= 8 and self.scrypt_p >= 1, "scrypt_r must be >= 8 and scrypt_p >= 1")


@dataclass(frozen=True)
class RateLimitPolicy:
    """Thresholds consumed by the threat detector."""

    window_seconds: int = 60
    max_requests: int = 120
    brute_force_threshold: int = 10
    password_spray_distinct_users: int = 5
    authz_denial_threshold: int = 10
    auto_block_seconds: int = 900

    def __post_init__(self) -> None:
        for name in (
            "window_seconds",
            "max_requests",
            "brute_force_threshold",
            "password_spray_distinct_users",
            "authz_denial_threshold",
            "auto_block_seconds",
        ):
            _require(getattr(self, name) >= 1, f"{name} must be >= 1")


@dataclass(frozen=True)
class ZeroTrustPolicy:
    """Contextual access requirements. Levels are ``DataClassification`` values.

    ``0`` = PUBLIC, ``1`` = INTERNAL, ``2`` = CONFIDENTIAL, ``3`` = RESTRICTED.
    Anything at or above the stated level triggers the control.
    """

    trusted_networks: tuple[str, ...] = ()
    mfa_required_from_level: int = 2
    trusted_device_required_from_level: int = 2
    network_restricted_from_level: int = 2
    default_resource_level: int = 2
    risk_step_up_threshold: float = 0.5
    risk_deny_threshold: float = 0.8

    def __post_init__(self) -> None:
        for name in (
            "mfa_required_from_level",
            "trusted_device_required_from_level",
            "network_restricted_from_level",
            "default_resource_level",
        ):
            _require(0 <= getattr(self, name) <= 3, f"{name} must be between 0 and 3")
        _require(
            0.0 < self.risk_step_up_threshold <= self.risk_deny_threshold <= 1.0,
            "risk thresholds must satisfy 0 < step_up <= deny <= 1",
        )


def _coerce(value: Any) -> Any:
    """JSON gives lists; policies use tuples so they stay hashable and frozen."""
    return tuple(value) if isinstance(value, list) else value


@dataclass(frozen=True)
class SecurityPolicy:
    """Aggregate root of all security controls.

    Build from code, from a mapping (for example parsed JSON/YAML), or use the
    :meth:`strict` preset. Unknown keys are rejected so typos cannot silently
    disable a control.
    """

    password: PasswordPolicy = field(default_factory=PasswordPolicy)
    lockout: LockoutPolicy = field(default_factory=LockoutPolicy)
    session: SessionPolicy = field(default_factory=SessionPolicy)
    crypto: CryptoPolicy = field(default_factory=CryptoPolicy)
    rate_limit: RateLimitPolicy = field(default_factory=RateLimitPolicy)
    zero_trust: ZeroTrustPolicy = field(default_factory=ZeroTrustPolicy)
    audit_required: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SecurityPolicy:
        """Construct a policy from nested plain data, rejecting unknown keys."""
        nested: dict[str, type[Any]] = {
            "password": PasswordPolicy,
            "lockout": LockoutPolicy,
            "session": SessionPolicy,
            "crypto": CryptoPolicy,
            "rate_limit": RateLimitPolicy,
            "zero_trust": ZeroTrustPolicy,
        }
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise PolicyViolation(f"unknown policy keys: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in nested and isinstance(value, Mapping):
                sub_cls = nested[key]
                bad = set(value) - {f.name for f in fields(sub_cls)}
                if bad:
                    raise PolicyViolation(f"unknown keys in '{key}': {sorted(bad)}")
                kwargs[key] = sub_cls(**{k: _coerce(v) for k, v in value.items()})
            else:
                kwargs[key] = _coerce(value)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data (round-trips through :meth:`from_mapping`)."""
        return asdict(self)

    @classmethod
    def strict(cls) -> SecurityPolicy:
        """High-assurance preset: longer passwords, tighter lockout and sessions."""
        return cls(
            password=PasswordPolicy(min_length=16, max_age_days=60),
            lockout=LockoutPolicy(max_failed_attempts=3),
            session=SessionPolicy(
                access_token_ttl_seconds=600,
                idle_timeout_seconds=900,
                absolute_lifetime_seconds=14400,
                max_concurrent_sessions=2,
            ),
        )
