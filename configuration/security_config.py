"""Security, encryption and token-lifetime configuration."""

from __future__ import annotations

import base64
import binascii
import secrets
from datetime import timedelta
from typing import ClassVar

from pydantic import Field, SecretStr, field_validator, model_validator

from .constants import (
    DEFAULT_JWT_AUDIENCE,
    DEFAULT_JWT_ISSUER,
    ENV_PREFIX_SECURITY,
    FERNET_KEY_BYTES,
    MIN_SECRET_KEY_LENGTH,
    MIN_SECRET_KEY_UNIQUE_CHARS,
    WEAK_SECRET_MARKERS,
    JWTAlgorithm,
    PasswordHashScheme,
    SameSite,
)
from .environment import EnvBaseModel


def generate_secret_key(nbytes: int = 48) -> str:
    """Cryptographically strong URL-safe secret (>= 64 chars by default)."""
    return secrets.token_urlsafe(nbytes)


def generate_encryption_key() -> str:
    """A fresh Fernet-compatible key (32 random bytes, url-safe base64)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(FERNET_KEY_BYTES)).decode("ascii")


class SecuritySettings(EnvBaseModel):
    """Security options (env prefix ``SECURITY_``; e.g. ``SECURITY_SECRET_KEY``)."""

    env_prefix: ClassVar[str] = ENV_PREFIX_SECURITY

    # Keys (``secret_key`` is required: there is deliberately no default)
    secret_key: SecretStr
    encryption_key: SecretStr | None = None

    # Tokens
    jwt_algorithm: JWTAlgorithm = JWTAlgorithm.HS256
    jwt_issuer: str = Field(default=DEFAULT_JWT_ISSUER, min_length=1, max_length=255)
    jwt_audience: str = Field(default=DEFAULT_JWT_AUDIENCE, min_length=1, max_length=255)
    access_token_ttl_minutes: int = Field(default=15, ge=1, le=1_440)
    refresh_token_ttl_days: int = Field(default=7, ge=1, le=90)
    password_reset_ttl_minutes: int = Field(default=30, ge=5, le=1_440)

    # Password hashing
    password_hash_scheme: PasswordHashScheme = PasswordHashScheme.ARGON2ID
    argon2_time_cost: int = Field(default=3, ge=1, le=10)
    argon2_memory_cost_kib: int = Field(default=65_536, ge=8_192, le=1_048_576)
    argon2_parallelism: int = Field(default=4, ge=1, le=16)
    bcrypt_rounds: int = Field(default=12, ge=10, le=16)
    password_min_length: int = Field(default=12, ge=8, le=128)

    # Abuse protection
    max_login_attempts: int = Field(default=5, ge=1, le=100)
    lockout_minutes: int = Field(default=15, ge=1, le=1_440)
    rate_limit_per_minute: int = Field(default=120, ge=1, le=100_000)

    # Transport / browser hardening
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    cors_origins: tuple[str, ...] = ()
    cookie_secure: bool = True
    cookie_samesite: SameSite = SameSite.LAX
    tls_required: bool = True
    hsts_max_age_seconds: int = Field(default=31_536_000, ge=0, le=63_072_000)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @field_validator("secret_key")
    @classmethod
    def _validate_secret_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value()
        if len(key) < MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                f"secret key must be at least {MIN_SECRET_KEY_LENGTH} characters "
                "(generate one with generate_secret_key())"
            )
        lowered = key.lower()
        if any(marker in lowered for marker in WEAK_SECRET_MARKERS):
            raise ValueError("secret key looks like a placeholder or guessable value")
        if len(set(key)) < MIN_SECRET_KEY_UNIQUE_CHARS:
            raise ValueError("secret key has too little entropy (too many repeated characters)")
        return value

    @field_validator("encryption_key")
    @classmethod
    def _validate_encryption_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        key = value.get_secret_value().strip()
        try:
            decoded = base64.urlsafe_b64decode(key + "=" * (-len(key) % 4))
        except (binascii.Error, ValueError):
            raise ValueError("encryption key must be url-safe base64") from None
        if len(decoded) != FERNET_KEY_BYTES:
            raise ValueError(
                f"encryption key must decode to {FERNET_KEY_BYTES} bytes "
                "(generate one with generate_encryption_key())"
            )
        return SecretStr(key)

    @field_validator("allowed_hosts")
    @classmethod
    def _validate_hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("allowed_hosts must contain at least one host")
        return value

    @field_validator("cors_origins")
    @classmethod
    def _validate_origins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for origin in value:
            if origin != "*" and not origin.startswith(("http://", "https://")):
                raise ValueError(f"invalid CORS origin {origin!r}; use http(s)://host or *")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> SecuritySettings:
        if self.refresh_token_ttl <= self.access_token_ttl:
            raise ValueError("refresh token lifetime must exceed the access token lifetime")
        if self.cookie_samesite is SameSite.NONE and not self.cookie_secure:
            raise ValueError("SameSite=None requires cookie_secure=true")
        return self

    # ------------------------------------------------------------------ #
    # Derived values
    # ------------------------------------------------------------------ #
    @property
    def access_token_ttl(self) -> timedelta:
        return timedelta(minutes=self.access_token_ttl_minutes)

    @property
    def refresh_token_ttl(self) -> timedelta:
        return timedelta(days=self.refresh_token_ttl_days)

    @property
    def password_reset_ttl(self) -> timedelta:
        return timedelta(minutes=self.password_reset_ttl_minutes)

    @property
    def lockout_duration(self) -> timedelta:
        return timedelta(minutes=self.lockout_minutes)

    @property
    def allows_any_origin(self) -> bool:
        return "*" in self.cors_origins

    @property
    def allows_any_host(self) -> bool:
        return "*" in self.allowed_hosts

    def signing_key(self) -> str:
        """Reveal the JWT signing key. Never log the return value."""
        return self.secret_key.get_secret_value()
