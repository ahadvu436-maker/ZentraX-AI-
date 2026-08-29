"""
ZentraX AI — Security Utilities
==================================
Password hashing, JWT access/refresh token management, and symmetric
encryption for data-at-rest — the cryptographic core of the privacy-first
platform.

Three independent concerns live here, each with its own key material:

1. Password hashing/verification         -> passlib (bcrypt)
2. JWT access & refresh token lifecycle   -> python-jose
3. Symmetric encryption of sensitive PII  -> cryptography.fernet

None of these primitives should be used outside this module — always go
through the functions below so hashing rounds, algorithms, and claim
structures stay consistent across the codebase.
"""

from __future__ import annotations

import base64
import binascii
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Final

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.exc import UnknownHashError
from pydantic import BaseModel, ConfigDict

from app.config.settings import settings

logger = logging.getLogger(__name__)


# ========================================================================
# Exceptions
# ========================================================================
class SecurityError(Exception):
    """Base class for all security-module errors."""


class TokenError(SecurityError):
    """Raised when a JWT is missing, malformed, expired, or otherwise invalid."""


class TokenExpiredError(TokenError):
    """Raised specifically when a JWT has expired."""


class EncryptionError(SecurityError):
    """Raised when symmetric encryption fails."""


class DecryptionError(SecurityError):
    """Raised when symmetric decryption fails (bad token, wrong key, tampering)."""


# ========================================================================
# 1. Password hashing & verification (passlib / bcrypt)
# ========================================================================
# bcrypt has a hard 72-byte input limit; passlib's bcrypt backend truncates
# silently in older configs, so we truncate deliberately and consistently.
_BCRYPT_MAX_BYTES: Final[int] = 72

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # cost factor; raise if hardware allows, re-tune periodically
)


def _truncate_for_bcrypt(password: str) -> str:
    """
    Truncate a password to bcrypt's 72-byte limit on UTF-8 encoded length,
    without splitting a multi-byte character.
    """
    encoded = password.encode("utf-8")
    if len(encoded) <= _BCRYPT_MAX_BYTES:
        return password
    truncated = encoded[:_BCRYPT_MAX_BYTES]
    # Back off until we land on a valid UTF-8 boundary.
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return ""  # pragma: no cover - unreachable for any non-empty input


def hash_password(plain_password: str) -> str:
    """
    Hash a plaintext password for storage.

    Raises:
        SecurityError: if `plain_password` is empty.
    """
    if not plain_password:
        raise SecurityError("Cannot hash an empty password.")
    return pwd_context.hash(_truncate_for_bcrypt(plain_password))


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.

    Returns False (never raises) for malformed/unknown hash formats, so
    callers can treat this as a simple boolean auth check.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return pwd_context.verify(_truncate_for_bcrypt(plain_password), hashed_password)
    except UnknownHashError:
        logger.warning("Password verification attempted against an unrecognized hash format.")
        return False


def needs_rehash(hashed_password: str) -> bool:
    """
    True if the stored hash was created with outdated parameters (e.g. a
    lower cost factor than current config) and should be re-hashed on next
    successful login.
    """
    try:
        return pwd_context.needs_update(hashed_password)
    except UnknownHashError:
        return True


# ========================================================================
# 2. JWT access & refresh tokens (python-jose)
# ========================================================================
class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


class TokenPayload(BaseModel):
    """Decoded, validated claims from a ZentraX JWT."""

    model_config = ConfigDict(frozen=True)

    sub: str  # subject — user id (string form of UUID)
    type: TokenType
    jti: str  # unique token id, enables server-side revocation/blacklisting
    iat: datetime
    exp: datetime


def _create_token(
    *,
    subject: uuid.UUID | str,
    token_type: TokenType,
    expires_delta: timedelta,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type.value,
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": now + expires_delta,
    }
    if extra_claims:
        # Never allow caller-supplied claims to override core identity/type claims.
        safe_extra = {
            k: v for k, v in extra_claims.items() if k not in payload
        }
        payload.update(safe_extra)

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY.get_secret_value(),
        algorithm=settings.JWT_ALGORITHM,
    )


def create_access_token(
    subject: uuid.UUID | str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a short-lived access token for API authorization."""
    return _create_token(
        subject=subject,
        token_type=TokenType.ACCESS,
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        extra_claims=extra_claims,
    )


def create_refresh_token(
    subject: uuid.UUID | str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a long-lived refresh token used solely to mint new access tokens."""
    return _create_token(
        subject=subject,
        token_type=TokenType.REFRESH,
        expires_delta=timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
        extra_claims=extra_claims,
    )


def decode_token(token: str, *, expected_type: TokenType | None = None) -> TokenPayload:
    """
    Decode and validate a JWT, returning its typed claims.

    Args:
        token: the encoded JWT string.
        expected_type: if given, raises TokenError when the token's `type`
            claim doesn't match (e.g. a refresh token presented where an
            access token is required).

    Raises:
        TokenExpiredError: the token's `exp` claim is in the past.
        TokenError: the token is malformed, has an invalid signature, or
            fails the `expected_type` check.
    """
    if not token:
        raise TokenError("Token is empty.")

    try:
        raw_claims = jwt.decode(
            token,
            settings.JWT_SECRET_KEY.get_secret_value(),
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("Token has expired.") from exc
    except JWTError as exc:
        raise TokenError(f"Token validation failed: {exc}") from exc

    try:
        payload = TokenPayload.model_validate(raw_claims)
    except Exception as exc:  # pydantic ValidationError
        raise TokenError(f"Token payload is malformed: {exc}") from exc

    if expected_type is not None and payload.type != expected_type:
        raise TokenError(
            f"Expected a '{expected_type.value}' token but received "
            f"'{payload.type.value}'."
        )

    return payload


# ========================================================================
# 3. Symmetric encryption for data at rest (cryptography.fernet)
# ========================================================================
# `MultiFernet` supports zero-downtime key rotation: put the new key first
# in KEY ROTATION order below and keep old keys for decrypting existing
# ciphertext until it's been re-encrypted. Only ONE key is configured by
# default (from settings); extend `_load_fernet_keys` if/when you rotate.


def _load_fernet_keys() -> list[Fernet]:
    """
    Build the list of Fernet instances used for encryption/decryption.

    The primary key comes from `settings.DATA_ENCRYPTION_KEY` and MUST be a
    valid 32-byte, URL-safe, base64-encoded key (i.e. `Fernet.generate_key()`
    output). This is validated eagerly so a misconfigured key fails at
    import time, not on the first encryption call in production.
    """
    raw_key = settings.DATA_ENCRYPTION_KEY.get_secret_value().encode("utf-8")
    try:
        # Validates base64 encoding and 32-byte length as a side effect.
        base64.urlsafe_b64decode(raw_key)
    except (binascii.Error, ValueError) as exc:
        raise SecurityError(
            "DATA_ENCRYPTION_KEY is not a valid Fernet key. Generate one with "
            "`cryptography.fernet.Fernet.generate_key()` and set it verbatim "
            "as the DATA_ENCRYPTION_KEY environment variable."
        ) from exc

    try:
        primary = Fernet(raw_key)
    except (ValueError, TypeError) as exc:
        raise SecurityError(f"Failed to initialize Fernet with configured key: {exc}") from exc

    # Add additional legacy keys here (oldest last) during key rotation, e.g.:
    # return [primary, Fernet(settings.DATA_ENCRYPTION_KEY_PREVIOUS.get_secret_value())]
    return [primary]


_fernet = MultiFernet(_load_fernet_keys())


def encrypt_data(plaintext: str) -> str:
    """
    Encrypt a plaintext string for storage (e.g. a sensitive user field).

    Returns a URL-safe base64 token (str) suitable for storing directly in
    a `Text`/`String` database column.

    Raises:
        EncryptionError: on any encryption failure.
    """
    if plaintext is None:
        raise EncryptionError("Cannot encrypt None.")
    try:
        token = _fernet.encrypt(plaintext.encode("utf-8"))
        return token.decode("utf-8")
    except Exception as exc:
        logger.error("Encryption failed.", exc_info=True)
        raise EncryptionError("Failed to encrypt data.") from exc


def decrypt_data(ciphertext: str, *, ttl_seconds: int | None = None) -> str:
    """
    Decrypt a value previously produced by `encrypt_data`.

    Args:
        ciphertext: the stored Fernet token.
        ttl_seconds: optional max age enforcement; omit for data-at-rest
            fields with no natural expiry (the common case for PII columns).

    Raises:
        DecryptionError: token is invalid, tampered with, encrypted under an
            unknown key, or (if `ttl_seconds` is set) has expired.
    """
    if not ciphertext:
        raise DecryptionError("Cannot decrypt an empty value.")
    try:
        plaintext = _fernet.decrypt(ciphertext.encode("utf-8"), ttl=ttl_seconds)
        return plaintext.decode("utf-8")
    except InvalidToken as exc:
        logger.warning("Decryption failed: invalid token, wrong key, or tampering detected.")
        raise DecryptionError(
            "Failed to decrypt data: invalid token, key mismatch, or the "
            "data has been tampered with."
        ) from exc
    except Exception as exc:
        logger.error("Decryption failed with an unexpected error.", exc_info=True)
        raise DecryptionError("Failed to decrypt data.") from exc


def generate_fernet_key() -> str:
    """
    Generate a new, valid Fernet key for use as `DATA_ENCRYPTION_KEY`.

    This is an operator/setup utility (e.g. run once via a management
    script when provisioning a new environment) — it is NOT called during
    normal application operation.
    """
    return Fernet.generate_key().decode("utf-8")