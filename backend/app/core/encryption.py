"""
backend/app/core/encryption.py

Encryption, password hashing, and secure token utilities for ZentraX AI.

Provides:
  - `SymmetricEncryptor`: Fernet-based authenticated encryption for data at
    rest (e.g. encrypting sensitive fields before writing to the database).
  - `PasswordHasher`: Argon2id-based password hashing/verification, with
    automatic rehash detection when parameters change.
  - Secure random token / API key generation helpers.
  - Constant-time comparison helper for anything else that needs it.

Key management:
  - The master key is read from the `ZENTRAX_ENCRYPTION_KEY` environment
    variable (or injected explicitly) and MUST be a urlsafe-base64-encoded
    32-byte key, e.g. generated once via `SymmetricEncryptor.generate_key()`
    and stored in a secrets manager (Vault, AWS Secrets Manager, etc.) —
    never committed to source control.
  - `MultiFernet` support is included so keys can be rotated without
    invalidating previously-encrypted data: new key first, old key(s) kept
    for decryption only until data is re-encrypted.

Dependencies: `cryptography`, `argon2-cffi`
    pip install cryptography argon2-cffi
"""

from __future__ import annotations

import base64
import hmac
import logging
import os
import secrets
from typing import Optional

from argon2 import PasswordHasher as _Argon2Hasher
from argon2.exceptions import InvalidHash, VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

logger = logging.getLogger("zentrax.encryption")


class EncryptionError(Exception):
    """Raised when encryption/decryption fails or is misconfigured."""


class PasswordVerificationError(Exception):
    """Raised when a password fails verification against a stored hash."""


# --------------------------------------------------------------------------- #
# Symmetric encryption (Fernet)
# --------------------------------------------------------------------------- #

class SymmetricEncryptor:
    """
    Authenticated symmetric encryption for sensitive data at rest.

    Usage:
        # One-time key generation (store the result securely, e.g. in a
        # secrets manager or the ZENTRAX_ENCRYPTION_KEY env var):
        key = SymmetricEncryptor.generate_key()

        encryptor = SymmetricEncryptor()  # reads ZENTRAX_ENCRYPTION_KEY by default
        token = encryptor.encrypt("sensitive value")
        plaintext = encryptor.decrypt(token)

    Key rotation:
        encryptor = SymmetricEncryptor(keys=[new_key, old_key])
        # Encrypts with new_key; can decrypt tokens produced by either key.
    """

    ENV_VAR = "ZENTRAX_ENCRYPTION_KEY"

    def __init__(self, keys: Optional[list[str | bytes]] = None):
        resolved_keys = keys or self._keys_from_env()
        if not resolved_keys:
            raise EncryptionError(
                f"No encryption key provided. Set the {self.ENV_VAR} environment "
                "variable or pass `keys` explicitly."
            )
        fernets = [Fernet(self._normalize_key(k)) for k in resolved_keys]
        self._cipher = fernets[0] if len(fernets) == 1 else MultiFernet(fernets)

    @staticmethod
    def generate_key() -> str:
        """Generate a new urlsafe-base64-encoded 32-byte Fernet key."""
        return Fernet.generate_key().decode("utf-8")

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a UTF-8 string, returning a urlsafe-base64 token string."""
        if plaintext is None:
            raise EncryptionError("Cannot encrypt None")
        try:
            token = self._cipher.encrypt(plaintext.encode("utf-8"))
            return token.decode("utf-8")
        except Exception as exc:  # defensive: never leak plaintext in errors
            logger.exception("Encryption failed")
            raise EncryptionError("Failed to encrypt value") from exc

    def decrypt(self, token: str, *, ttl_seconds: Optional[int] = None) -> str:
        """
        Decrypt a token produced by `encrypt`.
        `ttl_seconds`, if set, rejects tokens older than that age.
        """
        if not token:
            raise EncryptionError("Cannot decrypt empty token")
        try:
            plaintext = self._cipher.decrypt(token.encode("utf-8"), ttl=ttl_seconds)
            return plaintext.decode("utf-8")
        except InvalidToken as exc:
            raise EncryptionError("Invalid or tampered token, or key mismatch") from exc
        except Exception as exc:
            logger.exception("Decryption failed")
            raise EncryptionError("Failed to decrypt value") from exc

    def try_decrypt(self, token: str) -> Optional[str]:
        """Non-raising variant of `decrypt`; returns None on any failure."""
        try:
            return self.decrypt(token)
        except EncryptionError:
            return None

    @classmethod
    def _keys_from_env(cls) -> list[str]:
        raw = os.environ.get(cls.ENV_VAR)
        if not raw:
            return []
        # Support comma-separated keys for rotation: "new_key,old_key"
        return [k.strip() for k in raw.split(",") if k.strip()]

    @staticmethod
    def _normalize_key(key: str | bytes) -> bytes:
        if isinstance(key, str):
            key = key.encode("utf-8")
        try:
            # Validate it's proper urlsafe-base64 of the right length.
            decoded = base64.urlsafe_b64decode(key)
            if len(decoded) != 32:
                raise ValueError
        except Exception as exc:
            raise EncryptionError(
                "Encryption key must be a urlsafe-base64-encoded 32-byte key "
                "(use SymmetricEncryptor.generate_key())"
            ) from exc
        return key


# --------------------------------------------------------------------------- #
# Password hashing (Argon2id)
# --------------------------------------------------------------------------- #

class PasswordHasher:
    """
    Argon2id password hashing and verification.

    Usage:
        hasher = PasswordHasher()
        stored_hash = hasher.hash("user-supplied-password")
        hasher.verify(stored_hash, "user-supplied-password")  # raises on mismatch
        hasher.needs_rehash(stored_hash)  # True if params changed since hashing
    """

    def __init__(
        self,
        *,
        time_cost: int = 3,
        memory_cost_kib: int = 65536,  # 64 MiB
        parallelism: int = 4,
        hash_len: int = 32,
        salt_len: int = 16,
    ):
        self._hasher = _Argon2Hasher(
            time_cost=time_cost,
            memory_cost=memory_cost_kib,
            parallelism=parallelism,
            hash_len=hash_len,
            salt_len=salt_len,
        )

    def hash(self, password: str) -> str:
        if not password:
            raise ValueError("Password must not be empty")
        return self._hasher.hash(password)

    def verify(self, stored_hash: str, password: str) -> bool:
        """Returns True on success; raises PasswordVerificationError on mismatch/invalid hash."""
        try:
            return self._hasher.verify(stored_hash, password)
        except VerifyMismatchError as exc:
            raise PasswordVerificationError("Incorrect password") from exc
        except InvalidHash as exc:
            raise PasswordVerificationError("Stored hash is invalid or corrupted") from exc

    def verify_safe(self, stored_hash: str, password: str) -> bool:
        """Non-raising variant of `verify`; returns False on any failure."""
        try:
            return self.verify(stored_hash, password)
        except PasswordVerificationError:
            return False

    def needs_rehash(self, stored_hash: str) -> bool:
        """
        Returns True if `stored_hash` was produced with different parameters
        than this hasher is currently configured with (e.g. after tuning
        time_cost/memory_cost upward) — rehash on next successful login.
        """
        try:
            return self._hasher.check_needs_rehash(stored_hash)
        except InvalidHash:
            return True


# --------------------------------------------------------------------------- #
# Secure tokens & constant-time comparison
# --------------------------------------------------------------------------- #

def generate_secure_token(num_bytes: int = 32) -> str:
    """Generate a urlsafe-base64 random token (e.g. for password reset/email verification links)."""
    return secrets.token_urlsafe(num_bytes)


def generate_api_key(prefix: str = "zx", num_bytes: int = 32) -> str:
    """
    Generate a prefixed API key, e.g. 'zx_Xy1abcDEF...'.
    The prefix aids identification in logs/dashboards without revealing the secret.
    """
    return f"{prefix}_{secrets.token_urlsafe(num_bytes)}"


def constant_time_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison, e.g. for verifying webhook signatures or API keys."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Module-level convenience singletons
# --------------------------------------------------------------------------- #
# Lazily constructed so importing this module never fails just because the
# encryption key env var isn't set yet (e.g. during unrelated unit tests).

_encryptor_instance: Optional[SymmetricEncryptor] = None
_password_hasher_instance: Optional[PasswordHasher] = None


def get_encryptor() -> SymmetricEncryptor:
    """
    Shared SymmetricEncryptor instance, reading ZENTRAX_ENCRYPTION_KEY lazily.

    Example (in a user model or repository):
        from backend.app.core.encryption import get_encryptor

        encrypted_email = get_encryptor().encrypt(user.email)
    """
    global _encryptor_instance
    if _encryptor_instance is None:
        _encryptor_instance = SymmetricEncryptor()
    return _encryptor_instance


def get_password_hasher() -> PasswordHasher:
    """
    Shared PasswordHasher instance.

    Example (in an auth service):
        from backend.app.core.encryption import get_password_hasher

        hashed = get_password_hasher().hash(plain_password)
    """
    global _password_hasher_instance
    if _password_hasher_instance is None:
        _password_hasher_instance = PasswordHasher()
    return _password_hasher_instance