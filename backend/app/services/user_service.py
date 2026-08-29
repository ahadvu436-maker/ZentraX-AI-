"""
user_service.py

User account service for ZentraX AI.

Responsibilities:
    - Register new users with strong password hashing (Argon2id).
    - Authenticate users and issue short-lived JWT access tokens + refresh tokens.
    - Manage user profile data with a minimal, privacy-conscious data model.
    - Never store, log, or return plaintext passwords or raw tokens.

This module contains NO direct database driver code — it depends on a
`UserRepository` interface so it can sit on top of Postgres, Mongo, etc.
without change. See `app/repositories/` for concrete implementations.

Required third-party packages:
    pip install "passlib[argon2]" pyjwt
"""

from __future__ import annotations

import logging
import re
import secrets
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from passlib.context import CryptContext

logger = logging.getLogger("zentrax.user_service")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Argon2id is the current OWASP-recommended default for password hashing.
_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

ACCESS_TOKEN_TTL = timedelta(minutes=15)
REFRESH_TOKEN_TTL = timedelta(days=14)
JWT_ALGORITHM = "HS256"

MIN_PASSWORD_LENGTH = 12
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Passwords that are trivially guessable regardless of length — a tiny
# defense-in-depth check; pair this with a real breached-password list
# (e.g. HaveIBeenPwned's k-anonymity API) in production.
_COMMON_PASSWORDS = {"password123", "letmein123", "qwerty123456", "changeme123"}


class UserServiceError(Exception):
    """Base class for user-facing errors from this service."""


class EmailAlreadyRegisteredError(UserServiceError):
    pass


class InvalidCredentialsError(UserServiceError):
    """Deliberately generic — never reveal whether the email or password was wrong."""


class WeakPasswordError(UserServiceError):
    pass


class UserNotFoundError(UserServiceError):
    pass


class TokenInvalidError(UserServiceError):
    pass


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class UserRecord:
    """
    Persisted user representation.

    Only `password_hash` is stored for credentials — plaintext passwords are
    never persisted, logged, or held longer than the single call that hashes
    them. `email` is the only direct PII field by design; anything else
    (display name, avatar, etc.) lives in `profile` so it can be selectively
    redacted or deleted independently of the auth record (e.g. GDPR erasure).
    """
    user_id: str
    email: str
    password_hash: str
    created_at: datetime
    is_active: bool = True
    profile: dict = field(default_factory=dict)


@dataclass(slots=True)
class PublicUserProfile:
    """Safe-to-return view of a user — excludes password_hash and any
    internal-only fields."""
    user_id: str
    email: str
    display_name: Optional[str]
    created_at: datetime


@dataclass(slots=True)
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_seconds: int = int(ACCESS_TOKEN_TTL.total_seconds())


# -----------------------------------------------------------------------------
# Repository interface
# -----------------------------------------------------------------------------

class UserRepository(ABC):
    """Storage interface. Implement against your real database; this keeps
    UserService testable and DB-agnostic."""

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[UserRecord]: ...

    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[UserRecord]: ...

    @abstractmethod
    async def create(self, user: UserRecord) -> None: ...

    @abstractmethod
    async def update(self, user: UserRecord) -> None: ...

    @abstractmethod
    async def delete(self, user_id: str) -> None: ...


class InMemoryUserRepository(UserRepository):
    """Reference implementation for local dev/tests. Not durable — replace
    with a real database-backed repository in production."""

    def __init__(self) -> None:
        self._by_id: dict[str, UserRecord] = {}
        self._by_email: dict[str, str] = {}  # email -> user_id

    async def get_by_email(self, email: str) -> Optional[UserRecord]:
        user_id = self._by_email.get(email.lower())
        return self._by_id.get(user_id) if user_id else None

    async def get_by_id(self, user_id: str) -> Optional[UserRecord]:
        return self._by_id.get(user_id)

    async def create(self, user: UserRecord) -> None:
        self._by_id[user.user_id] = user
        self._by_email[user.email.lower()] = user.user_id

    async def update(self, user: UserRecord) -> None:
        self._by_id[user.user_id] = user

    async def delete(self, user_id: str) -> None:
        user = self._by_id.pop(user_id, None)
        if user:
            self._by_email.pop(user.email.lower(), None)


# -----------------------------------------------------------------------------
# User service
# -----------------------------------------------------------------------------

class UserService:
    """
    Handles registration, authentication, token issuance, and profile
    management.

    `jwt_secret` must come from a secrets manager / environment variable —
    never hardcode it. Rotate it periodically; doing so invalidates all
    outstanding tokens, so plan rotations accordingly.
    """

    def __init__(self, repository: UserRepository, *, jwt_secret: str, issuer: str = "zentrax-ai"):
        if not jwt_secret or len(jwt_secret) < 32:
            raise ValueError("jwt_secret must be set and at least 32 characters long.")
        self._repo = repository
        self._jwt_secret = jwt_secret
        self._issuer = issuer

    # -- Registration -----------------------------------------------------------

    async def register(self, email: str, password: str, display_name: Optional[str] = None) -> PublicUserProfile:
        """Create a new user account. Raises EmailAlreadyRegisteredError or
        WeakPasswordError as appropriate."""
        email = self._normalize_email(email)
        self._validate_password_strength(password)

        if await self._repo.get_by_email(email) is not None:
            # Generic message to the caller; specifics only in server logs.
            logger.info("registration_attempt_duplicate_email")
            raise EmailAlreadyRegisteredError("An account with this email already exists.")

        password_hash = _pwd_context.hash(password)
        # Defensive: drop the reference to the plaintext password ASAP.
        del password

        user = UserRecord(
            user_id=str(uuid.uuid4()),
            email=email,
            password_hash=password_hash,
            created_at=datetime.now(timezone.utc),
            profile={"display_name": display_name} if display_name else {},
        )
        await self._repo.create(user)
        logger.info("user_registered user_id=%s", user.user_id)
        return self._to_public_profile(user)

    # -- Authentication ---------------------------------------------------------

    async def authenticate(self, email: str, password: str) -> TokenPair:
        """
        Verify credentials and issue a new token pair.

        Always takes the same code path (including a hash comparison) whether
        or not the email exists, to avoid leaking account existence via
        response timing.
        """
        email = self._normalize_email(email)
        user = await self._repo.get_by_email(email)

        # Use a fixed dummy hash when the user doesn't exist so verification
        # still runs and takes comparable time either way.
        hash_to_check = user.password_hash if user else _pwd_context.hash(secrets.token_urlsafe(16))
        password_ok = _pwd_context.verify(password, hash_to_check)

        if not user or not password_ok or not user.is_active:
            del password
            logger.info("authentication_failed")
            raise InvalidCredentialsError("Incorrect email or password.")

        # Transparently upgrade the stored hash if the hashing scheme/params
        # have been strengthened since this user last logged in. Must happen
        # before we drop the plaintext password, since re-hashing needs it.
        if _pwd_context.needs_update(hash_to_check):
            user.password_hash = _pwd_context.hash(password)
            await self._repo.update(user)

        del password

        logger.info("authentication_succeeded user_id=%s", user.user_id)
        return self._issue_token_pair(user.user_id)

    async def refresh_access_token(self, refresh_token: str) -> TokenPair:
        """Exchange a valid refresh token for a new token pair (rotation)."""
        payload = self._decode_token(refresh_token, expected_type="refresh")
        user_id = payload["sub"]

        user = await self._repo.get_by_id(user_id)
        if user is None or not user.is_active:
            raise TokenInvalidError("User no longer exists or is inactive.")

        return self._issue_token_pair(user_id)

    def verify_access_token(self, access_token: str) -> str:
        """Validate an access token and return the user_id it authorizes.
        Raises TokenInvalidError if invalid/expired."""
        payload = self._decode_token(access_token, expected_type="access")
        return payload["sub"]

    # -- Profile management -------------------------------------------------------

    async def get_profile(self, user_id: str) -> PublicUserProfile:
        user = await self._repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError(f"No user with id={user_id}")
        return self._to_public_profile(user)

    async def update_profile(self, user_id: str, *, display_name: Optional[str] = None) -> PublicUserProfile:
        """Update mutable, non-sensitive profile fields. Email/password
        changes should go through their own dedicated, re-auth-guarded flows,
        not this generic method."""
        user = await self._repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError(f"No user with id={user_id}")

        if display_name is not None:
            user.profile["display_name"] = display_name.strip()[:120]

        await self._repo.update(user)
        logger.info("profile_updated user_id=%s", user_id)
        return self._to_public_profile(user)

    async def change_password(self, user_id: str, current_password: str, new_password: str) -> None:
        """Re-validates the current password before allowing a change —
        never allow a password change on a stolen access token alone."""
        user = await self._repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError(f"No user with id={user_id}")

        if not _pwd_context.verify(current_password, user.password_hash):
            raise InvalidCredentialsError("Current password is incorrect.")

        self._validate_password_strength(new_password)
        user.password_hash = _pwd_context.hash(new_password)
        del current_password, new_password

        await self._repo.update(user)
        logger.info("password_changed user_id=%s", user_id)

    async def delete_account(self, user_id: str) -> None:
        """Permanently erase a user's account and profile data (right to
        erasure). Related data in other services should be purged via events
        / a coordinated deletion workflow, not from here directly."""
        await self._repo.delete(user_id)
        logger.info("account_deleted user_id=%s", user_id)

    # -- Internal helpers -------------------------------------------------------------

    @staticmethod
    def _normalize_email(email: str) -> str:
        email = email.strip().lower()
        if not _EMAIL_RE.match(email):
            raise UserServiceError("Invalid email address format.")
        return email

    @staticmethod
    def _validate_password_strength(password: str) -> None:
        if len(password) < MIN_PASSWORD_LENGTH:
            raise WeakPasswordError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
        if password.lower() in _COMMON_PASSWORDS:
            raise WeakPasswordError("Password is too common. Choose a stronger password.")
        if not (re.search(r"[a-z]", password) and re.search(r"[A-Z]", password) and re.search(r"\d", password)):
            raise WeakPasswordError("Password must include upper, lower, and numeric characters.")

    @staticmethod
    def _to_public_profile(user: UserRecord) -> PublicUserProfile:
        return PublicUserProfile(
            user_id=user.user_id,
            email=user.email,
            display_name=user.profile.get("display_name"),
            created_at=user.created_at,
        )

    def _issue_token_pair(self, user_id: str) -> TokenPair:
        now = datetime.now(timezone.utc)
        access_payload = {
            "sub": user_id,
            "type": "access",
            "iss": self._issuer,
            "iat": now,
            "exp": now + ACCESS_TOKEN_TTL,
            "jti": secrets.token_hex(16),
        }
        refresh_payload = {
            "sub": user_id,
            "type": "refresh",
            "iss": self._issuer,
            "iat": now,
            "exp": now + REFRESH_TOKEN_TTL,
            "jti": secrets.token_hex(16),
        }
        access_token = jwt.encode(access_payload, self._jwt_secret, algorithm=JWT_ALGORITHM)
        refresh_token = jwt.encode(refresh_payload, self._jwt_secret, algorithm=JWT_ALGORITHM)
        return TokenPair(access_token=access_token, refresh_token=refresh_token)

    def _decode_token(self, token: str, *, expected_type: str) -> dict:
        try:
            payload = jwt.decode(
                token,
                self._jwt_secret,
                algorithms=[JWT_ALGORITHM],
                issuer=self._issuer,
                options={"require": ["exp", "iat", "sub", "type"]},
            )
        except jwt.PyJWTError as exc:
            logger.info("token_validation_failed error_type=%s", type(exc).__name__)
            raise TokenInvalidError("Invalid or expired token.") from exc

        if payload.get("type") != expected_type:
            raise TokenInvalidError(f"Expected a {expected_type} token.")
        return payload