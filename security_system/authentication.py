"""Authentication: password hashing, TOTP MFA, signed tokens and server-side sessions.

Zero-trust properties
---------------------
* Passwords are hashed with scrypt (memory-hard) using per-user random salts
  and verified in constant time. Unknown users cost the same time as known ones.
* MFA (RFC 6238 TOTP) with replay protection: a code is accepted at most once.
* Access tokens are short-lived HS256 JWTs with a pinned algorithm, issuer and
  audience. A token alone is **never** sufficient: every verification re-checks
  the server-side session (revocation, idle and absolute timeouts, device
  binding) and the live user record (disabled flag, current roles).
* Failed logins feed an account lockout policy.

The reference :class:`InMemoryUserStore` is for tests and single-process use.
Implement :class:`UserStore` over a database for production; note that
:class:`AuthenticationService` serialises state changes with a lock, so use
row-level locking or an atomic counter in the replacement store.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import secrets
import struct
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from .policies.security_policy import PolicyViolation, SecurityError, SecurityPolicy

__all__ = [
    "AuthenticationError",
    "AccountLockedError",
    "MFARequiredError",
    "InvalidTokenError",
    "PasswordHasher",
    "TOTP",
    "TokenService",
    "UserRecord",
    "UserStore",
    "InMemoryUserStore",
    "Session",
    "Principal",
    "AuthResult",
    "AuthenticationService",
]

_USERNAME_RE = re.compile(r"[a-z0-9][a-z0-9_.@-]{2,63}")


class AuthenticationError(SecurityError):
    """Generic authentication failure. Messages are intentionally non-specific."""


class AccountLockedError(AuthenticationError):
    """Account is temporarily locked by the lockout policy."""


class MFARequiredError(AuthenticationError):
    """A second factor is required (or not yet enrolled)."""


class InvalidTokenError(AuthenticationError):
    """Token is malformed, forged, expired, revoked or bound to another device."""


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)


class PasswordHasher:
    """scrypt password hashing in a self-describing format.

    Encoded form: ``$scrypt$ln=15,r=8,p=1$<salt>$<hash>``. Parameters are stored
    with the hash so they can be raised later; :meth:`needs_rehash` tells you
    when to upgrade a stored hash after a successful login.
    """

    _MAX_LOG_N = 17  # cap attacker-controlled cost parameters when verifying

    def __init__(self, *, n: int = 2**15, r: int = 8, p: int = 1, dklen: int = 32, salt_len: int = 16) -> None:
        if n < 2**14 or n & (n - 1):
            raise ValueError("scrypt n must be a power of two >= 2**14")
        self._n, self._r, self._p, self._dklen, self._salt_len = n, r, p, dklen, salt_len

    @staticmethod
    def _derive(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
        material = unicodedata.normalize("NFKC", password).encode("utf-8")
        return hashlib.scrypt(material, salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=256 * n * r)

    def hash(self, password: str) -> str:
        salt = secrets.token_bytes(self._salt_len)
        digest = self._derive(password, salt, self._n, self._r, self._p, self._dklen)
        log_n = self._n.bit_length() - 1
        return f"$scrypt$ln={log_n},r={self._r},p={self._p}${_b64e(salt)}${_b64e(digest)}"

    def verify(self, password: str, encoded: str) -> bool:
        """Constant-time check. Returns ``False`` (never raises) on malformed input."""
        try:
            _, scheme, params, salt_b64, hash_b64 = encoded.split("$")
            if scheme != "scrypt":
                return False
            values = dict(item.split("=") for item in params.split(","))
            log_n, r, p = int(values["ln"]), int(values["r"]), int(values["p"])
            if not (14 <= log_n <= self._MAX_LOG_N and 1 <= r <= 16 and 1 <= p <= 4):
                return False
            expected = _b64d(hash_b64)
            actual = self._derive(password, _b64d(salt_b64), 1 << log_n, r, p, len(expected))
            return hmac.compare_digest(actual, expected)
        except (ValueError, KeyError, binascii.Error, TypeError):
            return False

    def needs_rehash(self, encoded: str) -> bool:
        try:
            params = encoded.split("$")[2]
            values = dict(item.split("=") for item in params.split(","))
            return (int(values["ln"]), int(values["r"]), int(values["p"])) != (
                self._n.bit_length() - 1,
                self._r,
                self._p,
            )
        except (ValueError, KeyError, IndexError):
            return True


# ---------------------------------------------------------------------------
# TOTP (RFC 6238)
# ---------------------------------------------------------------------------
class TOTP:
    """Time-based one-time passwords compatible with standard authenticator apps."""

    def __init__(self, *, digits: int = 6, period: int = 30, window: int = 1) -> None:
        self.digits, self.period, self.window = digits, period, window

    @staticmethod
    def generate_secret() -> str:
        """160-bit random secret, base32 encoded (what authenticator apps expect)."""
        return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")

    @staticmethod
    def _key(secret_b32: str) -> bytes:
        return base64.b32decode(secret_b32.upper() + "=" * (-len(secret_b32) % 8))

    def _hotp(self, key: bytes, counter: int) -> str:
        digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
        offset = digest[-1] & 0x0F
        value = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
        return str(value % 10**self.digits).zfill(self.digits)

    def code(self, secret_b32: str, *, at: float | None = None) -> str:
        """Current code (used for enrolment checks and tests)."""
        moment = time.time() if at is None else at
        return self._hotp(self._key(secret_b32), int(moment // self.period))

    def verify(
        self,
        secret_b32: str,
        code: str,
        *,
        at: float | None = None,
        last_counter: int | None = None,
    ) -> int | None:
        """Return the matched time-step counter, or ``None`` if invalid or replayed."""
        if not (isinstance(code, str) and code.isascii() and code.isdigit() and len(code) == self.digits):
            return None
        key = self._key(secret_b32)
        current = int((time.time() if at is None else at) // self.period)
        matched: int | None = None
        for offset in range(-self.window, self.window + 1):  # no early exit: uniform timing
            counter = current + offset
            if hmac.compare_digest(self._hotp(key, counter), code) and matched is None:
                matched = counter
        if matched is None or (last_counter is not None and matched <= last_counter):
            return None
        return matched

    def provisioning_uri(self, secret_b32: str, account: str, issuer: str) -> str:
        """``otpauth://`` URI for QR-code enrolment."""
        from urllib.parse import quote

        return (
            f"otpauth://totp/{quote(issuer)}:{quote(account)}?secret={secret_b32}"
            f"&issuer={quote(issuer)}&digits={self.digits}&period={self.period}"
        )


# ---------------------------------------------------------------------------
# Tokens (JWT HS256, algorithm pinned)
# ---------------------------------------------------------------------------
def _json_b64(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _url_b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TokenService:
    """Issue and verify compact signed tokens. ``alg=none`` and algorithm switching are impossible."""

    _HEADER = {"alg": "HS256", "typ": "JWT"}

    def __init__(
        self,
        signing_key: bytes,
        *,
        issuer: str = "security_system",
        audience: str = "security_system",
        ttl_seconds: int = 900,
        clock_skew_seconds: int = 30,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if len(signing_key) < 32:
            raise ValueError("token signing key must be at least 32 bytes")
        self._key = signing_key
        self._issuer, self._audience = issuer, audience
        self._ttl, self._skew, self._clock = ttl_seconds, clock_skew_seconds, clock

    def issue(self, claims: dict[str, Any], *, ttl_seconds: int | None = None) -> str:
        now = int(self._clock())
        body = {
            **claims,
            "iss": self._issuer,
            "aud": self._audience,
            "iat": now,
            "nbf": now,
            "exp": now + (ttl_seconds if ttl_seconds is not None else self._ttl),
            "jti": uuid.uuid4().hex,
        }
        signing_input = f"{_json_b64(self._HEADER)}.{_json_b64(body)}"
        signature = hmac.new(self._key, signing_input.encode("ascii"), hashlib.sha256).digest()
        return f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"

    def verify(self, token: str) -> dict[str, Any]:
        """Return validated claims or raise :class:`InvalidTokenError`."""
        if not isinstance(token, str) or len(token) > 8192:
            raise InvalidTokenError("invalid token")
        try:
            header_b64, payload_b64, signature_b64 = token.split(".")
            if json.loads(_url_b64d(header_b64)) != self._HEADER:
                raise InvalidTokenError("invalid token")
            expected = hmac.new(self._key, f"{header_b64}.{payload_b64}".encode("ascii"), hashlib.sha256).digest()
            if not hmac.compare_digest(expected, _url_b64d(signature_b64)):
                raise InvalidTokenError("invalid token")
            claims = json.loads(_url_b64d(payload_b64))
        except InvalidTokenError:
            raise
        except (ValueError, TypeError, binascii.Error, UnicodeError) as exc:
            raise InvalidTokenError("invalid token") from exc
        if not isinstance(claims, dict):
            raise InvalidTokenError("invalid token")
        now = self._clock()
        try:
            if claims["iss"] != self._issuer or claims["aud"] != self._audience:
                raise InvalidTokenError("invalid token")
            if now >= float(claims["exp"]) + self._skew or now < float(claims["nbf"]) - self._skew:
                raise InvalidTokenError("token expired or not yet valid")
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidTokenError("invalid token") from exc
        return claims


# ---------------------------------------------------------------------------
# Users, sessions, principals
# ---------------------------------------------------------------------------
@dataclass
class UserRecord:
    user_id: str
    username: str
    password_hash: str
    roles: frozenset[str] = frozenset()
    attributes: dict[str, Any] = field(default_factory=dict)
    mfa_secret: str | None = None
    mfa_enabled: bool = False
    last_totp_counter: int | None = None
    failed_attempts: int = 0
    last_failed_at: float = 0.0
    locked_until: float = 0.0
    password_changed_at: float = 0.0
    disabled: bool = False


class UserStore(Protocol):
    """Persistence interface for users. Implement over a real database in production."""

    def get_by_username(self, username: str) -> UserRecord | None: ...
    def get_by_id(self, user_id: str) -> UserRecord | None: ...
    def save(self, user: UserRecord) -> None: ...


class InMemoryUserStore:
    """Reference store for tests and single-process deployments."""

    def __init__(self) -> None:
        self._by_id: dict[str, UserRecord] = {}
        self._by_name: dict[str, str] = {}
        self._lock = threading.Lock()

    def get_by_username(self, username: str) -> UserRecord | None:
        with self._lock:
            user_id = self._by_name.get(username)
            return self._by_id.get(user_id) if user_id else None

    def get_by_id(self, user_id: str) -> UserRecord | None:
        with self._lock:
            return self._by_id.get(user_id)

    def save(self, user: UserRecord) -> None:
        with self._lock:
            self._by_id[user.user_id] = user
            self._by_name[user.username] = user.user_id


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: float
    last_seen: float
    expires_at: float
    mfa_verified: bool
    device_id: str | None
    ip: str | None
    revoked: bool = False


@dataclass(frozen=True)
class Principal:
    """The verified identity attached to a request. Built fresh on every call."""

    user_id: str
    username: str
    roles: frozenset[str]
    session_id: str
    mfa_verified: bool
    device_id: str | None
    ip: str | None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthResult:
    access_token: str
    session_id: str
    user_id: str
    expires_at: float
    mfa_verified: bool


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class AuthenticationService:
    """Credential verification, MFA, session lifecycle and per-request verification."""

    def __init__(
        self,
        policy: SecurityPolicy,
        store: UserStore,
        token_service: TokenService,
        *,
        hasher: PasswordHasher | None = None,
        totp: TOTP | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._policy = policy
        self._store = store
        self._tokens = token_service
        self._hasher = hasher or PasswordHasher(
            n=policy.crypto.scrypt_n, r=policy.crypto.scrypt_r, p=policy.crypto.scrypt_p
        )
        self._totp = totp or TOTP()
        self._clock = clock
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        # Verified against when the username is unknown, so response time does not reveal existence.
        self._dummy_hash = self._hasher.hash(secrets.token_urlsafe(16))

    # ------------------------------------------------------------ provisioning
    @staticmethod
    def _normalise_username(username: str) -> str:
        if not isinstance(username, str):
            raise PolicyViolation("username must be a string")
        name = unicodedata.normalize("NFKC", username).strip().lower()
        if not _USERNAME_RE.fullmatch(name):
            raise PolicyViolation("username must be 3-64 characters of [a-z0-9_.@-]")
        return name

    def register_user(
        self,
        username: str,
        password: str,
        *,
        roles: Iterable[str] = (),
        attributes: dict[str, Any] | None = None,
    ) -> UserRecord:
        """Create a user after enforcing the password policy."""
        name = self._normalise_username(username)
        self._policy.password.assert_valid(password, username=name)
        with self._lock:
            if self._store.get_by_username(name) is not None:
                raise PolicyViolation("username is not available")
            user = UserRecord(
                user_id=uuid.uuid4().hex,
                username=name,
                password_hash=self._hasher.hash(password),
                roles=frozenset(roles),
                attributes=dict(attributes or {}),
                password_changed_at=self._clock(),
            )
            self._store.save(user)
            return user

    def begin_mfa_enrollment(self, user_id: str) -> str:
        """Generate a TOTP secret. MFA is not enforced until :meth:`confirm_mfa` succeeds."""
        with self._lock:
            user = self._require_user(user_id)
            user.mfa_secret = TOTP.generate_secret()
            user.mfa_enabled = False
            self._store.save(user)
            return user.mfa_secret

    def confirm_mfa(self, user_id: str, code: str) -> bool:
        """Activate MFA once the user proves their authenticator works."""
        with self._lock:
            user = self._require_user(user_id)
            if not user.mfa_secret:
                return False
            counter = self._totp.verify(user.mfa_secret, code, at=self._clock())
            if counter is None:
                return False
            user.mfa_enabled = True
            user.last_totp_counter = counter
            self._store.save(user)
            return True

    def _require_user(self, user_id: str) -> UserRecord:
        user = self._store.get_by_id(user_id)
        if user is None:
            raise AuthenticationError("unknown user")
        return user

    # ------------------------------------------------------------ authenticate
    def authenticate(
        self,
        username: str,
        password: str,
        *,
        mfa_code: str | None = None,
        device_id: str | None = None,
        ip: str | None = None,
    ) -> AuthResult:
        """Verify credentials (+ MFA) and open a session. Raises :class:`AuthenticationError`."""
        now = self._clock()
        with self._lock:
            try:
                name = self._normalise_username(username)
            except PolicyViolation:
                name = ""
            user = self._store.get_by_username(name) if name else None
            candidate = password if isinstance(password, str) else ""

            if user is None or user.disabled:
                self._hasher.verify(candidate, self._dummy_hash)  # equalise timing
                raise AuthenticationError("Invalid credentials")
            if user.locked_until > now:
                raise AccountLockedError("Account temporarily locked")
            if not isinstance(password, str) or not self._hasher.verify(password, user.password_hash):
                self._register_failure(user, now)
                raise AuthenticationError("Invalid credentials")

            mfa_ok = False
            if user.mfa_enabled and user.mfa_secret:
                if not mfa_code:
                    raise MFARequiredError("MFA code required")
                counter = self._totp.verify(user.mfa_secret, mfa_code, at=now, last_counter=user.last_totp_counter)
                if counter is None:
                    self._register_failure(user, now)
                    raise AuthenticationError("Invalid credentials")
                user.last_totp_counter = counter
                mfa_ok = True
            elif self._policy.session.require_mfa:
                raise MFARequiredError("MFA enrolment required")

            if (now - user.password_changed_at) > self._policy.password.max_age_days * 86400:
                raise AuthenticationError("Password expired")

            user.failed_attempts, user.last_failed_at, user.locked_until = 0, 0.0, 0.0
            if self._hasher.needs_rehash(user.password_hash):
                user.password_hash = self._hasher.hash(password)
            self._store.save(user)

            self._purge_sessions(now)
            self._enforce_concurrency(user.user_id)
            session_cfg = self._policy.session
            session = Session(
                session_id=uuid.uuid4().hex,
                user_id=user.user_id,
                created_at=now,
                last_seen=now,
                expires_at=now + session_cfg.absolute_lifetime_seconds,
                mfa_verified=mfa_ok,
                device_id=device_id,
                ip=ip,
            )
            self._sessions[session.session_id] = session
            ttl = int(min(session_cfg.access_token_ttl_seconds, session.expires_at - now))
            token = self._tokens.issue(
                {"sub": user.user_id, "sid": session.session_id, "mfa": mfa_ok}, ttl_seconds=ttl
            )
            return AuthResult(token, session.session_id, user.user_id, now + ttl, mfa_ok)

    def _register_failure(self, user: UserRecord, now: float) -> None:
        lockout = self._policy.lockout
        if now - user.last_failed_at > lockout.failure_window_seconds:
            user.failed_attempts = 0
        user.failed_attempts += 1
        user.last_failed_at = now
        if user.failed_attempts >= lockout.max_failed_attempts:
            user.locked_until = now + lockout.lockout_seconds
            user.failed_attempts = 0
        self._store.save(user)

    # ------------------------------------------------------------ verification
    def verify_token(self, token: str, *, device_id: str | None = None, ip: str | None = None) -> Principal:
        """Verify a token **and** its live session on every request (never trust, always verify)."""
        claims = self._tokens.verify(token)
        now = self._clock()
        with self._lock:
            session = self._sessions.get(str(claims.get("sid")))
            if session is None or session.revoked or session.user_id != claims.get("sub"):
                raise InvalidTokenError("invalid token")
            cfg = self._policy.session
            if now >= session.expires_at or (now - session.last_seen) > cfg.idle_timeout_seconds:
                session.revoked = True
                raise InvalidTokenError("session expired")
            if session.device_id is not None and device_id != session.device_id:
                session.revoked = True  # possible token theft: kill the session
                raise InvalidTokenError("invalid token")
            user = self._store.get_by_id(session.user_id)
            if user is None or user.disabled:
                session.revoked = True
                raise InvalidTokenError("invalid token")
            session.last_seen = now
            return Principal(
                user_id=user.user_id,
                username=user.username,
                roles=user.roles,
                session_id=session.session_id,
                mfa_verified=session.mfa_verified,
                device_id=session.device_id,
                ip=ip,
                attributes=dict(user.attributes),
            )

    # ---------------------------------------------------------------- lifecycle
    def revoke_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.revoked = True

    def revoke_user_sessions(self, user_id: str) -> int:
        with self._lock:
            count = 0
            for session in self._sessions.values():
                if session.user_id == user_id and not session.revoked:
                    session.revoked = True
                    count += 1
            return count

    def logout(self, token: str) -> None:
        claims = self._tokens.verify(token)
        self.revoke_session(str(claims.get("sid")))

    def disable_user(self, user_id: str) -> None:
        with self._lock:
            user = self._require_user(user_id)
            user.disabled = True
            self._store.save(user)
            self.revoke_user_sessions(user_id)

    def change_password(self, user_id: str, old_password: str, new_password: str) -> None:
        """Change a password, enforcing policy; all existing sessions are revoked."""
        with self._lock:
            user = self._require_user(user_id)
            if not self._hasher.verify(old_password, user.password_hash):
                self._register_failure(user, self._clock())
                raise AuthenticationError("Invalid credentials")
            if hmac.compare_digest(old_password.encode(), new_password.encode()):
                raise PolicyViolation("new password must differ from the old password")
            self._policy.password.assert_valid(new_password, username=user.username)
            user.password_hash = self._hasher.hash(new_password)
            user.password_changed_at = self._clock()
            self._store.save(user)
            self.revoke_user_sessions(user_id)

    def _purge_sessions(self, now: float) -> None:
        idle = self._policy.session.idle_timeout_seconds
        stale = [
            sid
            for sid, s in self._sessions.items()
            if s.revoked or now >= s.expires_at or (now - s.last_seen) > idle
        ]
        for sid in stale:
            del self._sessions[sid]

    def _enforce_concurrency(self, user_id: str) -> None:
        limit = self._policy.session.max_concurrent_sessions
        active = sorted((s for s in self._sessions.values() if s.user_id == user_id), key=lambda s: s.created_at)
        while len(active) >= limit:
            active.pop(0).revoked = True
