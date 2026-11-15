"""Secrets management: versioned, envelope-encrypted, never logged.

* Secret values are encrypted with envelope encryption before they reach any
  backend. The ciphertext is bound (AAD) to ``name`` and ``version``, so blobs
  cannot be swapped between secrets or rolled back to a different version.
* Values are returned as :class:`SecretStr`, which masks itself in ``repr``,
  ``str`` and logs; call :meth:`SecretStr.reveal` deliberately.
* Secrets can expire, be rotated (new version, old versions pruned) and deleted.
* Audit entries record *that* a secret was touched, never its value.

This class is a low-level service. Put access control in front of it
(``SecuritySystem.get_secret`` does exactly that).
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import re
import secrets as _secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .audit_logger import AuditLogger, AuditOutcome, AuditSeverity
from .encryption import DecryptionError, EncryptionService
from .policies.security_policy import SecurityError

__all__ = [
    "SecretsError",
    "SecretNotFoundError",
    "SecretExpiredError",
    "SecretStr",
    "SecretMetadata",
    "StoredSecret",
    "SecretBackend",
    "InMemorySecretBackend",
    "FileSecretBackend",
    "SecretsManager",
]

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]{0,127}")
_MAX_SECRET_BYTES = 64 * 1024


class SecretsError(SecurityError):
    """Generic secrets failure."""


class SecretNotFoundError(SecretsError):
    """No such secret or version."""


class SecretExpiredError(SecretsError):
    """The secret's validity period has ended."""


class SecretStr:
    """A string wrapper that refuses to reveal itself accidentally."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('**********')"

    def __str__(self) -> str:
        return "**********"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SecretStr) and hmac.compare_digest(
            self._value.encode("utf-8"), other._value.encode("utf-8")
        )

    __hash__ = None  # type: ignore[assignment]

    def __reduce__(self) -> Any:
        raise TypeError("SecretStr cannot be pickled")

    def __len__(self) -> int:
        return len(self._value)


@dataclass(frozen=True)
class SecretMetadata:
    name: str
    version: int
    created_at: float
    expires_at: float | None
    created_by: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoredSecret:
    metadata: SecretMetadata
    blob: bytes


class SecretBackend(Protocol):
    """Storage interface. Backends only ever see ciphertext."""

    def put(self, record: StoredSecret) -> None: ...
    def get(self, name: str, version: int | None) -> StoredSecret | None: ...
    def versions(self, name: str) -> list[int]: ...
    def delete_version(self, name: str, version: int) -> None: ...
    def delete(self, name: str) -> bool: ...
    def names(self) -> list[str]: ...


class InMemorySecretBackend:
    def __init__(self) -> None:
        self._data: dict[str, dict[int, StoredSecret]] = {}
        self._lock = threading.Lock()

    def put(self, record: StoredSecret) -> None:
        with self._lock:
            self._data.setdefault(record.metadata.name, {})[record.metadata.version] = record

    def get(self, name: str, version: int | None) -> StoredSecret | None:
        with self._lock:
            versions = self._data.get(name)
            if not versions:
                return None
            return versions.get(version if version is not None else max(versions))

    def versions(self, name: str) -> list[int]:
        with self._lock:
            return sorted(self._data.get(name, {}))

    def delete_version(self, name: str, version: int) -> None:
        with self._lock:
            self._data.get(name, {}).pop(version, None)

    def delete(self, name: str) -> bool:
        with self._lock:
            return self._data.pop(name, None) is not None

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._data)


class FileSecretBackend(InMemorySecretBackend):
    """JSON file of *ciphertext* records, written atomically with mode 0600."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._io_lock = threading.Lock()
        if self._path.exists():
            self._load()

    def put(self, record: StoredSecret) -> None:
        super().put(record)
        self._flush()

    def delete_version(self, name: str, version: int) -> None:
        super().delete_version(name, version)
        self._flush()

    def delete(self, name: str) -> bool:
        removed = super().delete(name)
        self._flush()
        return removed

    def _flush(self) -> None:
        with self._lock:
            payload = [
                {
                    "name": s.metadata.name,
                    "version": s.metadata.version,
                    "created_at": s.metadata.created_at,
                    "expires_at": s.metadata.expires_at,
                    "created_by": s.metadata.created_by,
                    "tags": list(s.metadata.tags),
                    "blob": base64.b64encode(s.blob).decode("ascii"),
                }
                for versions in self._data.values()
                for s in versions.values()
            ]
        with self._io_lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".secrets-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(tmp, 0o600)
                os.replace(tmp, self._path)
            except OSError as exc:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise SecretsError(f"cannot persist secrets: {exc}") from exc

    def _load(self) -> None:
        try:
            for item in json.loads(self._path.read_text(encoding="utf-8")):
                meta = SecretMetadata(
                    name=item["name"],
                    version=int(item["version"]),
                    created_at=float(item["created_at"]),
                    expires_at=None if item["expires_at"] is None else float(item["expires_at"]),
                    created_by=item["created_by"],
                    tags=tuple(item["tags"]),
                )
                super().put(StoredSecret(meta, base64.b64decode(item["blob"], validate=True)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SecretsError(f"cannot read secrets file: {exc}") from exc


class SecretsManager:
    """Versioned secret storage on top of :class:`EncryptionService`."""

    def __init__(
        self,
        encryption: EncryptionService,
        backend: SecretBackend | None = None,
        *,
        audit: AuditLogger | None = None,
        max_versions: int = 5,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_versions < 1:
            raise SecretsError("max_versions must be >= 1")
        self._enc = encryption
        self._backend = backend or InMemorySecretBackend()
        self._audit = audit
        self._max_versions = max_versions
        self._clock = clock
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _check_name(name: str) -> None:
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or ".." in name.split("/"):
            raise SecretsError("invalid secret name")

    @staticmethod
    def _aad(name: str, version: int) -> bytes:
        return f"secret:{name}:v{version}".encode("utf-8")

    @staticmethod
    def generate(length: int = 32) -> SecretStr:
        """Generate a URL-safe random secret (for example an API key)."""
        return SecretStr(_secrets.token_urlsafe(length))

    def _record(self, event: str, actor: str, name: str, outcome: AuditOutcome, **details: Any) -> None:
        if self._audit is not None:
            self._audit.log(
                event,
                actor=actor,
                action=event,
                resource=f"secret/{name}",
                outcome=outcome,
                severity=AuditSeverity.INFO if outcome is AuditOutcome.SUCCESS else AuditSeverity.WARNING,
                details=details,
            )

    # -------------------------------------------------------------------- API
    def set_secret(
        self,
        name: str,
        value: str | SecretStr,
        *,
        actor: str,
        ttl_seconds: float | None = None,
        tags: tuple[str, ...] = (),
    ) -> SecretMetadata:
        """Store a new version of ``name`` (this is also how you rotate)."""
        self._check_name(name)
        plain = value.reveal() if isinstance(value, SecretStr) else value
        if not isinstance(plain, str) or not plain:
            raise SecretsError("secret value must be a non-empty string")
        payload = plain.encode("utf-8")
        if len(payload) > _MAX_SECRET_BYTES:
            raise SecretsError("secret value too large")
        with self._lock:
            existing = self._backend.versions(name)
            version = (existing[-1] if existing else 0) + 1
            now = self._clock()
            meta = SecretMetadata(
                name=name,
                version=version,
                created_at=now,
                expires_at=None if ttl_seconds is None else now + ttl_seconds,
                created_by=actor,
                tags=tuple(tags),
            )
            blob = self._enc.encrypt_envelope(payload, aad=self._aad(name, version))
            self._backend.put(StoredSecret(meta, blob))
            for old in (existing + [version])[: -self._max_versions]:
                self._backend.delete_version(name, old)
        self._record("secret.set", actor, name, AuditOutcome.SUCCESS, version=version)
        return meta

    def get_secret(self, name: str, *, actor: str, version: int | None = None) -> SecretStr:
        """Decrypt and return a secret (latest version unless ``version`` is given)."""
        self._check_name(name)
        with self._lock:
            stored = self._backend.get(name, version)
        if stored is None:
            self._record("secret.get", actor, name, AuditOutcome.FAILURE, reason="not_found")
            raise SecretNotFoundError("secret not found")
        meta = stored.metadata
        if meta.expires_at is not None and self._clock() >= meta.expires_at:
            self._record("secret.get", actor, name, AuditOutcome.DENIED, reason="expired", version=meta.version)
            raise SecretExpiredError("secret has expired")
        try:
            plain = self._enc.decrypt_envelope(stored.blob, aad=self._aad(name, meta.version))
        except DecryptionError as exc:
            self._record("secret.get", actor, name, AuditOutcome.ERROR, reason="decrypt_failed", version=meta.version)
            raise SecretsError("secret could not be decrypted") from exc
        self._record("secret.get", actor, name, AuditOutcome.SUCCESS, version=meta.version)
        return SecretStr(plain.decode("utf-8"))

    def rotate_secret(self, name: str, new_value: str | SecretStr, *, actor: str, ttl_seconds: float | None = None) -> SecretMetadata:
        """Alias of :meth:`set_secret` that requires the secret to exist already."""
        with self._lock:
            if not self._backend.versions(name):
                raise SecretNotFoundError("secret not found")
            return self.set_secret(name, new_value, actor=actor, ttl_seconds=ttl_seconds)

    def delete_secret(self, name: str, *, actor: str) -> bool:
        self._check_name(name)
        with self._lock:
            removed = self._backend.delete(name)
        self._record("secret.delete", actor, name, AuditOutcome.SUCCESS if removed else AuditOutcome.FAILURE)
        return removed

    def list_secrets(self) -> list[str]:
        """Names only; values and metadata of other versions are never exposed here."""
        with self._lock:
            return self._backend.names()

    def metadata(self, name: str, version: int | None = None) -> SecretMetadata:
        self._check_name(name)
        with self._lock:
            stored = self._backend.get(name, version)
        if stored is None:
            raise SecretNotFoundError("secret not found")
        return stored.metadata

    def import_from_environment(self, mapping: Mapping[str, str], *, prefix: str, actor: str) -> list[str]:
        """Import ``PREFIX_NAME=value`` pairs (for example ``os.environ``) as secrets."""
        imported: list[str] = []
        for key, value in mapping.items():
            if key.startswith(prefix) and value:
                name = key[len(prefix):].lower()
                if _NAME_RE.fullmatch(name):
                    self.set_secret(name, value, actor=actor, tags=("env-import",))
                    imported.append(name)
        return imported
