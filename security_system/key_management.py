"""Key management: generation, versioning, rotation, wrapping and destruction.

Design
------
* Every data key is generated with ``secrets``/``os.urandom`` (CSPRNG).
* Keys are never held in the clear at rest: each is wrapped (AES-256-GCM)
  under a *master key*, with the key id bound as AAD so wrapped blobs cannot be
  swapped between records.
* Keys have a lifecycle: ``ACTIVE`` -> ``DECRYPT_ONLY`` -> ``DESTROYED``.
  Only ``ACTIVE`` keys encrypt; ``DECRYPT_ONLY`` keys remain readable so old
  data stays recoverable until it is re-encrypted.
* In production the master key should come from an HSM/KMS. This module takes
  the master key as bytes so any such provider can be plugged in.

Python cannot guarantee memory zeroisation (immutable ``bytes`` may be copied
by the interpreter). The master key is held in a ``bytearray`` that is wiped on
:meth:`KeyManager.close` as a best-effort measure.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .policies.security_policy import SecurityError

__all__ = [
    "KeyManagementError",
    "KeyState",
    "KeyPurpose",
    "KeyRecord",
    "KeyManager",
    "load_master_key_from_env",
]

_NONCE_LEN = 12
_KEY_LEN = 32


class KeyManagementError(SecurityError):
    """Raised for any key lifecycle or storage failure."""


class KeyState(str, Enum):
    ACTIVE = "active"
    DECRYPT_ONLY = "decrypt_only"
    DESTROYED = "destroyed"


class KeyPurpose(str, Enum):
    ENCRYPTION = "enc"  # data encryption keys
    SIGNING = "sig"  # HMAC keys
    WRAPPING = "kek"  # key-encryption keys for envelope encryption


@dataclass
class KeyRecord:
    """Metadata plus wrapped key material. The clear key is never stored here."""

    key_id: str
    name: str
    purpose: KeyPurpose
    version: int
    state: KeyState
    created_at: float
    rotate_after: float
    wrapped: bytes

    def metadata(self) -> dict[str, Any]:
        """Safe-to-log view without key material."""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "purpose": self.purpose.value,
            "version": self.version,
            "state": self.state.value,
            "created_at": self.created_at,
            "rotate_after": self.rotate_after,
        }


def load_master_key_from_env(variable: str = "SECURITY_MASTER_KEY") -> bytes:
    """Read a base64-encoded 32-byte master key from the environment."""
    raw = os.environ.get(variable)
    if not raw:
        raise KeyManagementError(f"environment variable {variable} is not set")
    try:
        key = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise KeyManagementError(f"{variable} is not valid base64") from exc
    if len(key) != _KEY_LEN:
        raise KeyManagementError(f"{variable} must decode to exactly {_KEY_LEN} bytes")
    return key


class KeyManager:
    """Thread-safe in-process key manager with optional encrypted persistence."""

    def __init__(
        self,
        master_key: bytes,
        *,
        keystore_path: str | Path | None = None,
        rotation_interval_days: int = 90,
        auto_rotate: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if len(master_key) != _KEY_LEN:
            raise KeyManagementError(f"master key must be exactly {_KEY_LEN} bytes")
        if rotation_interval_days < 1:
            raise KeyManagementError("rotation_interval_days must be >= 1")
        self._master = bytearray(master_key)
        self._rotation_seconds = rotation_interval_days * 86400
        self._auto_rotate = auto_rotate
        self._clock = clock
        self._lock = threading.RLock()
        self._records: dict[str, KeyRecord] = {}
        self._active: dict[tuple[KeyPurpose, str], str] = {}
        self._closed = False
        self._path = Path(keystore_path) if keystore_path else None
        if self._path is not None and self._path.exists():
            self._load()

    # ------------------------------------------------------------ master key
    @staticmethod
    def generate_master_key() -> bytes:
        """Create a fresh random 256-bit master key."""
        return secrets.token_bytes(_KEY_LEN)

    @staticmethod
    def derive_master_key(passphrase: str, salt: bytes, *, n: int = 2**15, r: int = 8, p: int = 1) -> bytes:
        """Derive a master key from a passphrase with scrypt. ``salt`` must be stored."""
        if len(salt) < 16:
            raise KeyManagementError("salt must be at least 16 bytes")
        return hashlib.scrypt(
            passphrase.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=_KEY_LEN, maxmem=256 * n * r
        )

    def close(self) -> None:
        """Wipe the master key (best effort) and refuse further use."""
        with self._lock:
            for index in range(len(self._master)):
                self._master[index] = 0
            self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise KeyManagementError("key manager is closed")

    # --------------------------------------------------------------- wrapping
    def _wrap(self, key_id: str, material: bytes) -> bytes:
        nonce = os.urandom(_NONCE_LEN)
        return nonce + AESGCM(bytes(self._master)).encrypt(nonce, material, key_id.encode("utf-8"))

    def _unwrap(self, key_id: str, wrapped: bytes) -> bytes:
        try:
            return AESGCM(bytes(self._master)).decrypt(
                wrapped[:_NONCE_LEN], wrapped[_NONCE_LEN:], key_id.encode("utf-8")
            )
        except (InvalidTag, ValueError) as exc:
            raise KeyManagementError(f"cannot unwrap key '{key_id}' (wrong master key or tampering)") from exc

    # -------------------------------------------------------------- lifecycle
    def _new_record(self, purpose: KeyPurpose, name: str, version: int) -> KeyRecord:
        now = self._clock()
        key_id = f"{purpose.value}:{name}:v{version}"
        material = secrets.token_bytes(_KEY_LEN)
        return KeyRecord(
            key_id=key_id,
            name=name,
            purpose=purpose,
            version=version,
            state=KeyState.ACTIVE,
            created_at=now,
            rotate_after=now + self._rotation_seconds,
            wrapped=self._wrap(key_id, material),
        )

    def _latest_version(self, purpose: KeyPurpose, name: str) -> int:
        versions = [r.version for r in self._records.values() if r.purpose is purpose and r.name == name]
        return max(versions, default=0)

    def create_key(self, purpose: KeyPurpose, name: str = "default") -> KeyRecord:
        """Create the first key for ``(purpose, name)``. Use :meth:`rotate` afterwards."""
        self._validate_name(name)
        with self._lock:
            self._check_open()
            if (purpose, name) in self._active:
                raise KeyManagementError(f"an active key for {purpose.value}/{name} already exists; rotate instead")
            record = self._new_record(purpose, name, self._latest_version(purpose, name) + 1)
            self._records[record.key_id] = record
            self._active[(purpose, name)] = record.key_id
            self._persist()
            return record

    def rotate(self, purpose: KeyPurpose, name: str = "default") -> KeyRecord:
        """Retire the active key to ``DECRYPT_ONLY`` and activate a new version."""
        with self._lock:
            self._check_open()
            current_id = self._active.get((purpose, name))
            if current_id is not None:
                self._records[current_id].state = KeyState.DECRYPT_ONLY
            record = self._new_record(purpose, name, self._latest_version(purpose, name) + 1)
            self._records[record.key_id] = record
            self._active[(purpose, name)] = record.key_id
            self._persist()
            return record

    def active_key_id(self, purpose: KeyPurpose, name: str = "default") -> str:
        """Return the active key id, creating or rotating the key when needed."""
        with self._lock:
            self._check_open()
            key_id = self._active.get((purpose, name))
            if key_id is None:
                return self.create_key(purpose, name).key_id
            if self._auto_rotate and self._clock() >= self._records[key_id].rotate_after:
                return self.rotate(purpose, name).key_id
            return key_id

    def get_material(self, key_id: str, *, for_encryption: bool = False) -> bytes:
        """Unwrap and return key bytes.

        ``for_encryption=True`` additionally requires the key to be ``ACTIVE``.
        """
        with self._lock:
            self._check_open()
            record = self._records.get(key_id)
            if record is None:
                raise KeyManagementError(f"unknown key '{key_id}'")
            if record.state is KeyState.DESTROYED:
                raise KeyManagementError(f"key '{key_id}' has been destroyed")
            if for_encryption and record.state is not KeyState.ACTIVE:
                raise KeyManagementError(f"key '{key_id}' is not active and cannot be used to encrypt")
            return self._unwrap(key_id, record.wrapped)

    def retire(self, key_id: str) -> None:
        """Force a key to ``DECRYPT_ONLY`` (for example after suspected exposure)."""
        with self._lock:
            self._check_open()
            record = self._records.get(key_id)
            if record is None:
                raise KeyManagementError(f"unknown key '{key_id}'")
            if record.state is KeyState.DESTROYED:
                return
            record.state = KeyState.DECRYPT_ONLY
            if self._active.get((record.purpose, record.name)) == key_id:
                del self._active[(record.purpose, record.name)]
            self._persist()

    def destroy(self, key_id: str) -> None:
        """Irreversibly delete key material. Data encrypted under it is lost."""
        with self._lock:
            self._check_open()
            record = self._records.get(key_id)
            if record is None:
                raise KeyManagementError(f"unknown key '{key_id}'")
            record.state = KeyState.DESTROYED
            record.wrapped = b""
            if self._active.get((record.purpose, record.name)) == key_id:
                del self._active[(record.purpose, record.name)]
            self._persist()

    def needs_rotation(self, key_id: str) -> bool:
        with self._lock:
            record = self._records.get(key_id)
            return record is not None and record.state is KeyState.ACTIVE and self._clock() >= record.rotate_after

    def list_keys(self) -> list[dict[str, Any]]:
        """Metadata for every key (never key material)."""
        with self._lock:
            return [r.metadata() for r in sorted(self._records.values(), key=lambda r: r.key_id)]

    # ------------------------------------------------------------ persistence
    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or len(name) > 64 or not all(c.isalnum() or c in "-_." for c in name):
            raise KeyManagementError("key name must be 1-64 characters of [A-Za-z0-9._-]")

    def _persist(self) -> None:
        if self._path is None:
            return
        payload = {
            "format": 1,
            "records": [
                {
                    **r.metadata(),
                    "wrapped": base64.b64encode(r.wrapped).decode("ascii"),
                }
                for r in self._records.values()
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self._path.parent, prefix=".keystore-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self._path)  # atomic on POSIX and Windows
        except OSError as exc:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise KeyManagementError(f"cannot write keystore: {exc}") from exc

    def _load(self) -> None:
        assert self._path is not None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data.get("format") != 1:
                raise KeyManagementError("unsupported keystore format")
            for item in data["records"]:
                record = KeyRecord(
                    key_id=item["key_id"],
                    name=item["name"],
                    purpose=KeyPurpose(item["purpose"]),
                    version=int(item["version"]),
                    state=KeyState(item["state"]),
                    created_at=float(item["created_at"]),
                    rotate_after=float(item["rotate_after"]),
                    wrapped=base64.b64decode(item["wrapped"], validate=True),
                )
                if record.state is not KeyState.DESTROYED:
                    self._unwrap(record.key_id, record.wrapped)  # fail fast on wrong master key
                self._records[record.key_id] = record
                if record.state is KeyState.ACTIVE:
                    self._active[(record.purpose, record.name)] = record.key_id
        except KeyManagementError:
            raise
        except (OSError, ValueError, KeyError, TypeError, binascii.Error) as exc:
            raise KeyManagementError(f"cannot read keystore: {exc}") from exc
