"""Authenticated encryption, envelope encryption, signing and key derivation.

Primitives (all from the ``cryptography`` library, never hand-rolled):

* AES-256-GCM and ChaCha20-Poly1305 for authenticated encryption (AEAD)
* HKDF-SHA256 for key separation
* HMAC-SHA256 for signatures, verified in constant time

Wire format (versioned so algorithms can be migrated)::

    "SSE1" | cipher_id (1) | key_id_len (1) | key_id | nonce (12) | ciphertext+tag

The whole header plus caller AAD is authenticated, so a ciphertext cannot be
re-labelled with a different key id or cipher, nor replayed under another
context (pass a context string such as ``b"user:42:profile"`` as ``aad``).

Random 96-bit nonces are safe for up to ~2**32 messages per key; the key
rotation policy in :class:`~Security_system.policies.CryptoPolicy` keeps usage
well below that bound.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from enum import Enum

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .key_management import KeyManagementError, KeyManager, KeyPurpose
from .policies.security_policy import CryptoPolicy, SecurityError

__all__ = [
    "EncryptionError",
    "DecryptionError",
    "Cipher",
    "EncryptionService",
]

_MAGIC = b"SSE1"
_ENV_MAGIC = b"SSE2"
_NONCE_LEN = 12


class EncryptionError(SecurityError):
    """Raised when encryption or signing cannot be performed."""


class DecryptionError(EncryptionError):
    """Raised on any decryption failure.

    The message is deliberately uniform: distinguishing "bad key", "bad tag" and
    "bad format" would give an attacker an oracle.
    """


class Cipher(str, Enum):
    AES_256_GCM = "AES-256-GCM"
    CHACHA20_POLY1305 = "ChaCha20-Poly1305"


_CIPHER_IDS: dict[Cipher, int] = {Cipher.AES_256_GCM: 1, Cipher.CHACHA20_POLY1305: 2}
_ID_TO_CIPHER: dict[int, Cipher] = {v: k for k, v in _CIPHER_IDS.items()}


def _aead(cipher: Cipher, key: bytes) -> AESGCM | ChaCha20Poly1305:
    if cipher is Cipher.AES_256_GCM:
        return AESGCM(key)
    return ChaCha20Poly1305(key)


def _full_aad(header: bytes, aad: bytes) -> bytes:
    # Length-prefix the caller AAD so (header, aad) pairs cannot be confused.
    return header + struct.pack(">I", len(aad)) + aad


class EncryptionService:
    """High-level encryption API backed by a :class:`KeyManager`."""

    def __init__(
        self,
        key_manager: KeyManager,
        policy: CryptoPolicy | None = None,
        *,
        default_cipher: Cipher = Cipher.AES_256_GCM,
    ) -> None:
        self._km = key_manager
        self._policy = policy or CryptoPolicy()
        self._check_cipher(default_cipher)
        self._default_cipher = default_cipher

    def _check_cipher(self, cipher: Cipher) -> None:
        if cipher.value not in self._policy.allowed_ciphers:
            raise EncryptionError(f"cipher {cipher.value} is not permitted by policy")

    # ------------------------------------------------------------------ AEAD
    def encrypt(
        self,
        plaintext: bytes,
        *,
        aad: bytes = b"",
        key_id: str | None = None,
        cipher: Cipher | None = None,
    ) -> bytes:
        """Encrypt ``plaintext`` under the active (or given) data key."""
        selected = cipher or self._default_cipher
        self._check_cipher(selected)
        try:
            kid = key_id or self._km.active_key_id(KeyPurpose.ENCRYPTION)
            key = self._km.get_material(kid, for_encryption=True)
        except KeyManagementError as exc:
            raise EncryptionError(str(exc)) from exc
        kid_bytes = kid.encode("utf-8")
        if len(kid_bytes) > 255:
            raise EncryptionError("key id too long")
        header = _MAGIC + bytes([_CIPHER_IDS[selected], len(kid_bytes)]) + kid_bytes
        nonce = os.urandom(_NONCE_LEN)
        ciphertext = _aead(selected, key).encrypt(nonce, plaintext, _full_aad(header, aad))
        return header + nonce + ciphertext

    def decrypt(self, blob: bytes, *, aad: bytes = b"") -> bytes:
        """Decrypt and authenticate a blob produced by :meth:`encrypt`."""
        try:
            if blob[:4] != _MAGIC:
                raise DecryptionError("decryption failed")
            cipher = _ID_TO_CIPHER[blob[4]]
            self._check_cipher(cipher)
            kid_len = blob[5]
            kid_end = 6 + kid_len
            kid = blob[6:kid_end].decode("utf-8")
            header = blob[:kid_end]
            nonce = blob[kid_end : kid_end + _NONCE_LEN]
            ciphertext = blob[kid_end + _NONCE_LEN :]
            if len(nonce) != _NONCE_LEN or not ciphertext:
                raise DecryptionError("decryption failed")
            key = self._km.get_material(kid)
            return _aead(cipher, key).decrypt(nonce, ciphertext, _full_aad(header, aad))
        except DecryptionError:
            raise
        except (InvalidTag, KeyError, IndexError, ValueError, UnicodeError, SecurityError) as exc:
            raise DecryptionError("decryption failed") from exc

    def encrypt_str(self, text: str, *, aad: bytes = b"") -> bytes:
        return self.encrypt(text.encode("utf-8"), aad=aad)

    def decrypt_str(self, blob: bytes, *, aad: bytes = b"") -> str:
        try:
            return self.decrypt(blob, aad=aad).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecryptionError("decryption failed") from exc

    # -------------------------------------------------------------- envelope
    def encrypt_envelope(self, plaintext: bytes, *, aad: bytes = b"") -> bytes:
        """Envelope encryption: a fresh data key per message, wrapped by a KEK.

        Use this for large or high-value payloads and for data you may need to
        re-protect cheaply: rotating the KEK only requires re-wrapping the small
        per-message data key, not re-encrypting the payload.
        """
        self._check_cipher(Cipher.AES_256_GCM)
        try:
            kek_id = self._km.active_key_id(KeyPurpose.WRAPPING)
            kek = self._km.get_material(kek_id, for_encryption=True)
        except KeyManagementError as exc:
            raise EncryptionError(str(exc)) from exc
        kek_bytes = kek_id.encode("utf-8")
        dek = AESGCM.generate_key(bit_length=256)
        wrap_nonce = os.urandom(_NONCE_LEN)
        wrapped = wrap_nonce + AESGCM(kek).encrypt(wrap_nonce, dek, kek_bytes)
        header = _ENV_MAGIC + bytes([len(kek_bytes)]) + kek_bytes + struct.pack(">H", len(wrapped)) + wrapped
        nonce = os.urandom(_NONCE_LEN)
        ciphertext = AESGCM(dek).encrypt(nonce, plaintext, _full_aad(header, aad))
        return header + nonce + ciphertext

    def decrypt_envelope(self, blob: bytes, *, aad: bytes = b"") -> bytes:
        """Reverse of :meth:`encrypt_envelope`."""
        try:
            if blob[:4] != _ENV_MAGIC:
                raise DecryptionError("decryption failed")
            kek_len = blob[4]
            kek_end = 5 + kek_len
            kek_id = blob[5:kek_end].decode("utf-8")
            (wrapped_len,) = struct.unpack(">H", blob[kek_end : kek_end + 2])
            wrapped_start = kek_end + 2
            wrapped_end = wrapped_start + wrapped_len
            wrapped = blob[wrapped_start:wrapped_end]
            header = blob[:wrapped_end]
            nonce = blob[wrapped_end : wrapped_end + _NONCE_LEN]
            ciphertext = blob[wrapped_end + _NONCE_LEN :]
            if len(nonce) != _NONCE_LEN or not ciphertext or len(wrapped) <= _NONCE_LEN:
                raise DecryptionError("decryption failed")
            kek = self._km.get_material(kek_id)
            dek = AESGCM(kek).decrypt(wrapped[:_NONCE_LEN], wrapped[_NONCE_LEN:], kek_id.encode("utf-8"))
            return AESGCM(dek).decrypt(nonce, ciphertext, _full_aad(header, aad))
        except DecryptionError:
            raise
        except (InvalidTag, struct.error, IndexError, ValueError, UnicodeError, SecurityError) as exc:
            raise DecryptionError("decryption failed") from exc

    # --------------------------------------------------------------- signing
    def sign(self, data: bytes, *, key_id: str | None = None) -> bytes:
        """HMAC-SHA256 signature, prefixed with the signing key id for rotation."""
        try:
            kid = key_id or self._km.active_key_id(KeyPurpose.SIGNING)
            key = self._km.get_material(kid, for_encryption=True)
        except KeyManagementError as exc:
            raise EncryptionError(str(exc)) from exc
        kid_bytes = kid.encode("utf-8")
        mac = hmac.new(key, data, hashlib.sha256).digest()
        return bytes([len(kid_bytes)]) + kid_bytes + mac

    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Constant-time verification. Returns ``False`` for any invalid input."""
        try:
            kid_len = signature[0]
            kid = signature[1 : 1 + kid_len].decode("utf-8")
            mac = signature[1 + kid_len :]
            if len(mac) != hashlib.sha256().digest_size:
                return False
            key = self._km.get_material(kid)
            return hmac.compare_digest(hmac.new(key, data, hashlib.sha256).digest(), mac)
        except (IndexError, UnicodeError, SecurityError):
            return False

    # ---------------------------------------------------------------- utility
    @staticmethod
    def hkdf_derive(ikm: bytes, *, info: bytes, length: int = 32, salt: bytes | None = None) -> bytes:
        """Derive a purpose-bound subkey (key separation) with HKDF-SHA256."""
        return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)

    @staticmethod
    def secure_compare(a: bytes, b: bytes) -> bool:
        """Constant-time equality."""
        return hmac.compare_digest(a, b)

    @staticmethod
    def random_bytes(length: int = 32) -> bytes:
        """CSPRNG bytes."""
        return os.urandom(length)
