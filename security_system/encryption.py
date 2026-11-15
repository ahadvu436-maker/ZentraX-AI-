import os
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    AESGCM = None

class EnvelopeCipher:
    """AES-256-GCM authenticated encryption. Requires `cryptography`; no insecure fallback."""
    def __init__(self, key: bytes):
        if AESGCM is None: raise RuntimeError("Install cryptography to use EnvelopeCipher")
        if len(key) != 32: raise ValueError("AES-256 key must be exactly 32 bytes")
        self._cipher = AESGCM(key)
    @staticmethod
    def generate_key(): return AESGCM.generate_key(bit_length=256) if AESGCM else os.urandom(32)
    def encrypt(self, plaintext: bytes, associated_data: bytes = b""):
        nonce = os.urandom(12)
        return nonce + self._cipher.encrypt(nonce, plaintext, associated_data)
    def decrypt(self, ciphertext: bytes, associated_data: bytes = b""):
        if len(ciphertext) < 28: raise ValueError("ciphertext too short")
        return self._cipher.decrypt(ciphertext[:12], ciphertext[12:], associated_data)
