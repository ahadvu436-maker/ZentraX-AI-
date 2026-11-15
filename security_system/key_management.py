import secrets, threading, time
class InMemoryKeyStore:
    """Development-only key store. Production deployments must use KMS/HSM."""
    def __init__(self): self._keys, self._lock = {}, threading.RLock()
    def create(self, key_id=None, length=32):
        key_id = key_id or secrets.token_hex(12)
        with self._lock:
            if key_id in self._keys: raise KeyError("key id already exists")
            self._keys[key_id] = {"key": secrets.token_bytes(length), "created": time.time(), "active": True}
        return key_id
    def get(self, key_id):
        with self._lock:
            item = self._keys.get(key_id)
            if not item or not item["active"]: raise KeyError("key unavailable")
            return bytes(item["key"])
    def revoke(self, key_id):
        with self._lock:
            if key_id not in self._keys: raise KeyError(key_id)
            self._keys[key_id]["active"] = False
