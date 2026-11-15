import base64, hashlib, hmac, json, time
from dataclasses import dataclass

@dataclass(frozen=True)
class Principal:
    subject: str
    roles: tuple[str, ...]
    expires_at: int

class Authenticator:
    """HMAC-signed compact bearer tokens. Use a high-entropy secret from a secret store."""
    def __init__(self, signing_key: bytes, issuer="zentrax", clock=time.time):
        if len(signing_key) < 32: raise ValueError("signing_key must be at least 32 bytes")
        self.key, self.issuer, self.clock = signing_key, issuer, clock

    @staticmethod
    def _b64(data): return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def issue(self, subject, roles, ttl_seconds=900):
        if not subject or ttl_seconds <= 0: raise ValueError("invalid subject or TTL")
        payload = {"sub": subject, "roles": sorted(set(roles)), "exp": int(self.clock()) + ttl_seconds, "iss": self.issuer}
        body = self._b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        sig = self._b64(hmac.new(self.key, body.encode(), hashlib.sha256).digest())
        return body + "." + sig

    def verify(self, token):
        try:
            body, signature = token.split(".", 1)
            expected = self._b64(hmac.new(self.key, body.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected): return None
            raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
            p = json.loads(raw)
            if p.get("iss") != self.issuer or int(p["exp"]) <= int(self.clock()): return None
            return Principal(str(p["sub"]), tuple(p.get("roles", [])), int(p["exp"]))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
