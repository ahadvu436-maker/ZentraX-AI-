"""Security settings. Secrets are ``SecretStr`` so they never leak via repr/logs."""
from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr

from .constants import INSECURE_SECRET_PLACEHOLDER


class SecuritySettings(BaseModel):
    secret_key: SecretStr = SecretStr(INSECURE_SECRET_PLACEHOLDER)
    jwt_algorithm: str = "HS256"
    access_token_ttl_s: int = Field(900, ge=60)
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost"])
    cors_origins: list[str] = Field(default_factory=list)
    rate_limit_per_minute: int = Field(120, ge=1)

    @property
    def has_default_secret(self) -> bool:
        v = self.secret_key.get_secret_value()
        return v == INSECURE_SECRET_PLACEHOLDER or len(v) < 32
