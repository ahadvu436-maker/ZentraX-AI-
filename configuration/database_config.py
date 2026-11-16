"""Database connection settings."""
from __future__ import annotations

from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, Field, SecretStr


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = Field(5432, ge=1, le=65535)
    user: str = "app"
    password: SecretStr = SecretStr("")
    name: str = "app"
    pool_size: int = Field(10, ge=1, le=200)
    pool_timeout_s: float = Field(30.0, gt=0)
    sslmode: Literal["disable", "require", "verify-ca", "verify-full"] = "disable"

    def dsn(self, *, masked: bool = True) -> str:
        pwd = "***" if masked else self.password.get_secret_value()
        return (
            f"postgresql://{quote(self.user)}:{quote(pwd, safe='*')}"
            f"@{self.host}:{self.port}/{self.name}?sslmode={self.sslmode}"
        )
