"""Deployment environment detection."""
from __future__ import annotations

import os
from enum import StrEnum


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION

    @property
    def is_deployed(self) -> bool:
        return self in (Environment.STAGING, Environment.PRODUCTION)


def current_environment(var: str = "APP_ENVIRONMENT") -> Environment:
    """Resolve the environment from ``var`` (defaults to development)."""
    raw = os.getenv(var, Environment.DEVELOPMENT.value).strip().lower()
    try:
        return Environment(raw)
    except ValueError as exc:
        allowed = ", ".join(e.value for e in Environment)
        raise ValueError(f"{var}={raw!r} is invalid; expected one of: {allowed}") from exc
