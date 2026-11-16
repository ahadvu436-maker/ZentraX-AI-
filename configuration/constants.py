"""Immutable project-wide constants. Nothing here reads the environment."""
from __future__ import annotations

from typing import Final

APP_NAME: Final[str] = "ai-platform"
ENV_PREFIX: Final[str] = "APP_"
ENV_NESTED_DELIMITER: Final[str] = "__"

DEFAULT_BATCH_SIZE: Final[int] = 500
DEFAULT_MAX_WORKERS: Final[int] = 4
DEFAULT_MAX_REJECT_RATIO: Final[float] = 0.2

HEALTH_PATH: Final[str] = "/healthz"
READINESS_PATH: Final[str] = "/readyz"
METRICS_PATH: Final[str] = "/metrics"

REQUEST_ID_HEADER: Final[str] = "X-Request-ID"
INSECURE_SECRET_PLACEHOLDER: Final[str] = "change-me"
