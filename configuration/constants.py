"""Global, import-safe system constants.

This module has **no** dependency on any other module of the package so it can
be imported from anywhere without risking circular imports. Nothing here reads
the environment or touches the filesystem.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Application identity
# --------------------------------------------------------------------------- #
APP_NAME_DEFAULT: Final[str] = "ZentraX-AI"
APP_VERSION_DEFAULT: Final[str] = "0.1.0"
SEMVER_PATTERN: Final[str] = (
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
BASE_DIR: Final[Path] = PACKAGE_DIR.parent
DEFAULT_DATA_DIR: Final[Path] = BASE_DIR / "data"
DEFAULT_LOG_DIR: Final[Path] = BASE_DIR / "logs"
DEFAULT_MODELS_DIR: Final[Path] = BASE_DIR / "models"

# --------------------------------------------------------------------------- #
# Environment variable naming
# --------------------------------------------------------------------------- #
ENV_VAR_ENVIRONMENT: Final[str] = "APP_ENV"
DOTENV_BASENAME: Final[str] = ".env"

ENV_PREFIX_APP: Final[str] = "APP_"
ENV_PREFIX_LOGGING: Final[str] = "LOG_"
ENV_PREFIX_DATABASE: Final[str] = "DATABASE_"
ENV_PREFIX_MODEL: Final[str] = "MODEL_"
ENV_PREFIX_SECURITY: Final[str] = "SECURITY_"
ENV_VAR_SECRET_KEY: Final[str] = "SECURITY_SECRET_KEY"

# --------------------------------------------------------------------------- #
# Time & size helpers
# --------------------------------------------------------------------------- #
SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 3_600
SECONDS_PER_DAY: Final[int] = 86_400
KIB: Final[int] = 1_024
MIB: Final[int] = 1_024 * KIB
GIB: Final[int] = 1_024 * MIB

# --------------------------------------------------------------------------- #
# Security constants
# --------------------------------------------------------------------------- #
MIN_SECRET_KEY_LENGTH: Final[int] = 32
MIN_SECRET_KEY_UNIQUE_CHARS: Final[int] = 10
FERNET_KEY_BYTES: Final[int] = 32
DEFAULT_JWT_ISSUER: Final[str] = "zentrax-ai"
DEFAULT_JWT_AUDIENCE: Final[str] = "zentrax-ai-clients"

#: Substrings that mark a secret key as a placeholder / guessable value.
WEAK_SECRET_MARKERS: Final[tuple[str, ...]] = (
    "changeme",
    "change-me",
    "change_me",
    "replace-me",
    "replace_me",
    "your-secret",
    "your_secret",
    "secret",
    "password",
    "example",
    "default",
    "12345",
    "qwerty",
    "letmein",
    "admin",
)

# --------------------------------------------------------------------------- #
# Database constants
# --------------------------------------------------------------------------- #
ALLOWED_DATABASE_SCHEMES: Final[frozenset[str]] = frozenset(
    {
        "sqlite",
        "sqlite+aiosqlite",
        "postgresql",
        "postgresql+psycopg",
        "postgresql+psycopg2",
        "postgresql+asyncpg",
        "mysql",
        "mysql+pymysql",
        "mysql+aiomysql",
        "mariadb",
        "mariadb+pymysql",
    }
)
DEFAULT_DATABASE_URL: Final[str] = "sqlite:///./data/app.db"

# --------------------------------------------------------------------------- #
# Logging constants
# --------------------------------------------------------------------------- #
DEFAULT_LOG_FILE_NAME: Final[str] = "app.log"
LOG_CONTEXT_CORRELATION_KEY: Final[str] = "correlation_id"
DEFAULT_QUIET_LOGGERS: Final[tuple[str, ...]] = ("urllib3", "asyncio", "botocore")


# --------------------------------------------------------------------------- #
# Shared enumerations
# --------------------------------------------------------------------------- #
class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    JSON = "json"
    CONSOLE = "console"


class Device(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class JWTAlgorithm(StrEnum):
    """Symmetric (HMAC) algorithms only: they sign with ``secret_key``."""

    HS256 = "HS256"
    HS384 = "HS384"
    HS512 = "HS512"


class SameSite(StrEnum):
    STRICT = "strict"
    LAX = "lax"
    NONE = "none"


class PasswordHashScheme(StrEnum):
    ARGON2ID = "argon2id"
    BCRYPT = "bcrypt"


class DatabaseSSLMode(StrEnum):
    DISABLE = "disable"
    PREFER = "prefer"
    REQUIRE = "require"
    VERIFY_CA = "verify-ca"
    VERIFY_FULL = "verify-full"
