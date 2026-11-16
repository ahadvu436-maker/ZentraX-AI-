"""Environment detection, ``.env`` loading and the env-aware base model.

Value precedence (highest first)::

    real process environment  >  .env.<environment>  >  .env  >  field defaults
                                                         (incl. environment profile)

``.env`` files are *read*, never injected into ``os.environ``, so loading
settings has no global side effects and is trivial to test.
"""

from __future__ import annotations

import json
import os
import types
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Final, Self, Union, get_args, get_origin

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, ValidationError

from .constants import BASE_DIR, DOTENV_BASENAME, ENV_VAR_ENVIRONMENT


class ConfigurationError(Exception):
    """Raised when configuration is missing, malformed or insecure."""


# --------------------------------------------------------------------------- #
# Environment mode
# --------------------------------------------------------------------------- #
_ALIASES: Final[dict[str, str]] = {
    "dev": "development",
    "develop": "development",
    "local": "development",
    "stage": "staging",
    "stg": "staging",
    "prod": "production",
    "live": "production",
    "test": "testing",
}

#: Environment specific *defaults* per config section. They sit below real
#: environment variables, so ops can always override them.
_PROFILES: Final[dict[str, dict[str, dict[str, Any]]]] = {
    "development": {
        "app": {"debug": True},
        "logging": {"level": "DEBUG", "format": "console"},
        "security": {"cookie_secure": False, "tls_required": False},
    },
    "testing": {
        "app": {"debug": False},
        "logging": {"level": "WARNING", "format": "console"},
        "security": {"cookie_secure": False, "tls_required": False},
    },
    "staging": {
        "app": {"debug": False},
        "logging": {"level": "INFO", "format": "json"},
    },
    "production": {
        "app": {"debug": False},
        "logging": {"level": "INFO", "format": "json"},
    },
}


class Environment(StrEnum):
    """Deployment mode. ``StrEnum`` so it serialises as a plain string."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

    @classmethod
    def parse(cls, value: str | Environment) -> Environment:
        """Parse a user supplied value (accepts aliases such as ``prod``)."""
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        key = _ALIASES.get(key, key)
        try:
            return cls(key)
        except ValueError:
            allowed = ", ".join(m.value for m in cls)
            raise ConfigurationError(
                f"Invalid environment {value!r}. Expected one of: {allowed}."
            ) from None

    @property
    def is_development(self) -> bool:
        return self is Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        return self is Environment.TESTING

    @property
    def is_staging(self) -> bool:
        return self is Environment.STAGING

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION

    @property
    def is_strict(self) -> bool:
        """Staging and production enforce hardened settings."""
        return self in (Environment.STAGING, Environment.PRODUCTION)

    def profile(self, section: str) -> dict[str, Any]:
        """Default values this environment applies to ``section``."""
        return dict(_PROFILES.get(self.value, {}).get(section, {}))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def resolve_path(value: Path | str, base: Path = BASE_DIR) -> Path:
    """Expand ``~`` and anchor relative paths at the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def resolve_environment(
    explicit: str | Environment | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_dir: Path | str | None = None,
) -> Environment:
    """Work out the active environment.

    Order: explicit argument, ``APP_ENV`` in the process environment,
    ``APP_ENV`` in the base ``.env`` file, then ``development``.
    """
    if explicit is not None:
        return Environment.parse(explicit)
    source = os.environ if environ is None else environ
    raw = source.get(ENV_VAR_ENVIRONMENT, "").strip()
    if not raw:
        directory = Path(env_dir) if env_dir else BASE_DIR
        raw = _read_dotenv(directory / DOTENV_BASENAME).get(
            ENV_VAR_ENVIRONMENT, ""
        ).strip()
    return Environment.parse(raw) if raw else Environment.DEVELOPMENT


def load_environment(
    explicit: str | Environment | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_dir: Path | str | None = None,
) -> tuple[Environment, dict[str, str]]:
    """Return ``(environment, merged variables)`` without mutating ``os.environ``."""
    source = dict(os.environ if environ is None else environ)
    environment = resolve_environment(explicit, environ=source, env_dir=env_dir)
    directory = Path(env_dir) if env_dir else BASE_DIR

    merged: dict[str, str] = {}
    merged.update(_read_dotenv(directory / DOTENV_BASENAME))
    merged.update(_read_dotenv(directory / f"{DOTENV_BASENAME}.{environment.value}"))
    merged.update(source)
    merged[ENV_VAR_ENVIRONMENT] = environment.value
    return environment, merged


# --------------------------------------------------------------------------- #
# Env-aware base model
# --------------------------------------------------------------------------- #
def _is_collection(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        return True
    if origin in (Union, types.UnionType):
        return any(_is_collection(arg) for arg in get_args(annotation))
    return annotation in (list, tuple, set, frozenset)


def _parse_collection(raw: str) -> list[str]:
    """Accept a JSON array (``["a","b"]``) or a comma separated list."""
    text = raw.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON list: {exc.msg}") from exc
        if not isinstance(parsed, list):
            raise ConfigurationError("Expected a JSON array.")
        return [str(item).strip() for item in parsed]
    return [part.strip() for part in text.split(",") if part.strip()]


def _format_errors(model: type[BaseModel], exc: ValidationError) -> str:
    """Readable, secret-free error summary (never echoes input values)."""
    prefix = getattr(model, "env_prefix", "")
    lines = [f"Invalid {model.__name__}:"]
    for err in exc.errors(include_url=False, include_context=False, include_input=False):
        loc = err.get("loc", ())
        where = f"{prefix}{loc[0]}".upper() if loc else model.__name__
        lines.append(f"  - {where}: {err['msg']}")
    return "\n".join(lines)


class EnvBaseModel(BaseModel):
    """Immutable, strictly validated settings section loaded from env vars.

    Each field ``foo`` is read from ``<env_prefix>FOO``. Empty variables are
    treated as unset. Collection fields accept JSON arrays or comma lists.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
        protected_namespaces=(),
    )

    env_prefix: ClassVar[str] = ""
    env_reserved: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def env_key(cls, field_name: str) -> str:
        return f"{cls.env_prefix}{field_name}".upper()

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        defaults: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> Self:
        """Build the section. Precedence: overrides > env vars > defaults."""
        source = os.environ if environ is None else environ
        data: dict[str, Any] = dict(defaults or {})
        for name, field in cls.model_fields.items():
            raw = source.get(cls.env_key(name))
            if raw is None or not raw.strip():
                continue
            if _is_collection(field.annotation):
                try:
                    data[name] = _parse_collection(raw)
                except ConfigurationError as exc:
                    raise ConfigurationError(
                        f"Invalid {cls.__name__}:\n  - {cls.env_key(name)}: {exc}"
                    ) from exc
            else:
                data[name] = raw
        data.update(overrides)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigurationError(_format_errors(cls, exc)) from exc

    @classmethod
    def unknown_env_keys(cls, environ: Mapping[str, str]) -> list[str]:
        """Variables using this section's prefix that match no field (typos)."""
        known = {cls.env_key(n) for n in cls.model_fields} | set(cls.env_reserved)
        prefix = cls.env_prefix.upper()
        return sorted(
            key for key in environ
            if key.upper().startswith(prefix) and key.upper() not in known
        )
