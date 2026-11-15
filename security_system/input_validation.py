"""Input validation and sanitisation.

Principle: *allow-list first*. Every validator states what is permitted and
rejects everything else. The injection signature scanner is a defence-in-depth
**signal** for the threat detector; it is not a substitute for parameterised
queries, output encoding and least-privilege execution.
"""

from __future__ import annotations

import html
import ipaddress
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .policies.security_policy import SecurityError

__all__ = ["ValidationError", "FieldSpec", "InputValidator"]

_USERNAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{2,63}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]{1,64}@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}")
_ACTION_RE = re.compile(r"[a-z][a-z0-9_.]{0,127}")
_RESOURCE_RE = re.compile(r"[A-Za-z0-9_.:/@\-]{1,256}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")

_INJECTION_SIGNATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sql_injection", re.compile(r"(?i)(?:\bunion\s+(?:all\s+)?select\b|\bor\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+|;\s*(?:drop|delete|truncate|insert|update)\b|--\s*$|/\*.*?\*/)")),
    ("xss", re.compile(r"(?i)(?:<\s*script|javascript\s*:|\bon[a-z]{3,20}\s*=|<\s*iframe|<\s*img[^>]+onerror)")),
    ("command_injection", re.compile(r"(?:[;&|]\s*(?:cat|ls|rm|curl|wget|nc|bash|sh|powershell|cmd)\b|\$\([^)]*\)|`[^`]*`)")),
    ("path_traversal", re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)|%2e%2e|\x00")),
    ("template_injection", re.compile(r"(?:\{\{.*?\}\}|\$\{.*?\}|<%.*?%>)")),
    ("ldap_injection", re.compile(r"[()*\\]\s*\(?\s*[|&!]")),
)


class ValidationError(SecurityError):
    """Raised when input does not satisfy its validation rule."""


@dataclass(frozen=True)
class FieldSpec:
    """Schema entry for :meth:`InputValidator.validate_mapping`."""

    validator: Callable[[Any], Any]
    required: bool = True


class InputValidator:
    """Collection of strict, allow-list based validators."""

    def __init__(self, *, max_scan_length: int = 100_000) -> None:
        self._max_scan_length = max_scan_length

    # ---------------------------------------------------------------- strings
    def string(
        self,
        value: Any,
        *,
        field: str = "value",
        min_length: int = 0,
        max_length: int = 1024,
        pattern: re.Pattern[str] | None = None,
        allow_newlines: bool = False,
        strip: bool = True,
    ) -> str:
        """Validate and normalise (NFKC) a string; reject control characters."""
        if not isinstance(value, str):
            raise ValidationError(f"{field} must be a string")
        if len(value) > max_length * 4:  # cheap pre-check before normalisation
            raise ValidationError(f"{field} is too long")
        text = unicodedata.normalize("NFKC", value)
        if strip:
            text = text.strip()
        if not (min_length <= len(text) <= max_length):
            raise ValidationError(f"{field} must be between {min_length} and {max_length} characters")
        for char in text:
            if allow_newlines and char in "\n\r\t":
                continue
            if unicodedata.category(char) in {"Cc", "Cf", "Cs", "Co", "Cn"}:
                raise ValidationError(f"{field} contains forbidden control or invisible characters")
        if pattern is not None and not pattern.fullmatch(text):
            raise ValidationError(f"{field} has an invalid format")
        return text

    def username(self, value: Any) -> str:
        return self.string(value, field="username", min_length=3, max_length=64, pattern=_USERNAME_RE)

    def email(self, value: Any) -> str:
        text = self.string(value, field="email", min_length=6, max_length=254, pattern=_EMAIL_RE)
        return text.lower()

    def action_name(self, value: Any) -> str:
        return self.string(value, field="action", min_length=1, max_length=128, pattern=_ACTION_RE)

    def resource_name(self, value: Any) -> str:
        """Concrete resource identifiers; wildcards are rejected on purpose."""
        text = self.string(value, field="resource", min_length=1, max_length=256, pattern=_RESOURCE_RE)
        if ".." in text.split("/"):
            raise ValidationError("resource contains path traversal")
        return text

    def identifier(self, value: Any, *, field: str = "identifier") -> str:
        return self.string(value, field=field, min_length=1, max_length=64, pattern=_IDENTIFIER_RE)

    # ---------------------------------------------------------------- numbers
    def integer(self, value: Any, *, minimum: int, maximum: int, field: str = "value") -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(f"{field} must be an integer")
        if not (minimum <= value <= maximum):
            raise ValidationError(f"{field} must be between {minimum} and {maximum}")
        return value

    # ----------------------------------------------------------- structured
    def ip_address(self, value: Any) -> str:
        text = self.string(value, field="ip", min_length=2, max_length=45)
        try:
            return str(ipaddress.ip_address(text))
        except ValueError as exc:
            raise ValidationError("ip must be a valid IPv4 or IPv6 address") from exc

    def uuid_str(self, value: Any, *, field: str = "id") -> str:
        text = self.string(value, field=field, min_length=32, max_length=36)
        try:
            return str(uuid.UUID(text))
        except ValueError as exc:
            raise ValidationError(f"{field} must be a UUID") from exc

    def validate_mapping(
        self,
        data: Any,
        schema: Mapping[str, FieldSpec],
        *,
        allow_extra: bool = False,
    ) -> dict[str, Any]:
        """Validate a dict against a schema. Unknown keys are rejected by default."""
        if not isinstance(data, Mapping):
            raise ValidationError("payload must be an object")
        extra = set(data) - set(schema)
        if extra and not allow_extra:
            raise ValidationError(f"unexpected fields: {sorted(str(k) for k in extra)}")
        result: dict[str, Any] = {}
        for key, spec in schema.items():
            if key not in data:
                if spec.required:
                    raise ValidationError(f"missing required field: {key}")
                continue
            result[key] = spec.validator(data[key])
        return result

    def check_limits(
        self,
        value: Any,
        *,
        max_depth: int = 10,
        max_items: int = 10_000,
        max_string: int = 65_536,
    ) -> None:
        """Bound the size and nesting of untrusted JSON-like data (DoS guard)."""
        count = 0
        stack: list[tuple[Any, int]] = [(value, 0)]
        while stack:
            item, depth = stack.pop()
            count += 1
            if count > max_items:
                raise ValidationError("payload has too many elements")
            if depth > max_depth:
                raise ValidationError("payload is nested too deeply")
            if isinstance(item, str) and len(item) > max_string:
                raise ValidationError("payload contains an oversized string")
            elif isinstance(item, Mapping):
                stack.extend((v, depth + 1) for v in item.values())
                stack.extend((k, depth + 1) for k in item.keys())
            elif isinstance(item, (list, tuple, set, frozenset)):
                stack.extend((v, depth + 1) for v in item)

    # ------------------------------------------------------------ filesystem
    @staticmethod
    def safe_join(base_dir: str | Path, user_path: str) -> Path:
        """Join ``user_path`` under ``base_dir``; reject traversal, absolute paths and symlink escapes."""
        if not isinstance(user_path, str) or "\x00" in user_path:
            raise ValidationError("invalid path")
        if Path(user_path).is_absolute() or re.match(r"^[A-Za-z]:", user_path) or user_path.startswith(("\\", "/")):
            raise ValidationError("absolute paths are not allowed")
        base = Path(base_dir).resolve()
        target = (base / user_path).resolve()
        if not target.is_relative_to(base):
            raise ValidationError("path escapes the permitted directory")
        return target

    # ----------------------------------------------------------- sanitising
    @staticmethod
    def escape_html(text: str) -> str:
        """Context-appropriate output encoding for HTML bodies and attributes."""
        return html.escape(text, quote=True)

    @staticmethod
    def sanitize_for_log(text: str, *, max_length: int = 256) -> str:
        """Neutralise log-forging characters and truncate."""
        cleaned = "".join(
            ch if (ch.isprintable() and ch not in "\r\n") else " " for ch in unicodedata.normalize("NFKC", text)
        )
        return cleaned[:max_length]

    def detect_injection(self, text: str) -> list[str]:
        """Return the names of attack signatures found in ``text`` (empty if none)."""
        if not isinstance(text, str):
            return []
        findings: list[str] = []
        if len(text) > self._max_scan_length:
            findings.append("oversized_payload")
            text = text[: self._max_scan_length]
        decoded = unicodedata.normalize("NFKC", text)
        for name, pattern in _INJECTION_SIGNATURES:
            if pattern.search(decoded):
                findings.append(name)
        return findings
