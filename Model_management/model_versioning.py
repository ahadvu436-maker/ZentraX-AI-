"""Semantic versioning, stage promotion, rollback and canary traffic splitting."""
from __future__ import annotations

import enum
import hashlib
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .model_core import VersionError

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.\-]+))?$")


@dataclass(frozen=True, order=True)
class SemVer:
    major: int
    minor: int
    patch: int
    # Release (no prerelease) sorts above prerelease: "" -> rank 1, text -> rank 0
    _rank: int = field(default=1, repr=False)
    prerelease: str = ""

    @classmethod
    def parse(cls, text: str) -> "SemVer":
        m = _SEMVER.match(text.strip())
        if not m:
            raise VersionError(f"Invalid semantic version: {text!r}")
        pre = m.group(4) or ""
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)), 0 if pre else 1, pre)

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.prerelease}" if self.prerelease else base


class VersionStage(str, enum.Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class VersionRecord:
    name: str
    version: SemVer
    model_id: str
    stage: VersionStage = VersionStage.DEVELOPMENT
    created_at: float = field(default_factory=time.time)
    notes: str = ""


class ModelVersionManager:
    """Tracks versions per logical model name. References look like ``name``
    (production, falling back to latest) or ``name@1.2.0``."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: Dict[str, Dict[SemVer, VersionRecord]] = {}
        self._prod_history: Dict[str, List[SemVer]] = {}
        self._canary: Dict[str, Tuple[SemVer, float]] = {}

    def register_version(self, name: str, version: str, model_id: str,
                         stage: VersionStage = VersionStage.DEVELOPMENT, notes: str = "") -> VersionRecord:
        v = SemVer.parse(version)
        with self._lock:
            versions = self._records.setdefault(name, {})
            if v in versions:
                raise VersionError(f"{name}@{v} already registered")
            rec = versions[v] = VersionRecord(name, v, model_id, VersionStage.DEVELOPMENT, notes=notes)
            if stage is not VersionStage.DEVELOPMENT:
                self.promote(name, str(v), stage)
            return rec

    def promote(self, name: str, version: str, stage: VersionStage) -> VersionRecord:
        with self._lock:
            rec = self._get(name, SemVer.parse(version))
            if stage is VersionStage.PRODUCTION:
                current = self._production(name)
                if current is not None and current.version != rec.version:
                    current.stage = VersionStage.STAGING
                hist = self._prod_history.setdefault(name, [])
                if not hist or hist[-1] != rec.version:
                    hist.append(rec.version)
            rec.stage = stage
            return rec

    def rollback(self, name: str) -> VersionRecord:
        """Restore the previous production version."""
        with self._lock:
            hist = self._prod_history.get(name, [])
            if len(hist) < 2:
                raise VersionError(f"No earlier production version for {name}")
            bad = hist.pop()
            self._get(name, bad).stage = VersionStage.ARCHIVED
            prev = self._get(name, hist[-1])
            prev.stage = VersionStage.PRODUCTION
            self._canary.pop(name, None)
            return prev

    def set_canary(self, name: str, version: str, weight: float) -> None:
        if not 0.0 < weight < 1.0:
            raise VersionError("canary weight must be between 0 and 1 (exclusive)")
        with self._lock:
            v = self._get(name, SemVer.parse(version)).version
            self._canary[name] = (v, weight)

    def clear_canary(self, name: str) -> None:
        with self._lock:
            self._canary.pop(name, None)

    def latest(self, name: str) -> VersionRecord:
        with self._lock:
            versions = [r for r in self._records.get(name, {}).values()
                        if r.stage is not VersionStage.ARCHIVED]
            if not versions:
                raise VersionError(f"No versions for {name}")
            return max(versions, key=lambda r: r.version)

    def list_versions(self, name: str) -> List[VersionRecord]:
        with self._lock:
            return sorted(self._records.get(name, {}).values(), key=lambda r: r.version)

    def knows(self, name: str) -> bool:
        with self._lock:
            return name in self._records

    def resolve(self, name: str, version: Optional[str] = None, *, key: Optional[str] = None) -> str:
        """Return the concrete model_id. Canary routing is deterministic per ``key``."""
        with self._lock:
            if version is not None:
                return self._get(name, SemVer.parse(version)).model_id
            canary = self._canary.get(name)
            if canary is not None:
                v, weight = canary
                roll = (int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                        if key else random.random())
                if roll < weight:
                    return self._get(name, v).model_id
            prod = self._production(name)
            return (prod or self.latest(name)).model_id

    def resolve_reference(self, ref: str, *, key: Optional[str] = None) -> str:
        """Resolve ``name`` / ``name@version``; unknown names pass through unchanged."""
        name, _, version = ref.partition("@")
        if not self.knows(name):
            return ref
        return self.resolve(name, version or None, key=key)

    # -- internals ---------------------------------------------------------- #
    def _get(self, name: str, v: SemVer) -> VersionRecord:
        try:
            return self._records[name][v]
        except KeyError:
            raise VersionError(f"Unknown version {name}@{v}") from None

    def _production(self, name: str) -> Optional[VersionRecord]:
        for r in self._records.get(name, {}).values():
            if r.stage is VersionStage.PRODUCTION:
                return r
        return None
