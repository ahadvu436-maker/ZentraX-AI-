"""Resource limits enforced with POSIX ``setrlimit`` inside sandboxed processes."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..execution_core import ResourceLimitError, SandboxError

try:  # POSIX only
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = ["ResourceLimits", "AppliedLimits"]

_MB = 1024 * 1024


@dataclass(frozen=True)
class AppliedLimits:
    """Which limits were enforced and which could not be."""

    applied: Tuple[str, ...]
    skipped: Tuple[str, ...]


@dataclass(frozen=True)
class ResourceLimits:
    """Limits for one sandboxed run (``None`` disables that limit).

    Attributes:
        cpu_seconds: CPU time (RLIMIT_CPU). SIGXCPU is sent at the limit, SIGKILL one second later.
        memory_bytes: Address-space cap (RLIMIT_AS, falling back to RLIMIT_DATA).
            Note: address space includes mapped libraries; allow generous headroom for
            heavy libraries (numpy, torch).
        max_file_size_bytes: Largest file the process may write (RLIMIT_FSIZE).
        max_open_files: Open file descriptors (RLIMIT_NOFILE).
        max_processes: RLIMIT_NPROC. Counts per *user*, not per sandbox - use with care.
        wall_time_seconds: Wall-clock cap enforced by the parent (not an rlimit).
        disable_core_dumps: Set RLIMIT_CORE to 0.
        require_enforcement: If True, failing to apply a requested limit raises
            :class:`ResourceLimitError` instead of silently running unrestricted.
    """

    cpu_seconds: Optional[int] = 30
    memory_bytes: Optional[int] = 512 * _MB
    max_file_size_bytes: Optional[int] = 64 * _MB
    max_open_files: Optional[int] = 256
    max_processes: Optional[int] = None
    wall_time_seconds: Optional[float] = 60.0
    disable_core_dumps: bool = True
    require_enforcement: bool = True

    def __post_init__(self) -> None:
        for name in ("cpu_seconds", "memory_bytes", "max_file_size_bytes", "max_open_files",
                     "max_processes", "wall_time_seconds"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")

    # -- presets ------------------------------------------------------------ #
    @classmethod
    def strict(cls) -> "ResourceLimits":
        return cls(cpu_seconds=5, memory_bytes=256 * _MB, max_file_size_bytes=8 * _MB,
                   max_open_files=64, wall_time_seconds=15.0)

    @classmethod
    def relaxed(cls) -> "ResourceLimits":
        return cls(cpu_seconds=300, memory_bytes=4096 * _MB, max_file_size_bytes=1024 * _MB,
                   max_open_files=1024, wall_time_seconds=600.0)

    @classmethod
    def unlimited(cls) -> "ResourceLimits":
        return cls(None, None, None, None, None, None, disable_core_dumps=False,
                   require_enforcement=False)

    @staticmethod
    def supported() -> bool:
        """True when rlimits can be applied on this platform."""
        return resource is not None

    def has_rlimits(self) -> bool:
        return any(v is not None for v in (self.cpu_seconds, self.memory_bytes,
                                           self.max_file_size_bytes, self.max_open_files,
                                           self.max_processes)) or self.disable_core_dumps

    # -- enforcement -------------------------------------------------------- #
    def apply(self) -> AppliedLimits:
        """Apply the limits to the *current* process. Call only in a child process."""
        if resource is None:
            if self.has_rlimits() and self.require_enforcement:
                raise SandboxError("resource limits are unsupported on this platform")
            return AppliedLimits((), ("all",))

        requests: List[Tuple[str, Tuple[str, ...], int, int]] = []
        if self.cpu_seconds is not None:
            requests.append(("cpu", ("RLIMIT_CPU",), self.cpu_seconds, self.cpu_seconds + 1))
        if self.memory_bytes is not None:
            requests.append(("memory", ("RLIMIT_AS", "RLIMIT_DATA"), self.memory_bytes, self.memory_bytes))
        if self.max_file_size_bytes is not None:
            requests.append(("file_size", ("RLIMIT_FSIZE",), self.max_file_size_bytes, self.max_file_size_bytes))
        if self.max_open_files is not None:
            requests.append(("open_files", ("RLIMIT_NOFILE",), self.max_open_files, self.max_open_files))
        if self.max_processes is not None:
            requests.append(("processes", ("RLIMIT_NPROC",), self.max_processes, self.max_processes))
        if self.disable_core_dumps:
            requests.append(("core", ("RLIMIT_CORE",), 0, 0))

        applied: List[str] = []
        skipped: List[str] = []
        for label, names, soft, hard in requests:
            ok, last_error = False, None
            for rname in names:
                rid = getattr(resource, rname, None)
                if rid is None:
                    continue
                try:
                    self._set(rid, soft, hard)
                    ok = True
                    break
                except (ValueError, OSError) as exc:
                    last_error = exc
            (applied if ok else skipped).append(label)
            if not ok:
                logger.warning("could not apply %s limit: %s", label, last_error)
                if self.require_enforcement and label != "core":
                    raise ResourceLimitError(f"cannot enforce '{label}' limit: {last_error}")
        return AppliedLimits(tuple(applied), tuple(skipped))

    @staticmethod
    def _set(rid: int, soft: int, hard: int) -> None:
        _, cur_hard = resource.getrlimit(rid)
        if cur_hard != resource.RLIM_INFINITY:
            hard = min(hard, cur_hard)
            soft = min(soft, hard)
        resource.setrlimit(rid, (soft, hard))
