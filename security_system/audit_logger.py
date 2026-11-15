"""Tamper-evident audit logging.

Each record carries an HMAC-SHA256 computed over its canonical JSON body, which
includes the previous record's MAC. This forms a hash chain: altering,
reordering or deleting any record from the middle breaks verification.

Limits worth knowing:

* Truncating the *tail* of the log cannot be detected from the log alone.
  Periodically ship :meth:`AuditLogger.head` (sequence + MAC) to an external,
  write-once store to anchor the chain.
* Details are passed through the privacy policy redactor before being written,
  so secrets and personal data do not leak into the log.
* Logging fails *closed*: if a record cannot be durably written an
  :class:`AuditError` is raised and the caller must not proceed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from .policies.privacy_policy import PrivacyPolicy
from .policies.security_policy import SecurityError

__all__ = [
    "AuditError",
    "AuditSeverity",
    "AuditOutcome",
    "AuditRecord",
    "AuditSink",
    "FileAuditSink",
    "VerificationResult",
    "AuditLogger",
]

_GENESIS = "0" * 64


class AuditError(SecurityError):
    """Raised when an audit record cannot be written."""


class AuditSeverity(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class AuditOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"


@dataclass(frozen=True)
class AuditRecord:
    seq: int
    timestamp: float
    event: str
    actor: str
    action: str
    resource: str
    outcome: str
    severity: int
    details: Mapping[str, Any]
    prev_mac: str
    mac: str

    def body(self) -> dict[str, Any]:
        """Everything except the MAC (what the MAC authenticates)."""
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event": self.event,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "severity": self.severity,
            "details": self.details,
            "prev_mac": self.prev_mac,
        }

    def to_json(self) -> str:
        return json.dumps({**self.body(), "mac": self.mac}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_json(cls, line: str) -> AuditRecord:
        data = json.loads(line)
        return cls(
            seq=int(data["seq"]),
            timestamp=float(data["timestamp"]),
            event=str(data["event"]),
            actor=str(data["actor"]),
            action=str(data["action"]),
            resource=str(data["resource"]),
            outcome=str(data["outcome"]),
            severity=int(data["severity"]),
            details=data["details"],
            prev_mac=str(data["prev_mac"]),
            mac=str(data["mac"]),
        )


class AuditSink(Protocol):
    """Destination for serialised records. Must raise on failure."""

    def write(self, line: str) -> None: ...


class FileAuditSink:
    """Append-only JSON-lines file, fsync'd on every record, mode 0600."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, line: str) -> None:
        try:
            os.write(self._fd, line.encode("utf-8") + b"\n")
            os.fsync(self._fd)
        except OSError as exc:
            raise AuditError(f"audit sink write failed: {exc}") from exc

    def read_last_line(self) -> str | None:
        """Last non-empty line (reads the final 64 KiB), used to resume the chain."""
        try:
            size = self._path.stat().st_size
            if size == 0:
                return None
            with open(self._path, "rb") as handle:
                handle.seek(max(0, size - 65536))
                lines = [ln for ln in handle.read().splitlines() if ln.strip()]
            return lines[-1].decode("utf-8") if lines else None
        except OSError as exc:
            raise AuditError(f"cannot read audit log: {exc}") from exc

    def close(self) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    checked: int
    first_bad_seq: int | None = None
    reason: str = ""


class AuditLogger:
    """Thread-safe, hash-chained audit logger."""

    def __init__(
        self,
        hmac_key: bytes,
        *,
        sink: AuditSink | None = None,
        privacy: PrivacyPolicy | None = None,
        max_memory_records: int = 10_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if len(hmac_key) < 32:
            raise AuditError("audit HMAC key must be at least 32 bytes")
        self._key = hmac_key
        self._sink = sink
        self._privacy = privacy or PrivacyPolicy()
        self._clock = clock
        self._lock = threading.Lock()
        self._records: deque[AuditRecord] = deque(maxlen=max_memory_records)
        self._seq = 0
        self._last_mac = _GENESIS
        self._resume()

    # ----------------------------------------------------------------- write
    def log(
        self,
        event: str,
        *,
        actor: str,
        action: str,
        resource: str = "-",
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Mapping[str, Any] | None = None,
    ) -> AuditRecord:
        """Append a record. Raises :class:`AuditError` if it cannot be persisted."""
        safe_details = self._normalise(self._privacy.redact_value(dict(details or {})))
        with self._lock:
            unsigned = AuditRecord(
                seq=self._seq + 1,
                timestamp=self._clock(),
                event=self._privacy.redact(event),
                actor=self._privacy.redact(actor),
                action=self._privacy.redact(action),
                resource=self._privacy.redact(resource),
                outcome=outcome.value,
                severity=int(severity),
                details=safe_details,
                prev_mac=self._last_mac,
                mac="",
            )
            record = AuditRecord(**{**unsigned.__dict__, "mac": self._mac(unsigned.body())})
            if self._sink is not None:
                self._sink.write(record.to_json())  # raises AuditError on failure
            self._seq = record.seq
            self._last_mac = record.mac
            self._records.append(record)
            return record

    # ---------------------------------------------------------------- reading
    def records(self) -> list[AuditRecord]:
        """Snapshot of the in-memory (most recent) records."""
        with self._lock:
            return list(self._records)

    def head(self) -> tuple[int, str]:
        """``(sequence, mac)`` of the newest record; anchor it externally."""
        with self._lock:
            return self._seq, self._last_mac

    # ------------------------------------------------------------ verification
    def verify(self) -> VerificationResult:
        """Verify the in-memory chain (from the oldest retained record)."""
        with self._lock:
            records = list(self._records)
        expected_prev = records[0].prev_mac if records else None
        return self.verify_records(records, self._key, expected_prev=expected_prev)

    @staticmethod
    def verify_records(
        records: Iterable[AuditRecord],
        hmac_key: bytes,
        *,
        expected_prev: str | None = _GENESIS,
    ) -> VerificationResult:
        """Verify an arbitrary record sequence against ``hmac_key``."""
        previous_mac = expected_prev
        previous_seq: int | None = None
        checked = 0
        for record in records:
            expected = hmac.new(hmac_key, AuditLogger._canonical(record.body()), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, record.mac):
                return VerificationResult(False, checked, record.seq, "MAC mismatch (record altered)")
            if previous_mac is not None and record.prev_mac != previous_mac:
                return VerificationResult(False, checked, record.seq, "chain broken (record removed or reordered)")
            if previous_seq is not None and record.seq != previous_seq + 1:
                return VerificationResult(False, checked, record.seq, "sequence gap")
            previous_mac, previous_seq = record.mac, record.seq
            checked += 1
        return VerificationResult(True, checked)

    @classmethod
    def verify_file(cls, path: str | Path, hmac_key: bytes) -> VerificationResult:
        """Verify an entire audit file from the genesis record."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                records = [AuditRecord.from_json(line) for line in handle if line.strip()]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return VerificationResult(False, 0, None, f"unreadable log: {exc}")
        return cls.verify_records(records, hmac_key, expected_prev=_GENESIS)

    # --------------------------------------------------------------- internals
    @staticmethod
    def _canonical(body: Mapping[str, Any]) -> bytes:
        return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    def _mac(self, body: Mapping[str, Any]) -> str:
        return hmac.new(self._key, self._canonical(body), hashlib.sha256).hexdigest()

    @staticmethod
    def _normalise(value: Any) -> Any:
        """Force JSON-safe types so a record re-read from disk verifies identically."""
        return json.loads(json.dumps(value, default=str, sort_keys=True))

    def _resume(self) -> None:
        reader = getattr(self._sink, "read_last_line", None)
        if reader is None:
            return
        last = reader()
        if last:
            try:
                record = AuditRecord.from_json(last)
            except (ValueError, KeyError, TypeError) as exc:
                raise AuditError("existing audit log is corrupt; refusing to append") from exc
            expected = hmac.new(self._key, self._canonical(record.body()), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, record.mac):
                raise AuditError("existing audit log failed verification; refusing to append")
            self._seq = record.seq
            self._last_mac = record.mac
