"""Baseline interface for access control."""
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class Result:
    status: str
    details: Mapping[str, Any]

class AccessControl:
    def inspect(self, payload: Mapping[str, Any] | None = None) -> Result:
        return Result("baseline_ready", {"module": "access_control", "input_keys": sorted((payload or {}).keys())})
