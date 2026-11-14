"""Baseline interface for authentication."""
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class Result:
    status: str
    details: Mapping[str, Any]

class Authentication:
    def inspect(self, payload: Mapping[str, Any] | None = None) -> Result:
        return Result("baseline_ready", {"module": "authentication", "input_keys": sorted((payload or {}).keys())})
