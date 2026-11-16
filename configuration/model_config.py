"""ML model / inference settings."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "default"
    model_version: str = "latest"
    model_dir: Path = Path("/var/lib/app/models")
    max_batch_size: int = Field(32, ge=1)
    inference_timeout_s: float = Field(10.0, gt=0)
    warmup_on_start: bool = True
