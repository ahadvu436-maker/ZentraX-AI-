"""AI/ML model hyper-parameters, runtime options and filesystem paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from .constants import (
    DEFAULT_MODELS_DIR,
    ENV_PREFIX_MODEL,
    SEMVER_PATTERN,
    Device,
    Precision,
)
from .environment import EnvBaseModel, resolve_path


class ModelSettings(EnvBaseModel):
    """Model options (env prefix ``MODEL_``; e.g. ``MODEL_LEARNING_RATE``)."""

    env_prefix: ClassVar[str] = ENV_PREFIX_MODEL

    # Identity & paths
    name: str = Field(default="base-model", pattern=r"^[A-Za-z0-9._/-]+$", max_length=128)
    version: str = Field(default="1.0.0", pattern=SEMVER_PATTERN)
    models_dir: Path = DEFAULT_MODELS_DIR
    checkpoint_dir: Path = DEFAULT_MODELS_DIR / "checkpoints"
    cache_dir: Path = DEFAULT_MODELS_DIR / ".cache"

    # Runtime
    device: Device = Device.AUTO
    precision: Precision = Precision.FP32
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    num_workers: int = Field(default=0, ge=0, le=256)

    # Architecture / data
    max_sequence_length: int = Field(default=2_048, ge=1, le=1_048_576)
    batch_size: int = Field(default=32, ge=1, le=65_536)
    eval_batch_size: int = Field(default=64, ge=1, le=65_536)

    # Optimisation
    learning_rate: float = Field(default=3e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    epochs: int = Field(default=3, ge=1, le=10_000)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=4_096)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    early_stopping_patience: int = Field(default=3, ge=0, le=1_000)

    # Generation / inference
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=100_000)
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
    max_new_tokens: int = Field(default=512, ge=1, le=1_048_576)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @field_validator("models_dir", "checkpoint_dir", "cache_dir")
    @classmethod
    def _resolve_dirs(cls, value: Path) -> Path:
        return resolve_path(value)

    @field_validator("device", "precision", mode="before")
    @classmethod
    def _lower_enums(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_consistency(self) -> ModelSettings:
        if self.max_new_tokens > self.max_sequence_length:
            raise ValueError("MODEL_MAX_NEW_TOKENS must not exceed MODEL_MAX_SEQUENCE_LENGTH")
        return self

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def model_dir(self) -> Path:
        """Directory holding artifacts of the active model name + version."""
        return self.models_dir / self.name / self.version

    def checkpoint_path(self, tag: str = "latest") -> Path:
        safe_tag = Path(tag).name  # prevents path traversal
        return self.checkpoint_dir / self.name / f"{self.version}-{safe_tag}.ckpt"

    def ensure_directories(self) -> None:
        for directory in (self.models_dir, self.checkpoint_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """Concrete device string. ``auto`` probes torch lazily (optional dep)."""
        if self.device is not Device.AUTO:
            return self.device.value
        try:
            import torch  # type: ignore[import-not-found,unused-ignore]
        except ImportError:
            return Device.CPU.value
        if torch.cuda.is_available():
            return Device.CUDA.value
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return Device.MPS.value
        return Device.CPU.value

    def training_kwargs(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "warmup_ratio": self.warmup_ratio,
            "dropout": self.dropout,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "early_stopping_patience": self.early_stopping_patience,
            "precision": self.precision.value,
            "seed": self.seed,
        }

    def generation_kwargs(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
