"""Provider for models on local disk, with path-confinement security."""
from __future__ import annotations

import gc
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Union

from ..model_core import (
    InferenceRequest,
    InvalidRequestError,
    ModelLoadError,
    ModelProvider,
    ModelSecurityError,
    ModelSpec,
    ModelTask,
    ProviderError,
    ProviderResult,
    ProviderType,
    estimate_tokens,
)
from ..model_optimizer import OptimizationConfig, Quantization

logger = logging.getLogger(__name__)


class LocalBackend(Protocol):
    """Pluggable runtime (transformers, llama.cpp, ONNX, ...)."""

    def load(self, spec: ModelSpec, path: Path) -> Any: ...
    def infer(self, spec: ModelSpec, handle: Any, request: InferenceRequest) -> ProviderResult: ...


_HF_TASKS: Mapping[ModelTask, str] = {
    ModelTask.TEXT_GENERATION: "text-generation",
    ModelTask.EMBEDDING: "feature-extraction",
    ModelTask.CLASSIFICATION: "text-classification",
}


class TransformersBackend:
    """HuggingFace ``pipeline`` backend. ``transformers`` is imported lazily and
    ``trust_remote_code`` is never enabled."""

    def load(self, spec: ModelSpec, path: Path) -> Any:
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ModelLoadError("The 'transformers' package is required for this backend") from exc
        task = _HF_TASKS.get(spec.task)
        if task is None:
            raise ModelLoadError(f"Task {spec.task.value} unsupported by transformers backend")
        return pipeline(task, model=str(path), trust_remote_code=False)

    def infer(self, spec: ModelSpec, handle: Any, request: InferenceRequest) -> ProviderResult:
        out = handle(request.inputs, **dict(request.parameters))
        text_in = request.inputs if isinstance(request.inputs, str) else repr(request.inputs)
        if spec.task is ModelTask.TEXT_GENERATION and isinstance(out, list) and out:
            result: Any = out[0].get("generated_text", out[0]) if isinstance(out[0], dict) else out[0]
        else:
            result = out
        return ProviderResult(result, estimate_tokens(text_in), estimate_tokens(result))


@dataclass
class _LocalHandle:
    backend_name: str
    backend_handle: Any


class LocalProvider(ModelProvider):
    provider_type = ProviderType.LOCAL

    def __init__(self, allowed_roots: Sequence[Union[str, Path]] = (),
                 backends: Optional[Mapping[str, LocalBackend]] = None) -> None:
        self._roots = [Path(r).expanduser().resolve() for r in allowed_roots]
        self._backends: Dict[str, LocalBackend] = {"transformers": TransformersBackend()}
        if backends:
            self._backends.update(backends)
        self._lock = threading.Lock()

    def register_backend(self, name: str, backend: LocalBackend) -> None:
        with self._lock:
            self._backends[name] = backend

    def load(self, spec: ModelSpec) -> _LocalHandle:
        path = self._safe_path(spec.location)
        name = str(spec.metadata.get("backend", "transformers"))
        backend = self._backend(name)
        try:
            return _LocalHandle(name, backend.load(spec, path))
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Local load failed for {spec.model_id}: {exc}") from exc

    def unload(self, spec: ModelSpec, handle: _LocalHandle) -> None:
        handle.backend_handle = None
        gc.collect()
        torch = sys.modules.get("torch")  # only touch torch if already imported
        if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, spec: ModelSpec, handle: _LocalHandle, request: InferenceRequest) -> ProviderResult:
        if handle.backend_handle is None:
            raise ProviderError(f"{spec.model_id} was unloaded", retryable=False)
        try:
            return self._backend(handle.backend_name).infer(spec, handle.backend_handle, request)
        except (InvalidRequestError, ProviderError):
            raise
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"Bad input for {spec.model_id}: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Local inference failed: {exc}", retryable=False) from exc

    def health_check(self, spec: ModelSpec, handle: _LocalHandle) -> bool:
        return handle.backend_handle is not None

    def optimize(self, spec: ModelSpec, handle: _LocalHandle, config: Any) -> _LocalHandle:
        """Backends may expose ``optimize(handle, config)``; otherwise a no-op."""
        backend = self._backend(handle.backend_name)
        fn = getattr(backend, "optimize", None)
        if callable(fn) and isinstance(config, OptimizationConfig) \
                and (config.quantization is not Quantization.NONE or config.max_batch_size > 1):
            return _LocalHandle(handle.backend_name, fn(handle.backend_handle, config))
        return handle

    # -- security ----------------------------------------------------------- #
    def _safe_path(self, location: str) -> Path:
        if not self._roots:
            raise ModelSecurityError("LocalProvider has no allowed_roots configured")
        try:
            path = Path(location).expanduser().resolve(strict=True)  # resolves symlinks / '..'
        except (OSError, RuntimeError) as exc:
            raise ModelLoadError(f"Model path not found: {location}") from exc
        if not any(path == r or r in path.parents for r in self._roots):
            raise ModelSecurityError(f"Path outside allowed roots: {location}")
        return path

    def _backend(self, name: str) -> LocalBackend:
        with self._lock:
            try:
                return self._backends[name]
            except KeyError:
                raise ModelLoadError(f"Unknown local backend: {name!r}") from None
