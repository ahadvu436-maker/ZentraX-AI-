"""Provider for user-supplied callables (in-process models, SDK wrappers, mocks)."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..model_core import (
    InferenceRequest,
    InvalidRequestError,
    ModelLoadError,
    ModelProvider,
    ModelSpec,
    ProviderError,
    ProviderResult,
    ProviderType,
    estimate_tokens,
)


@dataclass(frozen=True)
class CustomHandler:
    """``infer`` may return a ProviderResult or any raw value."""

    load: Callable[[ModelSpec], Any]
    infer: Callable[[Any, InferenceRequest], Any]
    unload: Optional[Callable[[Any], None]] = None
    health: Optional[Callable[[Any], bool]] = None


class CustomProvider(ModelProvider):
    provider_type = ProviderType.CUSTOM

    def __init__(self) -> None:
        self._handlers: Dict[str, CustomHandler] = {}
        self._lock = threading.Lock()

    def register_handler(self, model_id: str, handler: CustomHandler) -> None:
        with self._lock:
            self._handlers[model_id] = handler

    def unregister_handler(self, model_id: str) -> None:
        with self._lock:
            self._handlers.pop(model_id, None)

    def load(self, spec: ModelSpec) -> Any:
        handler = self._handler(spec)
        try:
            return handler.load(spec)
        except Exception as exc:
            raise ModelLoadError(f"Custom load failed for {spec.model_id}: {exc}") from exc

    def unload(self, spec: ModelSpec, handle: Any) -> None:
        handler = self._handler(spec)
        if handler.unload is not None:
            handler.unload(handle)

    def infer(self, spec: ModelSpec, handle: Any, request: InferenceRequest) -> ProviderResult:
        handler = self._handler(spec)
        try:
            raw = handler.infer(handle, request)
        except (InvalidRequestError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderError(f"Custom inference failed: {exc}", retryable=False) from exc
        if isinstance(raw, ProviderResult):
            return raw
        return ProviderResult(raw, estimate_tokens(request.inputs), estimate_tokens(raw))

    def health_check(self, spec: ModelSpec, handle: Any) -> bool:
        handler = self._handler(spec)
        return handler.health(handle) if handler.health else handle is not None

    def _handler(self, spec: ModelSpec) -> CustomHandler:
        with self._lock:
            try:
                return self._handlers[spec.model_id]
            except KeyError:
                raise ModelLoadError(f"No custom handler registered for {spec.model_id}") from None
