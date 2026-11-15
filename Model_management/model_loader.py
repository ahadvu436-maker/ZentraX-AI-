"""Model lifecycle: lazy loading, per-model load locks, LRU eviction, leases."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from .model_core import (
    ModelLoadError,
    ModelManagementError,
    ModelProvider,
    ModelSpec,
    ModelStatus,
    ProviderType,
)
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads models through providers and keeps at most ``max_loaded`` in memory.

    Models that are leased (in use by an inference) are never evicted.
    """

    def __init__(self, registry: ModelRegistry, providers: Mapping[ProviderType, ModelProvider],
                 max_loaded: int = 4) -> None:
        if max_loaded < 1:
            raise ValueError("max_loaded must be >= 1")
        self._registry = registry
        self._providers = dict(providers)
        self._max_loaded = max_loaded
        self._lock = threading.RLock()
        self._loaded: "OrderedDict[str, Tuple[ModelSpec, Any]]" = OrderedDict()
        self._leases: Dict[str, int] = {}
        self._load_locks: Dict[str, threading.Lock] = {}

    def provider_for(self, spec: ModelSpec) -> ModelProvider:
        try:
            return self._providers[spec.provider]
        except KeyError:
            raise ModelLoadError(f"No provider configured for {spec.provider.value}") from None

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._loaded

    def loaded_ids(self) -> List[str]:
        with self._lock:
            return list(self._loaded)

    def ensure_loaded(self, spec: ModelSpec) -> Any:
        return self._acquire(spec, lease=False)

    @contextmanager
    def lease(self, spec: ModelSpec) -> Iterator[Any]:
        """Yield the model handle and pin it against eviction while in use."""
        handle = self._acquire(spec, lease=True)
        try:
            yield handle
        finally:
            with self._lock:
                self._leases[spec.model_id] = max(0, self._leases.get(spec.model_id, 1) - 1)

    def replace_handle(self, model_id: str, handle: Any) -> None:
        with self._lock:
            if model_id not in self._loaded:
                raise ModelLoadError(f"{model_id} is not loaded")
            spec, _ = self._loaded[model_id]
            self._loaded[model_id] = (spec, handle)

    def unload(self, model_id: str) -> None:
        with self._lock:
            if self._leases.get(model_id, 0) > 0:
                raise ModelManagementError(f"{model_id} is in use and cannot be unloaded")
            item = self._loaded.pop(model_id, None)
        if item is not None:
            self._release(*item)
            self._registry.set_status(model_id, ModelStatus.UNLOADED)

    def unload_all(self) -> None:
        with self._lock:
            items = list(self._loaded.items())
            self._loaded.clear()
        for model_id, (spec, handle) in items:
            self._release(spec, handle)
            self._registry.set_status(model_id, ModelStatus.UNLOADED)

    # -- internals ---------------------------------------------------------- #
    def _acquire(self, spec: ModelSpec, *, lease: bool) -> Any:
        mid = spec.model_id
        with self._lock:
            if mid in self._loaded:
                self._loaded.move_to_end(mid)
                if lease:
                    self._leases[mid] = self._leases.get(mid, 0) + 1
                return self._loaded[mid][1]
            load_lock = self._load_locks.setdefault(mid, threading.Lock())

        with load_lock:  # serialise concurrent loads of the same model
            with self._lock:
                if mid in self._loaded:
                    self._loaded.move_to_end(mid)
                    if lease:
                        self._leases[mid] = self._leases.get(mid, 0) + 1
                    return self._loaded[mid][1]
            provider = self.provider_for(spec)
            self._registry.set_status(mid, ModelStatus.LOADING)
            try:
                handle = provider.load(spec)
            except ModelManagementError as exc:
                self._registry.set_status(mid, ModelStatus.FAILED, str(exc))
                raise
            except Exception as exc:  # provider bug / library error
                self._registry.set_status(mid, ModelStatus.FAILED, repr(exc))
                raise ModelLoadError(f"Failed to load {mid}: {exc}") from exc
            with self._lock:
                self._loaded[mid] = (spec, handle)
                if lease:
                    self._leases[mid] = self._leases.get(mid, 0) + 1
                evicted = self._evict_locked(keep=mid)
            self._registry.set_status(mid, ModelStatus.READY)
            for item in evicted:
                self._release(*item)
                self._registry.set_status(item[0].model_id, ModelStatus.UNLOADED)
            return handle

    def _evict_locked(self, keep: str) -> List[Tuple[ModelSpec, Any]]:
        evicted: List[Tuple[ModelSpec, Any]] = []
        while len(self._loaded) > self._max_loaded:
            victim = next(
                (k for k in self._loaded if k != keep and self._leases.get(k, 0) == 0), None
            )
            if victim is None:
                logger.warning("Loaded-model cap exceeded but all models are leased")
                break
            evicted.append(self._loaded.pop(victim))
            logger.info("Evicted model %s (LRU)", victim)
        return evicted

    def _release(self, spec: ModelSpec, handle: Any) -> None:
        try:
            self.provider_for(spec).unload(spec, handle)
        except Exception:  # never let cleanup break callers
            logger.exception("Error unloading %s", spec.model_id)
