"""Optimisation: response cache, optimisation configs/recommendations, idle eviction."""
from __future__ import annotations

import enum
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .model_core import (
    InferenceRequest,
    InferenceResponse,
    ModelManagementError,
    ProviderType,
)
from .model_loader import ModelLoader
from .model_registry import ModelRegistry


class Quantization(str, enum.Enum):
    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass(frozen=True)
class OptimizationConfig:
    """Passed verbatim to ``ModelProvider.optimize``; providers apply what they support."""

    quantization: Quantization = Quantization.NONE
    max_batch_size: int = 1
    warmup: bool = False


@dataclass(frozen=True)
class Recommendation:
    model_id: str
    action: str  # "quantize" | "unload_idle" | "investigate" | "enable_cache"
    reason: str


class ResponseCache:
    """Thread-safe TTL + LRU cache. Only used for requests with ``cacheable=True``."""

    def __init__(self, max_entries: int = 1024, ttl_s: float = 300.0) -> None:
        self._max = max_entries
        self._ttl = ttl_s
        self._data: "OrderedDict[str, Tuple[float, InferenceResponse]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(request: InferenceRequest) -> str:
        blob = json.dumps(
            {"i": request.inputs, "t": request.task.value, "m": request.model_id,
             "s": request.strategy.value, "p": dict(request.parameters)},
            sort_keys=True, default=repr,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[InferenceResponse]:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None or now - item[0] > self._ttl:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def put(self, key: str, response: InferenceResponse) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), response)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class ModelOptimizer:
    def __init__(self, registry: ModelRegistry, loader: ModelLoader, *,
                 cache: Optional[ResponseCache] = None, slow_latency_ms: float = 2000.0,
                 failure_rate_threshold: float = 0.2, min_calls: int = 10) -> None:
        self._registry = registry
        self._loader = loader
        self.cache = cache or ResponseCache()
        self._slow = slow_latency_ms
        self._fail_thr = failure_rate_threshold
        self._min_calls = min_calls

    def analyze(self, idle_after_s: float = 900.0) -> List[Recommendation]:
        """Inspect runtime metrics and suggest actions (does not apply them)."""
        recs: List[Recommendation] = []
        now = time.time()
        for mid in self._registry.list_ids():
            entry = self._registry.get_entry(mid)
            m, spec = entry.metrics, entry.spec
            if m.calls >= self._min_calls and m.failure_rate > self._fail_thr:
                recs.append(Recommendation(mid, "investigate", f"failure rate {m.failure_rate:.0%}"))
            if (spec.provider is ProviderType.LOCAL and m.avg_latency_ms
                    and m.avg_latency_ms > self._slow):
                recs.append(Recommendation(mid, "quantize", f"avg latency {m.avg_latency_ms:.0f} ms"))
            if self._loader.is_loaded(mid) and entry.last_used is not None \
                    and now - entry.last_used > idle_after_s:
                recs.append(Recommendation(mid, "unload_idle", f"idle {now - entry.last_used:.0f}s"))
        return recs

    def apply(self, model_id: str, config: OptimizationConfig) -> None:
        """Ask the model's provider to optimise the loaded handle in place."""
        spec = self._registry.get(model_id)
        if not self._loader.is_loaded(model_id):
            raise ModelManagementError(f"{model_id} must be loaded before optimisation")
        with self._loader.lease(spec) as handle:
            new_handle = self._loader.provider_for(spec).optimize(spec, handle, config)
        if new_handle is not handle:
            self._loader.replace_handle(model_id, new_handle)

    def evict_idle(self, idle_after_s: float = 900.0) -> List[str]:
        """Unload models unused for ``idle_after_s`` seconds. Returns unloaded ids."""
        unloaded: List[str] = []
        for rec in self.analyze(idle_after_s):
            if rec.action == "unload_idle":
                try:
                    self._loader.unload(rec.model_id)
                    unloaded.append(rec.model_id)
                except ModelManagementError:
                    continue  # in use; try next cycle
        return unloaded
