"""Builds an ordered execution plan (primary + fallbacks) for each request."""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Set

from .model_core import InferenceRequest, ModelSpec, NoModelAvailableError
from .model_registry import ModelRegistry
from .model_selector import ModelSelector


class CircuitBreaker:
    """Opens after ``failure_threshold`` consecutive failures; after
    ``reset_timeout_s`` it becomes half-open (one probe allowed through)."""

    def __init__(self, failure_threshold: int = 3, reset_timeout_s: float = 30.0) -> None:
        self._threshold = failure_threshold
        self._reset = reset_timeout_s
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "half_open" if time.monotonic() - self._opened_at >= self._reset else "open"

    def available(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._opened_at = time.monotonic()


class ModelRouter:
    def __init__(self, registry: ModelRegistry, selector: ModelSelector, *, max_chain: int = 3,
                 failure_threshold: int = 3, reset_timeout_s: float = 30.0) -> None:
        self._registry = registry
        self._selector = selector
        self._max_chain = max(1, max_chain)
        self._ft = failure_threshold
        self._rt = reset_timeout_s
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def plan(self, request: InferenceRequest) -> List[ModelSpec]:
        """Return models to try, in order. Raises NoModelAvailableError if empty."""
        chain: List[ModelSpec] = []
        exclude: Set[str] = set()
        if request.model_id:
            primary = self._registry.get(request.model_id)  # raises ModelNotFoundError
            chain.append(primary)
            exclude.add(primary.model_id)
        if not request.model_id or request.allow_fallback:
            chain.extend(self._ranked(request, exclude))
        if request.model_id and len(chain) > 1 and not self._breaker(chain[0].model_id).available():
            chain.pop(0)  # primary is tripped and alternatives exist
        chain = chain[: self._max_chain]
        if not chain:
            raise NoModelAvailableError(
                f"No available model for task={request.task.value} strategy={request.strategy.value}"
            )
        return chain

    def record_success(self, model_id: str) -> None:
        self._breaker(model_id).record_success()

    def record_failure(self, model_id: str) -> None:
        self._breaker(model_id).record_failure()

    def breaker_state(self, model_id: str) -> str:
        return self._breaker(model_id).state

    def _ranked(self, request: InferenceRequest, exclude: Set[str]) -> List[ModelSpec]:
        cands = [
            s for s in self._registry.find(task=request.task, capabilities=request.required_capabilities)
            if s.model_id not in exclude
            and self._breaker(s.model_id).available()
            and (request.max_cost_per_1k is None or s.cost_per_1k_tokens <= request.max_cost_per_1k)
        ]
        metrics = self._registry.all_metrics(s.model_id for s in cands)
        by_id = {s.model_id: s for s in cands}
        return [by_id[sc.model_id] for sc in self._selector.rank(cands, metrics, request.strategy)]

    def _breaker(self, model_id: str) -> CircuitBreaker:
        with self._lock:
            br = self._breakers.get(model_id)
            if br is None:
                br = self._breakers[model_id] = CircuitBreaker(self._ft, self._rt)
            return br
