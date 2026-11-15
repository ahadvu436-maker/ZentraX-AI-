"""Executes routed inference with timeouts, retries, fallbacks and metrics."""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import List

from .model_core import (
    InferenceError,
    InferenceRequest,
    InferenceResponse,
    InferenceTimeoutError,
    InvalidRequestError,
    ModelSpec,
    ProviderError,
    ProviderResult,
)
from .model_loader import ModelLoader
from .model_registry import ModelRegistry
from .model_router import ModelRouter

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, registry: ModelRegistry, loader: ModelLoader, router: ModelRouter, *,
                 max_workers: int = 8, max_retries: int = 1, backoff_s: float = 0.2) -> None:
        self._registry = registry
        self._loader = loader
        self._router = router
        self._max_retries = max(0, max_retries)
        self._backoff = backoff_s
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="infer")

    def execute(self, request: InferenceRequest) -> InferenceResponse:
        plan = self._router.plan(request)
        errors: List[str] = []
        attempts = 0
        for spec in plan:
            for attempt in range(self._max_retries + 1):
                attempts += 1
                start = time.perf_counter()
                try:
                    result = self._run_once(spec, request)
                except InvalidRequestError:
                    raise  # no point retrying or falling back
                except Exception as exc:
                    self._on_failure(spec, exc, errors)
                    retryable = isinstance(exc, ProviderError) and exc.retryable
                    if retryable and attempt < self._max_retries:
                        time.sleep(self._backoff * (2 ** attempt))
                        continue
                    break  # next model in chain
                latency_ms = (time.perf_counter() - start) * 1000.0
                return self._on_success(spec, request, result, latency_ms, attempts)
        raise InferenceError(f"All models failed for request {request.request_id}", errors)

    async def aexecute(self, request: InferenceRequest) -> InferenceResponse:
        return await asyncio.to_thread(self.execute, request)

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)

    # -- internals ---------------------------------------------------------- #
    def _attempt(self, spec: ModelSpec, request: InferenceRequest) -> ProviderResult:
        with self._loader.lease(spec) as handle:  # lease held until the worker finishes
            return self._loader.provider_for(spec).infer(spec, handle, request)

    def _run_once(self, spec: ModelSpec, request: InferenceRequest) -> ProviderResult:
        future = self._pool.submit(self._attempt, spec, request)
        try:
            return future.result(timeout=request.timeout_s)
        except FutureTimeout:
            future.cancel()  # cannot interrupt a running worker; result is discarded
            raise InferenceTimeoutError(
                f"{spec.model_id} exceeded {request.timeout_s}s"
            ) from None

    def _on_failure(self, spec: ModelSpec, exc: Exception, errors: List[str]) -> None:
        logger.warning("Inference failed on %s: %r", spec.model_id, exc)
        errors.append(f"{spec.model_id}: {exc!r}")
        self._registry.record_result(spec.model_id, success=False, error=repr(exc))
        self._router.record_failure(spec.model_id)

    def _on_success(self, spec: ModelSpec, request: InferenceRequest, result: ProviderResult,
                    latency_ms: float, attempts: int) -> InferenceResponse:
        tokens = result.tokens_in + result.tokens_out
        cost = tokens / 1000.0 * spec.cost_per_1k_tokens
        self._registry.record_result(spec.model_id, success=True, latency_ms=latency_ms,
                                     cost=cost, tokens=tokens)
        self._router.record_success(spec.model_id)
        return InferenceResponse(
            request_id=request.request_id, model_id=spec.model_id, provider=spec.provider,
            output=result.output, latency_ms=latency_ms, tokens_in=result.tokens_in,
            tokens_out=result.tokens_out, cost=cost, attempts=attempts, metadata=result.metadata,
        )
