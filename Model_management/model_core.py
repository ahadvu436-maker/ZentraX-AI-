"""Central types, errors, provider contract and the ``ModelManager`` facade.

This module deliberately imports no sibling modules at import time (only under
``TYPE_CHECKING`` and inside ``ModelManager.create_default``) so every other
module can import the shared types from here without circular imports.
"""
from __future__ import annotations

import abc
import asyncio
import dataclasses
import enum
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .inference_engine import InferenceEngine
    from .model_loader import ModelLoader
    from .model_optimizer import ModelOptimizer
    from .model_registry import ModelRegistry
    from .model_router import ModelRouter
    from .model_versioning import ModelVersionManager

logger = logging.getLogger(__name__)

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-/]{0,127}$")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ModelManagementError(Exception):
    """Base class for all package errors."""


class ModelNotFoundError(ModelManagementError):
    """Requested model is not registered."""


class ModelAlreadyRegisteredError(ModelManagementError):
    """A model with the same id is already registered."""


class ModelLoadError(ModelManagementError):
    """A provider failed to load a model."""


class ModelSecurityError(ModelManagementError):
    """A security policy (path, host, scheme, ...) was violated."""


class NoModelAvailableError(ModelManagementError):
    """Routing found no eligible model for the request."""


class VersionError(ModelManagementError):
    """Invalid or unknown model version operation."""


class InvalidRequestError(ModelManagementError):
    """Request is malformed; retrying or falling back will not help."""


class ProviderError(ModelManagementError):
    """Provider-side failure. ``retryable`` hints whether a retry may succeed."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class InferenceTimeoutError(ModelManagementError):
    """Inference did not finish within the request timeout."""


class InferenceError(ModelManagementError):
    """All routed models failed. ``errors`` lists per-attempt failures."""

    def __init__(self, message: str, errors: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.errors: Sequence[str] = tuple(errors)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class ModelTask(str, enum.Enum):
    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    CUSTOM = "custom"


class ProviderType(str, enum.Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    CUSTOM = "custom"


class ModelStatus(str, enum.Enum):
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    UNLOADED = "unloaded"
    DEPRECATED = "deprecated"


class RoutingStrategy(str, enum.Enum):
    PERFORMANCE = "performance"
    COST = "cost"
    AVAILABILITY = "availability"
    BALANCED = "balanced"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class ModelSpec:
    """Immutable description of a deployable model.

    ``location`` is a filesystem path (local), an https endpoint (cloud) or an
    opaque key (custom). Secrets must never be stored here; reference them by
    environment variable name in ``metadata`` (e.g. ``api_key_env``).
    """

    model_id: str
    name: str
    provider: ProviderType
    location: str
    task: ModelTask = ModelTask.TEXT_GENERATION
    version: str = "0.0.0"
    cost_per_1k_tokens: float = 0.0
    expected_latency_ms: Optional[float] = None
    max_context: int = 4096
    capabilities: FrozenSet[str] = frozenset()
    tags: FrozenSet[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _MODEL_ID_RE.match(self.model_id):
            raise ValueError(f"Invalid model_id: {self.model_id!r}")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if self.cost_per_1k_tokens < 0:
            raise ValueError("cost_per_1k_tokens must be >= 0")
        if self.expected_latency_ms is not None and self.expected_latency_ms < 0:
            raise ValueError("expected_latency_ms must be >= 0")
        if self.max_context <= 0:
            raise ValueError("max_context must be > 0")
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "tags", frozenset(self.tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, eq=False)
class InferenceRequest:
    """A single inference call."""

    inputs: Any
    task: ModelTask = ModelTask.TEXT_GENERATION
    model_id: Optional[str] = None  # explicit model, "name", or "name@version"
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    parameters: Mapping[str, Any] = field(default_factory=dict)
    required_capabilities: FrozenSet[str] = frozenset()
    max_cost_per_1k: Optional[float] = None
    timeout_s: float = 30.0
    allow_fallback: bool = True
    cacheable: bool = False
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.inputs is None:
            raise InvalidRequestError("inputs must not be None")
        if self.timeout_s <= 0:
            raise InvalidRequestError("timeout_s must be > 0")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "required_capabilities", frozenset(self.required_capabilities))


@dataclass(frozen=True)
class ProviderResult:
    """Normalised provider output."""

    output: Any
    tokens_in: int = 0
    tokens_out: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    model_id: str
    provider: ProviderType
    output: Any
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    attempts: int = 1
    cached: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


def estimate_tokens(value: Any) -> int:
    """Cheap token estimate (~4 chars/token) used when providers give no usage."""
    if value is None:
        return 0
    text = value if isinstance(value, str) else repr(value)
    return max(1, len(text) // 4) if text else 0


# --------------------------------------------------------------------------- #
# Provider contract
# --------------------------------------------------------------------------- #
class ModelProvider(abc.ABC):
    """Execution-environment adapter. Implementations must be thread-safe."""

    provider_type: ProviderType

    @abc.abstractmethod
    def load(self, spec: ModelSpec) -> Any:
        """Prepare the model and return an opaque handle."""

    @abc.abstractmethod
    def unload(self, spec: ModelSpec, handle: Any) -> None:
        """Release resources held by ``handle``."""

    @abc.abstractmethod
    def infer(self, spec: ModelSpec, handle: Any, request: InferenceRequest) -> ProviderResult:
        """Run one inference."""

    def health_check(self, spec: ModelSpec, handle: Any) -> bool:
        return handle is not None

    def optimize(self, spec: ModelSpec, handle: Any, config: Any) -> Any:
        """Return an optimised handle. Default: no-op."""
        return handle


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class ModelManager:
    """Single entry point coordinating registry, loader, router, engine,
    versioning and optimisation. Components are injected; use
    :meth:`create_default` for a ready-to-use wiring."""

    def __init__(
        self,
        registry: "ModelRegistry",
        loader: "ModelLoader",
        router: "ModelRouter",
        engine: "InferenceEngine",
        versions: Optional["ModelVersionManager"] = None,
        optimizer: Optional["ModelOptimizer"] = None,
    ) -> None:
        self.registry = registry
        self.loader = loader
        self.router = router
        self.engine = engine
        self.versions = versions
        self.optimizer = optimizer
        self._closed = False

    @classmethod
    def create_default(
        cls,
        *,
        local_roots: Sequence[str] = (),
        allowed_hosts: Optional[Sequence[str]] = None,
        max_loaded: int = 4,
        max_workers: int = 8,
        max_retries: int = 1,
        providers: Optional[Mapping[ProviderType, ModelProvider]] = None,
    ) -> "ModelManager":
        from .inference_engine import InferenceEngine
        from .model_loader import ModelLoader
        from .model_optimizer import ModelOptimizer
        from .model_registry import ModelRegistry
        from .model_router import ModelRouter
        from .model_selector import ModelSelector
        from .model_versioning import ModelVersionManager
        from .providers.cloud_provider import CloudProvider
        from .providers.custom_provider import CustomProvider
        from .providers.local_provider import LocalProvider

        registry = ModelRegistry()
        provs: Dict[ProviderType, ModelProvider] = {
            ProviderType.LOCAL: LocalProvider(allowed_roots=local_roots),
            ProviderType.CLOUD: CloudProvider(allowed_hosts=allowed_hosts),
            ProviderType.CUSTOM: CustomProvider(),
        }
        if providers:
            provs.update(providers)
        loader = ModelLoader(registry, provs, max_loaded=max_loaded)
        router = ModelRouter(registry, ModelSelector())
        engine = InferenceEngine(registry, loader, router, max_workers=max_workers, max_retries=max_retries)
        optimizer = ModelOptimizer(registry, loader)
        return cls(registry, loader, router, engine, ModelVersionManager(), optimizer)

    # -- lifecycle ---------------------------------------------------------- #
    def register_model(self, spec: ModelSpec, *, load: bool = False, replace: bool = False) -> ModelSpec:
        self._ensure_open()
        self.registry.register(spec, replace=replace)
        if self.versions is not None and spec.version != "0.0.0":
            try:
                self.versions.register_version(spec.name, spec.version, spec.model_id)
            except VersionError:
                logger.debug("version already tracked for %s", spec.model_id)
        if load:
            self.loader.ensure_loaded(spec)
        return spec

    def unregister_model(self, model_id: str) -> None:
        self._ensure_open()
        if self.loader.is_loaded(model_id):
            self.loader.unload(model_id)
        self.registry.unregister(model_id)

    def load_model(self, model_id: str) -> None:
        self._ensure_open()
        self.loader.ensure_loaded(self.registry.get(model_id))

    def unload_model(self, model_id: str) -> None:
        self._ensure_open()
        self.loader.unload(model_id)

    # -- inference ---------------------------------------------------------- #
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self._ensure_open()
        request = self._resolve(request)
        cache = self.optimizer.cache if (self.optimizer and request.cacheable) else None
        key = cache.make_key(request) if cache else None
        if cache is not None and key is not None:
            hit = cache.get(key)
            if hit is not None:
                return dataclasses.replace(hit, request_id=request.request_id, cached=True, latency_ms=0.0)
        response = self.engine.execute(request)
        if cache is not None and key is not None:
            cache.put(key, response)
        return response

    async def ainfer(self, request: InferenceRequest) -> InferenceResponse:
        return await asyncio.to_thread(self.infer, request)

    def generate(self, prompt: str, *, model_id: Optional[str] = None, **parameters: Any) -> InferenceResponse:
        """Convenience wrapper for text generation."""
        return self.infer(InferenceRequest(inputs=prompt, model_id=model_id, parameters=parameters))

    # -- introspection ------------------------------------------------------ #
    def health(self) -> Dict[str, Dict[str, Any]]:
        report: Dict[str, Dict[str, Any]] = {}
        for model_id in self.registry.list_ids():
            entry = self.registry.get_entry(model_id)
            m = entry.metrics
            report[model_id] = {
                "status": entry.status.value,
                "loaded": self.loader.is_loaded(model_id),
                "circuit": self.router.breaker_state(model_id),
                "calls": m.calls,
                "failures": m.failures,
                "avg_latency_ms": m.avg_latency_ms,
                "last_error": entry.last_error,
            }
        return report

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.engine.shutdown()
        self.loader.unload_all()

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- internals ---------------------------------------------------------- #
    def _resolve(self, request: InferenceRequest) -> InferenceRequest:
        if self.versions is None or not request.model_id:
            return request
        resolved = self.versions.resolve_reference(request.model_id, key=request.request_id)
        if resolved != request.model_id:
            return dataclasses.replace(request, model_id=resolved)
        return request

    def _ensure_open(self) -> None:
        if self._closed:
            raise ModelManagementError("ModelManager is closed")


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0
