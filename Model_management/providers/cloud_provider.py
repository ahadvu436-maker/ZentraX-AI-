"""Provider for remote HTTPS APIs (OpenAI-compatible by default). Stdlib only."""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Collection, Dict, Mapping, Optional
from urllib.parse import urlparse

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

logger = logging.getLogger(__name__)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so credentials are never forwarded to another host."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None


@dataclass(frozen=True)
class _CloudHandle:
    url: str
    headers: Mapping[str, str]
    api_style: str


class CloudProvider(ModelProvider):
    """``spec.location`` is the endpoint URL. ``spec.metadata`` may contain:
    ``api_key_env`` (env var holding the key), ``remote_model`` (provider model
    name), ``api_style`` (``openai`` | ``raw``)."""

    provider_type = ProviderType.CLOUD

    def __init__(self, allowed_hosts: Optional[Collection[str]] = None, *,
                 max_response_bytes: int = 10 * 1024 * 1024) -> None:
        self._hosts = {h.lower() for h in allowed_hosts} if allowed_hosts else None
        self._max_bytes = max_response_bytes
        self._opener = urllib.request.build_opener(_NoRedirect)

    def load(self, spec: ModelSpec) -> _CloudHandle:
        parsed = urlparse(spec.location)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ModelSecurityError("Cloud endpoints must use https")
        if parsed.username or parsed.password:
            raise ModelSecurityError("Credentials in URLs are not allowed")
        if self._hosts is not None and parsed.hostname.lower() not in self._hosts:
            raise ModelSecurityError(f"Host not allowed: {parsed.hostname}")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        env = spec.metadata.get("api_key_env")
        if env:
            key = os.environ.get(str(env))
            if not key:
                raise ModelLoadError(f"Environment variable {env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        return _CloudHandle(spec.location, headers, str(spec.metadata.get("api_style", "openai")))

    def unload(self, spec: ModelSpec, handle: _CloudHandle) -> None:
        return None  # stateless

    def infer(self, spec: ModelSpec, handle: _CloudHandle, request: InferenceRequest) -> ProviderResult:
        payload = self._payload(spec, handle, request)
        data = self._post(handle, payload, request.timeout_s)
        return self._parse(spec, handle, request, data)

    # -- internals ---------------------------------------------------------- #
    def _payload(self, spec: ModelSpec, handle: _CloudHandle, request: InferenceRequest) -> Dict[str, Any]:
        if handle.api_style == "raw":
            return {"inputs": request.inputs, "parameters": dict(request.parameters)}
        model = spec.metadata.get("remote_model", spec.name)
        if spec.task is ModelTask.EMBEDDING:
            return {"model": model, "input": request.inputs, **request.parameters}
        inputs = request.inputs
        if isinstance(inputs, str):
            messages = [{"role": "user", "content": inputs}]
        elif isinstance(inputs, list) and all(isinstance(m, dict) for m in inputs):
            messages = inputs
        else:
            raise InvalidRequestError("Chat inputs must be a string or a list of message dicts")
        return {"model": model, "messages": messages, **request.parameters}

    def _post(self, handle: _CloudHandle, payload: Mapping[str, Any], timeout_s: float) -> Any:
        req = urllib.request.Request(handle.url, data=json.dumps(payload).encode("utf-8"),
                                     headers=dict(handle.headers), method="POST")
        try:
            with self._opener.open(req, timeout=timeout_s) as resp:
                body = resp.read(self._max_bytes + 1)
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (408, 425, 429) or exc.code >= 500
            if not retryable and 400 <= exc.code < 500:
                raise InvalidRequestError(f"Remote rejected request (HTTP {exc.code})") from None
            raise ProviderError(f"Remote HTTP {exc.code}", retryable=retryable) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Network error: {exc}", retryable=True) from exc
        if len(body) > self._max_bytes:
            raise ProviderError("Response too large", retryable=False)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError("Invalid JSON from remote", retryable=False) from exc

    def _parse(self, spec: ModelSpec, handle: _CloudHandle, request: InferenceRequest, data: Any) -> ProviderResult:
        try:
            if handle.api_style == "raw":
                return ProviderResult(data, estimate_tokens(request.inputs), estimate_tokens(data))
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            if spec.task is ModelTask.EMBEDDING:
                out: Any = [d["embedding"] for d in data["data"]]
            else:
                out = data["choices"][0]["message"]["content"]
            return ProviderResult(
                out,
                int(usage.get("prompt_tokens", estimate_tokens(request.inputs))),
                int(usage.get("completion_tokens", estimate_tokens(out))),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("Unexpected response schema", retryable=False) from exc
