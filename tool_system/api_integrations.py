"""HTTP/API integration layer.

* :class:`HTTPClient` - stdlib-only client with timeouts, bounded response size, retries
  with back-off, auth helpers and **SSRF protection** (private/loopback/link-local
  addresses are blocked by default, also across redirects).
* Search providers (:class:`DuckDuckGoProvider`, :class:`BraveSearchProvider`,
  :class:`SearxngProvider`) behind the :class:`SearchProvider` interface.
* :class:`RestEndpointTool` - turns a declarative :class:`EndpointDefinition` into a
  fully fledged tool, so any REST API can be exposed to a model without new code.

Note: URL validation resolves DNS before connecting; a hostile DNS server could still
rebind between check and use.  For high-risk deployments also restrict egress at the
network level or use ``allowed_hosts``.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .tool_core import (
    BaseTool, ToolCategory, ToolContext, ToolError, ToolParameter, ToolPermissionError,
    ToolSpec, ToolValidationError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "APIError", "AuthConfig", "HTTPResponse", "HTTPClient", "SearchResult", "SearchProvider",
    "DuckDuckGoProvider", "BraveSearchProvider", "SearxngProvider", "create_search_provider",
    "EndpointDefinition", "RestEndpointTool",
]

_RETRY_STATUS = {429, 500, 502, 503, 504}


class APIError(ToolError):
    """An HTTP/API call failed."""

    code = "api_error"

    def __init__(self, message: str, status: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class AuthConfig:
    """Authentication for :class:`HTTPClient`.

    ``kind``: ``none`` | ``bearer`` | ``api_key`` (header or query param) | ``basic``.
    Prefer :meth:`from_env` so secrets never live in source code.
    """

    kind: str = "none"
    token: Optional[str] = None
    header_name: str = "X-API-Key"
    query_param: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    @classmethod
    def from_env(cls, var: str, kind: str = "bearer", **kw: Any) -> "AuthConfig":
        token = os.environ.get(var)
        if not token:
            raise ToolValidationError(f"environment variable {var} is not set")
        return cls(kind=kind, token=token, **kw)

    def apply(self, headers: Dict[str, str], params: Dict[str, Any]) -> None:
        if self.kind == "bearer":
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.kind == "api_key":
            if self.query_param:
                params[self.query_param] = self.token
            else:
                headers[self.header_name] = str(self.token)
        elif self.kind == "basic":
            raw = f"{self.username}:{self.password}".encode()
            headers["Authorization"] = "Basic " + base64.b64encode(raw).decode()
        elif self.kind != "none":
            raise ToolValidationError(f"unknown auth kind {self.kind!r}")


@dataclass
class HTTPResponse:
    """Response of :meth:`HTTPClient.request`."""

    status: int
    url: str
    headers: Dict[str, str]
    body: bytes
    elapsed: float
    truncated: bool = False

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    def text(self) -> str:
        charset = "utf-8"
        ctype = self.headers.get("content-type", "")
        if "charset=" in ctype:
            charset = ctype.split("charset=")[-1].split(";")[0].strip() or "utf-8"
        try:
            return self.body.decode(charset, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise APIError(f"response is not valid JSON: {exc}", self.status) from None


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validates every redirect target against the client's network policy."""

    def __init__(self, client: "HTTPClient") -> None:
        self._client = client

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        self._client.validate_url(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HTTPClient:
    """Small, safe HTTP client built on :mod:`urllib`.

    Args:
        base_url: Prepended to relative URLs.
        headers: Default headers.
        auth: Optional :class:`AuthConfig`.
        timeout: Per-request socket timeout (seconds).
        max_response_bytes: Hard cap on the body read (rest is dropped, ``truncated=True``).
        retries: Extra attempts on network errors and 429/5xx (exponential back-off).
        allow_private_networks: Permit loopback/private/link-local targets (default: blocked).
        allowed_hosts: If set, only these hostnames may be contacted.
    """

    def __init__(self, base_url: Optional[str] = None, headers: Optional[Mapping[str, str]] = None,
                 auth: Optional[AuthConfig] = None, timeout: float = 15.0,
                 max_response_bytes: int = 2_000_000, retries: int = 2, backoff: float = 0.5,
                 allow_private_networks: bool = False,
                 allowed_hosts: Optional[Sequence[str]] = None,
                 user_agent: str = "tool-system/1.0") -> None:
        self.base_url = base_url
        self.headers = {"User-Agent": user_agent, "Accept": "*/*", **(headers or {})}
        self.auth = auth
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.retries = max(0, retries)
        self.backoff = backoff
        self.allow_private_networks = allow_private_networks
        self.allowed_hosts = {h.lower() for h in allowed_hosts} if allowed_hosts else None
        self._opener = urllib.request.build_opener(_SafeRedirectHandler(self))

    # -- policy ------------------------------------------------------------- #
    def validate_url(self, url: str) -> None:
        """Raise :class:`ToolPermissionError` unless ``url`` is allowed by the network policy."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ToolPermissionError(f"only http/https URLs are allowed (got '{parsed.scheme}')")
        host = (parsed.hostname or "").lower()
        if not host:
            raise ToolValidationError("URL has no host")
        if self.allowed_hosts is not None and host not in self.allowed_hosts:
            raise ToolPermissionError(f"host '{host}' is not in the allow-list")
        if self.allow_private_networks:
            return
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                       type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise APIError(f"cannot resolve host '{host}': {exc}") from None
        for info in infos:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                    or ip.is_multicast or ip.is_unspecified):
                raise ToolPermissionError(
                    f"host '{host}' resolves to a non-public address ({ip}); blocked")

    # -- requests ----------------------------------------------------------- #
    def request(self, method: str, url: str, *, params: Optional[Mapping[str, Any]] = None,
                json_body: Any = None, data: Optional[bytes] = None,
                headers: Optional[Mapping[str, str]] = None, timeout: Optional[float] = None,
                raise_for_status: bool = True) -> HTTPResponse:
        """Perform a request with retries. Raises :class:`APIError` on failure."""
        full = urllib.parse.urljoin(self.base_url.rstrip("/") + "/", url.lstrip("/")) \
            if self.base_url and not urllib.parse.urlparse(url).scheme else url
        hdrs = {**self.headers, **(headers or {})}
        query: Dict[str, Any] = {k: v for k, v in (params or {}).items() if v is not None}
        if self.auth:
            self.auth.apply(hdrs, query)
        if query:
            sep = "&" if urllib.parse.urlparse(full).query else "?"
            full += sep + urllib.parse.urlencode(query, doseq=True)
        self.validate_url(full)
        body = data
        if json_body is not None:
            body = json.dumps(json_body).encode()
            hdrs.setdefault("Content-Type", "application/json")

        last: Optional[APIError] = None
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            req = urllib.request.Request(full, data=body, headers=hdrs, method=method.upper())
            try:
                with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                    raw = resp.read(self.max_response_bytes + 1)
                    out = HTTPResponse(
                        status=resp.status, url=resp.geturl(),
                        headers={k.lower(): v for k, v in resp.headers.items()},
                        body=raw[: self.max_response_bytes], elapsed=time.monotonic() - started,
                        truncated=len(raw) > self.max_response_bytes)
                    return out
            except urllib.error.HTTPError as exc:
                text = exc.read(2000).decode("utf-8", errors="replace")
                if exc.code in _RETRY_STATUS and attempt < self.retries:
                    delay = self._retry_after(exc) or self.backoff * (2 ** attempt)
                    logger.info("HTTP %s from %s; retrying in %.1fs", exc.code, full, delay)
                    time.sleep(min(delay, 10))
                    last = APIError(f"HTTP {exc.code}", exc.code, text)
                    continue
                if not raise_for_status:
                    return HTTPResponse(exc.code, full, {k.lower(): v for k, v in exc.headers.items()},
                                        text.encode(), time.monotonic() - started)
                raise APIError(f"HTTP {exc.code} {exc.reason}", exc.code, text) from None
            except ToolError:
                raise
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
                last = APIError(f"request failed: {getattr(exc, 'reason', exc)}")
                if attempt < self.retries:
                    time.sleep(min(self.backoff * (2 ** attempt), 10))
                    continue
        raise last or APIError("request failed")

    @staticmethod
    def _retry_after(exc: urllib.error.HTTPError) -> Optional[float]:
        value = exc.headers.get("Retry-After") if exc.headers else None
        try:
            return float(value) if value else None
        except ValueError:
            return None

    def get(self, url: str, **kw: Any) -> HTTPResponse:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw: Any) -> HTTPResponse:
        return self.request("POST", url, **kw)


# --------------------------------------------------------------------------- #
# Search providers
# --------------------------------------------------------------------------- #
@dataclass
class SearchResult:
    """One web search hit."""

    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchProvider:
    """Interface for web search back-ends."""

    name = "base"

    def search(self, query: str, max_results: int, client: HTTPClient) -> List[SearchResult]:
        raise NotImplementedError


class _DDGParser(HTMLParser):
    """Extracts result links/snippets from DuckDuckGo's HTML endpoint."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[SearchResult] = []
        self._mode: Optional[str] = None
        self._buf: List[str] = []
        self._href = ""

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "a" and "result__a" in cls:
            self._mode, self._buf, self._href = "title", [], a.get("href") or ""
        elif "result__snippet" in cls:
            self._mode, self._buf = "snippet", []

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._mode == "title" and tag == "a":
            self.results.append(SearchResult(" ".join("".join(self._buf).split()),
                                             self._clean(self._href)))
            self._mode = None
        elif self._mode == "snippet" and tag in ("a", "td", "div"):
            if self.results:
                self.results[-1].snippet = " ".join("".join(self._buf).split())
            self._mode = None

    @staticmethod
    def _clean(href: str) -> str:
        if href.startswith("//"):
            href = "https:" + href
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        return q["uddg"][0] if "uddg" in q else href


class DuckDuckGoProvider(SearchProvider):
    """Keyless search via DuckDuckGo's HTML endpoint (best effort; markup may change)."""

    name = "duckduckgo"

    def search(self, query: str, max_results: int, client: HTTPClient) -> List[SearchResult]:
        resp = client.get("https://html.duckduckgo.com/html/", params={"q": query})
        return self.parse(resp.text())[:max_results]

    @staticmethod
    def parse(html: str) -> List[SearchResult]:
        parser = _DDGParser()
        parser.feed(html)
        return [r for r in parser.results if r.url]


class BraveSearchProvider(SearchProvider):
    """Brave Search API (needs ``api_key`` or env ``BRAVE_API_KEY``)."""

    name = "brave"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            raise ToolValidationError("Brave provider needs api_key or BRAVE_API_KEY")

    def search(self, query: str, max_results: int, client: HTTPClient) -> List[SearchResult]:
        resp = client.get("https://api.search.brave.com/res/v1/web/search",
                          params={"q": query, "count": max_results},
                          headers={"X-Subscription-Token": str(self.api_key), "Accept": "application/json"})
        return self.parse(resp.json())[:max_results]

    @staticmethod
    def parse(payload: Mapping[str, Any]) -> List[SearchResult]:
        items = (payload.get("web") or {}).get("results") or []
        return [SearchResult(i.get("title", ""), i.get("url", ""), i.get("description", ""))
                for i in items if i.get("url")]


class SearxngProvider(SearchProvider):
    """Self-hosted SearXNG instance (JSON format must be enabled)."""

    name = "searxng"

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, max_results: int, client: HTTPClient) -> List[SearchResult]:
        resp = client.get(f"{self.base_url}/search", params={"q": query, "format": "json"})
        return self.parse(resp.json())[:max_results]

    @staticmethod
    def parse(payload: Mapping[str, Any]) -> List[SearchResult]:
        return [SearchResult(i.get("title", ""), i.get("url", ""), i.get("content", ""))
                for i in payload.get("results") or [] if i.get("url")]


def create_search_provider(name: str, **config: Any) -> SearchProvider:
    """Factory: ``duckduckgo`` | ``brave`` | ``searxng``."""
    providers: Dict[str, Callable[..., SearchProvider]] = {
        "duckduckgo": DuckDuckGoProvider, "brave": BraveSearchProvider, "searxng": SearxngProvider}
    try:
        return providers[name.lower()](**config)
    except KeyError:
        raise ToolValidationError(f"unknown search provider '{name}'; choose from {sorted(providers)}") from None
    except TypeError as exc:
        raise ToolValidationError(f"bad config for provider '{name}': {exc}") from None


# --------------------------------------------------------------------------- #
# Declarative REST tools
# --------------------------------------------------------------------------- #
@dataclass
class EndpointDefinition:
    """Declarative description of one REST endpoint.

    ``url`` may contain ``{placeholders}`` filled from ``path`` parameters.  ``locations``
    maps each parameter name to ``path`` | ``query`` | ``body`` | ``header`` (default:
    ``query`` for GET/DELETE, ``body`` otherwise).
    """

    name: str
    description: str
    method: str
    url: str
    parameters: Sequence[ToolParameter] = ()
    locations: Mapping[str, str] = field(default_factory=dict)
    auth: Optional[AuthConfig] = None
    headers: Mapping[str, str] = field(default_factory=dict)
    permissions: Sequence[str] = ("net",)
    timeout: float = 15.0
    allow_private_networks: bool = False
    allowed_hosts: Optional[Sequence[str]] = None


class RestEndpointTool(BaseTool):
    """A :class:`BaseTool` generated from an :class:`EndpointDefinition`.

    Register instances explicitly (they are not auto-discovered)::

        registry.register(RestEndpointTool(EndpointDefinition(...)))
    """

    def __init__(self, endpoint: EndpointDefinition, config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(config)
        self.endpoint = endpoint
        self.spec = ToolSpec(  # instance-level spec
            name=endpoint.name, description=endpoint.description,
            parameters=tuple(endpoint.parameters), category=ToolCategory.API,
            required_permissions=frozenset(endpoint.permissions), timeout=endpoint.timeout + 5)
        self._client = HTTPClient(
            auth=endpoint.auth, headers=endpoint.headers, timeout=endpoint.timeout,
            allow_private_networks=endpoint.allow_private_networks,
            allowed_hosts=endpoint.allowed_hosts)

    def run(self, context: ToolContext, **arguments: Any) -> Any:
        ep = self.endpoint
        default_loc = "query" if ep.method.upper() in ("GET", "DELETE", "HEAD") else "body"
        buckets: Dict[str, Dict[str, Any]] = {"path": {}, "query": {}, "body": {}, "header": {}}
        for key, value in arguments.items():
            if value is None:
                continue
            loc = ep.locations.get(key, default_loc)
            if loc not in buckets:
                raise ToolValidationError(f"bad location '{loc}' for parameter '{key}'")
            buckets[loc][key] = value
        url = ep.url
        for key, value in buckets["path"].items():
            url = url.replace("{" + key + "}", urllib.parse.quote(str(value), safe=""))
        if "{" in url and "}" in url:
            raise ToolValidationError("unfilled path placeholder in URL")
        context.check_cancelled()
        resp = self._client.request(
            ep.method, url, params=buckets["query"] or None, json_body=buckets["body"] or None,
            headers={k: str(v) for k, v in buckets["header"].items()})
        if "json" in resp.content_type:
            return {"status": resp.status, "data": resp.json()}
        return {"status": resp.status, "text": resp.text()[:20000], "truncated": resp.truncated}
