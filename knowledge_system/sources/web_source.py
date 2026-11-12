"""Ingest web pages as documents (SSRF-safe, robots.txt-aware, rate limited)."""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, Iterable, Iterator, Optional, Sequence, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from . import BaseSource, RawDocument, html_to_text, make_doc_id

logger = logging.getLogger(__name__)

__all__ = ["WebSource", "WebSourceError"]

_TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown")


class WebSourceError(Exception):
    """Raised for blocked or failed web requests."""


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validates every redirect target before following it."""

    def __init__(self, validator: Any) -> None:
        super().__init__()
        self._validator = validator

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        self._validator(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WebSource(BaseSource):
    """Fetch pages (optionally crawling same-site links) as documents.

    Args:
        urls: Seed URLs (http/https only).
        allowed_domains: If set, only these domains (and subdomains) are fetched.
        crawl_depth: 0 = seeds only; N follows links N levels deep.
        max_pages: Hard cap on pages fetched per ``fetch()`` call.
        respect_robots: Honour robots.txt.
        min_interval: Minimum seconds between requests to the same host.
        allow_private_hosts: Disable the private-network block (dev/testing only).

    Note: the hostname is resolved before connecting; a hostile DNS server could
    still rebind between check and connect. For untrusted URLs, also enforce
    egress rules at the network layer.
    """

    source_type = "web"

    def __init__(
        self,
        urls: Sequence[str],
        *,
        allowed_domains: Optional[Iterable[str]] = None,
        crawl_depth: int = 0,
        max_pages: int = 50,
        timeout: float = 15.0,
        max_bytes: int = 5 * 1024 * 1024,
        retries: int = 2,
        backoff: float = 1.0,
        min_interval: float = 1.0,
        respect_robots: bool = True,
        allow_private_hosts: bool = False,
        user_agent: str = "ZentraX-AI-KnowledgeBot/1.0",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or "web")
        self.urls = list(urls)
        self.allowed_domains: Set[str] = {d.lower().lstrip(".") for d in (allowed_domains or [])}
        self.crawl_depth = max(0, crawl_depth)
        self.max_pages = max(1, max_pages)
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.retries = max(0, retries)
        self.backoff = backoff
        self.min_interval = max(0.0, min_interval)
        self.respect_robots = respect_robots
        self.allow_private_hosts = allow_private_hosts
        self.user_agent = user_agent
        self._last_hit: Dict[str, float] = {}
        self._robots: Dict[str, Optional[RobotFileParser]] = {}
        self._opener = urllib.request.build_opener(_SafeRedirectHandler(self._assert_safe))

    # ------------------------------------------------------------------ #
    # Safety checks
    # ------------------------------------------------------------------ #
    def _domain_ok(self, host: str) -> bool:
        if not self.allowed_domains:
            return True
        host = host.lower()
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)

    def _assert_safe(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise WebSourceError(f"Blocked scheme: {parts.scheme!r}")
        host = parts.hostname
        if not host:
            raise WebSourceError("URL has no host")
        if not self._domain_ok(host):
            raise WebSourceError(f"Domain not allowed: {host}")
        if self.allow_private_hosts:
            return
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise WebSourceError(f"Cannot resolve {host}: {exc}") from exc
        for info in infos:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            mapped = getattr(ip, "ipv4_mapped", None)
            if mapped is not None:
                ip = mapped
            if not ip.is_global:
                raise WebSourceError(f"Blocked non-public address for {host}: {ip}")

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #
    def _throttle(self, host: str) -> None:
        wait = self._last_hit.get(host, 0.0) + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    def _http_get(self, url: str, max_bytes: Optional[int] = None) -> Tuple[Any, bytes, bool]:
        """Return (response-info, body, truncated)."""
        limit = max_bytes or self.max_bytes
        host = urlsplit(url).hostname or ""
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            self._assert_safe(url)
            self._throttle(host)
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
                },
            )
            try:
                with self._opener.open(req, timeout=self.timeout) as resp:
                    body = resp.read(limit + 1)
                    truncated = len(body) > limit
                    info = {
                        "status": getattr(resp, "status", 200),
                        "headers": resp.headers,
                        "url": resp.geturl(),
                    }
                    return info, body[:limit], truncated
            except urllib.error.HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise WebSourceError(f"HTTP {exc.code} for {url}") from exc
                last_exc = exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_exc = exc
            if attempt < self.retries:
                time.sleep(self.backoff * (2 ** attempt))
        raise WebSourceError(f"Failed to fetch {url}: {last_exc}")

    def _robots_ok(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser: Optional[RobotFileParser] = RobotFileParser()
            try:
                _, body, _ = self._http_get(origin + "/robots.txt", max_bytes=512 * 1024)
                parser.parse(body.decode("utf-8", errors="replace").splitlines())  # type: ignore[union-attr]
            except WebSourceError as exc:
                if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
                    parser.parse(["User-agent: *", "Disallow: /"])  # type: ignore[union-attr]
                else:
                    parser = None  # missing/unreachable robots.txt: allow
            self._robots[origin] = parser
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    @staticmethod
    def _decode(body: bytes, content_type: str) -> str:
        match = re.search(r"charset=([\w\-]+)", content_type, re.IGNORECASE)
        if not match:
            head = body[:2048].decode("ascii", errors="ignore")
            match = re.search(r"<meta[^>]+charset=[\"']?([\w\-]+)", head, re.IGNORECASE)
        for enc in ([match.group(1)] if match else []) + ["utf-8"]:
            try:
                return body.decode(enc, errors="replace")
            except LookupError:
                continue
        return body.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    def health_check(self) -> bool:
        return bool(self.urls)

    def fetch(self) -> Iterator[RawDocument]:
        seed_hosts = {urlsplit(u).hostname for u in self.urls}
        queue: Deque[Tuple[str, int]] = deque((urldefrag(u)[0], 0) for u in self.urls)
        seen: Set[str] = {u for u, _ in queue}
        fetched = 0
        while queue and fetched < self.max_pages:
            url, depth = queue.popleft()
            try:
                self._assert_safe(url)
                if not self._robots_ok(url):
                    logger.info("Blocked by robots.txt: %s", url)
                    continue
                info, body, truncated = self._http_get(url)
            except WebSourceError as exc:
                logger.warning("Skipping %s: %s", url, exc)
                continue
            content_type = info["headers"].get("Content-Type", "text/html")
            if not content_type.lower().startswith(_TEXT_TYPES):
                logger.info("Skipping %s: unsupported content type %s", url, content_type)
                continue
            fetched += 1
            raw = self._decode(body, content_type)
            if content_type.lower().startswith("text/plain") or content_type.lower().startswith("text/markdown"):
                title, text, links = url, raw, []
            else:
                parsed = html_to_text(raw)
                title, text, links = parsed.title or url, parsed.text, parsed.links
            if text.strip():
                final_url = info["url"]
                yield RawDocument(
                    id=make_doc_id(urldefrag(final_url)[0]),
                    uri=final_url,
                    content=text,
                    source_type=self.source_type,
                    title=title,
                    metadata={
                        "status": info["status"],
                        "content_type": content_type,
                        "truncated": truncated,
                        "requested_url": url,
                    },
                )
            if depth < self.crawl_depth:
                for href in links:
                    nxt = urldefrag(urljoin(info["url"], href))[0]
                    parts = urlsplit(nxt)
                    if parts.scheme not in ("http", "https") or nxt in seen:
                        continue
                    host = parts.hostname or ""
                    if self.allowed_domains:
                        if not self._domain_ok(host):
                            continue
                    elif host not in seed_hosts:
                        continue
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))
