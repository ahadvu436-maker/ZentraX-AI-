"""Browser tool: fetch a web page and extract readable text, title or links.

Static HTML only (no JavaScript execution).  Network policy (SSRF protection, allow-lists)
comes from :class:`~tool_system.api_integrations.HTTPClient`.

Config keys: ``timeout``, ``max_bytes``, ``allow_private_networks``, ``allowed_hosts``,
``user_agent``.
"""
from __future__ import annotations

import urllib.parse
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

from ..api_integrations import HTTPClient
from ..tool_core import (
    BaseTool, ToolCategory, ToolContext, ToolExecutionError, ToolParameter, ToolSpec,
)

_SKIP = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
          "header", "footer", "ul", "ol", "table", "pre", "blockquote"}
_TEXT_TYPES = ("text/", "application/xhtml+xml", "application/json", "application/xml")


class _PageParser(HTMLParser):
    """Collects title, visible text and links from HTML."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.links: List[Dict[str, str]] = []
        self._text: List[str] = []
        self._skip = 0
        self._in_title = False
        self._href: Optional[str] = None
        self._anchor: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _SKIP and tag != "head":
            self._skip += 1
        if tag in _BLOCK:
            self._text.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            if href and not href.startswith(("javascript:", "mailto:", "#")):
                self._href, self._anchor = urllib.parse.urljoin(self.base_url, href), []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _SKIP and tag != "head" and self._skip:
            self._skip -= 1
        if tag in _BLOCK:
            self._text.append("\n")
        if tag == "a" and self._href:
            self.links.append({"text": " ".join("".join(self._anchor).split()), "url": self._href})
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._skip:
            return
        self._text.append(data)
        if self._href is not None:
            self._anchor.append(data)

    @property
    def text(self) -> str:
        raw = "".join(self._text)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        out: List[str] = []
        for line in lines:
            if line or (out and out[-1]):
                out.append(line)
        return "\n".join(out).strip()


class BrowserTool(BaseTool):
    """Fetches web pages (static HTML) for reading."""

    spec = ToolSpec(
        name="browser",
        description=("Fetch a web page over HTTP(S) and return its title and readable text, or its "
                     "links. Does not run JavaScript. Use for reading a specific URL."),
        parameters=(
            ToolParameter("url", "string", "Absolute http(s) URL to fetch.", max_length=2048),
            ToolParameter("extract", "string", "What to return.", required=False, default="text",
                          enum=("text", "links", "title")),
            ToolParameter("max_chars", "integer", "Maximum characters of text to return.",
                          required=False, default=8000, minimum=100, maximum=100_000),
        ),
        category=ToolCategory.BROWSER, tags=("web", "http"),
        required_permissions=frozenset({"net"}), timeout=30.0,
    )

    def setup(self) -> None:
        self._client = HTTPClient(
            timeout=float(self.config.get("timeout", 15)),
            max_response_bytes=int(self.config.get("max_bytes", 2_000_000)),
            allow_private_networks=bool(self.config.get("allow_private_networks", False)),
            allowed_hosts=self.config.get("allowed_hosts"),
            user_agent=self.config.get("user_agent", "tool-system-browser/1.0"),
            headers={"Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5"})

    def run(self, context: ToolContext, url: str, extract: str = "text", max_chars: int = 8000, **_: Any) -> Any:
        context.check_cancelled()
        resp = self._client.get(url)
        ctype = resp.content_type
        if ctype and not ctype.startswith(_TEXT_TYPES):
            raise ToolExecutionError(f"unsupported content type '{ctype}' (text pages only)")
        body = resp.text()
        if "html" not in ctype and ctype:  # plain text / json / xml: return as is
            return {"url": resp.url, "status": resp.status, "title": "", "content": body[:max_chars],
                    "truncated": len(body) > max_chars or resp.truncated}
        parser = _PageParser(resp.url)
        parser.feed(body)
        parser.close()
        title = " ".join(parser.title.split())
        out: Dict[str, Any] = {"url": resp.url, "status": resp.status, "title": title}
        if extract == "title":
            return out
        if extract == "links":
            seen, links = set(), []
            for link in parser.links:
                if link["url"] not in seen:
                    seen.add(link["url"])
                    links.append(link)
            out["links"] = links[:200]
            out["truncated"] = len(links) > 200
            return out
        text = parser.text
        out["content"] = text[:max_chars]
        out["truncated"] = len(text) > max_chars or resp.truncated
        return out
