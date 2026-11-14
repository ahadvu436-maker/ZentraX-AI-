"""Web search tool backed by pluggable providers (DuckDuckGo, Brave, SearXNG).

Config keys: ``provider`` (default ``duckduckgo``), ``provider_config`` (kwargs for the
provider, e.g. ``{"api_key": ...}`` or ``{"base_url": ...}``), ``timeout``,
``allow_private_networks`` (needed for a SearXNG on localhost).
"""
from __future__ import annotations

from typing import Any

from ..api_integrations import HTTPClient, SearchProvider, create_search_provider
from ..tool_core import BaseTool, ToolCategory, ToolContext, ToolParameter, ToolSpec


class SearchTool(BaseTool):
    """Searches the web and returns titles, URLs and snippets."""

    spec = ToolSpec(
        name="web_search",
        description=("Search the web and return a list of results (title, url, snippet). Use it to "
                     "find sources, then fetch promising URLs with the browser tool."),
        parameters=(
            ToolParameter("query", "string", "Search query.", max_length=500),
            ToolParameter("max_results", "integer", "How many results to return.", required=False,
                          default=5, minimum=1, maximum=20),
        ),
        category=ToolCategory.SEARCH, tags=("web", "search"),
        required_permissions=frozenset({"net"}), timeout=30.0,
    )

    def setup(self) -> None:
        self._provider: SearchProvider = create_search_provider(
            self.config.get("provider", "duckduckgo"), **dict(self.config.get("provider_config", {})))
        self._client = HTTPClient(
            timeout=float(self.config.get("timeout", 15)),
            allow_private_networks=bool(self.config.get("allow_private_networks", False)),
            user_agent="Mozilla/5.0 (compatible; tool-system-search/1.0)")

    def run(self, context: ToolContext, query: str, max_results: int = 5, **_: Any) -> Any:
        if not query.strip():
            from ..tool_core import ToolValidationError
            raise ToolValidationError("query must not be empty")
        context.check_cancelled()
        results = self._provider.search(query.strip(), max_results, self._client)
        return {"query": query, "provider": self._provider.name,
                "results": [r.to_dict() for r in results]}
