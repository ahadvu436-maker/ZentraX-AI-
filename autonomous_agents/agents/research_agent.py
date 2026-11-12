"""ResearchAgent: gathers, ranks and summarises information.

The agent is provider-agnostic: pass any number of async ``search_providers``
(``async def provider(query, limit) -> list[dict]`` returning dicts with
``title`` / ``url`` / ``snippet`` and optional ``score``), and/or supply
``documents`` in the task payload. An optional ``llm`` produces abstractive
summaries; without one the agent falls back to extractive summarisation.

Payload actions
---------------
``research`` (default)  search + rank + summarise      keys: ``query``, ``max_results``, ``documents``
``search``              search + rank only             keys: ``query``, ``max_results``, ``documents``
``summarize``           summarise supplied text        keys: ``text`` | ``documents``, ``focus``
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..agent_core import AgentConfig, BaseAgent, Reasoner, Task, call_reasoner

__all__ = ["ResearchAgent", "SearchProvider"]

from typing import Awaitable, Callable

SearchProvider = Callable[[str, int], Awaitable[List[Dict[str, Any]]]]

_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i in is it its of on or our she that the their "
    "them they this to was we were what when where which who will with you your about into than then there these "
    "those how why can could should would may might do does did not no".split()
)


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _overlap(query_tokens: set, text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(_tokens(text))) / len(query_tokens)


class ResearchAgent(BaseAgent):
    agent_type = "research"
    DEFAULT_CAPABILITIES = frozenset({"research", "web_search", "summarization", "fact_gathering"})

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        agent_id: Optional[str] = None,
        search_providers: Sequence[SearchProvider] = (),
        llm: Optional[Reasoner] = None,
        max_results: int = 10,
        provider_timeout: float = 20.0,
        max_summary_sentences: int = 5,
    ) -> None:
        config = config or AgentConfig(
            name="research-agent", capabilities=self.DEFAULT_CAPABILITIES, max_concurrent_tasks=4
        )
        super().__init__(config, agent_id)
        self._providers: List[SearchProvider] = list(search_providers)
        self._llm = llm
        self._max_results = max_results
        self._provider_timeout = provider_timeout
        self._max_sentences = max_summary_sentences

    def add_provider(self, provider: SearchProvider) -> None:
        self._providers.append(provider)

    # ------------------------------------------------------------------ #

    async def handle_task(self, task: Task) -> Dict[str, Any]:
        action = str(task.payload.get("action", "research")).lower()
        if action in ("research", "search"):
            query = self._query(task)
            limit = int(task.payload.get("max_results", self._max_results))
            results, warnings = await self._collect(query, limit, task.payload.get("documents"))
            output: Dict[str, Any] = {
                "query": query,
                "result_count": len(results),
                "results": results,
                "sources": [{"title": r["title"], "url": r["url"]} for r in results if r["url"] or r["title"]],
                "warnings": warnings,
            }
            if action == "research":
                output["summary"] = await self._summarise(query, results)
            return output
        if action == "summarize":
            return await self._action_summarize(task)
        raise ValueError(f"unknown research action: {action!r}")

    @staticmethod
    def _query(task: Task) -> str:
        p = task.payload
        query = p.get("query") or p.get("instructions") or p.get("goal") or task.description
        if not query or not str(query).strip():
            raise ValueError("research task requires a 'query' in its payload")
        return str(query).strip()

    async def _action_summarize(self, task: Task) -> Dict[str, Any]:
        p = task.payload
        focus = str(p.get("focus", "") or "")
        items: List[Dict[str, Any]] = []
        if p.get("text"):
            items.append(self._normalise({"title": "input", "snippet": str(p["text"])}, "input"))
        for doc in p.get("documents") or []:
            normalised = self._normalise(doc if isinstance(doc, Mapping) else {"snippet": str(doc)}, "input")
            if normalised:
                items.append(normalised)
        items = [i for i in items if i]
        if not items:
            raise ValueError("summarize requires 'text' or 'documents'")
        return {"focus": focus, "summary": await self._summarise(focus, items), "source_count": len(items)}

    # ------------------------------------------------------------------ #
    # Collection and ranking
    # ------------------------------------------------------------------ #

    async def _collect(
        self, query: str, limit: int, documents: Optional[Sequence[Any]]
    ) -> "tuple[List[Dict[str, Any]], List[str]]":
        warnings: List[str] = []
        raw: List[Dict[str, Any]] = []

        for doc in documents or []:
            item = self._normalise(doc if isinstance(doc, Mapping) else {"snippet": str(doc)}, "local")
            if item:
                raw.append(item)

        if self._providers:

            async def call(provider: SearchProvider) -> List[Dict[str, Any]]:
                return await asyncio.wait_for(provider(query, limit), timeout=self._provider_timeout)

            outcomes = await asyncio.gather(*(call(p) for p in self._providers), return_exceptions=True)
            for index, outcome in enumerate(outcomes):
                if isinstance(outcome, Exception):
                    warnings.append(f"provider #{index} failed: {type(outcome).__name__}: {outcome}")
                    continue
                for entry in outcome or []:
                    item = self._normalise(entry, f"provider-{index}")
                    if item:
                        raw.append(item)
        elif not raw:
            warnings.append("no search providers configured and no documents supplied")

        return self._rank(query, raw, limit), warnings

    @staticmethod
    def _normalise(entry: Mapping[str, Any], source: str) -> Optional[Dict[str, Any]]:
        title = str(entry.get("title") or "").strip()
        url = str(entry.get("url") or "").strip()
        snippet = str(entry.get("snippet") or entry.get("content") or entry.get("text") or "").strip()
        if not (title or snippet):
            return None
        try:
            base = float(entry.get("score", 0) or 0)
        except (TypeError, ValueError):
            base = 0.0
        return {"title": title, "url": url, "snippet": snippet, "source": source, "provider_score": base}

    @staticmethod
    def _rank(query: str, items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        q = set(_tokens(query))
        best: Dict[str, Dict[str, Any]] = {}
        for item in items:
            key = item["url"].lower().rstrip("/") or item["title"].lower() or item["snippet"][:80].lower()
            score = (
                0.6 * _overlap(q, f"{item['title']} {item['snippet']}")
                + 0.4 * _overlap(q, item["title"])
                + 0.05 * max(0.0, min(1.0, item["provider_score"]))
            )
            candidate = {**item, "relevance": round(score, 4)}
            if key not in best or candidate["relevance"] > best[key]["relevance"]:
                best[key] = candidate
        ranked = sorted(best.values(), key=lambda r: r["relevance"], reverse=True)
        return ranked[: max(1, limit)]

    # ------------------------------------------------------------------ #
    # Summarisation
    # ------------------------------------------------------------------ #

    async def _summarise(self, query: str, items: List[Dict[str, Any]]) -> str:
        if not items:
            return ""
        if self._llm is not None:
            context = "\n\n".join(f"[{i}] {r['title']}\n{r['snippet']}" for i, r in enumerate(items, 1))
            focus = f"Focus on: {query}\n\n" if query else ""
            prompt = (
                "Summarise the numbered sources below into a concise, factual synthesis. "
                "Cite sources like [1]. Do not invent facts.\n\n" + focus + context
            )
            try:
                text = (await call_reasoner(self._llm, prompt)).strip()
                if text:
                    return text
            except Exception:  # noqa: BLE001
                self.log.exception("LLM summarisation failed; using extractive fallback")
        return self._extractive_summary(query, items)

    def _extractive_summary(self, query: str, items: List[Dict[str, Any]]) -> str:
        q = set(_tokens(query))
        scored: List[tuple] = []
        seen: set = set()
        order = 0
        for number, item in enumerate(items, 1):
            sentences = re.split(r"(?<=[.!?])\s+", item["snippet"]) if item["snippet"] else [item["title"]]
            for position, sentence in enumerate(s.strip() for s in sentences):
                if len(sentence) < 15 or sentence.lower() in seen:
                    continue
                seen.add(sentence.lower())
                score = _overlap(q, sentence) if q else 1.0 / (1 + position)
                scored.append((score, order, f"{sentence} [{number}]"))
                order += 1
        top = sorted(scored, key=lambda s: s[0], reverse=True)[: self._max_sentences]
        return " ".join(s[2] for s in sorted(top, key=lambda s: s[1]))
