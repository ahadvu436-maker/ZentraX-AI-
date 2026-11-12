"""Memory consolidation: the "sleep cycle" of the architecture.

Each run (1) expires short-term items, (2) promotes valuable ones to
long-term / episodic / semantic memory, (3) clusters related episodes into
semantic summaries, (4) merges near-duplicates and (5) applies forgetting.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .episodic_memory import EpisodicMemory
from .long_term_memory import LongTermMemory
from .memory_core import MemoryConfig, MemoryItem, MemoryType, Embedder, cosine_similarity, l2_normalize, tokenize
from .semantic_memory import SemanticMemory
from .short_term_memory import ShortTermMemory
from .working_memory import WorkingMemory

logger = logging.getLogger("zentrax.memory.consolidation")

Summarizer = Callable[[Sequence[MemoryItem]], Awaitable[str]]

__all__ = ["ConsolidationReport", "MemoryConsolidator", "Summarizer"]


@dataclass
class ConsolidationReport:
    started_at: float
    duration: float = 0.0
    expired_stm: int = 0
    pruned_working: int = 0
    promoted: int = 0
    reinforced: int = 0
    summarized: int = 0
    merged: int = 0
    forgotten: int = 0
    skipped: bool = False
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class _Cluster:
    centroid: List[float]
    members: List[MemoryItem] = field(default_factory=list)

    def add(self, item: MemoryItem) -> None:
        n = len(self.members)
        merged = [(c * n + v) / (n + 1) for c, v in zip(self.centroid, item.embedding or self.centroid)]
        self.centroid = l2_normalize(merged)
        self.members.append(item)


def _duplicate_pairs(vectors: List[List[float]], threshold: float) -> List[tuple]:
    pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if cosine_similarity(vectors[i], vectors[j]) >= threshold:
                pairs.append((i, j))
    return pairs


class MemoryConsolidator:
    def __init__(
        self,
        *,
        stm: ShortTermMemory,
        working: WorkingMemory,
        ltm: LongTermMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        embedder: Embedder,
        config: MemoryConfig,
        summarizer: Optional[Summarizer] = None,
    ) -> None:
        self.stm, self.working, self.ltm = stm, working, ltm
        self.episodic, self.semantic = episodic, semantic
        self.embedder = embedder
        self.config = config
        self.summarizer = summarizer  # plug in an LLM summariser; default is extractive
        self.last_report: Optional[ConsolidationReport] = None
        self._lock = asyncio.Lock()
        self._promoted = 0
        self._reinforced = 0

    # ------------------------------- promotion -------------------------------- #
    def _worthy(self, item: MemoryItem) -> bool:
        cfg = self.config
        return not item.metadata.get("ephemeral") and (
            item.importance >= cfg.promote_importance
            or item.access_count >= cfg.promote_access_count
            or item.metadata.get("promote") is True
        )

    async def handle_evicted(self, item: MemoryItem) -> None:
        """STM eviction/expiry hook: rescue items that deserve long-term storage."""
        if self._worthy(item):
            await self._promote(item)

    async def _promote(self, item: MemoryItem) -> None:
        kind = str(item.metadata.get("kind", "")).lower()
        if kind in {"event", "episode", "episodic"}:
            target: LongTermMemory = self.episodic
        elif kind in {"fact", "knowledge", "semantic"}:
            target = self.semantic
        else:
            target = self.ltm
        existing = await target.find_duplicate(item.content)
        if existing is not None:
            await target.reinforce(existing.id, boost=0.05)
            self._reinforced += 1
            return
        md = {**item.metadata, "promoted_from": item.memory_type.value, "promoted_at": time.time()}
        if target is self.semantic:
            await self.semantic.add_fact(
                item.content, subject=md.get("subject"), predicate=md.get("predicate"), obj=md.get("object"),
                confidence=float(md.get("confidence", 0.7)), importance=item.importance,
                tags=item.tags, source=item.source, metadata=md,
            )
        else:
            await target.add(dataclasses.replace(item, memory_type=target.memory_type, ttl=None, metadata=md))
        self._promoted += 1

    # ---------------------------------- run ----------------------------------- #
    async def run(self) -> ConsolidationReport:
        report = ConsolidationReport(started_at=time.time())
        if self._lock.locked():
            report.skipped = True
            report.errors.append("consolidation already running")
            return report
        async with self._lock:
            t0 = time.perf_counter()
            promoted0, reinforced0 = self._promoted, self._reinforced
            steps = (
                ("expire", self._step_expire),
                ("promote", self._step_promote),
                ("summarize", self._step_summarize),
                ("merge", self._step_merge),
                ("decay", self._step_decay),
            )
            for name, step in steps:
                try:
                    await step(report)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one failing step must not stop the rest
                    logger.exception("consolidation step %s failed", name)
                    report.errors.append(f"{name}: {exc!r}")
            report.promoted = self._promoted - promoted0
            report.reinforced = self._reinforced - reinforced0
            report.duration = time.perf_counter() - t0
            self.last_report = report
        logger.info("consolidation: %s", report.as_dict())
        return report

    async def _step_expire(self, report: ConsolidationReport) -> None:
        report.pruned_working = len(await self.working.decay_and_prune())
        report.expired_stm = len(await self.stm.expire())

    async def _step_promote(self, report: ConsolidationReport) -> None:
        done: List[str] = []
        for item in await self.stm.candidates(self.config.promote_importance, self.config.promote_access_count):
            await self._promote(item)
            done.append(item.id)
        await self.stm.remove_many(done)

    # ----------------------- episodes -> semantic knowledge ----------------------- #
    async def _step_summarize(self, report: ConsolidationReport) -> None:
        cfg = self.config
        episodes = [e for e in await self.episodic.unconsolidated(limit=500) if e.embedding]
        if len(episodes) < cfg.min_cluster_size:
            return
        clusters: List[_Cluster] = []
        for ep in episodes:  # chronological, so clustering is deterministic
            best, best_sim = None, 0.0
            for cluster in clusters:
                sim = cosine_similarity(cluster.centroid, ep.embedding)
                if sim > best_sim:
                    best, best_sim = cluster, sim
            if best is not None and best_sim >= cfg.cluster_similarity:
                best.add(ep)
            else:
                clusters.append(_Cluster(centroid=list(ep.embedding or []), members=[ep]))
        for cluster in clusters:
            if len(cluster.members) < cfg.min_cluster_size:
                continue
            text = await self._summarize(cluster)
            tag_counts = Counter(t for m in cluster.members for t in m.tags)
            common_tags = [t for t, c in tag_counts.items() if c >= max(2, len(cluster.members) // 2)]
            fact = await self.semantic.add_fact(
                text,
                confidence=0.6,
                importance=sum(m.importance for m in cluster.members) / len(cluster.members),
                tags=["episodic_summary", *common_tags],
                source="consolidation",
                metadata={"kind": "episodic_summary", "derived_from": [m.id for m in cluster.members]},
            )
            await self.episodic.mark_consolidated(cluster.members, fact.id)
            report.summarized += 1

    async def _summarize(self, cluster: _Cluster) -> str:
        members = cluster.members
        if self.summarizer is not None:
            try:
                return await self.summarizer(members)
            except Exception:  # noqa: BLE001
                logger.exception("custom summarizer failed; falling back to extractive summary")
        central = max(members, key=lambda m: cosine_similarity(cluster.centroid, m.embedding))
        terms = Counter(tok for m in members for tok in tokenize(m.content))
        keywords = ", ".join(t for t, _ in terms.most_common(5))
        outcomes = list(dict.fromkeys(str(m.metadata["outcome"]) for m in members if m.metadata.get("outcome")))[:3]
        text = f"Recurring pattern across {len(members)} episodes: {central.content}"
        if keywords:
            text += f" (key terms: {keywords})"
        if outcomes:
            text += f" Outcomes seen: {'; '.join(outcomes)}."
        return text

    # ------------------------------ near-duplicate merge ------------------------------ #
    async def _step_merge(self, report: ConsolidationReport) -> None:
        for mem in (self.ltm, self.semantic):
            items = [
                i for i in await mem.list_items(limit=self.config.dedupe_scan_limit, include_embedding=True)
                if i.embedding
            ]
            if len(items) < 2:
                continue
            pairs = await asyncio.to_thread(
                _duplicate_pairs, [i.embedding for i in items], self.config.duplicate_similarity  # type: ignore[misc]
            )
            gone: set = set()
            for a, b in pairs:
                x, y = items[a], items[b]
                if x.id in gone or y.id in gone:
                    continue
                keeper, loser = (x, y) if (x.importance, x.access_count) >= (y.importance, y.access_count) else (y, x)
                await mem.update_fields(
                    keeper.id,
                    importance=max(keeper.importance, loser.importance),
                    access_count=keeper.access_count + loser.access_count,
                    tags=list(dict.fromkeys([*keeper.tags, *loser.tags])),
                    metadata={**keeper.metadata, "merged_from": [*keeper.metadata.get("merged_from", []), loser.id]},
                )
                await mem.delete(loser.id)
                gone.add(loser.id)
                report.merged += 1

    async def _step_decay(self, report: ConsolidationReport) -> None:
        for mem in (self.ltm, self.episodic, self.semantic):
            report.forgotten += await mem.apply_decay()
