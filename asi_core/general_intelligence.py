"""General intelligence loop: perceive -> abstract -> transfer -> act -> reflect."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .capability_manager import CapabilityManager
from .cognitive_architecture import CognitiveArchitecture, CognitiveContext
from .cross_domain_transfer import CrossDomainTransferEngine, TransferResult


@dataclass
class Problem:
    domain: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    permissions: frozenset[str] = frozenset({"internal"})
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Solution:
    problem_id: str
    success: bool
    output: Any = None
    capability: str | None = None
    transfers: list[TransferResult] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    error: str | None = None


class GeneralIntelligence:
    def __init__(self, architecture: CognitiveArchitecture,
                 capabilities: CapabilityManager,
                 transfer: CrossDomainTransferEngine) -> None:
        self.arch, self.caps, self.transfer = architecture, capabilities, transfer

    async def solve(self, problem: Problem) -> Solution:
        ctx = CognitiveContext(task_id=problem.id, data={"input": problem.description})
        ctx = await self.arch.run_pipeline(ctx)
        transfers = self.transfer.transfer(problem.domain, problem.description)
        ctx.trace.append(f"retrieved {len(transfers)} transferable knowledge units")

        candidates = self.caps.find_by_domain(problem.domain)
        if not candidates:
            return Solution(problem.id, False, trace=ctx.trace, transfers=transfers,
                            error=f"no capability for domain '{problem.domain}'")
        cap = candidates[0]
        try:
            out = await self.caps.invoke(
                cap.descriptor.name, problem.permissions,
                problem=problem, context=ctx.data, transfers=transfers)
        except Exception as e:  # reflect: record failure, never crash the core
            ctx.trace.append(f"reflection: {cap.descriptor.name} failed")
            return Solution(problem.id, False, capability=cap.descriptor.name,
                            transfers=transfers, trace=ctx.trace, error=repr(e))
        ctx.trace.append("reflection: success -> knowledge stored")
        self.transfer.add_knowledge(problem.domain, problem.description)
        return Solution(problem.id, True, out, cap.descriptor.name, transfers, ctx.trace)
