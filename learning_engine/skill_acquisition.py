"""Skill acquisition: discover, practise, select and compose reusable skills."""
from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .learning_core import BaseLearner, EventBus, LearnerMetrics, LearningMode, Sample

logger = logging.getLogger("zentrax.learning.skills")

TOPIC_SKILL_EXECUTED = "skill.executed"
TOPIC_SKILL_CREATED = "skill.created"


@dataclass
class SkillOutcome:
    success: bool
    reward: float = 0.0
    output: Any = None
    error: Optional[str] = None


SkillExecutor = Callable[[Dict[str, Any]], Awaitable[SkillOutcome]]


@dataclass
class Skill:
    name: str
    description: str = ""
    skill_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prerequisites: List[str] = field(default_factory=list)  # skill ids
    components: List[str] = field(default_factory=list)  # skill ids, for composites
    tags: List[str] = field(default_factory=list)
    proficiency: float = 0.0
    attempts: int = 0
    successes: int = 0
    avg_reward: float = 0.0
    centroid: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "prerequisites": list(self.prerequisites),
            "components": list(self.components),
            "tags": list(self.tags),
            "proficiency": self.proficiency,
            "attempts": self.attempts,
            "successes": self.successes,
            "avg_reward": self.avg_reward,
            "centroid": self.centroid,
            "created_at": self.created_at,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Skill":
        c = d.get("centroid")
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            skill_id=d["skill_id"],
            prerequisites=list(d.get("prerequisites", [])),
            components=list(d.get("components", [])),
            tags=list(d.get("tags", [])),
            proficiency=float(d.get("proficiency", 0.0)),
            attempts=int(d.get("attempts", 0)),
            successes=int(d.get("successes", 0)),
            avg_reward=float(d.get("avg_reward", 0.0)),
            centroid=None if c is None else np.asarray(c, dtype=np.float64),
            created_at=float(d.get("created_at", time.time())),
            last_used=d.get("last_used"),
        )


class SkillLibrary:
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def add(self, skill: Skill) -> Skill:
        self._skills[skill.skill_id] = skill
        return skill

    def get(self, key: str) -> Optional[Skill]:
        if key in self._skills:
            return self._skills[key]
        return next((s for s in self._skills.values() if s.name == key), None)

    def remove(self, skill_id: str) -> None:
        self._skills.pop(skill_id, None)

    def all(self) -> List[Skill]:
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)


class SkillAcquirer(BaseLearner):
    """Learns which skills work where, and creates new skills from successful experience."""

    def __init__(
        self,
        name: str = "skills",
        mastery_threshold: float = 0.8,
        min_mastery_attempts: int = 5,
        proficiency_alpha: float = 0.1,
        exploration: float = 0.5,
        discovery_min_examples: int = 8,
        execution_timeout: float = 30.0,
        bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(name, LearningMode.SKILL, bus)
        self.routes = {LearningMode.SKILL}
        self.library = SkillLibrary()
        self.mastery_threshold = mastery_threshold
        self.min_mastery_attempts = min_mastery_attempts
        self.proficiency_alpha = proficiency_alpha
        self.exploration = exploration
        self.discovery_min_examples = discovery_min_examples
        self.execution_timeout = execution_timeout
        self._executors: Dict[str, SkillExecutor] = {}
        self._candidates: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._total_attempts = 0

    # -- registration ----------------------------------------------------- #
    def add_skill(
        self,
        name: str,
        executor: Optional[SkillExecutor] = None,
        description: str = "",
        prerequisites: Sequence[str] = (),
        tags: Sequence[str] = (),
        centroid: Optional[np.ndarray] = None,
    ) -> Skill:
        if self.library.get(name) is not None:
            raise ValueError(f"Skill '{name}' already exists")
        prereq_ids = []
        for p in prerequisites:
            dep = self.library.get(p)
            if dep is None:
                raise KeyError(f"Unknown prerequisite '{p}'")
            prereq_ids.append(dep.skill_id)
        skill = self.library.add(
            Skill(
                name=name,
                description=description,
                prerequisites=prereq_ids,
                tags=list(tags),
                centroid=None if centroid is None else np.asarray(centroid, dtype=np.float64).ravel(),
            )
        )
        if executor is not None:
            self._executors[skill.skill_id] = executor
        return skill

    def bind_executor(self, skill_key: str, executor: SkillExecutor) -> None:
        skill = self.library.get(skill_key)
        if skill is None:
            raise KeyError(skill_key)
        self._executors[skill.skill_id] = executor

    def compose(self, name: str, component_keys: Sequence[str], description: str = "") -> Skill:
        """Create a composite skill that runs components in order, stopping on first failure."""
        comps = []
        for key in component_keys:
            c = self.library.get(key)
            if c is None:
                raise KeyError(f"Unknown component '{key}'")
            comps.append(c)
        composite = self.add_skill(name, description=description or f"Composite of {len(comps)} skills", tags=["composite"])
        composite.components = [c.skill_id for c in comps]
        composite.prerequisites = [c.skill_id for c in comps]

        async def _run(context: Dict[str, Any]) -> SkillOutcome:
            ctx, total, output = dict(context), 0.0, None
            for comp in comps:
                outcome = await self.execute(comp.skill_id, ctx)
                total += outcome.reward
                if not outcome.success:
                    return SkillOutcome(False, total, output, f"component '{comp.name}' failed: {outcome.error}")
                output = outcome.output
                ctx["previous_output"] = output
            return SkillOutcome(True, total, output)

        self._executors[composite.skill_id] = _run
        return composite

    # -- state ------------------------------------------------------------ #
    def is_mastered(self, skill: Skill) -> bool:
        return skill.attempts >= self.min_mastery_attempts and skill.proficiency >= self.mastery_threshold

    def unmet_prerequisites(self, skill: Skill) -> List[str]:
        missing = []
        for pid in skill.prerequisites:
            dep = self.library.get(pid)
            if dep is None or not self.is_mastered(dep):
                missing.append(dep.name if dep else pid)
        return missing

    def _record(self, skill: Skill, success: bool, reward: float, features: Optional[np.ndarray]) -> None:
        skill.attempts += 1
        skill.successes += int(success)
        self._total_attempts += 1
        skill.last_used = time.time()
        a = max(self.proficiency_alpha, 1.0 / skill.attempts)
        skill.proficiency += a * (float(success) - skill.proficiency)
        skill.avg_reward += a * (reward - skill.avg_reward)
        if success and features is not None and features.size:
            if skill.centroid is None or skill.centroid.size != features.size:
                skill.centroid = features.copy()
            else:
                skill.centroid += 0.1 * (features - skill.centroid)

    # -- execution & selection --------------------------------------------- #
    async def execute(
        self, skill_key: str, context: Optional[Dict[str, Any]] = None, features: Optional[Any] = None
    ) -> SkillOutcome:
        skill = self.library.get(skill_key)
        if skill is None:
            raise KeyError(f"Unknown skill '{skill_key}'")
        missing = self.unmet_prerequisites(skill)
        if missing:
            return SkillOutcome(False, 0.0, None, f"unmet prerequisites: {', '.join(missing)}")
        executor = self._executors.get(skill.skill_id)
        if executor is None:
            return SkillOutcome(False, 0.0, None, "no executor bound")
        try:
            outcome = await asyncio.wait_for(executor(dict(context or {})), self.execution_timeout)
        except Exception as exc:  # noqa: BLE001
            outcome = SkillOutcome(False, -1.0, None, repr(exc))
        feats = None if features is None else np.asarray(features, dtype=np.float64).ravel()
        async with self._lock:
            self._record(skill, outcome.success, outcome.reward, feats)
            self.updates += 1
        if self.bus is not None:
            await self.bus.publish(
                TOPIC_SKILL_EXECUTED,
                {"skill": skill.name, "success": outcome.success, "reward": outcome.reward, "error": outcome.error},
                self.name,
            )
        return outcome

    def _score(self, skill: Skill, x: Optional[np.ndarray], explore: bool) -> float:
        sim = 0.5
        if x is not None and skill.centroid is not None and skill.centroid.size == x.size:
            denom = float(np.linalg.norm(x) * np.linalg.norm(skill.centroid))
            sim = 0.5 * (1.0 + float(x @ skill.centroid) / denom) if denom > 1e-12 else 0.5
        score = sim * (0.2 + 0.8 * skill.proficiency)
        if explore:
            score += self.exploration * math.sqrt(2.0 * math.log(self._total_attempts + 2) / (skill.attempts + 1))
        return score

    def _ranked(self, features: Optional[Any], explore: bool) -> List[Skill]:
        x = None if features is None else np.asarray(features, dtype=np.float64).ravel()
        usable = [
            s for s in self.library.all() if s.skill_id in self._executors and not self.unmet_prerequisites(s)
        ]
        return sorted(usable, key=lambda s: self._score(s, x, explore), reverse=True)

    async def select(self, features: Optional[Any] = None, explore: bool = True) -> Optional[Skill]:
        async with self._lock:
            ranked = self._ranked(features, explore)
        return ranked[0] if ranked else None

    # -- BaseLearner ------------------------------------------------------- #
    async def partial_fit(self, samples: Sequence[Sample]) -> LearnerMetrics:
        used, rewards, created = 0, [], []
        async with self._lock:
            for s in samples:
                key = s.metadata.get("skill_id") or s.metadata.get("skill")
                skill = self.library.get(key) if key else None
                reward = float(s.reward) if s.reward is not None else 0.0
                if skill is not None:
                    success = bool(s.metadata.get("success", reward > 0))
                    self._record(skill, success, reward, s.features)
                    used += 1
                    rewards.append(reward)
                elif s.reward is not None and s.reward > 0 and "cluster" in s.metadata:
                    cid = str(s.metadata["cluster"])
                    self._candidates[cid].append(s.features)
                    used += 1
                    rewards.append(reward)
                    if len(self._candidates[cid]) >= self.discovery_min_examples:
                        created.append(self._promote_candidate(cid))
        for skill in created:
            if skill is not None and self.bus is not None:
                await self.bus.publish(TOPIC_SKILL_CREATED, {"skill": skill.name, "id": skill.skill_id}, self.name)
        if not used:
            return self._metrics(0)
        return self._metrics(
            used,
            reward=float(np.mean(rewards)) if rewards else None,
            extra={"skills": len(self.library), "discovered": len([c for c in created if c])},
        )

    def _promote_candidate(self, cluster_id: str) -> Optional[Skill]:
        examples = self._candidates.pop(cluster_id, [])
        name = f"auto_{cluster_id}"
        if not examples or self.library.get(name) is not None:
            return None
        centroid = np.mean(np.vstack(examples), axis=0)
        return self.library.add(
            Skill(name=name, description=f"Discovered from cluster {cluster_id}", tags=["discovered"], centroid=centroid)
        )

    async def predict(self, features: Any) -> List[Dict[str, Any]]:
        async with self._lock:
            ranked = self._ranked(features, explore=False)[:3]
            x = np.asarray(features, dtype=np.float64).ravel()
            return [
                {"skill_id": s.skill_id, "name": s.name, "score": self._score(s, x, False), "proficiency": s.proficiency}
                for s in ranked
            ]

    def get_hparams(self) -> Dict[str, float]:
        return {"exploration": self.exploration, "proficiency_alpha": self.proficiency_alpha}

    def set_hparams(self, params: Mapping[str, float]) -> None:
        if "exploration" in params:
            self.exploration = float(max(params["exploration"], 0.0))
        if "proficiency_alpha" in params:
            self.proficiency_alpha = float(np.clip(params["proficiency_alpha"], 0.01, 1.0))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "skills": [s.to_dict() for s in self.library.all()],
            "total_attempts": self._total_attempts,
            "hparams": self.get_hparams(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # Restores statistics; executors must be re-bound by the application via bind_executor().
        for d in state.get("skills", []):
            existing = self.library.get(d["skill_id"]) or self.library.get(d["name"])
            restored = Skill.from_dict(d)
            if existing is not None and existing.skill_id != restored.skill_id:
                executor = self._executors.pop(existing.skill_id, None)
                self.library.remove(existing.skill_id)
                if executor is not None:
                    self._executors[restored.skill_id] = executor
            self.library.add(restored)
        self._total_attempts = int(state.get("total_attempts", 0))
        self.set_hparams(state.get("hparams", {}))
