"""Adaptation engine: watches learner metrics, detects drift/plateaus and retunes hyper-parameters."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .learning_core import (
    TOPIC_METRICS,
    BaseLearner,
    EventBus,
    LearnerMetrics,
    LearningEvent,
)

logger = logging.getLogger("zentrax.learning.adaptation")

TOPIC_DRIFT = "adaptation.drift"
TOPIC_APPLIED = "adaptation.applied"


class PageHinkley:
    """Page-Hinkley test for an *increase* in a stream's mean (e.g. loss rising)."""

    def __init__(self, delta: float = 0.005, threshold: float = 0.5, min_samples: int = 10, alpha: float = 0.99) -> None:
        self.delta, self.threshold, self.min_samples, self.alpha = delta, threshold, min_samples, alpha
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.cum_min = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum = self.alpha * self.cum + (x - self.mean - self.delta)
        self.cum_min = min(self.cum_min, self.cum)
        if self.n >= self.min_samples and self.cum - self.cum_min > self.threshold:
            self.reset()
            return True
        return False


@dataclass
class AdaptationRule:
    """How one hyper-parameter reacts to a trigger. Ops: mul | add | set | max | min."""

    key: str
    on_drift: Optional[Tuple[str, float]] = None
    on_plateau: Optional[Tuple[str, float]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_rel: Optional[float] = None  # relative to the baseline value
    max_rel: Optional[float] = None


def default_rules() -> List[AdaptationRule]:
    return [
        AdaptationRule("lr", on_drift=("mul", 2.0), on_plateau=("mul", 0.5), min_rel=0.01, max_rel=10.0),
        AdaptationRule("epsilon", on_drift=("max", 0.6), min_value=0.01, max_value=1.0),
        AdaptationRule("replay_ratio", on_drift=("mul", 1.5), min_value=0.1, max_value=2.0),
        AdaptationRule("novelty_sigma", on_drift=("mul", 0.9), on_plateau=("mul", 1.05), min_value=1.5, max_value=6.0),
    ]


@dataclass
class AdaptationConfig:
    plateau_patience: int = 20
    plateau_min_delta: float = 1e-3
    recover_after: int = 10
    cooldown_updates: int = 5
    drift_delta: float = 0.005
    drift_threshold: float = 0.5
    rules: List[AdaptationRule] = field(default_factory=default_rules)


@dataclass
class _LearnerState:
    detector: PageHinkley
    baseline: Dict[str, float]
    best: float = float("inf")
    stale: int = 0
    improving: int = 0
    cooldown: int = 0
    drifts: int = 0


class AdaptationEngine:
    """Subscribes to learner metrics and applies rule-based hyper-parameter adaptation."""

    def __init__(
        self,
        bus: EventBus,
        resolver: Callable[[str], Optional[BaseLearner]],
        config: Optional[AdaptationConfig] = None,
    ) -> None:
        self.bus = bus
        self.resolver = resolver
        self.config = config or AdaptationConfig()
        self._state: Dict[str, _LearnerState] = {}
        self._running = False

    async def start(self) -> None:
        if not self._running:
            self.bus.subscribe(TOPIC_METRICS, self._on_metrics)
            self._running = True

    async def stop(self) -> None:
        if self._running:
            self.bus.unsubscribe(TOPIC_METRICS, self._on_metrics)
            self._running = False

    # -- signal handling -------------------------------------------------- #
    @staticmethod
    def _signal(m: LearnerMetrics) -> Optional[float]:
        if m.loss is not None:
            return float(m.loss)
        if m.reward is not None:
            return -float(m.reward)
        return None

    def _state_for(self, learner: BaseLearner) -> _LearnerState:
        st = self._state.get(learner.name)
        if st is None:
            st = _LearnerState(
                detector=PageHinkley(self.config.drift_delta, self.config.drift_threshold),
                baseline=dict(learner.get_hparams()),
            )
            self._state[learner.name] = st
        return st

    async def _on_metrics(self, event: LearningEvent) -> None:
        metrics = event.payload
        if not isinstance(metrics, LearnerMetrics):
            return
        learner = self.resolver(metrics.learner)
        signal = self._signal(metrics)
        if learner is None or signal is None:
            return
        st = self._state_for(learner)
        cfg = self.config
        st.cooldown = max(0, st.cooldown - 1)

        if signal < st.best - cfg.plateau_min_delta:
            st.best, st.stale, st.improving = signal, 0, st.improving + 1
        else:
            st.stale, st.improving = st.stale + 1, 0

        if st.detector.update(signal) and st.cooldown == 0:
            st.drifts += 1
            st.best, st.stale, st.cooldown = signal, 0, cfg.cooldown_updates
            await self.bus.publish(TOPIC_DRIFT, {"learner": learner.name, "signal": signal}, "adaptation")
            await self._apply(learner, st, "drift")
        elif st.stale >= cfg.plateau_patience and st.cooldown == 0:
            st.stale, st.cooldown = 0, cfg.cooldown_updates
            await self._apply(learner, st, "plateau")
        elif st.improving >= cfg.recover_after:
            st.improving = 0
            await self._relax(learner, st)

    # -- hyper-parameter changes ----------------------------------------- #
    @staticmethod
    def _op(current: float, op: Tuple[str, float]) -> float:
        name, v = op
        return {"mul": current * v, "add": current + v, "set": v, "max": max(current, v), "min": min(current, v)}[name]

    def _clamp(self, rule: AdaptationRule, value: float, baseline: Optional[float]) -> float:
        lo, hi = rule.min_value, rule.max_value
        if baseline is not None and baseline > 0:
            if rule.min_rel is not None:
                lo = max(lo, baseline * rule.min_rel) if lo is not None else baseline * rule.min_rel
            if rule.max_rel is not None:
                hi = min(hi, baseline * rule.max_rel) if hi is not None else baseline * rule.max_rel
        if lo is not None:
            value = max(value, lo)
        if hi is not None:
            value = min(value, hi)
        return value

    async def _apply(self, learner: BaseLearner, st: _LearnerState, trigger: str) -> Dict[str, float]:
        current = learner.get_hparams()
        updates: Dict[str, float] = {}
        for rule in self.config.rules:
            op = rule.on_drift if trigger == "drift" else rule.on_plateau
            if op is None or rule.key not in current:
                continue
            new = self._clamp(rule, self._op(current[rule.key], op), st.baseline.get(rule.key))
            if abs(new - current[rule.key]) > 1e-12:
                updates[rule.key] = new
        if updates:
            learner.set_hparams(updates)
            await self.bus.publish(
                TOPIC_APPLIED,
                {"learner": learner.name, "trigger": trigger, "changes": updates},
                "adaptation",
            )
        return updates

    async def _relax(self, learner: BaseLearner, st: _LearnerState) -> None:
        current = learner.get_hparams()
        updates = {
            k: cur + (st.baseline[k] - cur) * 0.5
            for k, cur in current.items()
            if k in st.baseline and abs(cur - st.baseline[k]) > 1e-9 * max(1.0, abs(st.baseline[k]))
        }
        if updates:
            learner.set_hparams(updates)
            await self.bus.publish(
                TOPIC_APPLIED, {"learner": learner.name, "trigger": "recover", "changes": updates}, "adaptation"
            )

    # -- manual / contextual API ----------------------------------------- #
    async def adapt_to_context(self, learner_name: str, context: Mapping[str, float]) -> Dict[str, float]:
        """React to external signals: ``novelty`` > 0.7 behaves like drift, ``stability`` > 0.8 relaxes."""
        learner = self.resolver(learner_name)
        if learner is None:
            return {}
        st = self._state_for(learner)
        if context.get("novelty", 0.0) > 0.7:
            return await self._apply(learner, st, "drift")
        if context.get("stability", 0.0) > 0.8:
            await self._relax(learner, st)
        return {}

    def override(self, learner_name: str, params: Mapping[str, float], rebaseline: bool = True) -> None:
        learner = self.resolver(learner_name)
        if learner is None:
            raise KeyError(learner_name)
        learner.set_hparams(params)
        if rebaseline:
            self._state_for(learner).baseline.update({k: float(v) for k, v in params.items()})

    def report(self) -> Dict[str, Any]:
        return {
            name: {"drifts": st.drifts, "best": None if st.best == float("inf") else st.best, "baseline": st.baseline}
            for name, st in self._state.items()
        }
