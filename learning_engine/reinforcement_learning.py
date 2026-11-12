"""Reinforcement learning: replay buffer, linear double-Q agent and async episode runner."""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from .learning_core import BaseLearner, EventBus, LearnerMetrics, LearningMode, Sample

logger = logging.getLogger("zentrax.learning.rl")


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    mean_loss: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    async def reset(self) -> np.ndarray: ...

    async def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]: ...


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int = 0) -> None:
        self._items: Deque[Transition] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def push(self, t: Transition) -> None:
        self._items.append(t)

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, n: int) -> List[Transition]:
        n = min(n, len(self._items))
        idx = self._rng.choice(len(self._items), size=n, replace=False)
        return [self._items[int(i)] for i in idx]


class LinearQNetwork:
    """Q(s, a) = W[a] . s + b[a] with a slowly-updated target copy (double-Q targets)."""

    def __init__(self, n_features: int, n_actions: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.01, (n_actions, n_features))
        self.b = np.zeros(n_actions)
        self.Wt, self.bt = self.W.copy(), self.b.copy()

    def q(self, S: np.ndarray, target: bool = False) -> np.ndarray:
        W, b = (self.Wt, self.bt) if target else (self.W, self.b)
        return S @ W.T + b

    def update(
        self,
        S: np.ndarray,
        A: np.ndarray,
        R: np.ndarray,
        S2: np.ndarray,
        D: np.ndarray,
        gamma: float,
        lr: float,
    ) -> float:
        n = len(S)
        best = self.q(S2).argmax(axis=1)
        q_next = self.q(S2, target=True)[np.arange(n), best]
        target = R + gamma * (1.0 - D) * q_next
        pred = self.q(S)[np.arange(n), A]
        td = np.clip(target - pred, -1.0, 1.0)
        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)
        np.add.at(gW, A, td[:, None] * S)
        np.add.at(gb, A, td)
        self.W = np.clip(self.W + lr * gW / n, -1e3, 1e3)
        self.b = np.clip(self.b + lr * gb / n, -1e3, 1e3)
        return float(np.abs(target - pred).mean())

    def sync(self, tau: float) -> None:
        self.Wt = tau * self.W + (1 - tau) * self.Wt
        self.bt = tau * self.b + (1 - tau) * self.bt


class RLLearner(BaseLearner):
    """Off-policy value learner. Samples carry state in ``features`` and the action in metadata."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        name: str = "reinforcement",
        lr: float = 0.01,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        warmup: int = 64,
        train_steps_per_call: int = 8,
        train_every: int = 4,
        target_tau: float = 0.05,
        reward_clip: float = 10.0,
        bus: Optional[EventBus] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(name, LearningMode.REINFORCEMENT, bus)
        self.routes = {LearningMode.REINFORCEMENT}
        self.n_features, self.n_actions = n_features, n_actions
        self.lr, self.gamma = lr, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.batch_size, self.warmup = batch_size, warmup
        self.train_steps_per_call = train_steps_per_call
        self.train_every = train_every
        self.target_tau = target_tau
        self.reward_clip = reward_clip
        self.net = LinearQNetwork(n_features, n_actions, seed)
        self.buffer = ReplayBuffer(buffer_size, seed)
        self._rng = np.random.default_rng(seed)
        self.total_steps = 0

    # -- helpers --------------------------------------------------------- #
    def _clean_state(self, s: Any) -> np.ndarray:
        arr = np.clip(np.asarray(s, dtype=np.float64).ravel(), -10.0, 10.0)
        if arr.size != self.n_features:
            raise ValueError(f"Expected state of size {self.n_features}, got {arr.size}")
        return arr

    def _to_transition(self, s: Sample) -> Optional[Transition]:
        if s.reward is None or "action" not in s.metadata:
            return None
        action = int(s.metadata["action"])
        if not 0 <= action < self.n_actions or s.features.size != self.n_features:
            return None
        nxt = s.metadata.get("next_state")
        next_state = s.features if nxt is None else self._clean_state(nxt)
        reward = float(np.clip(s.reward, -self.reward_clip, self.reward_clip))
        return Transition(self._clean_state(s.features), action, reward, next_state, bool(s.metadata.get("done", False)))

    def _train_sync(self, steps: int) -> Optional[float]:
        if len(self.buffer) < max(self.warmup, 2):
            return None
        losses = []
        for _ in range(steps):
            batch = self.buffer.sample(self.batch_size)
            S = np.vstack([t.state for t in batch])
            A = np.array([t.action for t in batch], dtype=np.int64)
            R = np.array([t.reward for t in batch])
            S2 = np.vstack([t.next_state for t in batch])
            D = np.array([float(t.done) for t in batch])
            losses.append(self.net.update(S, A, R, S2, D, self.gamma, self.lr))
            self.net.sync(self.target_tau)
        return float(np.mean(losses))

    # -- acting ---------------------------------------------------------- #
    async def act(self, state: Any, explore: bool = True) -> int:
        s = self._clean_state(state)
        async with self._lock:
            if explore and self._rng.random() < self.epsilon:
                action = int(self._rng.integers(self.n_actions))
            else:
                action = int(self.net.q(s[None, :])[0].argmax())
            if explore:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    async def observe(self, transition: Transition, train: bool = True) -> Optional[float]:
        async with self._lock:
            self.buffer.push(transition)
            self.total_steps += 1
            if not train or self.total_steps % self.train_every:
                return None
            return await asyncio.to_thread(self._train_sync, 1)

    async def run_episode(
        self, env: Environment, max_steps: int = 500, explore: bool = True, learn: bool = True
    ) -> EpisodeResult:
        state = self._clean_state(await env.reset())
        total, losses, steps = 0.0, [], 0
        info: Dict[str, Any] = {}
        for steps in range(1, max_steps + 1):
            action = await self.act(state, explore)
            next_state, reward, done, info = await env.step(action)
            next_state = self._clean_state(next_state)
            total += float(reward)
            if learn:
                loss = await self.observe(
                    Transition(state, action, float(np.clip(reward, -self.reward_clip, self.reward_clip)), next_state, bool(done))
                )
                if loss is not None:
                    losses.append(loss)
            state = next_state
            if done:
                break
        return EpisodeResult(total, steps, float(np.mean(losses)) if losses else None, info)

    # -- BaseLearner ------------------------------------------------------ #
    async def partial_fit(self, samples: Sequence[Sample]) -> LearnerMetrics:
        transitions = [t for t in (self._to_transition(s) for s in samples) if t is not None]
        if not transitions:
            return self._metrics(0)
        async with self._lock:
            for t in transitions:
                self.buffer.push(t)
            self.total_steps += len(transitions)
            loss = await asyncio.to_thread(self._train_sync, self.train_steps_per_call)
        return self._metrics(
            len(transitions),
            loss=loss,
            reward=float(np.mean([t.reward for t in transitions])),
            extra={"epsilon": self.epsilon, "buffer": len(self.buffer)},
        )

    async def predict(self, features: Any) -> Dict[str, Any]:
        s = self._clean_state(features)
        async with self._lock:
            q = self.net.q(s[None, :])[0]
        return {"action": int(q.argmax()), "q_values": q.tolist()}

    async def evaluate(self, samples: Sequence[Sample]) -> LearnerMetrics:
        transitions = [t for t in (self._to_transition(s) for s in samples) if t is not None]
        if not transitions:
            return self._metrics(0)
        async with self._lock:
            errs = []
            for t in transitions:
                q_next = self.net.q(t.next_state[None, :], target=True).max()
                target = t.reward + self.gamma * (1.0 - float(t.done)) * q_next
                errs.append(abs(target - self.net.q(t.state[None, :])[0, t.action]))
        return self._metrics(len(transitions), loss=float(np.mean(errs)), reward=float(np.mean([t.reward for t in transitions])))

    def get_hparams(self) -> Dict[str, float]:
        return {
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": float(self.batch_size),
        }

    def set_hparams(self, params: Mapping[str, float]) -> None:
        if "lr" in params:
            self.lr = float(np.clip(params["lr"], 1e-6, 1.0))
        if "gamma" in params:
            self.gamma = float(np.clip(params["gamma"], 0.0, 0.9999))
        if "epsilon" in params:
            self.epsilon = float(np.clip(params["epsilon"], 0.0, 1.0))
        if "epsilon_min" in params:
            self.epsilon_min = float(np.clip(params["epsilon_min"], 0.0, 1.0))
        if "epsilon_decay" in params:
            self.epsilon_decay = float(np.clip(params["epsilon_decay"], 0.5, 1.0))
        if "batch_size" in params:
            self.batch_size = int(max(1, params["batch_size"]))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "n_actions": self.n_actions,
            "W": self.net.W,
            "b": self.net.b,
            "Wt": self.net.Wt,
            "bt": self.net.bt,
            "total_steps": self.total_steps,
            "hparams": self.get_hparams(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["n_features"]) != self.n_features or int(state["n_actions"]) != self.n_actions:
            raise ValueError("Checkpoint shape does not match RLLearner configuration")
        self.net.W = np.asarray(state["W"], dtype=np.float64)
        self.net.b = np.asarray(state["b"], dtype=np.float64)
        self.net.Wt = np.asarray(state["Wt"], dtype=np.float64)
        self.net.bt = np.asarray(state["bt"], dtype=np.float64)
        self.total_steps = int(state.get("total_steps", 0))
        self.set_hparams(state.get("hparams", {}))
