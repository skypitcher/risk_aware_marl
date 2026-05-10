from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sat_net.sim_kernel import FLOWLET_DELIVERED, FLOWLET_DROPPED


@dataclass(slots=True)
class RewardConfig:
    """Configurable reward shaping for per-flowlet routing transitions."""

    progress_weight: float = 1.0
    delay_weight: float = 1.0
    queue_cost_weight: float = 0.25
    delivered_bonus: float = 1.0
    dropped_penalty: float = 1.0
    truncated_penalty: float = 0.0
    delay_norm: float = 1000.0
    cost_norm: float = 1000.0
    cost_limit: float = 10.0

    @classmethod
    def from_config(cls, config: Any) -> "RewardConfig":
        reward = _get(config, "reward", None)
        source = reward if reward is not None else config

        def value(name: str, default: float) -> float:
            if hasattr(source, "get"):
                return float(_get(source, name, default))
            return float(default)

        delay_norm = value("delay_norm", float(_get(config, "delay_norm", 1000.0)))
        return cls(
            progress_weight=value("progress_weight", 1.0),
            delay_weight=value("delay_weight", 1.0),
            queue_cost_weight=value("queue_cost_weight", 0.25),
            delivered_bonus=value("delivered_bonus", float(_get(config, "delivered_bonus", 1.0))),
            dropped_penalty=value("dropped_penalty", float(_get(config, "dropped_penalty", 1.0))),
            truncated_penalty=value("truncated_penalty", 0.0),
            delay_norm=delay_norm,
            cost_norm=value("cost_norm", delay_norm),
            cost_limit=value("cost_limit", float(_get(config, "cost_limit", 10.0))),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class TransitionReward:
    reward: float
    cost: float
    progress_reward: float
    delay_penalty: float
    queue_penalty: float
    terminal_reward: float


@dataclass(slots=True)
class RewardStats:
    transitions: int = 0
    terminal_transitions: int = 0
    delivered_transitions: int = 0
    dropped_transitions: int = 0
    truncated_transitions: int = 0
    reward_sum: float = 0.0
    cost_sum: float = 0.0
    progress_reward_sum: float = 0.0
    delay_penalty_sum: float = 0.0
    queue_penalty_sum: float = 0.0
    terminal_reward_sum: float = 0.0

    def add(self, value: TransitionReward, terminal_status: int | None = None, truncated: bool = False) -> None:
        self.transitions += 1
        self.reward_sum += value.reward
        self.cost_sum += value.cost
        self.progress_reward_sum += value.progress_reward
        self.delay_penalty_sum += value.delay_penalty
        self.queue_penalty_sum += value.queue_penalty
        self.terminal_reward_sum += value.terminal_reward
        if truncated:
            self.truncated_transitions += 1
        if terminal_status is not None:
            self.terminal_transitions += 1
            if terminal_status == FLOWLET_DELIVERED:
                self.delivered_transitions += 1
            elif terminal_status == FLOWLET_DROPPED:
                self.dropped_transitions += 1

    def to_dict(self) -> dict[str, float | int]:
        transitions = max(self.transitions, 1)
        return {
            "transitions": self.transitions,
            "terminal_transitions": self.terminal_transitions,
            "delivered_transitions": self.delivered_transitions,
            "dropped_transitions": self.dropped_transitions,
            "truncated_transitions": self.truncated_transitions,
            "reward_sum": self.reward_sum,
            "reward_mean": self.reward_sum / transitions,
            "cost_sum": self.cost_sum,
            "cost_mean": self.cost_sum / transitions,
            "progress_reward_mean": self.progress_reward_sum / transitions,
            "delay_penalty_mean": self.delay_penalty_sum / transitions,
            "queue_penalty_mean": self.queue_penalty_sum / transitions,
            "terminal_reward_mean": self.terminal_reward_sum / transitions,
        }


def compute_transition_reward(
    config: RewardConfig,
    previous_remaining_distance: float,
    next_remaining_distance: float,
    initial_distance: float,
    delta_delay: float,
    delta_queue_cost: float,
    terminal_status: int | None = None,
    truncated: bool = False,
) -> TransitionReward:
    raw_initial = max(float(initial_distance), 1e-6)
    previous_distance = _finite_or_default(previous_remaining_distance, raw_initial)
    next_distance = _finite_or_default(next_remaining_distance, previous_distance)
    initial = max(raw_initial, abs(previous_distance), abs(next_distance), 1e-6)
    progress = (previous_distance - next_distance) / initial
    progress_reward = config.progress_weight * progress

    delay_penalty = config.delay_weight * max(float(delta_delay), 0.0) / max(config.delay_norm, 1e-6)
    queue_penalty = config.queue_cost_weight * max(float(delta_queue_cost), 0.0) / max(config.cost_norm, 1e-6)
    terminal_reward = 0.0
    if terminal_status == FLOWLET_DELIVERED:
        terminal_reward += config.delivered_bonus
    elif terminal_status == FLOWLET_DROPPED:
        terminal_reward -= config.dropped_penalty
    if truncated:
        terminal_reward -= config.truncated_penalty

    cost = max(float(delta_queue_cost), 0.0) / max(config.cost_norm, 1e-6)
    reward = progress_reward + terminal_reward - delay_penalty - queue_penalty
    return TransitionReward(
        reward=float(reward),
        cost=float(cost),
        progress_reward=float(progress_reward),
        delay_penalty=float(delay_penalty),
        queue_penalty=float(queue_penalty),
        terminal_reward=float(terminal_reward),
    )


def _finite_or_default(value: float, default: float) -> float:
    value = float(value)
    if np.isfinite(value):
        return value
    return float(default)


def _get(source: Any, key: str, default: Any) -> Any:
    if hasattr(source, "get"):
        return source.get(key, default)
    return default
