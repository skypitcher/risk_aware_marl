from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sat_net.sim_kernel import FLOWLET_DELIVERED, FLOWLET_DROPPED


@dataclass(slots=True)
class RewardConfig:
    """Legacy routing reward scale used by the original packet-level agents."""

    delay_norm: float = 100.0
    cost_limit: float = 10.0

    @classmethod
    def from_config(cls, config: Any) -> "RewardConfig":
        reward = _get(config, "reward", None)
        source = reward if reward is not None else config

        def value(name: str, default: float) -> float:
            if hasattr(source, "get"):
                return float(_get(source, name, default))
            return float(default)

        delay_norm = value("delay_norm", float(_get(config, "delay_norm", 100.0)))
        return cls(
            delay_norm=delay_norm,
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
    previous_best_distance: float,
    current_distance: float,
    initial_distance: float,
    delta_delay: float,
    delta_queue_cost: float,
    flowlet_size: float,
    ttl_remaining: float,
    terminal_status: int | None = None,
    truncated: bool = False,
    queue_delay_in_reward: bool = False,
) -> TransitionReward:
    initial = max(float(initial_distance), 1e-6)
    previous_best = _finite_or_default(previous_best_distance, initial)
    current = _finite_or_default(current_distance, previous_best)
    progress = max(0.0, previous_best - current) / initial
    progress_reward = progress

    delta_delay = max(float(delta_delay), 0.0)
    delta_queue_cost = max(float(delta_queue_cost), 0.0)
    delay_ms_for_reward = delta_delay if queue_delay_in_reward else max(delta_delay - delta_queue_cost, 0.0)
    delay_penalty = delay_ms_for_reward / max(config.delay_norm, 1e-6)
    queue_penalty = 0.0

    reached_goal = 0.0
    if terminal_status == FLOWLET_DELIVERED:
        reached_goal = 1.0
    elif terminal_status == FLOWLET_DROPPED:
        reached_goal = -1.0

    terminal_reward = reached_goal * (1.0 + float(flowlet_size))
    reward = progress_reward + terminal_reward - delay_penalty
    if terminal_status == FLOWLET_DROPPED:
        current_progress = current / initial
        reward = -current_progress - float(ttl_remaining) * 5.0 / max(config.delay_norm, 1e-6)
        terminal_reward = reward

    cost = delta_queue_cost / max(config.delay_norm, 1e-6)
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
