from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sat_net.agent.base_agent import BaseAgent, RoutingBatch
from sat_net.routing_env import RoutingEnv
from sat_net.stats import Metrics


@dataclass(slots=True)
class EpisodeResult:
    """Result of one MARL episode."""

    metrics: Metrics
    info: dict
    agent_stats: dict
    step_stats: dict
    elapsed_seconds: float
    seed: int | None
    train: bool


@dataclass(slots=True)
class EvaluationResult:
    """Aggregated metrics over multiple evaluation episodes."""

    episodes: list[EpisodeResult]
    throughput_mean: float
    throughput_std: float
    drop_rate_mean: float
    drop_rate_std: float
    e2e_delay_mean: float
    e2e_delay_std: float
    cost_mean: float


def run_marl_episode(
    env: RoutingEnv,
    agent: BaseAgent,
    seed: int | None = None,
    start_time: float | None = None,
    train: bool = False,
) -> EpisodeResult:
    """Run one MARL episode: reset env, collect batched agent actions, step env, update agent."""

    if train:
        agent.set_train()
    else:
        agent.set_eval()

    wall_start = time.time()
    agent.on_episode_start()
    observation, info = env.reset(
        seed=seed,
        start_time=start_time,
        options={"include_spf_table": agent.requires_shortest_path_table},
    )

    terminated = False
    truncated = False
    step_count = 0
    decision_count = 0
    decision_batches = 0
    active_agent_sum = 0
    max_active_agents = 0
    while not (terminated or truncated):
        if not _is_empty_observation(observation):
            decision_count += observation.decision_count
            decision_batches += 1
            active_agents = len(observation.active_agent_ids)
            active_agent_sum += active_agents
            max_active_agents = max(max_active_agents, active_agents)
        action = _empty_action() if _is_empty_observation(observation) else agent.act(observation)
        observation, _reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if train and env.flowlets is not None:
            agent.observe_flowlet_outcomes(env.flowlets, env.current_time)
            agent.on_train_signal()

    if env.flowlets is not None:
        agent.on_episode_end(env.flowlets, env.current_time)
    if train:
        agent.on_train_signal()

    return EpisodeResult(
        metrics=env.calc_metrics(),
        info=info,
        agent_stats=agent.get_train_stats(),
        step_stats={
            "steps": step_count,
            "decision_batches": decision_batches,
            "decisions": decision_count,
            "active_agents_mean": active_agent_sum / max(decision_batches, 1),
            "active_agents_max": max_active_agents,
        },
        elapsed_seconds=time.time() - wall_start,
        seed=seed,
        train=train,
    )


def evaluate_agent(
    env: RoutingEnv,
    agent: BaseAgent,
    seeds: Iterable[int],
    start_time: float | None = 0,
) -> EvaluationResult:
    """Evaluate one agent over multiple deterministic seeds without mutating training buffers."""

    episodes = [
        run_marl_episode(env=env, agent=agent, seed=seed, start_time=start_time, train=False)
        for seed in seeds
    ]
    throughputs = np.array([episode.metrics.throughput for episode in episodes], dtype=np.float64)
    drop_rates = np.array([episode.metrics.drop_rate for episode in episodes], dtype=np.float64)
    e2e_delays = np.array([episode.metrics.e2e_delay_mean for episode in episodes], dtype=np.float64)
    costs = np.array([episode.metrics.queue_delay_mean for episode in episodes], dtype=np.float64)
    return EvaluationResult(
        episodes=episodes,
        throughput_mean=float(throughputs.mean()) if len(throughputs) else 0.0,
        throughput_std=float(throughputs.std()) if len(throughputs) else 0.0,
        drop_rate_mean=float(drop_rates.mean()) if len(drop_rates) else 0.0,
        drop_rate_std=float(drop_rates.std()) if len(drop_rates) else 0.0,
        e2e_delay_mean=float(e2e_delays.mean()) if len(e2e_delays) else 0.0,
        e2e_delay_std=float(e2e_delays.std()) if len(e2e_delays) else 0.0,
        cost_mean=float(costs.mean()) if len(costs) else 0.0,
    )


def _empty_action() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _is_empty_observation(observation: RoutingBatch | None) -> bool:
    return observation is None or observation.decision_count == 0
