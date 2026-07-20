from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from sat_net.agent.base_agent import BaseAgent, RoutingBatch
from sat_net.routing_env import RoutingEnv
from sat_net.stats import Metrics


@dataclass(slots=True)
class RolloutResult:
    """Result of one finite rollout window from the continuing routing process."""

    metrics: Metrics
    info: dict
    agent_stats: dict
    step_stats: dict
    elapsed_seconds: float
    seed: int | Iterable[int] | None
    train: bool


@dataclass(slots=True)
class EvaluationResult:
    """Aggregated metrics over multiple evaluation rollouts."""

    rollouts: list[RolloutResult]
    throughput_mean: float
    throughput_std: float
    drop_rate_mean: float
    drop_rate_std: float
    e2e_delay_mean: float
    e2e_delay_std: float
    cost_mean: float
    delivery_rate_mean: float = 0.0
    delivery_rate_std: float = 0.0
    pending_rate_mean: float = 0.0
    pending_rate_std: float = 0.0


def run_marl_rollout(
    env: RoutingEnv,
    agent: BaseAgent,
    seed: int | Iterable[int] | None = None,
    start_time: float | None = None,
    train: bool = False,
    max_steps: int | None = None,
    duration_seconds: float | None = None,
    reset_options: dict | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> RolloutResult:
    """Run one rollout window from a continuing MARL routing process."""

    num_envs = _num_envs(env)
    if train:
        agent.set_train()
    else:
        agent.set_eval()

    if duration_seconds is not None:
        env.set_duration_seconds(duration_seconds)
    traffic_until_time_ms = None
    if duration_seconds is not None:
        start_time_ms = 0.0 if start_time is None else float(start_time)
        traffic_until_time_ms = start_time_ms + float(duration_seconds) * 1000.0

    wall_start = time.time()
    if train:
        agent.on_rollout_start()
    reset_options = {} if reset_options is None else dict(reset_options)
    reset_options.update(
        {
            "include_spf_table": agent.requires_shortest_path_table,
            "traffic_until_time_ms": traffic_until_time_ms,
        }
    )
    observation, info = env.reset(
        seed=seed,
        start_time=start_time,
        options=reset_options,
    )
    window_start_time_ms = float(env.start_time)

    terminated = False
    truncated = False
    step_count = 0
    decision_count = 0
    decision_batches = 0
    active_agent_sum = 0
    max_active_agents = 0
    while not (terminated or truncated) and (
        max_steps is None or _env_interaction_steps(step_count, num_envs) < max_steps
    ):
        if not _is_empty_observation(observation):
            decision_count += observation.decision_count
            decision_batches += 1
            active_agents = len(observation.active_agent_ids)
            active_agent_sum += active_agents
            max_active_agents = max(max_active_agents, active_agents)
        action = _empty_action() if _is_empty_observation(observation) else agent.act(observation)
        observation, _reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if progress_callback is not None:
            progress_callback(step_count)
        if train and env.flowlets is not None:
            agent.observe_flowlet_outcomes(env.flowlets, env.current_time)
            agent.on_train_signal(steps=num_envs)

    total_env_steps = _env_interaction_steps(step_count, num_envs)
    max_steps_reached = max_steps is not None and total_env_steps >= max_steps and not (terminated or truncated)
    if train and env.flowlets is not None:
        agent.on_rollout_end(env.flowlets, env.current_time)
    if train:
        agent.on_train_signal(force=True)

    elapsed_seconds = time.time() - wall_start
    duration_seconds = max(float(env.current_time) - window_start_time_ms, 0.0) / 1000.0
    aggregate_duration_seconds = duration_seconds * num_envs
    return RolloutResult(
        metrics=env.calc_metrics(),
        info=info,
        agent_stats=agent.get_train_stats() if train else {},
        step_stats={
            "steps": total_env_steps,
            "env_interaction_steps": total_env_steps,
            "vector_steps": step_count,
            "num_envs": num_envs,
            "start_time_ms": window_start_time_ms,
            "end_time_ms": float(env.current_time),
            "duration_ms": duration_seconds * 1000.0,
            "aggregate_duration_ms": aggregate_duration_seconds * 1000.0,
            "sim_speed": aggregate_duration_seconds / max(elapsed_seconds, 1e-12),
            "decision_batches": decision_batches,
            "decisions": decision_count,
            "active_agents_mean": active_agent_sum / max(decision_batches, 1),
            "active_agents_max": max_active_agents,
            "max_steps_reached": max_steps_reached,
        },
        elapsed_seconds=elapsed_seconds,
        seed=seed,
        train=train,
    )


def run_marl_steps(
    env: RoutingEnv,
    agent: BaseAgent,
    train: bool = False,
    max_steps: int | None = None,
    until_time_ms: float | None = None,
    finalize: bool = False,
    progress_callback: Callable[[int], None] | None = None,
) -> RolloutResult:
    """Advance an already-reset continuing environment without resetting it."""

    num_envs = _num_envs(env)
    if train:
        agent.set_train()
    else:
        agent.set_eval()

    wall_start = time.time()
    window_start_time_ms = float(env.current_time)

    terminated = False
    truncated = False
    info = {}
    step_count = 0
    decision_count = 0
    decision_batches = 0
    active_agent_sum = 0
    max_active_agents = 0
    while not (terminated or truncated):
        if max_steps is not None and _env_interaction_steps(step_count, num_envs) >= max_steps:
            break
        if until_time_ms is not None and float(env.current_time) >= float(until_time_ms):
            break

        observation = env.observation
        if not _is_empty_observation(observation):
            decision_count += observation.decision_count
            decision_batches += 1
            active_agents = len(observation.active_agent_ids)
            active_agent_sum += active_agents
            max_active_agents = max(max_active_agents, active_agents)
        action = _empty_action() if _is_empty_observation(observation) else agent.act(observation)
        _observation, _reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if progress_callback is not None:
            progress_callback(step_count)
        if train and env.flowlets is not None:
            agent.observe_flowlet_outcomes(env.flowlets, env.current_time)
            agent.on_train_signal(steps=num_envs)

    total_env_steps = _env_interaction_steps(step_count, num_envs)
    if train and finalize and env.flowlets is not None:
        agent.on_rollout_end(env.flowlets, env.current_time)
    if train:
        agent.on_train_signal(force=True)

    elapsed_seconds = time.time() - wall_start
    duration_seconds = max(float(env.current_time) - window_start_time_ms, 0.0) / 1000.0
    aggregate_duration_seconds = duration_seconds * num_envs
    return RolloutResult(
        metrics=env.calc_metrics(),
        info=info or getattr(env, "_build_step_info")(),
        agent_stats=agent.get_train_stats() if train else {},
        step_stats={
            "steps": total_env_steps,
            "env_interaction_steps": total_env_steps,
            "vector_steps": step_count,
            "num_envs": num_envs,
            "start_time_ms": window_start_time_ms,
            "end_time_ms": float(env.current_time),
            "duration_ms": duration_seconds * 1000.0,
            "aggregate_duration_ms": aggregate_duration_seconds * 1000.0,
            "sim_speed": aggregate_duration_seconds / max(elapsed_seconds, 1e-12),
            "decision_batches": decision_batches,
            "decisions": decision_count,
            "active_agents_mean": active_agent_sum / max(decision_batches, 1),
            "active_agents_max": max_active_agents,
            "max_steps_reached": max_steps is not None and total_env_steps >= max_steps,
        },
        elapsed_seconds=elapsed_seconds,
        seed=None,
        train=train,
    )


def evaluate_agent(
    env: RoutingEnv,
    agent: BaseAgent,
    seeds: Iterable[int],
    start_time: float | None = 0,
    duration_seconds: float | None = None,
) -> EvaluationResult:
    """Evaluate one agent over multiple deterministic rollout windows without mutating training buffers."""

    rollouts = [
        run_marl_rollout(
            env=env,
            agent=agent,
            seed=seed,
            start_time=start_time,
            train=False,
            duration_seconds=duration_seconds,
        )
        for seed in seeds
    ]
    throughputs = np.array([rollout.metrics.throughput for rollout in rollouts], dtype=np.float64)
    delivery_rates = np.array([rollout.metrics.delivery_rate for rollout in rollouts], dtype=np.float64)
    pending_rates = np.array([rollout.metrics.pending_rate for rollout in rollouts], dtype=np.float64)
    drop_rates = np.array([rollout.metrics.drop_rate for rollout in rollouts], dtype=np.float64)
    e2e_delays = np.array([rollout.metrics.e2e_delay_mean for rollout in rollouts], dtype=np.float64)
    costs = np.array([rollout.metrics.cost_mean for rollout in rollouts], dtype=np.float64)
    return EvaluationResult(
        rollouts=rollouts,
        throughput_mean=float(throughputs.mean()) if len(throughputs) else 0.0,
        throughput_std=float(throughputs.std()) if len(throughputs) else 0.0,
        drop_rate_mean=float(drop_rates.mean()) if len(drop_rates) else 0.0,
        drop_rate_std=float(drop_rates.std()) if len(drop_rates) else 0.0,
        e2e_delay_mean=float(e2e_delays.mean()) if len(e2e_delays) else 0.0,
        e2e_delay_std=float(e2e_delays.std()) if len(e2e_delays) else 0.0,
        cost_mean=float(costs.mean()) if len(costs) else 0.0,
        delivery_rate_mean=float(delivery_rates.mean()) if len(delivery_rates) else 0.0,
        delivery_rate_std=float(delivery_rates.std()) if len(delivery_rates) else 0.0,
        pending_rate_mean=float(pending_rates.mean()) if len(pending_rates) else 0.0,
        pending_rate_std=float(pending_rates.std()) if len(pending_rates) else 0.0,
    )


def _empty_action() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _is_empty_observation(observation: RoutingBatch | None) -> bool:
    return observation is None or observation.decision_count == 0


def _num_envs(env: RoutingEnv) -> int:
    return max(int(getattr(env, "num_envs", 1)), 1)


def _env_interaction_steps(wrapper_steps: int, num_envs: int) -> int:
    return int(wrapper_steps) * max(int(num_envs), 1)
