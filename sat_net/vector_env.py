from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sat_net.agent.base_agent import ACTION_COUNT, RoutingBatch, RoutingDecision
from sat_net.routing_env import RoutingEnv
from sat_net.stats import ContinuingMetricsAccumulator, Metrics
from sat_net.util import NamedDict


DAY_MS = 24 * 60 * 60 * 1000


def seeded_start_offsets_ms(seeds: list[int] | np.ndarray, span_ms: float = DAY_MS) -> np.ndarray:
    span = max(int(round(float(span_ms))), 1)
    offsets = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) % (2**32))
        offsets.append(int(rng.integers(0, span)))
    return np.asarray(offsets, dtype=np.float64)


@dataclass(slots=True)
class VectorFlowletView:
    """Flowlet states for a flattened vector environment."""

    env_flowlets: list
    current_times: np.ndarray


class VectorRoutingEnv:
    """Flattened vector wrapper over multiple routing environments.

    This is the first vectorization layer: each sub-environment starts at a
    different UTC offset, and their decision batches are concatenated for one
    shared MARL policy. Flowlet ids remain local to each env and are disambiguated
    by RoutingBatch.env_ids.
    """

    def __init__(
        self,
        config: NamedDict,
        num_envs: int,
        utc_offset_span_ms: float = 5_400_000.0,
        seed_stride: int = 100_000,
        tf_writer: Any | None = None,
    ):
        self.config = config
        self.num_envs = max(int(num_envs), 1)
        self.utc_offset_span_ms = float(utc_offset_span_ms)
        self.seed_stride = int(seed_stride)
        self.tf_writer = tf_writer

        self.start_time_offsets_ms = self._build_start_offsets()
        self.envs = [
            RoutingEnv(self._env_config_for_index(env_id), tf_writer=tf_writer if env_id == 0 else None)
            for env_id in range(self.num_envs)
        ]
        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_dim
        self.slot_ms = self.envs[0].slot_ms
        self.start_time = 0.0
        self.current_time = 0.0
        self._include_spf_table = False
        self._pending_routing_batch = self._empty_routing_batch()
        self._schedule_duration_ms: float | None = None
        self._active = np.ones(self.num_envs, dtype=bool)

    def set_duration_seconds(self, seconds: float) -> None:
        for env in self.envs:
            env.set_duration_seconds(seconds)

    def clear_duration_limit(self) -> None:
        for env in self.envs:
            env.clear_duration_limit()

    def reset(self, seed=None, start_time=None, options: dict | None = None, include_spf_table: bool | None = None):
        options = {} if options is None else dict(options)
        if include_spf_table is None:
            include_spf_table = bool(options.get("include_spf_table", False))
        self._include_spf_table = bool(include_spf_table)
        env_seeds = options.pop("env_seeds", None)
        if env_seeds is None and isinstance(seed, (list, tuple, np.ndarray)):
            env_seeds = seed
        if env_seeds is not None and len(env_seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} env seeds, got {len(env_seeds)}.")
        env_start_offsets_ms = options.pop("env_start_offsets_ms", None)
        if env_start_offsets_ms is not None:
            env_start_offsets_ms = np.asarray(env_start_offsets_ms, dtype=np.float64)
            if len(env_start_offsets_ms) != self.num_envs:
                raise ValueError(
                    f"Expected {self.num_envs} env start offsets, got {len(env_start_offsets_ms)}."
                )
        else:
            env_start_offsets_ms = self.start_time_offsets_ms

        base_start_time = 0.0 if start_time is None else float(start_time)
        traffic_until_time_ms = options.get("traffic_until_time_ms", None)
        self._schedule_duration_ms = (
            None if traffic_until_time_ms is None else max(float(traffic_until_time_ms) - base_start_time, 0.0)
        )
        self.start_time = base_start_time
        self._active.fill(True)

        observations = []
        for env_id, env in enumerate(self.envs):
            local_start = base_start_time + float(env_start_offsets_ms[env_id])
            local_options = dict(options)
            if self._schedule_duration_ms is not None:
                local_options["traffic_until_time_ms"] = local_start + self._schedule_duration_ms
            if env_seeds is not None:
                local_seed = int(env_seeds[env_id])
            else:
                local_seed = None if seed is None else int(seed) + env_id * self.seed_stride
            observation, _info = env.reset(
                seed=local_seed,
                start_time=local_start,
                options=local_options,
                include_spf_table=include_spf_table,
            )
            observations.append(observation)

        self._pending_routing_batch = self._concat_batches(observations)
        self._refresh_vector_clock()
        return self._pending_routing_batch, self._build_step_info()

    def step(self, action=None):
        if not self._active.any():
            return (
                self._pending_routing_batch,
                np.empty(0, dtype=np.float32),
                True,
                False,
                self._build_step_info(),
            )

        batch = self.observation
        actions = self._normalize_action(action, batch.decision_count)
        observations = []
        rewards = []
        truncated = []

        for env_id, env in enumerate(self.envs):
            if not self._active[env_id]:
                observations.append(env.observation)
                rewards.append(np.empty(0, dtype=np.float32))
                truncated.append(False)
                continue

            rows = np.flatnonzero(batch.row_env_ids == env_id)
            env_action = RoutingDecision(next_hop_sat_ids=actions[rows])
            observation, reward, _done, cut, _info = env.step(env_action)
            observations.append(observation)
            rewards.append(reward)
            truncated.append(bool(cut))

            if self._schedule_duration_ms is not None:
                elapsed_ms = max(float(env.current_time - env.start_time), 0.0)
                if elapsed_ms >= self._schedule_duration_ms and observation.decision_count == 0:
                    self._active[env_id] = False

        self._pending_routing_batch = self._concat_batches(observations)
        self._refresh_vector_clock()
        reward = np.concatenate(rewards) if rewards else np.empty(0, dtype=np.float32)
        return self._pending_routing_batch, reward, not self._active.any(), any(truncated), self._build_step_info()

    @property
    def observation(self) -> RoutingBatch:
        return self._pending_routing_batch

    @property
    def flowlets(self) -> VectorFlowletView:
        return VectorFlowletView(
            env_flowlets=[env.flowlets for env in self.envs],
            current_times=np.array([env.current_time for env in self.envs], dtype=np.float64),
        )

    @property
    def terminated(self) -> bool:
        return all(env.terminated for env in self.envs)

    def calc_metrics(self) -> Metrics:
        accumulator = ContinuingMetricsAccumulator()
        for env in self.envs:
            duration_ms = max(float(env.current_time - env.start_time), 0.0)
            accumulator.add(env.calc_metrics(), duration_ms)
        return accumulator.to_metrics()

    def get_flowlet_dataframe(self) -> pd.DataFrame:
        frames = []
        for env_id, env in enumerate(self.envs):
            frame = env.get_flowlet_dataframe()
            if frame.empty:
                continue
            frame.insert(0, "env_id", env_id)
            frame.insert(1, "env_start_time", env.start_time)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def save_flowlets_to_csv(self, file_path: str) -> None:
        self.get_flowlet_dataframe().to_csv(file_path, index=False)

    def _build_start_offsets(self) -> np.ndarray:
        if self.num_envs <= 1:
            return np.zeros(1, dtype=np.float64)
        return np.linspace(0.0, self.utc_offset_span_ms, self.num_envs, endpoint=False, dtype=np.float64)

    def _env_config_for_index(self, env_id: int) -> NamedDict:
        config = NamedDict(self.config.to_dict())
        if env_id > 0:
            config.verbose = False
        return config

    def _refresh_vector_clock(self) -> None:
        elapsed = [max(float(env.current_time - env.start_time), 0.0) for env in self.envs]
        self.current_time = self.start_time + (min(elapsed) if elapsed else 0.0)

    def _build_step_info(self) -> dict:
        batch = self.observation
        elapsed = np.array([max(float(env.current_time - env.start_time), 0.0) for env in self.envs], dtype=np.float64)
        return {
            "time_ms": self.current_time,
            "step": int(min(getattr(env, "_step_index", 0) for env in self.envs)),
            "num_envs": self.num_envs,
            "env_time_ms_min": float(elapsed.min()) if len(elapsed) else 0.0,
            "env_time_ms_max": float(elapsed.max()) if len(elapsed) else 0.0,
            "decision_count": batch.decision_count,
            "active_agent_count": len(np.unique(np.column_stack((batch.row_env_ids, batch.agent_ids)), axis=0))
            if batch.decision_count
            else 0,
            "route_count": batch.decision_count,
            "terminated": not self._active.any(),
        }

    def _normalize_action(self, action, expected_count: int) -> np.ndarray:
        if action is None:
            if expected_count == 0:
                return np.empty(0, dtype=np.int64)
            raise ValueError(f"Expected {expected_count} actions, got None.")
        if isinstance(action, RoutingDecision):
            action = action.next_hop_sat_ids
        actions = np.asarray(action, dtype=np.int64)
        if actions.ndim != 1:
            raise ValueError(f"Actions must be a 1-D array, got shape {actions.shape}.")
        if len(actions) != expected_count:
            raise ValueError(f"Expected {expected_count} actions, got {len(actions)}.")
        return actions

    def _concat_batches(self, batches: list[RoutingBatch]) -> RoutingBatch:
        nonempty = [(env_id, batch) for env_id, batch in enumerate(batches) if batch.decision_count > 0]
        if not nonempty:
            return self._empty_routing_batch()

        env_ids = np.concatenate(
            [np.full(batch.decision_count, env_id, dtype=np.int64) for env_id, batch in nonempty]
        )
        current_times = np.concatenate(
            [np.full(batch.decision_count, batch.current_time, dtype=np.float64) for _env_id, batch in nonempty]
        )

        def cat(name: str):
            return np.concatenate([getattr(batch, name) for _env_id, batch in nonempty])

        def cat_optional(name: str, dtype) -> np.ndarray | None:
            values = [getattr(batch, name) for _env_id, batch in nonempty]
            if any(value is None for value in values):
                return None
            return np.concatenate(values).astype(dtype, copy=False)

        region_tables = None
        region_versions = None
        if any(batch.region_next_hop_table is not None for batch in batches):
            region_tables = tuple(batch.region_next_hop_table for batch in batches)
            region_versions = np.array(
                [int(batch.region_next_hop_version) for batch in batches],
                dtype=np.int64,
            )

        return RoutingBatch(
            flowlet_ids=cat("flowlet_ids"),
            current_sat_ids=cat("current_sat_ids"),
            source_region_ids=cat("source_region_ids"),
            target_region_ids=cat("target_region_ids"),
            target_access_sat_ids=cat("target_access_sat_ids"),
            neighbor_sat_ids=cat("neighbor_sat_ids"),
            neighbor_link_ids=cat("neighbor_link_ids"),
            action_mask=cat("action_mask"),
            neighbor_queue_load=cat("neighbor_queue_load"),
            neighbor_link_capacity=cat("neighbor_link_capacity"),
            neighbor_link_delay=cat("neighbor_link_delay"),
            neighbor_link_free_time=cat("neighbor_link_free_time"),
            flowlet_size=cat("flowlet_size"),
            packet_count=cat("packet_count"),
            is_normal=cat("is_normal"),
            creation_time=cat("creation_time"),
            ttl=cat("ttl"),
            current_time=float(current_times.min()),
            region_next_hop_tables=region_tables,
            region_next_hop_versions=region_versions,
            observations=cat_optional("observations", np.float32),
            hops=cat_optional("hops", np.int16),
            queue_delay=cat_optional("queue_delay", np.float64),
            transmission_delay=cat_optional("transmission_delay", np.float64),
            propagation_delay=cat_optional("propagation_delay", np.float64),
            total_queue_cost=cat_optional("total_queue_cost", np.float64),
            remaining_gcd=cat_optional("remaining_gcd", np.float64),
            shortest_gcd=cat_optional("shortest_gcd", np.float64),
            initial_gcd=cat_optional("initial_gcd", np.float64),
            last_action1=cat_optional("last_action1", np.int64),
            last_action2=cat_optional("last_action2", np.int64),
            last_node1=cat_optional("last_node1", np.int64),
            last_node2=cat_optional("last_node2", np.int64),
            env_ids=env_ids,
            current_times=current_times,
        )

    def _empty_routing_batch(self) -> RoutingBatch:
        neighbor_i = np.empty((0, ACTION_COUNT), dtype=np.int64)
        neighbor_f = np.empty((0, ACTION_COUNT), dtype=np.float64)
        region_tables = None
        region_versions = None
        if getattr(self, "_include_spf_table", False):
            region_tables = tuple(env.observation.region_next_hop_table for env in self.envs)
            region_versions = np.array(
                [int(env.observation.region_next_hop_version) for env in self.envs],
                dtype=np.int64,
            )
        return RoutingBatch(
            flowlet_ids=np.empty(0, dtype=np.int64),
            current_sat_ids=np.empty(0, dtype=np.int64),
            source_region_ids=np.empty(0, dtype=np.int64),
            target_region_ids=np.empty(0, dtype=np.int64),
            target_access_sat_ids=np.empty(0, dtype=np.int64),
            neighbor_sat_ids=neighbor_i,
            neighbor_link_ids=neighbor_i.astype(np.int32, copy=True),
            action_mask=np.empty((0, ACTION_COUNT), dtype=bool),
            neighbor_queue_load=neighbor_f,
            neighbor_link_capacity=neighbor_f.copy(),
            neighbor_link_delay=neighbor_f.copy(),
            neighbor_link_free_time=neighbor_f.copy(),
            flowlet_size=np.empty(0, dtype=np.float64),
            packet_count=np.empty(0, dtype=np.int64),
            is_normal=np.empty(0, dtype=bool),
            creation_time=np.empty(0, dtype=np.float64),
            ttl=np.empty(0, dtype=np.int16),
            current_time=self.current_time,
            region_next_hop_tables=region_tables,
            region_next_hop_versions=region_versions,
            observations=np.empty((0, self.obs_dim), dtype=np.float32),
            hops=np.empty(0, dtype=np.int16),
            queue_delay=np.empty(0, dtype=np.float64),
            transmission_delay=np.empty(0, dtype=np.float64),
            propagation_delay=np.empty(0, dtype=np.float64),
            total_queue_cost=np.empty(0, dtype=np.float64),
            remaining_gcd=np.empty(0, dtype=np.float64),
            shortest_gcd=np.empty(0, dtype=np.float64),
            initial_gcd=np.empty(0, dtype=np.float64),
            last_action1=np.empty(0, dtype=np.int64),
            last_action2=np.empty(0, dtype=np.int64),
            last_node1=np.empty(0, dtype=np.int64),
            last_node2=np.empty(0, dtype=np.int64),
            env_ids=np.empty(0, dtype=np.int64),
            current_times=np.empty(0, dtype=np.float64),
        )
