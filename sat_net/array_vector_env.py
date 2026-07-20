from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from sat_net.agent.base_agent import ACTION_COUNT, RoutingBatch, RoutingDecision
from sat_net.config import PROJECT_ROOT
from sat_net.flowlet_status import (
    FLOWLET_AT_NODE,
    FLOWLET_DELIVERED,
    FLOWLET_DROPPED,
    FLOWLET_NOT_STARTED,
    FLOWLET_ON_LINK,
)
from sat_net.geometric import EARTH_R_KM, GM_EARTH, LIGHT_SPEED_MS
from sat_net.network import SatelliteNetwork
from sat_net.numba_kernels import (
    apply_routing_actions_kernel,
    build_action_mask_kernel,
    build_legacy_observations_kernel,
    deliver_visible_kernel,
    drop_disconnected_kernel,
    handle_arrivals_kernel,
    refresh_link_state_kernel,
    refresh_node_queue_features_kernel,
    release_transmitted_kernel,
)
from sat_net.stats import Metrics
from sat_net.traffic_region import TrafficRegionModel
from sat_net.util import NamedDict, NetworkError


DAY_MS = 24 * 60 * 60 * 1000

NODE_POS_X = 0
NODE_POS_Y = 1
NODE_POS_Z = 2
NODE_QUEUE_LOAD = 3
NODE_QUEUE_REMAINING = 4
NODE_FEATURE_DIM = 5

LINK_CONNECTED = 0
LINK_DELAY = 1
LINK_QUEUE_LOAD = 2
LINK_QUEUE_REMAINING = 3
LINK_FREE_TIME_DELTA = 4
LINK_CAPACITY = 5
LINK_DATA_RATE = 6
LINK_FEATURE_DIM = 7


def seeded_start_offsets_ms(seeds: list[int] | np.ndarray, span_ms: float = DAY_MS) -> np.ndarray:
    span = max(int(round(float(span_ms))), 1)
    offsets = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) % (2**32))
        offsets.append(int(rng.integers(0, span)))
    return np.asarray(offsets, dtype=np.float64)


class ArrayVectorRoutingEnv:
    """Fixed-concurrency NumPy vector environment for MARL routing."""

    def __init__(
        self,
        config: NamedDict,
        num_envs: int,
        utc_offset_span_ms: float = 5_400_000.0,
        seed_stride: int = 100_000,
        tf_writer: Any | None = None,
        device: str | None = None,
    ):
        self.config = NamedDict(config.to_dict())
        self.config.verbose = bool(self.config.get("verbose", False))
        self.num_envs = max(int(num_envs), 1)
        self.utc_offset_span_ms = float(utc_offset_span_ms)
        self.seed_stride = int(seed_stride)
        self.tf_writer = tf_writer
        self.device = "cpu"
        self.dtype = np.float32

        self.network_config = self.config.network
        self.traffic_config = self.config.traffic
        self.network = self._create_network(self.network_config)
        self.traffic_model = TrafficRegionModel.from_config(self.traffic_config, PROJECT_ROOT)

        self.slot_ms = float(self.traffic_config.get("slot_ms", 1.0))
        self.mean_packets_per_flowlet = float(self.traffic_config.get("mean_packets_per_flowlet", 16.0))
        self.access_data_rate = float(self.traffic_config.get("access_data_rate", 1.0))
        self.default_ttl = int(self.config.default_ttl)
        self.delay_norm = float(self.config.get("delay_norm", 100.0))
        self.update_interval_ms = float(self.config.update_interval_ms)
        self.topology_update_slots = max(int(round(self.update_interval_ms / max(self.slot_ms, 1e-9))), 1)
        self.prob_normal_packet = float(self.config.prob_normal_packet)
        self.normal_packet_size = float(self.config.normal_packet_size)
        self.small_packet_size = float(self.config.small_packet_size)
        self.concurrent_flowlets_per_env = int(self.traffic_config.get("concurrent_flowlets_per_env", 8192))
        if self.concurrent_flowlets_per_env <= 0:
            raise ValueError(
                "traffic.concurrent_flowlets_per_env must be positive, "
                f"got {self.concurrent_flowlets_per_env}."
            )
        self.capacity = self.concurrent_flowlets_per_env

        self.num_satellites = int(self.network.num_satellites)
        self.num_nodes = int(self.network.num_nodes)
        self.num_links = int(self.network.num_links)
        self.num_regions = len(self.traffic_model.regions)
        self.obs_dim = 94
        self.action_dim = ACTION_COUNT
        self.time_limit = np.inf
        self.verbose = bool(self.config.get("verbose", False))

        self.start_time_offsets_ms = self._build_start_offsets()
        self._include_spf_table = False
        self._schedule_duration_ms: float | None = None
        self._active = np.ones(self.num_envs, dtype=bool)
        self.storage_overflow_flowlets = 0
        self.generated_flowlets_total = 0
        self.start_time = 0.0
        self.current_time = 0.0
        self._step_index = 0
        self.topology_update_steps = 0
        self._region_next_hop_cache_key: int | None = None
        self._region_next_hop_tables_cache: np.ndarray | None = None
        self._region_next_hop_versions_cache: np.ndarray | None = None
        self._pending_routing_batch = self._empty_routing_batch()
        self._rng = np.random.default_rng(0)

        self._setup_static_arrays()
        self._allocate_state()

    @staticmethod
    def _create_network(network_config: NamedDict) -> SatelliteNetwork:
        return SatelliteNetwork(
            altitude=int(network_config.get("altitude", 550)),
            inclination=int(network_config.get("inclination", 53)),
            num_orbits=int(network_config.get("num_orbits", 24)),
            num_sats_per_orbit=int(network_config.get("num_sats_per_orbit", 24)),
            phasing=int(network_config.get("phasing", 3)),
            min_elevation_angle_deg=int(network_config.get("min_elevation_angle_deg", 15)),
            link_buffer_size=float(network_config.get("link_buffer_size", 10.0)),
            isl_data_rate=float(network_config.get("isl_data_rate", 1.0)),
        )

    def set_duration_seconds(self, seconds: float) -> None:
        self.time_limit = max(float(seconds), 0.0) * 1000.0

    def clear_duration_limit(self) -> None:
        self.time_limit = np.inf

    def reset(self, seed=None, start_time=None, options: dict | None = None, include_spf_table: bool | None = None):
        options = {} if options is None else dict(options)
        if include_spf_table is None:
            include_spf_table = bool(options.get("include_spf_table", False))
        self._include_spf_table = bool(include_spf_table)

        env_start_offsets_ms = options.pop("env_start_offsets_ms", None)
        if env_start_offsets_ms is None:
            env_start_offsets_ms = self.start_time_offsets_ms
        env_start_offsets_ms = np.asarray(env_start_offsets_ms, dtype=np.float64)
        if len(env_start_offsets_ms) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} start offsets, got {len(env_start_offsets_ms)}.")

        env_seeds = options.pop("env_seeds", None)
        if env_seeds is None and isinstance(seed, (list, tuple, np.ndarray)):
            env_seeds = seed
        if env_seeds is not None:
            seed_values = np.asarray(env_seeds, dtype=np.int64)
            base_seed = int(np.sum(seed_values * (np.arange(len(seed_values), dtype=np.int64) + 1)) % (2**63 - 1))
        else:
            base_seed = 0 if seed is None else int(seed)
        self._rng = np.random.default_rng(int(base_seed) % (2**32))

        base_start = 0.0 if start_time is None else float(start_time)
        self.start_times = (base_start + env_start_offsets_ms).astype(np.float64, copy=False)
        self.start_time = base_start
        self.current_times = self.start_times.copy()
        self.current_time = base_start
        self._schedule_duration_ms = None
        if options.get("traffic_until_time_ms", None) is not None:
            self._schedule_duration_ms = max(float(options["traffic_until_time_ms"]) - base_start, 0.0)
        self._active.fill(True)
        self._step_index = 0
        self.topology_update_steps = 0
        self.storage_overflow_flowlets = 0
        self.generated_flowlets_total = 0
        self._invalidate_region_next_hop_cache()
        self._reset_state_arrays()
        self._update_topology(force=True)
        self._fill_free_flowlet_slots()
        self._pending_routing_batch = self._build_routing_batch()
        self._refresh_flowlet_counts()
        return self._pending_routing_batch, self._build_step_info(sync=True)

    def step(self, action=None):
        self._recycle_terminal_slots()
        batch = self.observation
        actions = self._normalize_action(action, batch.batch_size)
        if batch.batch_size:
            self._apply_routing_actions(batch, actions)

        self._step_index += 1
        self.current_times = self.start_times + float(self._step_index) * self.slot_ms
        self.current_time = float(self.start_time + self._step_index * self.slot_ms)
        if self._step_index % self.topology_update_slots == 0:
            self._update_topology(force=True)
            self.topology_update_steps += 1
            self._drop_flowlets_on_disconnected_links()

        self._release_transmitted_flowlets()
        self._handle_arrivals()
        if self._schedule_duration_ms is None or self._step_index * self.slot_ms < self._schedule_duration_ms:
            self._fill_free_flowlet_slots()
        else:
            self._active &= self._has_pending_flowlets_by_env()

        self._pending_routing_batch = self._build_routing_batch()
        self._refresh_flowlet_counts()
        terminated = False if self._schedule_duration_ms is None else bool(not self._active.any())
        truncated = bool(np.isfinite(self.time_limit) and self._step_index * self.slot_ms >= self.time_limit)
        reward = np.empty(batch.batch_size, dtype=np.float32)
        return self._pending_routing_batch, reward, terminated, truncated, self._build_step_info(sync=False)

    @property
    def observation(self) -> RoutingBatch:
        return self._pending_routing_batch

    @property
    def flowlets(self) -> "ArrayVectorRoutingEnv":
        return self

    @property
    def count(self) -> int:
        return self.capacity

    @property
    def nodes(self):
        return self.node_state

    @property
    def links(self):
        return self.link_state

    @property
    def terminated(self) -> bool:
        return bool(not self._active.any())

    def calc_metrics(self, env_id: int | None = None) -> Metrics:
        elapsed_ms = float(self._elapsed_ms().sum() if env_id is None else self._elapsed_ms()[int(env_id)])
        generated = self._counter_int("generated", env_id)
        generated_normal = self._counter_int("generated_normal", env_id)
        generated_small = self._counter_int("generated_small", env_id)
        delivered = self._counter_int("delivered", env_id)
        delivered_normal = self._counter_int("delivered_normal", env_id)
        delivered_small = self._counter_int("delivered_small", env_id)
        dropped = self._counter_int("dropped", env_id)
        dropped_normal = self._counter_int("dropped_normal", env_id)
        dropped_small = self._counter_int("dropped_small", env_id)
        ttl_dropped = self._counter_int("dropped_by_ttl", env_id)
        pending = max(generated - delivered - dropped, 0)
        pending_normal = max(generated_normal - delivered_normal - dropped_normal, 0)
        pending_small = max(generated_small - delivered_small - dropped_small, 0)
        total_delay = self._counter_float("e2e_delay_sum", env_id)
        queue_delay = self._counter_float("queue_delay_sum", env_id)
        transmission_delay = self._counter_float("transmission_delay_sum", env_id)
        propagation_delay = self._counter_float("propagation_delay_sum", env_id)
        normal_delay = self._counter_float("normal_e2e_delay_sum", env_id)
        normal_queue = self._counter_float("normal_queue_delay_sum", env_id)
        normal_tx = self._counter_float("normal_transmission_delay_sum", env_id)
        normal_prop = self._counter_float("normal_propagation_delay_sum", env_id)
        small_delay = self._counter_float("small_e2e_delay_sum", env_id)
        small_queue = self._counter_float("small_queue_delay_sum", env_id)
        small_tx = self._counter_float("small_transmission_delay_sum", env_id)
        small_prop = self._counter_float("small_propagation_delay_sum", env_id)
        cost = self._counter_float("cost_sum", env_id)
        normal_cost = self._counter_float("cost_normal_sum", env_id)
        small_cost = self._counter_float("cost_small_sum", env_id)
        elapsed_seconds = max(elapsed_ms / 1000.0, 1e-12)
        delivered_mbit = self._counter_float("delivered_mbit", env_id)
        return Metrics(
            generated=generated,
            generated_normal_packet=generated_normal,
            generated_small_packet=generated_small,
            delivered=delivered,
            delivered_normal_packet=delivered_normal,
            delivered_small_packet=delivered_small,
            dropped=dropped,
            dropped_by_ttl=ttl_dropped,
            dropped_normal_packet=dropped_normal,
            dropped_small_packet=dropped_small,
            pending=pending,
            pending_normal_packet=pending_normal,
            pending_small_packet=pending_small,
            throughput=delivered_mbit / elapsed_seconds,
            service_rate=delivered / elapsed_seconds,
            delivery_rate=delivered / generated if generated else 0.0,
            drop_rate=dropped / generated if generated else 0.0,
            pending_rate=pending / generated if generated else 0.0,
            normal_packet_delivery_rate=delivered_normal / generated_normal if generated_normal else 0.0,
            normal_packet_drop_rate=dropped_normal / generated_normal if generated_normal else 0.0,
            normal_packet_pending_rate=pending_normal / generated_normal if generated_normal else 0.0,
            small_packet_delivery_rate=delivered_small / generated_small if generated_small else 0.0,
            small_packet_drop_rate=dropped_small / generated_small if generated_small else 0.0,
            small_packet_pending_rate=pending_small / generated_small if generated_small else 0.0,
            e2e_delay_mean=total_delay / delivered if delivered else 0.0,
            queue_delay_mean=queue_delay / delivered if delivered else 0.0,
            transmission_delay_mean=transmission_delay / delivered if delivered else 0.0,
            propagation_delay_mean=propagation_delay / delivered if delivered else 0.0,
            normal_packet_e2e_delay_mean=normal_delay / delivered_normal if delivered_normal else 0.0,
            normal_packet_queue_delay_mean=normal_queue / delivered_normal if delivered_normal else 0.0,
            normal_packet_transmission_delay_mean=normal_tx / delivered_normal if delivered_normal else 0.0,
            normal_packet_propagation_delay_mean=normal_prop / delivered_normal if delivered_normal else 0.0,
            small_packet_e2e_delay_mean=small_delay / delivered_small if delivered_small else 0.0,
            small_packet_queue_delay_mean=small_queue / delivered_small if delivered_small else 0.0,
            small_packet_transmission_delay_mean=small_tx / delivered_small if delivered_small else 0.0,
            small_packet_propagation_delay_mean=small_prop / delivered_small if delivered_small else 0.0,
            cost_mean=cost / delivered if delivered else 0.0,
            cost_small_packet_mean=small_cost / delivered_small if delivered_small else 0.0,
            cost_normal_packet_mean=normal_cost / delivered_normal if delivered_normal else 0.0,
        )

    def get_flowlet_dataframe(self, env_id: int | None = None) -> pd.DataFrame:
        if env_id is None:
            frames = []
            for item in range(self.num_envs):
                frame = self.get_flowlet_dataframe(env_id=item)
                if not frame.empty:
                    frame.insert(0, "env_id", item)
                    frame.insert(1, "env_start_time", float(self.start_times[item]))
                    frames.append(frame)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        env_id = int(env_id)
        active = self.status[env_id] != FLOWLET_NOT_STARTED
        if not active.any():
            return pd.DataFrame()
        data = {
            "slot_id": np.flatnonzero(active),
            "source_id": self.source_id[env_id, active],
            "source_region_id": self.source_region_id[env_id, active],
            "target_region_id": self.target_region_id[env_id, active],
            "packet_count": self.packet_count[env_id, active],
            "packet_size": self.packet_size[env_id, active],
            "is_normal_packet": self.is_normal[env_id, active],
            "size": self.size[env_id, active],
            "creation_time": self.creation_time[env_id, active],
            "delivery_time": self.delivery_time[env_id, active],
            "drop_time": self.drop_time[env_id, active],
            "status": self.status[env_id, active],
            "drop_reason": self.drop_reason[env_id, active],
            "hops": self.hops[env_id, active],
            "queue_delay": self.queue_delay[env_id, active],
            "transmission_delay": self.transmission_delay[env_id, active],
            "propagation_delay": self.propagation_delay[env_id, active],
            "total_queue_cost": self.total_queue_cost[env_id, active],
            "first_access_delay": self.first_access_delay[env_id, active],
            "final_access_delay": self.final_access_delay[env_id, active],
        }
        frame = pd.DataFrame(data)
        frame["delivered"] = frame["status"] == FLOWLET_DELIVERED
        frame["dropped"] = frame["status"] == FLOWLET_DROPPED
        frame["total_delay"] = frame["queue_delay"] + frame["transmission_delay"] + frame["propagation_delay"]
        return frame

    def save_flowlets_to_csv(self, file_path: str) -> None:
        self.get_flowlet_dataframe().to_csv(file_path, index=False)

    def _setup_static_arrays(self) -> None:
        net = self.network
        self.orbit_radius = float(net.orbit_radius)
        self.orbit_cycle_ms = float(net.orbit_cycle * 1000.0)
        self.max_access_range = float(net.max_access_range)
        self.max_isl_range = float(net.max_isl_range)
        self.sat_altitudes = np.asarray(net._satellite_altitudes, dtype=np.float64)
        self.sat_raan_rad = np.asarray(net._satellite_raan_rad, dtype=np.float64)
        self.sat_inc_rad = np.asarray(net._satellite_inc_rad, dtype=np.float64)
        self.sat_true_anomaly = np.asarray(net._satellite_true_anomaly, dtype=np.float64)
        self.link_source_ids = np.asarray(net._link_source_ids, dtype=np.int64)
        self.link_sink_ids = np.asarray(net._link_sink_ids, dtype=np.int64)
        self.link_capacity = np.asarray(net._link_capacity_array, dtype=np.float32)
        self.link_data_rate = np.asarray(net._link_data_rate_array, dtype=np.float32)
        self.neighbor_sat_ids = np.asarray(net.neighbor_sat_ids, dtype=np.int64)
        self.neighbor_link_ids, self.direction_by_link_id = self._build_neighbor_link_arrays()
        valid_neighbor_links = self.neighbor_link_ids >= 0
        safe_neighbor_links = np.maximum(self.neighbor_link_ids, 0)
        self.node_total_link_capacity = np.where(
            valid_neighbor_links,
            self.link_capacity[safe_neighbor_links],
            0.0,
        ).sum(axis=1).astype(np.float32, copy=False)
        self.region_positions = np.asarray([region.position for region in self.traffic_model.regions], dtype=np.float32)
        self.region_latitudes = np.asarray([region.latitude for region in self.traffic_model.regions], dtype=np.float64)
        self.region_longitudes = np.asarray([region.longitude for region in self.traffic_model.regions], dtype=np.float64)
        self.region_distance_matrix = np.asarray(self.traffic_model.distance_matrix_km, dtype=np.float32)
        flat_od = np.asarray(self.traffic_model._base_flat_od, dtype=np.float64)
        flat_od = flat_od / max(float(flat_od.sum()), 1e-12)
        self.flat_od_probs = flat_od
        self.flat_od_cdf = np.cumsum(flat_od)
        self.flat_od_cdf[-1] = 1.0

    def _allocate_state(self) -> None:
        shape = (self.num_envs, self.capacity)
        link_shape = (self.num_envs, self.num_links)
        self.flat_env_ids = np.repeat(np.arange(self.num_envs, dtype=np.int64), self.capacity)
        self.flat_flowlet_ids = np.tile(np.arange(self.capacity, dtype=np.int64), self.num_envs)
        self.flowlet_counts = np.zeros(self.num_envs, dtype=np.int64)
        total_slots = self.num_envs * self.capacity
        self._batch_neighbor_sat_ids = np.full((total_slots, ACTION_COUNT), -1, dtype=np.int64)
        self._batch_neighbor_link_ids = np.full((total_slots, ACTION_COUNT), -1, dtype=np.int64)
        self._batch_action_mask = np.zeros((total_slots, ACTION_COUNT), dtype=bool)
        self._batch_at_node = np.zeros(total_slots, dtype=bool)
        self._batch_deliverable = np.zeros(total_slots, dtype=bool)
        self._batch_route_mask = np.zeros(total_slots, dtype=bool)
        self._batch_no_target_access = np.zeros(total_slots, dtype=bool)
        self._batch_decision_mask = np.zeros(total_slots, dtype=bool)
        self._batch_current_sats_safe = np.zeros(total_slots, dtype=np.int64)
        self._batch_target_regions_safe = np.zeros(total_slots, dtype=np.int64)
        self._batch_target_access_sats = np.full(total_slots, -1, dtype=np.int64)
        self._batch_remaining_gcd = np.full(total_slots, np.inf, dtype=np.float32)
        self._compact_rows = np.arange(total_slots, dtype=np.int64)
        self._compact_decision_mask = np.ones(total_slots, dtype=bool)
        self.start_times = np.zeros(self.num_envs, dtype=np.float64)
        self.current_times = np.zeros(self.num_envs, dtype=np.float64)
        self.status = np.full(shape, FLOWLET_NOT_STARTED, dtype=np.int8)
        self.creation_time = np.zeros(shape, dtype=np.float64)
        self.source_region_id = np.full(shape, -1, dtype=np.int64)
        self.target_region_id = np.full(shape, -1, dtype=np.int64)
        self.source_id = np.full(shape, -1, dtype=np.int64)
        self.current_sat = np.full(shape, -1, dtype=np.int64)
        self.next_sat = np.full(shape, -1, dtype=np.int64)
        self.link_id = np.full(shape, -1, dtype=np.int64)
        self.packet_count = np.ones(shape, dtype=np.int64)
        self.packet_size = np.zeros(shape, dtype=np.float32)
        self.is_normal = np.zeros(shape, dtype=bool)
        self.size = np.zeros(shape, dtype=np.float32)
        self.ttl = np.full(shape, self.default_ttl, dtype=np.int16)
        self.hops = np.zeros(shape, dtype=np.int16)
        self.queue_delay = np.zeros(shape, dtype=np.float32)
        self.transmission_delay = np.zeros(shape, dtype=np.float32)
        self.propagation_delay = np.zeros(shape, dtype=np.float32)
        self.total_queue_cost = np.zeros(shape, dtype=np.float32)
        self.first_access_delay = np.zeros(shape, dtype=np.float32)
        self.final_access_delay = np.zeros(shape, dtype=np.float32)
        self.delivery_time = np.full(shape, np.nan, dtype=np.float64)
        self.drop_time = np.full(shape, np.nan, dtype=np.float64)
        self.drop_reason = np.full(shape, -1, dtype=np.int16)
        self.transmit_end_time = np.full(shape, np.inf, dtype=np.float64)
        self.arrival_time = np.full(shape, np.inf, dtype=np.float64)
        self.link_released = np.ones(shape, dtype=bool)
        self.scheduled_prop_delay = np.zeros(shape, dtype=np.float32)
        self.remaining_gcd = np.full(shape, np.inf, dtype=np.float32)
        self.shortest_gcd = np.full(shape, np.inf, dtype=np.float32)
        self.initial_gcd = np.ones(shape, dtype=np.float32)
        self.last_action1 = np.full(shape, -1, dtype=np.int64)
        self.last_action2 = np.full(shape, -1, dtype=np.int64)
        self.last_node1 = np.full(shape, -1, dtype=np.int64)
        self.last_node2 = np.full(shape, -1, dtype=np.int64)

        self.queue_load = np.zeros(link_shape, dtype=np.float32)
        self.free_time = np.zeros(link_shape, dtype=np.float64)
        self._schedule_tx_delta = np.zeros(link_shape, dtype=np.float32)
        self._schedule_base_time = np.zeros(link_shape, dtype=np.float64)
        self._schedule_token = np.zeros(link_shape, dtype=np.int64)
        self._schedule_token_value = 0
        self.link_connected = np.ones(link_shape, dtype=bool)
        self.link_delay = np.zeros(link_shape, dtype=np.float32)
        self.nearest_region_sat_ids = np.full((self.num_envs, self.num_regions), -1, dtype=np.int64)
        self.nearest_region_sat_distances = np.full((self.num_envs, self.num_regions), np.inf, dtype=np.float32)
        self.sat_positions = np.zeros((self.num_envs, self.num_satellites, 3), dtype=np.float32)
        self.node_queue_load = np.zeros((self.num_envs, self.num_satellites), dtype=np.float32)
        self.node_queue_remaining = np.zeros((self.num_envs, self.num_satellites), dtype=np.float32)
        self.node_state = np.zeros((self.num_envs, self.num_satellites, NODE_FEATURE_DIM), dtype=np.float32)
        self.link_state = np.zeros((self.num_envs, self.num_links, LINK_FEATURE_DIM), dtype=np.float32)
        initial_obs_capacity = max(1, min(total_slots, 8192))
        self.observation_state = np.zeros((initial_obs_capacity, self.obs_dim), dtype=np.float32)
        self.region_sat_distances = np.full(
            (self.num_envs, self.num_regions, self.num_satellites),
            np.inf,
            dtype=np.float32,
        )
        self.region_sat_gcd_degrees = np.full(
            (self.num_envs, self.num_regions, self.num_satellites),
            np.inf,
            dtype=np.float32,
        )
        self._allocate_metric_counters()

    def _reset_state_arrays(self) -> None:
        self.flowlet_counts.fill(0)
        self.status.fill(FLOWLET_NOT_STARTED)
        self.creation_time.fill(0.0)
        self.source_region_id.fill(-1)
        self.target_region_id.fill(-1)
        self.source_id.fill(-1)
        self.current_sat.fill(-1)
        self.next_sat.fill(-1)
        self.link_id.fill(-1)
        self.packet_count.fill(1)
        self.packet_size.fill(0.0)
        self.is_normal.fill(False)
        self.size.fill(0.0)
        self.ttl.fill(self.default_ttl)
        self.hops.fill(0)
        self.queue_delay.fill(0.0)
        self.transmission_delay.fill(0.0)
        self.propagation_delay.fill(0.0)
        self.total_queue_cost.fill(0.0)
        self.first_access_delay.fill(0.0)
        self.final_access_delay.fill(0.0)
        self.delivery_time.fill(np.nan)
        self.drop_time.fill(np.nan)
        self.drop_reason.fill(-1)
        self.transmit_end_time.fill(np.inf)
        self.arrival_time.fill(np.inf)
        self.link_released.fill(True)
        self.scheduled_prop_delay.fill(0.0)
        self.remaining_gcd.fill(np.inf)
        self.shortest_gcd.fill(np.inf)
        self.initial_gcd.fill(1.0)
        self.last_action1.fill(-1)
        self.last_action2.fill(-1)
        self.last_node1.fill(-1)
        self.last_node2.fill(-1)
        self.queue_load.fill(0.0)
        self.free_time.fill(0.0)
        self._schedule_tx_delta.fill(0.0)
        self._schedule_base_time.fill(0.0)
        self._schedule_token.fill(0)
        self._schedule_token_value = 0
        self.node_queue_load.fill(0.0)
        self.node_queue_remaining.fill(0.0)
        self.node_state.fill(0.0)
        self.link_state.fill(0.0)
        self.observation_state.fill(0.0)
        self.region_sat_distances.fill(np.inf)
        self.region_sat_gcd_degrees.fill(np.inf)
        self._reset_metric_counters()
        self._refresh_flowlet_counts()

    def _allocate_metric_counters(self) -> None:
        int_names = (
            "generated",
            "generated_normal",
            "generated_small",
            "delivered",
            "delivered_normal",
            "delivered_small",
            "dropped",
            "dropped_by_ttl",
            "dropped_normal",
            "dropped_small",
        )
        float_names = (
            "delivered_mbit",
            "e2e_delay_sum",
            "queue_delay_sum",
            "transmission_delay_sum",
            "propagation_delay_sum",
            "normal_e2e_delay_sum",
            "normal_queue_delay_sum",
            "normal_transmission_delay_sum",
            "normal_propagation_delay_sum",
            "small_e2e_delay_sum",
            "small_queue_delay_sum",
            "small_transmission_delay_sum",
            "small_propagation_delay_sum",
            "cost_sum",
            "cost_normal_sum",
            "cost_small_sum",
        )
        self._metric_int_counters = {name: np.zeros(self.num_envs, dtype=np.int64) for name in int_names}
        self._metric_float_counters = {name: np.zeros(self.num_envs, dtype=np.float64) for name in float_names}

    def _reset_metric_counters(self) -> None:
        for counter in self._metric_int_counters.values():
            counter.fill(0)
        for counter in self._metric_float_counters.values():
            counter.fill(0.0)

    def _counter_int(self, name: str, env_id: int | None) -> int:
        counter = self._metric_int_counters[name]
        return int(counter.sum() if env_id is None else counter[int(env_id)])

    def _counter_float(self, name: str, env_id: int | None) -> float:
        counter = self._metric_float_counters[name]
        return float(counter.sum() if env_id is None else counter[int(env_id)])

    def _update_topology(self, force: bool = False) -> None:
        self.sat_positions = self._satellite_positions(self.current_times)
        source_pos = self.sat_positions[:, self.link_source_ids]
        sink_pos = self.sat_positions[:, self.link_sink_ids]
        distances = np.linalg.norm(source_pos - sink_pos, axis=-1)
        self.link_connected = distances <= self.max_isl_range
        self.link_delay = (distances / LIGHT_SPEED_MS).astype(np.float32, copy=False)
        self.node_state[:, :, NODE_POS_X:NODE_POS_Z + 1] = self.sat_positions / self.orbit_radius
        self._refresh_link_state_features()
        self._update_region_access()
        self._invalidate_region_next_hop_cache()

    def _invalidate_region_next_hop_cache(self) -> None:
        self._region_next_hop_cache_key = None
        self._region_next_hop_tables_cache = None
        self._region_next_hop_versions_cache = None

    def _update_region_access(self) -> None:
        chunk_size = int(self.traffic_config.get("region_chunk_size", 32))
        chunk_size = max(chunk_size, 1)
        max_access2 = self.max_access_range * self.max_access_range
        nearest_ids = []
        nearest_distances = []
        sat_xy = np.sqrt(
            self.sat_positions[:, :, 0] * self.sat_positions[:, :, 0]
            + self.sat_positions[:, :, 1] * self.sat_positions[:, :, 1]
        )
        sat_lat = np.arctan2(self.sat_positions[:, :, 2], sat_xy)
        sat_lon = np.arctan2(self.sat_positions[:, :, 1], self.sat_positions[:, :, 0])
        region_lat = np.deg2rad(self.region_latitudes)
        region_lon = np.deg2rad(self.region_longitudes)
        for start in range(0, self.num_regions, chunk_size):
            end = min(start + chunk_size, self.num_regions)
            region_pos = self.region_positions[start:end]
            diff = self.sat_positions[:, None, :, :] - region_pos[None, :, None, :]
            dist2 = np.sum(diff * diff, axis=-1)
            self.region_sat_distances[:, start:end, :] = np.sqrt(np.maximum(dist2, 0.0)).astype(np.float32, copy=False)
            dlat = region_lat[None, start:end, None] - sat_lat[:, None, :]
            dlon = region_lon[None, start:end, None] - sat_lon[:, None, :]
            a = (
                np.sin(dlat * 0.5) ** 2
                + np.cos(sat_lat[:, None, :])
                * np.cos(region_lat[None, start:end, None])
                * np.sin(dlon * 0.5) ** 2
            )
            self.region_sat_gcd_degrees[:, start:end, :] = np.rad2deg(
                2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
            ).astype(np.float32, copy=False)
            masked = np.where(dist2 <= max_access2, dist2, np.inf)
            min_cols = np.argmin(masked, axis=-1)
            min_dist2 = np.take_along_axis(masked, min_cols[..., None], axis=-1)[..., 0]
            visible = np.isfinite(min_dist2)
            nearest_ids.append(np.where(visible, min_cols, -1).astype(np.int64, copy=False))
            nearest_distances.append(np.sqrt(np.maximum(min_dist2, 0.0)).astype(np.float32, copy=False))
        self.nearest_region_sat_ids = np.concatenate(nearest_ids, axis=1)
        self.nearest_region_sat_distances = np.concatenate(nearest_distances, axis=1)

    def _fill_free_flowlet_slots(self) -> None:
        env_ids, flowlet_ids = np.nonzero(self._free_slot_mask())
        n = len(flowlet_ids)
        if n == 0:
            return
        self._reset_flowlet_slots(env_ids, flowlet_ids)
        pair_ids = self._sample_od_pair_ids(n)
        source_regions = pair_ids // self.num_regions
        target_regions = pair_ids % self.num_regions
        normal = self._rng.random(n) < self.prob_normal_packet
        packet_size = np.where(normal, self.normal_packet_size, self.small_packet_size).astype(np.float32, copy=False)
        packet_count = np.maximum(self._rng.poisson(self.mean_packets_per_flowlet, size=n).astype(np.int64, copy=False), 1)
        size = (packet_size * packet_count.astype(np.float32, copy=False)).astype(np.float32, copy=False)
        current_time = self.current_times[env_ids]

        source_sat = self.nearest_region_sat_ids[env_ids, source_regions]
        visible = source_sat >= 0
        source_distance = self.nearest_region_sat_distances[env_ids, source_regions]
        source_prop_delay = source_distance / LIGHT_SPEED_MS
        source_tx_delay = size / max(self.access_data_rate, 1e-9)
        initial_gcd = np.maximum(self.region_distance_matrix[source_regions, target_regions], 1e-6)

        self.status[env_ids, flowlet_ids] = FLOWLET_AT_NODE
        self.creation_time[env_ids, flowlet_ids] = current_time
        self.source_region_id[env_ids, flowlet_ids] = source_regions
        self.target_region_id[env_ids, flowlet_ids] = target_regions
        self.source_id[env_ids, flowlet_ids] = np.where(visible, source_sat, -1)
        self.current_sat[env_ids, flowlet_ids] = np.where(visible, source_sat, -1)
        self.packet_count[env_ids, flowlet_ids] = packet_count
        self.packet_size[env_ids, flowlet_ids] = packet_size
        self.is_normal[env_ids, flowlet_ids] = normal
        self.size[env_ids, flowlet_ids] = size
        self.ttl[env_ids, flowlet_ids] = self.default_ttl
        self.first_access_delay[env_ids, flowlet_ids] = np.where(visible, source_prop_delay + source_tx_delay, 0.0)
        self.propagation_delay[env_ids, flowlet_ids] = np.where(visible, source_prop_delay, 0.0)
        self.transmission_delay[env_ids, flowlet_ids] = np.where(visible, source_tx_delay, 0.0)
        self.initial_gcd[env_ids, flowlet_ids] = initial_gcd
        self.shortest_gcd[env_ids, flowlet_ids] = initial_gcd
        self._record_generated(env_ids, packet_count, normal)
        self._drop(env_ids[~visible], flowlet_ids[~visible], current_time[~visible], NetworkError.NO_AVAILABLE_SAT)

    def _free_slot_mask(self) -> np.ndarray:
        return self.status == FLOWLET_NOT_STARTED

    def _recycle_terminal_slots(self) -> None:
        env_ids, flowlet_ids = np.nonzero((self.status == FLOWLET_DELIVERED) | (self.status == FLOWLET_DROPPED))
        if len(flowlet_ids):
            self._reset_flowlet_slots(env_ids, flowlet_ids)

    def _reset_flowlet_slots(self, env_ids: np.ndarray, flowlet_ids: np.ndarray) -> None:
        self.status[env_ids, flowlet_ids] = FLOWLET_NOT_STARTED
        self.creation_time[env_ids, flowlet_ids] = 0.0
        self.source_region_id[env_ids, flowlet_ids] = -1
        self.target_region_id[env_ids, flowlet_ids] = -1
        self.source_id[env_ids, flowlet_ids] = -1
        self.current_sat[env_ids, flowlet_ids] = -1
        self.next_sat[env_ids, flowlet_ids] = -1
        self.link_id[env_ids, flowlet_ids] = -1
        self.packet_count[env_ids, flowlet_ids] = 1
        self.packet_size[env_ids, flowlet_ids] = 0.0
        self.is_normal[env_ids, flowlet_ids] = False
        self.size[env_ids, flowlet_ids] = 0.0
        self.ttl[env_ids, flowlet_ids] = self.default_ttl
        self.hops[env_ids, flowlet_ids] = 0
        self.queue_delay[env_ids, flowlet_ids] = 0.0
        self.transmission_delay[env_ids, flowlet_ids] = 0.0
        self.propagation_delay[env_ids, flowlet_ids] = 0.0
        self.total_queue_cost[env_ids, flowlet_ids] = 0.0
        self.first_access_delay[env_ids, flowlet_ids] = 0.0
        self.final_access_delay[env_ids, flowlet_ids] = 0.0
        self.delivery_time[env_ids, flowlet_ids] = np.nan
        self.drop_time[env_ids, flowlet_ids] = np.nan
        self.drop_reason[env_ids, flowlet_ids] = -1
        self.transmit_end_time[env_ids, flowlet_ids] = np.inf
        self.arrival_time[env_ids, flowlet_ids] = np.inf
        self.link_released[env_ids, flowlet_ids] = True
        self.scheduled_prop_delay[env_ids, flowlet_ids] = 0.0
        self.remaining_gcd[env_ids, flowlet_ids] = np.inf
        self.shortest_gcd[env_ids, flowlet_ids] = np.inf
        self.initial_gcd[env_ids, flowlet_ids] = 1.0
        self.last_action1[env_ids, flowlet_ids] = -1
        self.last_action2[env_ids, flowlet_ids] = -1
        self.last_node1[env_ids, flowlet_ids] = -1
        self.last_node2[env_ids, flowlet_ids] = -1

    def _record_generated(self, env_ids: np.ndarray, packet_count: np.ndarray, is_normal: np.ndarray) -> None:
        if len(env_ids) == 0:
            return
        np.add.at(self._metric_int_counters["generated"], env_ids, packet_count)
        np.add.at(self._metric_int_counters["generated_normal"], env_ids, np.where(is_normal, packet_count, 0))
        np.add.at(self._metric_int_counters["generated_small"], env_ids, np.where(~is_normal, packet_count, 0))
        self.generated_flowlets_total += int(len(env_ids))

    def _record_delivered(self, env_ids: np.ndarray, flowlet_ids: np.ndarray) -> None:
        if len(env_ids) == 0:
            return
        weights = self.packet_count[env_ids, flowlet_ids]
        is_normal = self.is_normal[env_ids, flowlet_ids]
        total_delay = self.queue_delay[env_ids, flowlet_ids] + self.transmission_delay[env_ids, flowlet_ids] + self.propagation_delay[env_ids, flowlet_ids]
        np.add.at(self._metric_int_counters["delivered"], env_ids, weights)
        np.add.at(self._metric_int_counters["delivered_normal"], env_ids, np.where(is_normal, weights, 0))
        np.add.at(self._metric_int_counters["delivered_small"], env_ids, np.where(~is_normal, weights, 0))
        np.add.at(self._metric_float_counters["delivered_mbit"], env_ids, self.size[env_ids, flowlet_ids])
        weight_f = weights.astype(np.float64, copy=False)
        np.add.at(self._metric_float_counters["e2e_delay_sum"], env_ids, total_delay * weight_f)
        np.add.at(self._metric_float_counters["queue_delay_sum"], env_ids, self.queue_delay[env_ids, flowlet_ids] * weight_f)
        np.add.at(self._metric_float_counters["transmission_delay_sum"], env_ids, self.transmission_delay[env_ids, flowlet_ids] * weight_f)
        np.add.at(self._metric_float_counters["propagation_delay_sum"], env_ids, self.propagation_delay[env_ids, flowlet_ids] * weight_f)
        np.add.at(self._metric_float_counters["cost_sum"], env_ids, self.total_queue_cost[env_ids, flowlet_ids] * weight_f)
        normal_w = np.where(is_normal, weight_f, 0.0)
        small_w = np.where(~is_normal, weight_f, 0.0)
        np.add.at(self._metric_float_counters["normal_e2e_delay_sum"], env_ids, total_delay * normal_w)
        np.add.at(self._metric_float_counters["normal_queue_delay_sum"], env_ids, self.queue_delay[env_ids, flowlet_ids] * normal_w)
        np.add.at(self._metric_float_counters["normal_transmission_delay_sum"], env_ids, self.transmission_delay[env_ids, flowlet_ids] * normal_w)
        np.add.at(self._metric_float_counters["normal_propagation_delay_sum"], env_ids, self.propagation_delay[env_ids, flowlet_ids] * normal_w)
        np.add.at(self._metric_float_counters["cost_normal_sum"], env_ids, self.total_queue_cost[env_ids, flowlet_ids] * normal_w)
        np.add.at(self._metric_float_counters["small_e2e_delay_sum"], env_ids, total_delay * small_w)
        np.add.at(self._metric_float_counters["small_queue_delay_sum"], env_ids, self.queue_delay[env_ids, flowlet_ids] * small_w)
        np.add.at(self._metric_float_counters["small_transmission_delay_sum"], env_ids, self.transmission_delay[env_ids, flowlet_ids] * small_w)
        np.add.at(self._metric_float_counters["small_propagation_delay_sum"], env_ids, self.propagation_delay[env_ids, flowlet_ids] * small_w)
        np.add.at(self._metric_float_counters["cost_small_sum"], env_ids, self.total_queue_cost[env_ids, flowlet_ids] * small_w)

    def _record_dropped(self, env_ids: np.ndarray, flowlet_ids: np.ndarray, reason: NetworkError) -> None:
        if len(env_ids) == 0:
            return
        weights = self.packet_count[env_ids, flowlet_ids]
        is_normal = self.is_normal[env_ids, flowlet_ids]
        np.add.at(self._metric_int_counters["dropped"], env_ids, weights)
        np.add.at(self._metric_int_counters["dropped_normal"], env_ids, np.where(is_normal, weights, 0))
        np.add.at(self._metric_int_counters["dropped_small"], env_ids, np.where(~is_normal, weights, 0))
        if reason == NetworkError.TTL_EXPIRED:
            np.add.at(self._metric_int_counters["dropped_by_ttl"], env_ids, weights)

    def _apply_routing_actions(self, batch: RoutingBatch, actions: np.ndarray) -> None:
        env_ids = np.asarray(batch.row_env_ids, dtype=np.int64)
        flowlet_ids = np.asarray(batch.flowlet_ids, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        if batch.decision_mask is None:
            decision_mask = np.asarray(batch.action_mask, dtype=bool).any(axis=1)
        else:
            decision_mask = np.asarray(batch.decision_mask, dtype=bool)
        self._schedule_token_value += 1
        apply_routing_actions_kernel(
            env_ids,
            flowlet_ids,
            actions,
            decision_mask,
            self.current_times,
            self.status,
            self.current_sat,
            self.neighbor_sat_ids,
            self.neighbor_link_ids,
            self.direction_by_link_id,
            self.link_connected,
            self.link_capacity,
            self.link_data_rate,
            self.link_delay,
            self.size,
            self._schedule_tx_delta,
            self._schedule_base_time,
            self._schedule_token,
            int(self._schedule_token_value),
            self.packet_count,
            self.is_normal,
            self.queue_load,
            self.free_time,
            self.queue_delay,
            self.transmission_delay,
            self.total_queue_cost,
            self.link_id,
            self.next_sat,
            self.transmit_end_time,
            self.arrival_time,
            self.scheduled_prop_delay,
            self.link_released,
            self.last_action1,
            self.last_action2,
            self.last_node1,
            self.last_node2,
            self.drop_time,
            self.drop_reason,
            self._metric_int_counters["dropped"],
            self._metric_int_counters["dropped_normal"],
            self._metric_int_counters["dropped_small"],
        )

    def _release_transmitted_flowlets(self) -> None:
        release_transmitted_kernel(
            self.status,
            self.link_released,
            self.transmit_end_time,
            self.current_times,
            self.link_id,
            self.size,
            self.queue_load,
        )

    def _handle_arrivals(self) -> None:
        handle_arrivals_kernel(
            self.status,
            self.current_sat,
            self.next_sat,
            self.link_id,
            self.hops,
            self.ttl,
            self.propagation_delay,
            self.queue_delay,
            self.scheduled_prop_delay,
            self.arrival_time,
            self.current_times,
            self.packet_count,
            self.is_normal,
            self.drop_time,
            self.drop_reason,
            self._metric_int_counters["dropped"],
            self._metric_int_counters["dropped_normal"],
            self._metric_int_counters["dropped_small"],
            self._metric_int_counters["dropped_by_ttl"],
        )

    def _drop_flowlets_on_disconnected_links(self) -> None:
        drop_disconnected_kernel(
            self.status,
            self.link_released,
            self.link_id,
            self.current_times,
            self.size,
            self.queue_load,
            self.link_connected,
            self.packet_count,
            self.is_normal,
            self.drop_time,
            self.drop_reason,
            self._metric_int_counters["dropped"],
            self._metric_int_counters["dropped_normal"],
            self._metric_int_counters["dropped_small"],
        )

    def _build_routing_batch(self) -> RoutingBatch:
        env_ids = self.flat_env_ids
        flowlet_ids = self.flat_flowlet_ids
        status = self.status.reshape(-1)
        at_node = self._batch_at_node
        np.equal(status, FLOWLET_AT_NODE, out=at_node)
        raw_current_sats = self.current_sat.reshape(-1)
        raw_target_regions = self.target_region_id.reshape(-1)
        current_sats = self._batch_current_sats_safe
        target_regions = self._batch_target_regions_safe
        np.maximum(raw_current_sats, 0, out=current_sats)
        np.maximum(raw_target_regions, 0, out=target_regions)
        remaining_gcd = self._batch_remaining_gcd
        np.copyto(remaining_gcd, self.remaining_gcd.reshape(-1))
        at_rows = np.flatnonzero(at_node)
        if len(at_rows):
            remaining_at = self._remaining_target_distances(env_ids[at_rows], current_sats[at_rows], target_regions[at_rows])
            remaining_gcd[at_rows] = remaining_at
            self.remaining_gcd[env_ids[at_rows], flowlet_ids[at_rows]] = remaining_at

        deliverable = self._batch_deliverable
        deliverable.fill(False)
        if len(at_rows):
            deliverable[at_rows] = self._region_sat_visible(env_ids[at_rows], target_regions[at_rows], current_sats[at_rows])
        deliver_visible_kernel(
            deliverable,
            env_ids,
            flowlet_ids,
            self.current_times,
            self.status,
            self.current_sat,
            self.target_region_id,
            self.region_sat_distances,
            self.size,
            self.packet_count,
            self.is_normal,
            self.queue_delay,
            self.transmission_delay,
            self.propagation_delay,
            self.total_queue_cost,
            self.final_access_delay,
            self.delivery_time,
            self._metric_int_counters["delivered"],
            self._metric_int_counters["delivered_normal"],
            self._metric_int_counters["delivered_small"],
            self._metric_float_counters["delivered_mbit"],
            self._metric_float_counters["e2e_delay_sum"],
            self._metric_float_counters["queue_delay_sum"],
            self._metric_float_counters["transmission_delay_sum"],
            self._metric_float_counters["propagation_delay_sum"],
            self._metric_float_counters["normal_e2e_delay_sum"],
            self._metric_float_counters["normal_queue_delay_sum"],
            self._metric_float_counters["normal_transmission_delay_sum"],
            self._metric_float_counters["normal_propagation_delay_sum"],
            self._metric_float_counters["small_e2e_delay_sum"],
            self._metric_float_counters["small_queue_delay_sum"],
            self._metric_float_counters["small_transmission_delay_sum"],
            self._metric_float_counters["small_propagation_delay_sum"],
            self._metric_float_counters["cost_sum"],
            self._metric_float_counters["cost_normal_sum"],
            self._metric_float_counters["cost_small_sum"],
            self.access_data_rate,
            LIGHT_SPEED_MS,
        )

        route_mask = self._batch_route_mask
        np.logical_not(deliverable, out=route_mask)
        np.logical_and(route_mask, at_node, out=route_mask)
        target_access_sats = self._batch_target_access_sats
        target_access_sats.fill(-1)
        route_rows = np.flatnonzero(route_mask)
        if len(route_rows):
            target_access_sats[route_rows] = self.nearest_region_sat_ids[env_ids[route_rows], target_regions[route_rows]]
        decision_mask = self._batch_decision_mask
        np.greater_equal(target_access_sats, 0, out=decision_mask)
        np.logical_and(decision_mask, route_mask, out=decision_mask)
        no_target_access = self._batch_no_target_access
        np.less(target_access_sats, 0, out=no_target_access)
        np.logical_and(no_target_access, route_mask, out=no_target_access)
        self._drop(
            env_ids[no_target_access],
            flowlet_ids[no_target_access],
            self.current_times[env_ids[no_target_access]],
            NetworkError.NO_AVAILABLE_SAT,
        )
        decision_rows_full = np.flatnonzero(decision_mask)
        decision_count = len(decision_rows_full)
        compact_rows = self._compact_rows[:decision_count]
        compact_decision_mask = self._compact_decision_mask[:decision_count]
        self._refresh_link_state_features()

        row_env_ids = env_ids[decision_rows_full]
        row_flowlet_ids = flowlet_ids[decision_rows_full]
        row_current_sats = current_sats[decision_rows_full]
        row_target_regions = target_regions[decision_rows_full]
        row_target_access_sats = target_access_sats[decision_rows_full]
        row_remaining_gcd = remaining_gcd[decision_rows_full]
        neighbor_sat_ids = self._batch_neighbor_sat_ids[:decision_count]
        neighbor_link_ids = self._batch_neighbor_link_ids[:decision_count]
        action_mask = self._batch_action_mask[:decision_count]
        build_action_mask_kernel(
            neighbor_sat_ids,
            neighbor_link_ids,
            action_mask,
            row_env_ids,
            row_flowlet_ids,
            row_current_sats,
            row_target_regions,
            compact_decision_mask,
            self.neighbor_sat_ids,
            self.neighbor_link_ids,
            self.link_connected,
            self.region_sat_distances,
            self.last_node1,
            self.max_access_range,
        )

        region_next_hop_tables, region_next_hop_versions = self._build_region_next_hop_tables() if self._include_spf_table else (None, None)
        batch = RoutingBatch(
            flowlet_ids=row_flowlet_ids,
            current_sat_ids=row_current_sats,
            source_region_ids=self.source_region_id.reshape(-1)[decision_rows_full],
            target_region_ids=row_target_regions,
            target_access_sat_ids=row_target_access_sats,
            neighbor_sat_ids=neighbor_sat_ids,
            neighbor_link_ids=neighbor_link_ids,
            action_mask=action_mask,
            neighbor_queue_load=None,
            neighbor_link_capacity=None,
            neighbor_link_delay=None,
            neighbor_link_free_time=None,
            flowlet_size=self.size.reshape(-1)[decision_rows_full],
            packet_count=self.packet_count.reshape(-1)[decision_rows_full],
            is_normal=self.is_normal.reshape(-1)[decision_rows_full],
            creation_time=self.creation_time.reshape(-1)[decision_rows_full],
            ttl=self.ttl.reshape(-1)[decision_rows_full],
            current_time=float(self.current_time),
            region_next_hop_tables=region_next_hop_tables,
            region_next_hop_versions=region_next_hop_versions,
            node_state=self.node_state,
            link_state=self.link_state,
            observations=None,
            hops=self.hops.reshape(-1)[decision_rows_full],
            queue_delay=self.queue_delay.reshape(-1)[decision_rows_full],
            transmission_delay=self.transmission_delay.reshape(-1)[decision_rows_full],
            propagation_delay=self.propagation_delay.reshape(-1)[decision_rows_full],
            total_queue_cost=self.total_queue_cost.reshape(-1)[decision_rows_full],
            remaining_gcd=row_remaining_gcd,
            shortest_gcd=self.shortest_gcd.reshape(-1)[decision_rows_full],
            initial_gcd=self.initial_gcd.reshape(-1)[decision_rows_full],
            last_action1=self.last_action1.reshape(-1)[decision_rows_full],
            last_action2=self.last_action2.reshape(-1)[decision_rows_full],
            last_node1=self.last_node1.reshape(-1)[decision_rows_full],
            last_node2=self.last_node2.reshape(-1)[decision_rows_full],
            env_ids=row_env_ids,
            current_times=self.current_times[row_env_ids],
            decision_mask=compact_decision_mask,
            decision_rows=compact_rows,
        )
        batch.observations = self._build_legacy_observations(
            env_ids=row_env_ids,
            flowlet_ids=row_flowlet_ids,
            current_sats=row_current_sats,
            target_regions=row_target_regions,
            neighbor_sat_ids=neighbor_sat_ids,
            neighbor_link_ids=neighbor_link_ids,
            action_mask=action_mask,
            remaining_gcd=row_remaining_gcd,
            decision_mask=compact_decision_mask,
            decision_rows=compact_rows,
        )
        return batch

    def _build_region_next_hop_tables(self) -> tuple[np.ndarray, np.ndarray]:
        cache_key = int(self.topology_update_steps)
        if (
            self._region_next_hop_cache_key == cache_key
            and self._region_next_hop_tables_cache is not None
            and self._region_next_hop_versions_cache is not None
        ):
            return self._region_next_hop_tables_cache, self._region_next_hop_versions_cache
        versions = np.full(self.num_envs, cache_key, dtype=np.int64)
        tables = np.full(
            (self.num_envs, self.num_regions, self.num_satellites),
            -1,
            dtype=np.int32,
        )
        for env_id in range(self.num_envs):
            tables[env_id] = self._build_region_next_hop_table_for_env(env_id)
        self._region_next_hop_cache_key = cache_key
        self._region_next_hop_tables_cache = tables
        self._region_next_hop_versions_cache = versions
        return self._region_next_hop_tables_cache, self._region_next_hop_versions_cache

    def _build_region_next_hop_table_for_env(self, env_id: int) -> np.ndarray:
        access_sats = self.nearest_region_sat_ids[int(env_id)].astype(np.int64, copy=False)
        valid_regions = access_sats >= 0
        table = np.full((self.num_regions, self.num_satellites), -1, dtype=np.int32)
        if not valid_regions.any():
            return table
        connected = self.link_connected[int(env_id)]
        weights = self.link_delay[int(env_id)].astype(np.float64, copy=False)
        edge_mask = connected & np.isfinite(weights)
        if not edge_mask.any():
            return table
        reverse_graph = csr_matrix(
            (weights[edge_mask], (self.link_sink_ids[edge_mask], self.link_source_ids[edge_mask])),
            shape=(self.num_satellites, self.num_satellites),
        )
        target_sinks = np.unique(access_sats[valid_regions]).astype(np.int32, copy=False)
        _distances, predecessors = dijkstra(
            csgraph=reverse_graph,
            directed=True,
            indices=target_sinks,
            return_predecessors=True,
        )
        if predecessors.ndim == 1:
            predecessors = predecessors[np.newaxis, :]
        next_hop_rows = predecessors.astype(np.int32, copy=False)
        next_hop_rows[next_hop_rows < 0] = -1
        next_hop_rows[np.arange(len(target_sinks)), target_sinks] = -1
        sink_to_row = {int(sink): row for row, sink in enumerate(target_sinks)}
        for region_id in np.flatnonzero(valid_regions):
            table[int(region_id)] = next_hop_rows[sink_to_row[int(access_sats[region_id])]]
        return table

    def _deliver(self, env_ids: np.ndarray, flowlet_ids: np.ndarray) -> None:
        if len(flowlet_ids) == 0:
            return
        current_sats = self.current_sat[env_ids, flowlet_ids]
        target_regions = self.target_region_id[env_ids, flowlet_ids]
        distance = self._region_sat_distance(env_ids, target_regions, current_sats)
        final_prop_delay = distance / LIGHT_SPEED_MS
        final_tx_delay = self.size[env_ids, flowlet_ids] / max(self.access_data_rate, 1e-9)
        final_delay = final_prop_delay + final_tx_delay
        self.final_access_delay[env_ids, flowlet_ids] = final_delay
        self.propagation_delay[env_ids, flowlet_ids] += final_prop_delay
        self.transmission_delay[env_ids, flowlet_ids] += final_tx_delay
        self.delivery_time[env_ids, flowlet_ids] = self.current_times[env_ids] + final_delay
        self._record_delivered(env_ids, flowlet_ids)
        self.status[env_ids, flowlet_ids] = FLOWLET_DELIVERED

    def _drop(self, env_ids: np.ndarray, flowlet_ids: np.ndarray, current_times: np.ndarray, reason: NetworkError) -> None:
        if len(flowlet_ids) == 0:
            return
        live = (self.status[env_ids, flowlet_ids] != FLOWLET_DROPPED) & (self.status[env_ids, flowlet_ids] != FLOWLET_DELIVERED)
        env_ids = env_ids[live]
        flowlet_ids = flowlet_ids[live]
        current_times = current_times[live]
        if len(flowlet_ids) == 0:
            return
        self._record_dropped(env_ids, flowlet_ids, reason)
        self.status[env_ids, flowlet_ids] = FLOWLET_DROPPED
        self.drop_time[env_ids, flowlet_ids] = current_times
        self.drop_reason[env_ids, flowlet_ids] = int(reason)

    def _build_legacy_observations(
        self,
        env_ids: np.ndarray,
        flowlet_ids: np.ndarray,
        current_sats: np.ndarray,
        target_regions: np.ndarray,
        neighbor_sat_ids: np.ndarray,
        neighbor_link_ids: np.ndarray,
        action_mask: np.ndarray,
        remaining_gcd: np.ndarray,
        decision_mask: np.ndarray | None = None,
        decision_rows: np.ndarray | None = None,
    ) -> np.ndarray:
        n = len(flowlet_ids)
        self._ensure_observation_capacity(n)
        obs = self.observation_state[:n]
        # This fixed buffer is reused across steps; only decision rows are current.
        if decision_rows is not None:
            rows = decision_rows
        else:
            rows = np.arange(n, dtype=np.int64) if decision_mask is None else np.flatnonzero(decision_mask)
        if len(rows) == 0:
            return obs
        self._refresh_node_queue_features()
        build_legacy_observations_kernel(
            obs,
            rows,
            env_ids,
            flowlet_ids,
            current_sats,
            target_regions,
            neighbor_sat_ids,
            neighbor_link_ids,
            remaining_gcd,
            self.node_state,
            self.link_state,
            self.region_positions,
            self.region_sat_distances,
            self.region_sat_gcd_degrees,
            self.current_times,
            self.creation_time,
            self.is_normal,
            self.size,
            self.ttl,
            self.queue_delay,
            self.transmission_delay,
            self.propagation_delay,
            self.initial_gcd,
            self.last_action1,
            self.last_action2,
            self.last_node1,
            self.last_node2,
            self.delay_norm,
            float(self.default_ttl),
            self.orbit_radius,
            self.orbit_cycle_ms,
            self.max_access_range,
        )
        return obs

    def _ensure_observation_capacity(self, required_rows: int) -> None:
        if required_rows <= self.observation_state.shape[0]:
            return
        total_slots = self.num_envs * self.capacity
        new_rows = min(max(required_rows, self.observation_state.shape[0] * 2), total_slots)
        self.observation_state = np.zeros((new_rows, self.obs_dim), dtype=np.float32)

    def _refresh_node_queue_features(self) -> None:
        refresh_node_queue_features_kernel(
            self.node_state,
            self.node_queue_load,
            self.node_queue_remaining,
            self.queue_load,
            self.neighbor_link_ids,
            self.node_total_link_capacity,
        )

    def _refresh_link_state_features(self) -> None:
        refresh_link_state_kernel(
            self.link_state,
            self.link_connected,
            self.link_delay,
            self.queue_load,
            self.free_time,
            self.current_times,
            self.link_capacity,
            self.link_data_rate,
        )

    def _cached_node_queue_features(self, env_ids: np.ndarray, sat_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = sat_ids >= 0
        safe_sat_ids = np.maximum(sat_ids, 0)
        load = self.node_state[env_ids, safe_sat_ids, NODE_QUEUE_LOAD]
        remaining = self.node_state[env_ids, safe_sat_ids, NODE_QUEUE_REMAINING]
        return np.where(valid, load, 0.0), np.where(valid, remaining, 0.0)

    def _neighbor_deliverable_to_target(self, env_ids: np.ndarray, target_regions: np.ndarray, neighbor_sat_ids: np.ndarray) -> np.ndarray:
        valid = neighbor_sat_ids >= 0
        safe_neighbor = np.maximum(neighbor_sat_ids, 0)
        flat_env = np.broadcast_to(env_ids[:, None], safe_neighbor.shape)
        flat_targets = np.broadcast_to(target_regions[:, None], safe_neighbor.shape)
        visible = self._region_sat_visible(flat_env.reshape(-1), flat_targets.reshape(-1), safe_neighbor.reshape(-1)).reshape(neighbor_sat_ids.shape)
        return valid & visible

    def _region_sat_visible(self, env_ids: np.ndarray, region_ids: np.ndarray, sat_ids: np.ndarray) -> np.ndarray:
        return self._region_sat_distance(env_ids, region_ids, sat_ids) <= self.max_access_range

    def _region_sat_distance(self, env_ids: np.ndarray, region_ids: np.ndarray, sat_ids: np.ndarray) -> np.ndarray:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        region_ids = np.asarray(region_ids, dtype=np.int64)
        sat_ids = np.asarray(sat_ids, dtype=np.int64)
        valid = (sat_ids >= 0) & (region_ids >= 0)
        safe_regions = np.maximum(region_ids, 0)
        safe_sats = np.maximum(sat_ids, 0)
        distances = self.region_sat_distances[env_ids, safe_regions, safe_sats]
        return np.where(valid, distances, np.inf).astype(np.float32, copy=False)

    def _remaining_target_distances(self, env_ids: np.ndarray, sat_ids: np.ndarray, target_regions: np.ndarray) -> np.ndarray:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        sat_ids = np.asarray(sat_ids, dtype=np.int64)
        target_regions = np.asarray(target_regions, dtype=np.int64)
        valid = (sat_ids >= 0) & (target_regions >= 0)
        safe_sats = np.maximum(sat_ids, 0)
        safe_regions = np.maximum(target_regions, 0)
        distances = self.region_sat_gcd_degrees[env_ids, safe_regions, safe_sats]
        return np.where(valid, distances, np.inf).astype(np.float32, copy=False)

    def _satellite_positions(self, times_ms: np.ndarray) -> np.ndarray:
        times_ms = np.asarray(times_ms, dtype=np.float64)
        semi_major_axis_m = (EARTH_R_KM + self.sat_altitudes) * 1000.0
        orbit_cycles = 2.0 * np.pi * np.sqrt(np.power(semi_major_axis_m, 3) / GM_EARTH)
        theta = np.remainder(
            self.sat_true_anomaly[None, :] + (360.0 / orbit_cycles[None, :]) * times_ms[:, None] / 1000.0,
            360.0,
        )
        theta_rad = np.deg2rad(theta)
        orbit_radius = EARTH_R_KM + self.sat_altitudes
        cos_raan = np.cos(self.sat_raan_rad)[None, :]
        sin_raan = np.sin(self.sat_raan_rad)[None, :]
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        cos_inc = np.cos(self.sat_inc_rad)[None, :]
        sin_inc = np.sin(self.sat_inc_rad)[None, :]
        x_eci = orbit_radius[None, :] * (cos_raan * cos_theta - sin_raan * sin_theta * cos_inc)
        y_eci = orbit_radius[None, :] * (sin_raan * cos_theta + cos_raan * sin_theta * cos_inc)
        z_eci = orbit_radius[None, :] * sin_theta * sin_inc
        theta_earth = 7.2921150e-5 * times_ms / 1000.0
        cos_earth = np.cos(theta_earth)[:, None]
        sin_earth = np.sin(theta_earth)[:, None]
        x_ecef = x_eci * cos_earth + y_eci * sin_earth
        y_ecef = -x_eci * sin_earth + y_eci * cos_earth
        return np.stack((x_ecef, y_ecef, z_eci), axis=-1).astype(np.float32, copy=False)

    def _sample_od_pair_ids(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        uniform = self._rng.random(count)
        return np.searchsorted(self.flat_od_cdf, uniform, side="left").astype(np.int64, copy=False)

    def _build_neighbor_link_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        source = np.asarray(self.network._link_source_ids, dtype=np.int64)
        sink = np.asarray(self.network._link_sink_ids, dtype=np.int64)
        id_by_pair = np.full((self.num_nodes, self.num_nodes), -1, dtype=np.int32)
        id_by_pair[source, sink] = np.arange(len(source), dtype=np.int32)
        neighbor_sat_ids = np.asarray(self.network.neighbor_sat_ids, dtype=np.int64)
        neighbor_link_ids = np.full(neighbor_sat_ids.shape, -1, dtype=np.int32)
        valid = neighbor_sat_ids >= 0
        if valid.any():
            rows = np.broadcast_to(np.arange(self.num_nodes, dtype=np.int64)[:, None], neighbor_sat_ids.shape)
            neighbor_link_ids[valid] = id_by_pair[rows[valid], neighbor_sat_ids[valid]]
        direction_by_link_id = np.full(self.num_links, -1, dtype=np.int64)
        direction_cols = np.broadcast_to(np.arange(ACTION_COUNT, dtype=np.int64), neighbor_sat_ids.shape)
        valid_link = neighbor_link_ids >= 0
        direction_by_link_id[neighbor_link_ids[valid_link]] = direction_cols[valid_link]
        return neighbor_link_ids.astype(np.int64, copy=False), direction_by_link_id

    def _normalize_action(self, action, expected_count: int) -> np.ndarray:
        if action is None:
            if expected_count == 0:
                return np.empty(0, dtype=np.int64)
            raise ValueError(f"Expected {expected_count} actions, got None.")
        if isinstance(action, RoutingDecision):
            action = action.next_hop_sat_ids
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        actions = np.asarray(action, dtype=np.int64)
        if actions.ndim != 1:
            raise ValueError(f"Actions must be a 1-D array, got shape {tuple(actions.shape)}.")
        if len(actions) != expected_count:
            raise ValueError(f"Expected {expected_count} actions, got {len(actions)}.")
        return actions

    def _empty_routing_batch(self) -> RoutingBatch:
        long_0 = np.empty(0, dtype=np.int64)
        bool_0a = np.empty((0, ACTION_COUNT), dtype=bool)
        long_0a = np.empty((0, ACTION_COUNT), dtype=np.int64)
        float_0 = np.empty(0, dtype=np.float32)
        float_0a = np.empty((0, ACTION_COUNT), dtype=np.float32)
        return RoutingBatch(
            flowlet_ids=long_0,
            current_sat_ids=long_0,
            source_region_ids=long_0,
            target_region_ids=long_0,
            target_access_sat_ids=long_0,
            neighbor_sat_ids=long_0a,
            neighbor_link_ids=long_0a,
            action_mask=bool_0a,
            neighbor_queue_load=float_0a,
            neighbor_link_capacity=float_0a,
            neighbor_link_delay=float_0a,
            neighbor_link_free_time=float_0a,
            flowlet_size=float_0,
            packet_count=long_0,
            is_normal=np.empty(0, dtype=bool),
            creation_time=float_0,
            ttl=long_0,
            current_time=self.current_time,
            region_next_hop_tables=None,
            region_next_hop_versions=None,
            observations=np.empty((0, self.obs_dim), dtype=np.float32),
            hops=long_0,
            queue_delay=float_0,
            transmission_delay=float_0,
            propagation_delay=float_0,
            total_queue_cost=float_0,
            remaining_gcd=float_0,
            shortest_gcd=float_0,
            initial_gcd=float_0,
            last_action1=long_0,
            last_action2=long_0,
            last_node1=long_0,
            last_node2=long_0,
            env_ids=long_0,
            current_times=float_0,
            decision_mask=np.empty(0, dtype=bool),
            decision_rows=long_0,
        )

    def _build_step_info(self, sync: bool = False) -> dict:
        batch = self.observation
        info = {
            "time_ms": self.current_time,
            "step": self._step_index,
            "num_envs": self.num_envs,
            "backend": "numpy_array",
            "device": "cpu",
            "decision_count": batch.active_decision_count,
            "route_count": batch.active_decision_count,
            "batch_rows": batch.batch_size,
            "flowlet_capacity": int(self.capacity * self.num_envs),
            "concurrent_flowlets_per_env": int(self.concurrent_flowlets_per_env),
            "closed_loop_traffic": True,
            "generated_flowlets_total": self.generated_flowlets_total,
            "storage_overflow_flowlets": self.storage_overflow_flowlets,
            "topology_updates": self.topology_update_steps,
            "terminated": False,
        }
        if not sync:
            return info
        elapsed = self._elapsed_ms()
        active_flowlet_rows = int(self.flowlet_counts.sum())
        info.update(
            {
                "env_time_ms_min": float(elapsed.min()) if len(elapsed) else 0.0,
                "env_time_ms_max": float(elapsed.max()) if len(elapsed) else 0.0,
                "active_agent_count": batch.active_decision_count,
                "active_flowlet_rows": active_flowlet_rows,
                "flowlet_rows": active_flowlet_rows,
                "terminated": self.terminated,
            }
        )
        return info

    def _has_pending_flowlets_by_env(self) -> np.ndarray:
        pending = (self.status == FLOWLET_AT_NODE) | (self.status == FLOWLET_ON_LINK)
        return pending.any(axis=1)

    def _refresh_flowlet_counts(self) -> None:
        active = (self.status == FLOWLET_AT_NODE) | (self.status == FLOWLET_ON_LINK)
        self.flowlet_counts = active.sum(axis=1).astype(np.int64, copy=False)

    def _elapsed_ms(self) -> np.ndarray:
        elapsed = np.maximum(self.current_times - self.start_times, 0.0)
        if np.isfinite(self.time_limit):
            elapsed = np.minimum(elapsed, float(self.time_limit))
        return elapsed

    def _build_start_offsets(self) -> np.ndarray:
        if self.num_envs <= 1:
            return np.zeros(1, dtype=np.float64)
        return np.linspace(0.0, self.utc_offset_span_ms, self.num_envs, endpoint=False, dtype=np.float64)
