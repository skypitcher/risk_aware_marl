import heapq
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sat_net.network import SatelliteNetwork
from sat_net.agent.base_agent import ACTION_COUNT, RoutingBatch, RoutingDecision
from sat_net.sim_kernel import (
    FLOWLET_AT_NODE,
    FLOWLET_DELIVERED,
    FLOWLET_DROPPED,
    FLOWLET_NOT_STARTED,
    FLOWLET_ON_LINK,
    FlowletState,
    LinkState,
    activate_flowlets_at_slot_ids,
    build_routing_batch,
    create_flowlet_state,
    create_link_state,
    deliver_flowlet_ids,
    drop_flowlet_ids,
    handle_flowlet_arrival_ids,
    partition_deliverable_flowlets,
    apply_no_route_mask,
    prepare_link_schedule_inputs,
    prepare_route_candidate_mask,
    release_transmitted_flowlet_ids,
    schedule_flowlets_by_link,
)
from sat_net.stats import Metrics
from sat_net.traffic_region import TrafficRegionModel
from sat_net.util import NamedDict, NetworkError, ms2str


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RoutingEnv:
    """
    Fixed-step slot environment for data-oriented satellite routing simulation.
    """

    def __init__(self, config: NamedDict, tf_writer: Any | None = None):
        """
        Initialize the environment.

        Args:
            config: The configuration used for this environment.
        """
        self.config = config
        self.network_config: NamedDict = self.config.network
        self.tf_writer = tf_writer

        self.np_random = None
        self._set_seed(self.config.get("seed", default=None))

        self.network = self._create_network()
        self.traffic_config: NamedDict = self.config.traffic
        self.traffic_model = TrafficRegionModel.from_config(self.traffic_config, PROJECT_ROOT)
        self.packet_rate_per_ms: float = self.traffic_config.get("packet_rate_per_ms", 1.0)
        self.slot_ms: float = self.traffic_config.get("slot_ms", 5.0)
        self.mean_packets_per_flowlet: float = self.traffic_config.get("mean_packets_per_flowlet", 16.0)
        self.eager_spf_region_threshold: int = self.traffic_config.get(
            "eager_spf_region_threshold",
            64,
        )
        self.access_data_rate: float = self.traffic_config.get("access_data_rate", 1.0)
        self.prob_normal_packet: float = self.config.prob_normal_packet
        self.normal_packet_size = self.config.normal_packet_size
        self.small_packet_size = self.config.small_packet_size

        self.default_ttl: int = self.config.default_ttl
        self.delay_norm: float = self.config.get("delay_norm", 100.0)

        self.start_time = 0.0  # the start timestamp of the simulation. randomize this to init the topology
        self.current_time = 0.0  #  # the current time offset, in milliseconds
        self.topology_update_steps = 0
        self.time_limit = float(self.config.time_limit_seconds * 1000.0)

        self.action_dim = ACTION_COUNT  # N, E, S, W - fixed for satellite routing
        self.obs_dim = 94

        self.verbose = self.config.verbose
        self.update_interval_ms = self.config.update_interval_ms
        self.progress_interval_seconds = float(self.config.get("progress_interval_seconds", 60.0))

        self.next_flowlet_id = 0
        self._region_positions = np.array(
            [region.position for region in self.traffic_model.regions],
            dtype=np.float64,
        )
        self._region_latitudes = np.array(
            [region.latitude for region in self.traffic_model.regions],
            dtype=np.float64,
        )
        self._region_longitudes = np.array(
            [region.longitude for region in self.traffic_model.regions],
            dtype=np.float64,
        )
        self._region_distance_matrix = self._build_region_distance_matrix()
        self._sat_ids = np.empty(0, dtype=np.int64)
        self._sat_id_to_col_array = np.empty(0, dtype=np.int64)
        self._region_sat_distance2 = np.empty((len(self.traffic_model.regions), 0))
        self._region_sat_visible = np.zeros((len(self.traffic_model.regions), 0), dtype=bool)
        self._nearest_region_sat_ids = np.full(len(self.traffic_model.regions), -1, dtype=np.int64)
        self._nearest_region_sat_distances = np.full(len(self.traffic_model.regions), np.inf)
        self._region_next_hop_table = np.empty((len(self.traffic_model.regions), 0), dtype=np.int64)
        self._region_next_hop_ready = np.zeros(len(self.traffic_model.regions), dtype=bool)
        self._region_next_hop_eager_ready = False
        self._region_next_hop_version = 0
        self._region_access_cache_time: float | None = None
        self._array_metrics: Metrics | None = None
        self._flowlets: FlowletState | None = None
        self._links: LinkState | None = None
        self._include_spf_table = False
        self._episode_started = False
        self._episode_done = False
        self._step_index = 0
        self._num_steps = 0
        self._end_time = 0.0
        self._next_topology_time = 0.0
        self._wall_start_time = 0.0
        self._progress_interval_ms = 0.0
        self._next_progress_time = 0.0
        self._pending_routing_batch: RoutingBatch | None = None
        self._arrival_events: dict[int, list[np.ndarray]] = {}
        self._release_events: dict[int, list[np.ndarray]] = {}
        self._arrival_event_heap: list[int] = []
        self._release_event_heap: list[int] = []
        self._on_link_id_chunks: list[np.ndarray] = []
        self._creation_event_slots = np.empty(0, dtype=np.int64)
        self._creation_event_cursor = 0

    def _create_network(self):
        """Create the network based on the configuration."""
        return SatelliteNetwork(
            altitude=self.network_config.altitude,
            inclination=self.network_config.inclination,
            num_orbits=self.network_config.num_orbits,
            num_sats_per_orbit=self.network_config.num_sats_per_orbit,
            phasing=self.network_config.phasing,
            min_elevation_angle_deg=self.network_config.min_elevation_angle_deg,
            link_buffer_size=self.network_config.link_buffer_size,
            isl_data_rate=self.network_config.isl_data_rate,
        )

    def _refresh_satellite_index_cache(self):
        self._sat_ids = np.asarray(self.network.satellite_ids, dtype=np.int64)
        self._sat_id_to_col_array = np.full(self.network.num_nodes, -1, dtype=np.int64)
        if len(self._sat_ids) > 0:
            self._sat_id_to_col_array[self._sat_ids] = np.arange(len(self._sat_ids), dtype=np.int64)

    def _set_seed(self, seed):
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

    def _build_region_distance_matrix(self) -> np.ndarray:
        regions = self.traffic_model.regions
        distances = np.zeros((len(regions), len(regions)), dtype=np.float64)
        for i, source in enumerate(regions):
            for j, target in enumerate(regions):
                if i != j:
                    distances[i, j] = source.distance_to(target)
        return distances

    def reset(self, seed=None, start_time=None, options: dict | None = None, include_spf_table: bool | None = None):
        """
        Reset the multi-agent environment and return the first decision batch.

        Returns:
            A tuple of (RoutingBatch, info). Each RoutingBatch row is one
            satellite-agent decision for one flowlet.
        """
        self._set_seed(seed)
        options = {} if options is None else options
        if include_spf_table is None:
            include_spf_table = bool(options.get("include_spf_table", False))

        self.network = self._create_network()

        self.topology_update_steps = 0

        self.next_flowlet_id = 0
        self._array_metrics = None
        self._flowlets = None
        self._links = None
        self._pending_routing_batch = None
        self._arrival_events = {}
        self._release_events = {}
        self._arrival_event_heap = []
        self._release_event_heap = []
        self._on_link_id_chunks = []
        self._creation_event_slots = np.empty(0, dtype=np.int64)
        self._creation_event_cursor = 0
        self._episode_started = False
        self._episode_done = False

        if start_time is None:
            self.start_time = 0
        else:
            self.start_time = start_time
        self.current_time = self.start_time
        self.network.update_topology(self.start_time, None)
        self._refresh_satellite_index_cache()
        self._update_region_access_cache()
        self._start_episode(include_spf_table=include_spf_table)
        return self._pending_routing_batch, self._build_step_info()

    def _start_episode(self, include_spf_table: bool = False):
        self._include_spf_table = bool(include_spf_table)
        self._build_link_array_cache()
        self._generate_flowlet_state()

        self._num_steps = int(np.ceil(self.time_limit / self.slot_ms))
        self._step_index = 0
        self._end_time = self.start_time + self.time_limit
        self._next_topology_time = self.start_time + self.update_interval_ms
        self._wall_start_time = time.perf_counter()
        self._progress_interval_ms = max(self.progress_interval_seconds * 1000.0, 0.0)
        self._next_progress_time = self.start_time
        self._episode_started = True
        self._episode_done = False

        if self.verbose:
            print("Env is in slot-array evaluation mode", flush=True)
            print(
                f"Generated {self._flowlets.count:,} flowlets over {self._num_steps:,} slots "
                f"({self.time_limit / 1000.0:.1f}s simulated)",
                flush=True,
            )
        self._pending_routing_batch = self._prepare_step_observation()

    def step(self, action=None):
        """
        Advance one simulation slot.

        Args:
            action: Either a row-aligned action-index array with one entry per
                RoutingBatch row, a MARL dict keyed by satellite-agent id, or a
                RoutingDecision for agent adapters that already return next-hop
                satellite ids.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if not self._episode_started:
            raise RuntimeError("Call reset() before step().")
        if self._episode_done:
            return (
                self._pending_routing_batch or self._empty_routing_batch(),
                np.empty(0, dtype=np.float32),
                True,
                False,
                self._build_step_info(),
            )

        batch = self._pending_routing_batch or self._empty_routing_batch()
        act = self._normalize_step_action(action, len(batch.flowlet_ids))
        reward = np.zeros(len(batch.flowlet_ids), dtype=np.float32)
        if len(batch.flowlet_ids) > 0:
            self._apply_routing_actions(batch, act)

        self._step_index += 1
        if self._step_index >= self._num_steps:
            self.current_time = self._end_time
            self._episode_done = True
            self._pending_routing_batch = self._empty_routing_batch()
            if self.verbose and self._progress_interval_ms > 0:
                self._print_progress(
                    step=self._num_steps,
                    num_steps=self._num_steps,
                    wall_start_time=self._wall_start_time,
            )
            self._array_metrics = self._calc_array_metrics()
            return self._pending_routing_batch, reward, True, False, self._build_step_info()

        self._pending_routing_batch = self._prepare_step_observation()
        return self._pending_routing_batch, reward, False, False, self._build_step_info()

    @property
    def observation(self) -> RoutingBatch:
        return self._pending_routing_batch or self._empty_routing_batch()

    @property
    def flowlets(self) -> FlowletState | None:
        return self._flowlets

    @property
    def links(self) -> LinkState | None:
        return self._links

    @property
    def terminated(self) -> bool:
        return self._episode_done

    def _build_link_array_cache(self):
        self._links = create_link_state(
            source_ids=self.network._link_source_ids,
            sink_ids=self.network._link_sink_ids,
            data_rate=self.network._link_data_rate_array,
            capacity=self.network._link_capacity_array,
            connected=self.network._link_connected_array,
            delay=self.network._link_delay_array,
            num_nodes=self.network.num_nodes,
            neighbor_sat_ids=self.network.neighbor_sat_ids,
        )

        self._refresh_link_state_arrays()

    def _refresh_link_state_arrays(self):
        if self._links is None:
            raise RuntimeError("Link state has not been initialized.")
        connected = getattr(self.network, "_link_connected_array", None)
        delay = getattr(self.network, "_link_delay_array", None)
        if connected is not None and delay is not None and len(connected) == self._links.count:
            self._links.connected = connected
            self._links.delay = delay
            return

        self._links.connected = self.network._link_connected_array
        self._links.delay = self.network._link_delay_array

    def _generate_flowlet_state(self):
        num_slots = int(np.ceil(self.time_limit / self.slot_ms))
        expected_flowlets_per_slot = (
            self.packet_rate_per_ms * self.slot_ms / max(self.mean_packets_per_flowlet, 1.0)
        )
        slot_counts = self.np_random.poisson(lam=expected_flowlets_per_slot, size=num_slots)
        num_flowlets = int(slot_counts.sum())
        self.next_flowlet_id = num_flowlets

        source_region_ids = self.traffic_model.sample_source_ids(self.np_random, num_flowlets).astype(np.int64)
        target_region_ids = self.traffic_model.sample_target_ids(self.np_random, source_region_ids).astype(np.int64)
        is_normal = self.np_random.uniform(size=num_flowlets) < self.prob_normal_packet
        packet_size = np.where(is_normal, self.normal_packet_size, self.small_packet_size).astype(np.float64)
        packet_count = np.maximum(
            1,
            self.np_random.poisson(lam=self.mean_packets_per_flowlet, size=num_flowlets),
        ).astype(np.int64)

        self._flowlets = create_flowlet_state(
            slot_counts=slot_counts,
            source_region_ids=source_region_ids,
            target_region_ids=target_region_ids,
            is_normal=is_normal,
            packet_size=packet_size,
            packet_count=packet_count,
            default_ttl=self.default_ttl,
            start_time=self.start_time,
            slot_ms=self.slot_ms,
        )
        self._creation_event_slots = np.flatnonzero(slot_counts > 0).astype(np.int64, copy=False)
        self._creation_event_cursor = 0

    def _release_transmitted_flowlets(self):
        if self._flowlets is None or self._links is None:
            return
        release_ids = self._pop_event_ids(self._release_events, self._step_index)
        if len(release_ids) == 0:
            return
        release_transmitted_flowlet_ids(
            flowlets=self._flowlets,
            links=self._links,
            flowlet_ids=release_ids,
            current_time=self.current_time,
        )

    def _handle_flowlet_arrival_ids(self) -> np.ndarray:
        if self._flowlets is None:
            return np.empty(0, dtype=np.int64)
        arrival_ids = self._pop_event_ids(self._arrival_events, self._step_index)
        if len(arrival_ids) == 0:
            return arrival_ids
        was_on_link = self._flowlets.status[arrival_ids] == FLOWLET_ON_LINK
        route_ready_ids = handle_flowlet_arrival_ids(
            flowlets=self._flowlets,
            flowlet_ids=arrival_ids,
            current_time=self.current_time,
            ttl_expired_reason=int(NetworkError.TTL_EXPIRED),
        )
        ttl_dropped = arrival_ids[
            was_on_link
            & (self._flowlets.status[arrival_ids] == FLOWLET_DROPPED)
            & (self._flowlets.drop_reason[arrival_ids] == int(NetworkError.TTL_EXPIRED))
        ]
        self._refresh_flowlet_remaining_gcd(ttl_dropped)
        return route_ready_ids

    def _activate_flowlets_at_current_slot_ids(self) -> np.ndarray:
        if self._flowlets is None:
            return np.empty(0, dtype=np.int64)

        slot_idx = int(round((self.current_time - self.start_time) / self.slot_ms))
        return activate_flowlets_at_slot_ids(
            flowlets=self._flowlets,
            slot_idx=slot_idx,
            current_time=self.current_time,
            nearest_region_sat_ids=self._nearest_region_sat_ids,
            nearest_region_sat_distances=self._nearest_region_sat_distances,
            region_distance_matrix=self._region_distance_matrix,
            access_data_rate=self.access_data_rate,
            no_available_sat_reason=int(NetworkError.NO_AVAILABLE_SAT),
        )

    def _prepare_step_observation(self) -> RoutingBatch:
        if self._flowlets is None or self._links is None:
            return self._empty_routing_batch()
        while self._step_index < self._num_steps:
            self.current_time = self.start_time + self._step_index * self.slot_ms
            if self.current_time >= self._end_time:
                self._episode_done = True
                return self._empty_routing_batch()

            if self.current_time >= self._next_topology_time:
                while self.current_time >= self._next_topology_time:
                    self._next_topology_time += self.update_interval_ms
                self.network.update_topology(self.current_time, None)
                self.topology_update_steps += 1
                self._update_region_access_cache()
                self._refresh_link_state_arrays()
                self._drop_flowlets_on_disconnected_links()

            self._release_transmitted_flowlets()
            arrival_ids = self._handle_flowlet_arrival_ids()
            activated_ids = self._activate_flowlets_at_current_slot_ids()
            if len(arrival_ids) == 0:
                route_ready_ids = activated_ids
            elif len(activated_ids) == 0:
                route_ready_ids = arrival_ids
            else:
                route_ready_ids = np.concatenate((arrival_ids, activated_ids))

            batch = self._build_step_routing_batch(route_ready_ids)
            self._maybe_print_progress()
            if batch.decision_count > 0:
                return batch

            next_step = self._next_activity_step(self._step_index + 1)
            if next_step <= self._step_index:
                next_step = self._step_index + 1
            self._step_index = next_step

        self.current_time = self._end_time
        self._episode_done = True
        return self._empty_routing_batch()

    def _maybe_print_progress(self):
        if not (self.verbose and self._progress_interval_ms > 0 and self.current_time >= self._next_progress_time):
            return
        self._print_progress(
            step=self._step_index + 1,
            num_steps=self._num_steps,
            wall_start_time=self._wall_start_time,
        )
        while self._next_progress_time <= self.current_time:
            self._next_progress_time += self._progress_interval_ms

    def _next_activity_step(self, lower_bound: int) -> int:
        if lower_bound >= self._num_steps:
            return self._num_steps
        candidates = [self._num_steps]

        creation_step = self._peek_creation_step(lower_bound)
        if creation_step is not None:
            candidates.append(creation_step)

        arrival_step = self._peek_event_step(self._arrival_events, self._arrival_event_heap, lower_bound)
        if arrival_step is not None:
            candidates.append(arrival_step)

        release_step = self._peek_event_step(self._release_events, self._release_event_heap, lower_bound)
        if release_step is not None:
            candidates.append(release_step)

        if self._next_topology_time < self._end_time:
            candidates.append(self._slot_index_for_time(self._next_topology_time))

        if self.verbose and self._progress_interval_ms > 0 and self._next_progress_time < self._end_time:
            candidates.append(self._slot_index_for_time(self._next_progress_time))

        return max(lower_bound, min(candidates))

    def _build_step_routing_batch(self, at_node_ids: np.ndarray) -> RoutingBatch:
        if len(at_node_ids) == 0:
            return self._empty_routing_batch()

        self._refresh_flowlet_remaining_gcd(at_node_ids)
        delivered_ids, route_ids = self._partition_deliverable_flowlets(at_node_ids)
        if len(delivered_ids) > 0:
            self._deliver_flowlet_ids(delivered_ids)
        if len(route_ids) == 0:
            return self._empty_routing_batch()

        current_sats = self._flowlets.current_sat[route_ids]
        target_regions = self._flowlets.target_region_id[route_ids]
        candidate_local_mask, target_access_sats = prepare_route_candidate_mask(
            flowlets=self._flowlets,
            route_flowlet_ids=route_ids,
            nearest_region_sat_ids=self._nearest_region_sat_ids,
            current_time=self.current_time,
            no_available_sat_reason=int(NetworkError.NO_AVAILABLE_SAT),
        )
        candidate_ids = route_ids[candidate_local_mask]
        if len(candidate_ids) == 0:
            return self._empty_routing_batch()

        current_sats = current_sats[candidate_local_mask]
        target_regions = target_regions[candidate_local_mask]
        if self._include_spf_table and not self._region_next_hop_eager_ready:
            self._ensure_region_next_hops(target_regions)

        return self._build_routing_batch(
            flowlet_ids=candidate_ids,
            current_sats=current_sats,
            target_regions=target_regions,
            target_access_sats=target_access_sats[candidate_local_mask],
            include_spf_table=self._include_spf_table,
        )

    def _apply_routing_actions(self, batch: RoutingBatch, act: np.ndarray):
        candidate_ids = batch.flowlet_ids
        if len(act) != len(candidate_ids):
            raise ValueError(f"Expected {len(candidate_ids)} actions, got {len(act)}.")
        if len(candidate_ids) == 0:
            return

        routable_local_mask = apply_no_route_mask(
            flowlets=self._flowlets,
            candidate_ids=candidate_ids,
            act=act,
            current_time=self.current_time,
            failed_to_find_next_hop_reason=int(NetworkError.FAILED_TO_FIND_NEXT_HOP),
        )
        routable_ids = candidate_ids[routable_local_mask]
        if len(routable_ids) == 0:
            return

        scheduled_ids, link_ids, scheduled_act = prepare_link_schedule_inputs(
            flowlets=self._flowlets,
            links=self._links,
            routable_ids=routable_ids,
            current_sats=batch.current_sat_ids[routable_local_mask],
            act=act[routable_local_mask],
            neighbor_sat_ids_by_node=self.network.neighbor_sat_ids,
            current_time=self.current_time,
            invalid_next_hop_reason=int(NetworkError.INVALID_NEXT_HOP),
        )
        if len(scheduled_ids) == 0:
            return

        accepted_ids, rejected_ids = self._schedule_flowlets_by_link(
            flowlet_ids=scheduled_ids,
            link_ids=link_ids,
            act=scheduled_act,
        )
        if len(accepted_ids) > 0:
            self._schedule_flowlet_events(accepted_ids)
        if len(rejected_ids) > 0:
            self._drop_flowlet_ids(rejected_ids, NetworkError.LINK_FULL)

    def _normalize_step_action(self, action, expected_count: int) -> np.ndarray:
        if action is None:
            if expected_count == 0:
                return np.empty(0, dtype=np.int64)
            raise ValueError(f"Expected {expected_count} actions, got None.")
        if isinstance(action, RoutingDecision):
            return np.asarray(action.next_hop_sat_ids, dtype=np.int64)
        if isinstance(action, Mapping):
            action = self._action_dict_to_row_actions(action, expected_count)

        actions = np.asarray(action, dtype=np.int64)
        if actions.ndim != 1:
            raise ValueError(f"Actions must be a 1-D array, got shape {actions.shape}.")
        if len(actions) != expected_count:
            raise ValueError(f"Expected {expected_count} actions, got {len(actions)}.")
        if expected_count == 0:
            return np.empty(0, dtype=np.int64)

        batch = self.observation
        rows = np.arange(expected_count)
        valid = (actions >= 0) & (actions < ACTION_COUNT)
        act = np.full(expected_count, -1, dtype=np.int64)
        if valid.any():
            act[valid] = batch.neighbor_sat_ids[rows[valid], actions[valid]]
        return act

    def _action_dict_to_row_actions(self, action_by_agent: Mapping, expected_count: int) -> np.ndarray:
        if expected_count == 0:
            return np.empty(0, dtype=np.int64)

        batch = self.observation
        actions = np.full(expected_count, -1, dtype=np.int64)
        unique_agents = np.unique(batch.agent_ids)
        for agent_id in unique_agents:
            agent_key = int(agent_id)
            if agent_key not in action_by_agent:
                raise ValueError(f"Missing action for active satellite agent {agent_key}.")

            rows = np.flatnonzero(batch.agent_ids == agent_id)
            value = np.asarray(action_by_agent[agent_key], dtype=np.int64)
            if value.ndim == 0:
                actions[rows] = int(value)
            elif value.ndim == 1 and len(value) == len(rows):
                actions[rows] = value
            else:
                raise ValueError(
                    f"Action for satellite agent {agent_key} must be scalar or length {len(rows)}, "
                    f"got shape {value.shape}."
                )
        return actions

    def _empty_routing_batch(self) -> RoutingBatch:
        neighbor_i = np.empty((0, ACTION_COUNT), dtype=np.int64)
        neighbor_f = np.empty((0, ACTION_COUNT), dtype=np.float64)
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
            region_next_hop_table=self._region_next_hop_table if self._include_spf_table else None,
            region_next_hop_version=self._region_next_hop_version if self._include_spf_table else 0,
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
        )

    def _build_step_info(self) -> dict:
        batch = self._pending_routing_batch
        decision_count = 0 if batch is None else batch.decision_count
        active_agent_count = 0 if batch is None else len(batch.active_agent_ids)
        return {
            "time_ms": self.current_time,
            "step": self._step_index,
            "num_steps": self._num_steps,
            "progress": self._step_index / max(self._num_steps, 1),
            "decision_count": decision_count,
            "active_agent_count": active_agent_count,
            "route_count": decision_count,
            "topology_updates": self.topology_update_steps,
            "terminated": self._episode_done,
        }

    def _partition_deliverable_flowlets(self, flowlet_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return partition_deliverable_flowlets(
            flowlets=self._flowlets,
            flowlet_ids=flowlet_ids,
            sat_id_to_col_array=self._sat_id_to_col_array,
            region_sat_visible=self._region_sat_visible,
        )

    def _deliver_flowlet_ids(self, flowlet_ids: np.ndarray):
        deliver_flowlet_ids(
            flowlets=self._flowlets,
            flowlet_ids=flowlet_ids,
            current_time=self.current_time,
            sat_id_to_col_array=self._sat_id_to_col_array,
            region_sat_distance2=self._region_sat_distance2,
            access_data_rate=self.access_data_rate,
        )

    def _build_routing_batch(
        self,
        flowlet_ids: np.ndarray,
        current_sats: np.ndarray,
        target_regions: np.ndarray,
        target_access_sats: np.ndarray,
        include_spf_table: bool,
    ):
        batch = build_routing_batch(
            flowlets=self._flowlets,
            links=self._links,
            flowlet_ids=flowlet_ids,
            current_sats=current_sats,
            target_regions=target_regions,
            target_access_sats=target_access_sats,
            neighbor_sat_ids_by_node=self.network.neighbor_sat_ids,
            current_time=self.current_time,
            region_next_hop_table=self._region_next_hop_table if include_spf_table else None,
            region_next_hop_version=self._region_next_hop_version if include_spf_table else 0,
        )
        batch.remaining_gcd = self._remaining_target_distances(current_sats, target_regions)
        batch.shortest_gcd = self._flowlets.shortest_gcd[flowlet_ids].copy()
        self._apply_legacy_action_mask(batch)
        batch.observations = self._build_legacy_observations(batch)
        return batch

    def _remaining_target_distances(self, current_sats: np.ndarray, target_regions: np.ndarray) -> np.ndarray:
        if len(current_sats) == 0:
            return np.empty(0, dtype=np.float64)
        sat_lon, sat_lat = self._satellite_projected_lon_lat(current_sats)
        target_lon = self._region_longitudes[target_regions]
        target_lat = self._region_latitudes[target_regions]
        return self._great_circle_distance_deg(sat_lon, sat_lat, target_lon, target_lat)

    def _refresh_flowlet_remaining_gcd(self, flowlet_ids: np.ndarray) -> None:
        if self._flowlets is None or len(flowlet_ids) == 0:
            return
        valid = self._flowlets.current_sat[flowlet_ids] >= 0
        if not valid.any():
            return
        ids = flowlet_ids[valid]
        self._flowlets.remaining_gcd[ids] = self._remaining_target_distances(
            self._flowlets.current_sat[ids],
            self._flowlets.target_region_id[ids],
        )

    def _apply_legacy_action_mask(self, batch: RoutingBatch) -> None:
        if batch.decision_count == 0:
            return
        deliverable = self._neighbor_deliverable_to_target(batch.target_region_ids, batch.neighbor_sat_ids)
        last_node1 = self._optional_int(batch.last_node1, batch.decision_count)
        loopback = batch.neighbor_sat_ids == last_node1[:, None]
        batch.action_mask &= ~(loopback & ~deliverable)

    def _build_legacy_observations(self, batch: RoutingBatch) -> np.ndarray:
        n = batch.decision_count
        obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        if n == 0:
            return obs

        current_pos = self.network._satellite_positions_by_id[batch.current_sat_ids] / self.network.orbit_radius
        target_pos = self._region_positions[batch.target_region_ids] / self.network.orbit_radius
        relative_pos = current_pos - target_pos
        relative_distance = np.linalg.norm(relative_pos, axis=1)

        orbit_cycle_ms = max(float(self.network.orbit_cycle * 1000.0), 1e-9)
        time_prog = np.full(n, (self.current_time % orbit_cycle_ms) / orbit_cycle_ms, dtype=np.float64)
        initial_gcd = np.maximum(self._optional_float(batch.initial_gcd, n), 1e-6)
        remaining_gcd = self._optional_float(batch.remaining_gcd, n)
        current_progress = remaining_gcd / initial_gcd

        current_load, current_remaining = self._node_queue_features(batch.current_sat_ids)
        total_delay = (
            self._optional_float(batch.queue_delay, n)
            + self._optional_float(batch.transmission_delay, n)
            + self._optional_float(batch.propagation_delay, n)
        )
        queue_delay = self._optional_float(batch.queue_delay, n)
        ttl = batch.ttl.astype(np.float64, copy=False)
        last_action1 = self._optional_int(batch.last_action1, n).astype(np.float64)
        last_action2 = self._optional_int(batch.last_action2, n).astype(np.float64)
        last_node1 = self._optional_int(batch.last_node1, n).astype(np.float64)
        last_node2 = self._optional_int(batch.last_node2, n).astype(np.float64)

        obs[:, 0:26] = np.column_stack(
            (
                time_prog,
                current_pos[:, 0],
                current_pos[:, 1],
                current_pos[:, 2],
                target_pos[:, 0],
                target_pos[:, 1],
                target_pos[:, 2],
                relative_pos[:, 0],
                relative_pos[:, 1],
                relative_pos[:, 2],
                relative_distance,
                current_progress,
                current_load,
                current_remaining,
                (self.current_time - batch.creation_time) / self.delay_norm,
                batch.is_normal.astype(np.float64, copy=False),
                batch.flowlet_size,
                ttl,
                float(self.default_ttl) - ttl,
                ttl / max(float(self.default_ttl), 1e-6),
                total_delay / self.delay_norm,
                queue_delay / self.delay_norm,
                last_action1,
                last_node1,
                last_action2,
                last_node2,
            )
        )

        neighbor_ids = batch.neighbor_sat_ids
        valid_neighbor = neighbor_ids >= 0
        safe_neighbor_ids = np.where(valid_neighbor, neighbor_ids, 0)
        neighbor_pos = self.network._satellite_positions_by_id[safe_neighbor_ids] / self.network.orbit_radius
        neighbor_pos = np.where(valid_neighbor[:, :, None], neighbor_pos, 0.0)
        neighbor_relative = neighbor_pos - target_pos[:, None, :]
        neighbor_relative_distance = np.linalg.norm(neighbor_relative, axis=2)

        neighbor_gcd = np.zeros((n, ACTION_COUNT), dtype=np.float64)
        if valid_neighbor.any():
            flat_targets = np.broadcast_to(batch.target_region_ids[:, None], neighbor_ids.shape)
            neighbor_gcd[valid_neighbor] = self._remaining_target_distances(
                neighbor_ids[valid_neighbor],
                flat_targets[valid_neighbor],
            )
        neighbor_progress = neighbor_gcd / initial_gcd[:, None]

        valid_link = batch.neighbor_link_ids >= 0
        safe_link_ids = np.where(valid_link, batch.neighbor_link_ids, 0)
        link_remaining = np.where(
            valid_link,
            np.maximum(batch.neighbor_link_capacity - batch.neighbor_queue_load, 0.0),
            0.0,
        )
        link_data_rate = np.where(valid_link, self._links.data_rate[safe_link_ids], 1.0)
        normalized_queue_delay = np.where(
            valid_link,
            np.maximum(batch.neighbor_link_free_time - self.current_time, 0.0) / self.delay_norm,
            0.0,
        )
        normalized_transmit_time = batch.flowlet_size[:, None] / np.maximum(link_data_rate, 1e-9) / self.delay_norm
        normalized_propagation_delay = np.where(
            valid_link,
            batch.neighbor_link_delay / self.delay_norm,
            0.0,
        )
        sink_load, sink_remaining = self._node_queue_features(safe_neighbor_ids.reshape(-1))
        sink_load = sink_load.reshape(n, ACTION_COUNT)
        sink_remaining = sink_remaining.reshape(n, ACTION_COUNT)
        has_enough_capacity = (link_remaining >= batch.flowlet_size[:, None]).astype(np.float64)
        target_access = self._neighbor_deliverable_to_target(batch.target_region_ids, neighbor_ids).astype(np.float64)
        looped = (
            (neighbor_ids == self._optional_int(batch.last_node1, n)[:, None])
            | (neighbor_ids == self._optional_int(batch.last_node2, n)[:, None])
        ).astype(np.float64)

        cursor = 26
        for action_idx in range(ACTION_COUNT):
            obs[:, cursor : cursor + 17] = np.column_stack(
                (
                    neighbor_pos[:, action_idx, 0],
                    neighbor_pos[:, action_idx, 1],
                    neighbor_pos[:, action_idx, 2],
                    neighbor_relative[:, action_idx, 0],
                    neighbor_relative[:, action_idx, 1],
                    neighbor_relative[:, action_idx, 2],
                    neighbor_relative_distance[:, action_idx],
                    neighbor_progress[:, action_idx],
                    normalized_queue_delay[:, action_idx],
                    normalized_transmit_time[:, action_idx],
                    normalized_propagation_delay[:, action_idx],
                    sink_load[:, action_idx],
                    sink_remaining[:, action_idx],
                    link_remaining[:, action_idx],
                    has_enough_capacity[:, action_idx],
                    looped[:, action_idx],
                    target_access[:, action_idx],
                )
            )
            cursor += 17

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _neighbor_deliverable_to_target(self, target_regions: np.ndarray, neighbor_sat_ids: np.ndarray) -> np.ndarray:
        deliverable = np.zeros(neighbor_sat_ids.shape, dtype=bool)
        valid_neighbor = neighbor_sat_ids >= 0
        if not valid_neighbor.any():
            return deliverable
        safe_neighbor = np.where(valid_neighbor, neighbor_sat_ids, 0)
        cols = self._sat_id_to_col_array[safe_neighbor]
        valid = valid_neighbor & (cols >= 0)
        if valid.any():
            target_matrix = np.broadcast_to(target_regions[:, None], neighbor_sat_ids.shape)
            deliverable[valid] = self._region_sat_visible[target_matrix[valid], cols[valid]]
        return deliverable

    def _node_queue_features(self, sat_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sat_ids = np.asarray(sat_ids, dtype=np.int64)
        if len(sat_ids) == 0 or self._links is None:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        neighbor_link_ids = self._links.neighbor_link_ids[sat_ids]
        valid = neighbor_link_ids >= 0
        safe_link_ids = np.where(valid, neighbor_link_ids, 0)
        queue = np.where(valid, self._links.queue_load[safe_link_ids], 0.0)
        capacity = np.where(valid, self._links.capacity[safe_link_ids], 0.0)
        total_capacity = capacity.sum(axis=1)
        total_queue = queue.sum(axis=1)
        load = np.divide(total_queue, total_capacity, out=np.zeros_like(total_queue), where=total_capacity > 0)
        remaining = np.maximum(total_capacity - total_queue, 0.0)
        return load, remaining

    def _satellite_projected_lon_lat(self, sat_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = self.network._satellite_positions_by_id[sat_ids]
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
        return lon, lat

    @staticmethod
    def _great_circle_distance_deg(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
        lon1_rad = np.radians(lon1)
        lat1_rad = np.radians(lat1)
        lon2_rad = np.radians(lon2)
        lat2_rad = np.radians(lat2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))

    @staticmethod
    def _optional_float(values: np.ndarray | None, length: int) -> np.ndarray:
        if values is None:
            return np.zeros(length, dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _optional_int(values: np.ndarray | None, length: int) -> np.ndarray:
        if values is None:
            return np.full(length, -1, dtype=np.int64)
        return np.asarray(values, dtype=np.int64)

    def _event_slots_for_time(self, timestamps: np.ndarray) -> np.ndarray:
        slots = np.ceil((timestamps - self.start_time) / self.slot_ms - 1e-9).astype(np.int64)
        return np.clip(slots, 0, self._num_steps)

    def _slot_index_for_time(self, timestamp: float) -> int:
        slot = int(np.ceil((timestamp - self.start_time) / self.slot_ms - 1e-9))
        return min(max(slot, 0), self._num_steps)

    def _add_event_ids(
        self,
        events: dict[int, list[np.ndarray]],
        event_heap: list[int],
        event_slots: np.ndarray,
        flowlet_ids: np.ndarray,
    ) -> None:
        if len(flowlet_ids) == 0:
            return
        valid = (event_slots >= self._step_index) & (event_slots < self._num_steps)
        if not valid.any():
            return
        slots = event_slots[valid]
        ids = flowlet_ids[valid]
        order = np.argsort(slots, kind="stable")
        sorted_slots = slots[order]
        sorted_ids = ids[order]
        starts = np.r_[0, np.flatnonzero(sorted_slots[1:] != sorted_slots[:-1]) + 1]
        ends = np.r_[starts[1:], len(sorted_slots)]
        for start, end in zip(starts, ends):
            slot = int(sorted_slots[start])
            if slot not in events:
                events[slot] = []
                heapq.heappush(event_heap, slot)
            events[slot].append(sorted_ids[start:end].copy())

    @staticmethod
    def _pop_event_ids(events: dict[int, list[np.ndarray]], step_index: int) -> np.ndarray:
        batches = events.pop(step_index, None)
        if not batches:
            return np.empty(0, dtype=np.int64)
        if len(batches) == 1:
            return batches[0]
        return np.concatenate(batches)

    def _peek_creation_step(self, lower_bound: int) -> int | None:
        slots = self._creation_event_slots
        cursor = self._creation_event_cursor
        while cursor < len(slots) and int(slots[cursor]) < lower_bound:
            cursor += 1
        self._creation_event_cursor = cursor
        if cursor >= len(slots):
            return None
        return int(slots[cursor])

    @staticmethod
    def _peek_event_step(
        events: dict[int, list[np.ndarray]],
        event_heap: list[int],
        lower_bound: int,
    ) -> int | None:
        while event_heap and (event_heap[0] not in events or event_heap[0] < lower_bound):
            heapq.heappop(event_heap)
        if not event_heap:
            return None
        return int(event_heap[0])

    def _schedule_flowlet_events(self, accepted_ids: np.ndarray) -> None:
        if self._flowlets is None or len(accepted_ids) == 0:
            return
        self._add_event_ids(
            events=self._release_events,
            event_heap=self._release_event_heap,
            event_slots=self._event_slots_for_time(self._flowlets.transmit_end_time[accepted_ids]),
            flowlet_ids=accepted_ids,
        )
        self._add_event_ids(
            events=self._arrival_events,
            event_heap=self._arrival_event_heap,
            event_slots=self._event_slots_for_time(self._flowlets.arrival_time[accepted_ids]),
            flowlet_ids=accepted_ids,
        )
        self._on_link_id_chunks.append(accepted_ids.copy())

    def _compact_on_link_ids(self) -> np.ndarray:
        if self._flowlets is None or not self._on_link_id_chunks:
            return np.empty(0, dtype=np.int64)
        if len(self._on_link_id_chunks) == 1:
            candidate_ids = self._on_link_id_chunks[0]
        else:
            candidate_ids = np.concatenate(self._on_link_id_chunks)
        if len(candidate_ids) == 0:
            self._on_link_id_chunks = []
            return candidate_ids
        active_ids = candidate_ids[self._flowlets.status[candidate_ids] == FLOWLET_ON_LINK]
        self._on_link_id_chunks = [active_ids.copy()] if len(active_ids) > 0 else []
        return active_ids

    def _schedule_flowlets_by_link(
        self,
        flowlet_ids: np.ndarray,
        link_ids: np.ndarray,
        act: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return schedule_flowlets_by_link(
            flowlets=self._flowlets,
            links=self._links,
            flowlet_ids=flowlet_ids,
            link_ids=link_ids,
            act=act,
            current_time=self.current_time,
        )

    def _ensure_region_next_hops(self, target_regions: np.ndarray):
        missing_mask = ~self._region_next_hop_ready[target_regions]
        if not missing_mask.any():
            return

        missing_regions = np.unique(target_regions[missing_mask])
        if len(missing_regions) == 0:
            return

        access_sat_ids = self._nearest_region_sat_ids[missing_regions]
        valid = access_sat_ids >= 0
        if not valid.any():
            return

        missing_regions = missing_regions[valid]
        access_sat_ids = access_sat_ids[valid]
        self.network.precompute_shortest_next_hops(access_sat_ids)
        self._region_next_hop_table[missing_regions] = self.network.shortest_next_hop_rows(access_sat_ids)
        self._region_next_hop_ready[missing_regions] = True
        self._region_next_hop_version += 1

    def _drop_flowlets_on_disconnected_links(self):
        if self._flowlets is None or self._links is None:
            return
        on_link = self._compact_on_link_ids()
        if len(on_link) == 0:
            return
        link_ids = self._flowlets.link_id[on_link]
        dropped = on_link[~self._links.connected[link_ids]]
        if len(dropped) == 0:
            return
        unreleased = dropped[~self._flowlets.link_released[dropped]]
        if len(unreleased) > 0:
            np.add.at(self._links.queue_load, self._flowlets.link_id[unreleased], -self._flowlets.size[unreleased])
            self._flowlets.link_released[unreleased] = True
        self._links.queue_load[~self._links.connected] = 0.0
        self._links.free_time[~self._links.connected] = self.current_time
        drop_flowlet_ids(
            flowlets=self._flowlets,
            flowlet_ids=dropped,
            current_time=self.current_time,
            reason=int(NetworkError.LINK_DISCONNECTED),
        )
        self._compact_on_link_ids()

    def _drop_flowlet_ids(self, flowlet_ids: np.ndarray, reason: NetworkError):
        if self._flowlets is None:
            return
        drop_flowlet_ids(
            flowlets=self._flowlets,
            flowlet_ids=flowlet_ids,
            current_time=self.current_time,
            reason=int(reason),
        )

    def _calc_array_metrics(self) -> Metrics:
        flowlets = self._flowlets
        if flowlets is None:
            return Metrics()
        if len(flowlets.status) == 0:
            return Metrics()

        generated_mask = flowlets.status != FLOWLET_NOT_STARTED
        delivered_mask = flowlets.status == FLOWLET_DELIVERED
        dropped_mask = flowlets.status == FLOWLET_DROPPED
        normal_mask = flowlets.is_normal
        small_mask = ~normal_mask
        weights = flowlets.packet_count

        delivered_delay = (
            flowlets.queue_delay + flowlets.transmission_delay + flowlets.propagation_delay
        )

        def weighted_sum(mask, values=None):
            if values is None:
                return int(weights[mask].sum())
            return float((values[mask] * weights[mask]).sum())

        generated = weighted_sum(generated_mask)
        delivered = weighted_sum(delivered_mask)
        dropped = weighted_sum(dropped_mask)
        delivered_normal = weighted_sum(delivered_mask & normal_mask)
        delivered_small = weighted_sum(delivered_mask & small_mask)
        generated_normal = weighted_sum(generated_mask & normal_mask)
        generated_small = weighted_sum(generated_mask & small_mask)
        dropped_normal = weighted_sum(dropped_mask & normal_mask)
        dropped_small = weighted_sum(dropped_mask & small_mask)
        ttl_dropped = weighted_sum(dropped_mask & (flowlets.drop_reason == int(NetworkError.TTL_EXPIRED)))

        total_delay = weighted_sum(delivered_mask, delivered_delay)
        queue_delay = weighted_sum(delivered_mask, flowlets.queue_delay)
        transmission_delay = weighted_sum(delivered_mask, flowlets.transmission_delay)
        propagation_delay = weighted_sum(delivered_mask, flowlets.propagation_delay)
        normal_delay = weighted_sum(delivered_mask & normal_mask, delivered_delay)
        normal_queue = weighted_sum(delivered_mask & normal_mask, flowlets.queue_delay)
        normal_tx = weighted_sum(delivered_mask & normal_mask, flowlets.transmission_delay)
        normal_prop = weighted_sum(delivered_mask & normal_mask, flowlets.propagation_delay)
        small_delay = weighted_sum(delivered_mask & small_mask, delivered_delay)
        small_queue = weighted_sum(delivered_mask & small_mask, flowlets.queue_delay)
        small_tx = weighted_sum(delivered_mask & small_mask, flowlets.transmission_delay)
        small_prop = weighted_sum(delivered_mask & small_mask, flowlets.propagation_delay)
        cost = weighted_sum(delivered_mask, flowlets.total_queue_cost)
        normal_cost = weighted_sum(delivered_mask & normal_mask, flowlets.total_queue_cost)
        small_cost = weighted_sum(delivered_mask & small_mask, flowlets.total_queue_cost)

        elapsed_ms = min(max(float(self.current_time - self.start_time), 0.0), self.time_limit)
        elapsed_seconds = max(elapsed_ms / 1000.0, 1e-12)
        throughput = float(flowlets.size[delivered_mask].sum()) / elapsed_seconds

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
            throughput=throughput,
            service_rate=delivered / elapsed_seconds,
            delivery_rate=delivered / generated if generated else 0.0,
            drop_rate=dropped / generated if generated else 0.0,
            normal_packet_delivery_rate=delivered_normal / generated_normal if generated_normal else 0.0,
            normal_packet_drop_rate=dropped_normal / generated_normal if generated_normal else 0.0,
            small_packet_delivery_rate=delivered_small / generated_small if generated_small else 0.0,
            small_packet_drop_rate=dropped_small / generated_small if generated_small else 0.0,
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

    def calc_metrics(self) -> Metrics:
        if self._array_metrics is not None:
            return self._array_metrics
        if self._flowlets is not None:
            return self._calc_array_metrics()
        return Metrics()

    def _print_current_metrics(self):
        metrics = self.calc_metrics()
        self._print(metrics.get_summary() + " " * 4, end="\r")

    def _print_progress(self, step: int, num_steps: int, wall_start_time: float):
        flowlets = self._flowlets
        if flowlets is None:
            return

        status = flowlets.status
        not_started = int(np.count_nonzero(status == FLOWLET_NOT_STARTED))
        at_node = int(np.count_nonzero(status == FLOWLET_AT_NODE))
        on_link = int(np.count_nonzero(status == FLOWLET_ON_LINK))
        delivered = int(np.count_nonzero(status == FLOWLET_DELIVERED))
        dropped = int(np.count_nonzero(status == FLOWLET_DROPPED))
        generated = flowlets.count - not_started
        active = at_node + on_link

        sim_elapsed_ms = min(max(self.current_time - self.start_time, 0.0), self.time_limit)
        progress = sim_elapsed_ms / max(self.time_limit, 1e-12)
        wall_elapsed = time.perf_counter() - wall_start_time
        eta = wall_elapsed * (1.0 - progress) / progress if progress > 1e-9 else None
        eta_text = f"{eta:.1f}s" if eta is not None else "n/a"
        rss_mb = self._get_max_rss_mb()
        rss_text = f" max_rss={rss_mb:.0f}MB" if rss_mb is not None else ""

        self._print(
            f"progress {progress * 100.0:6.2f}% "
            f"step={min(step, num_steps):,}/{num_steps:,} "
            f"wall={wall_elapsed:.1f}s eta={eta_text} "
            f"flowlets={generated:,}/{flowlets.count:,} "
            f"active={active:,}(node={at_node:,},link={on_link:,}) "
            f"ok={delivered:,} drop={dropped:,} "
            f"topo={self.topology_update_steps}{rss_text}"
        )

    @staticmethod
    def _get_max_rss_mb() -> float | None:
        try:
            import resource
        except ImportError:
            return None

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rss <= 0:
            return None
        if rss > 10_000_000:
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0

    def _print(self, line: str, end=None):
        kwargs = {"flush": True}
        if end is not None:
            kwargs["end"] = end
        print(
            f"{ms2str(self.start_time)}+{ms2str(self.current_time-self.start_time)}: {line}",
            **kwargs,
        )

    def _update_region_access_cache(self):
        if len(self._sat_ids) == 0:
            num_regions = len(self.traffic_model.regions)
            self._region_sat_distance2 = np.empty((num_regions, 0))
            self._region_sat_visible = np.zeros((num_regions, 0), dtype=bool)
            self._nearest_region_sat_ids = np.full(num_regions, -1, dtype=np.int64)
            self._nearest_region_sat_distances = np.full(num_regions, np.inf)
            self._region_next_hop_table = np.empty((num_regions, 0), dtype=np.int64)
            self._region_next_hop_ready = np.zeros(num_regions, dtype=bool)
            self._region_next_hop_eager_ready = False
            self._region_next_hop_version += 1
            self._region_access_cache_time = self.current_time
            return

        sat_positions = self.network._satellite_positions_by_id[self._sat_ids]
        delta = self._region_positions[:, None, :] - sat_positions[None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        visible = distance2 <= self.network.max_access_range * self.network.max_access_range
        masked_distance2 = np.where(visible, distance2, np.inf)
        nearest_cols = np.argmin(masked_distance2, axis=1)
        nearest_distance2 = masked_distance2[np.arange(masked_distance2.shape[0]), nearest_cols]
        nearest_distances = np.sqrt(nearest_distance2)
        nearest_sat_ids = self._sat_ids[nearest_cols].copy()
        nearest_sat_ids[~np.isfinite(nearest_distances)] = -1

        self._region_sat_distance2 = distance2
        self._region_sat_visible = visible
        self._nearest_region_sat_ids = nearest_sat_ids
        self._nearest_region_sat_distances = nearest_distances
        self._region_access_cache_time = self.current_time

        table_shape = (len(self.traffic_model.regions), len(self._sat_id_to_col_array))
        if self._region_next_hop_table.shape != table_shape:
            self._region_next_hop_table = np.empty(table_shape, dtype=np.int64)
        self._region_next_hop_table.fill(-1)
        self._region_next_hop_version += 1
        if len(self._region_next_hop_ready) != len(self.traffic_model.regions):
            self._region_next_hop_ready = np.zeros(len(self.traffic_model.regions), dtype=bool)
        else:
            self._region_next_hop_ready.fill(False)
        self._region_next_hop_eager_ready = False
        if len(self.traffic_model.regions) <= self.eager_spf_region_threshold:
            valid_region_ids = np.flatnonzero(nearest_sat_ids >= 0)
            if len(valid_region_ids) > 0:
                self._ensure_region_next_hops(valid_region_ids)
            self._region_next_hop_eager_ready = True

    def get_flowlet_dataframe(self) -> pd.DataFrame:
        flowlets = self._flowlets
        if flowlets is None or flowlets.count == 0:
            return pd.DataFrame()

        status = flowlets.status
        queue_delay = flowlets.queue_delay
        transmission_delay = flowlets.transmission_delay
        propagation_delay = flowlets.propagation_delay
        total_delay = queue_delay + transmission_delay + propagation_delay
        return pd.DataFrame(
            {
                "flowlet_id": np.arange(len(status)),
                "source_id": flowlets.source_id,
                "source_region_id": flowlets.source_region_id,
                "target_region_id": flowlets.target_region_id,
                "packet_count": flowlets.packet_count,
                "packet_size": flowlets.packet_size,
                "is_normal_packet": flowlets.is_normal,
                "size": flowlets.size,
                "creation_time": flowlets.creation_time,
                "delivery_time": flowlets.delivery_time,
                "total_delay": np.where(status == FLOWLET_DELIVERED, total_delay, np.nan),
                "queue_delay": queue_delay,
                "transmission_delay": transmission_delay,
                "propagation_delay": propagation_delay,
                "hops": flowlets.hops,
                "ttl": flowlets.ttl,
                "ttl_max": self.default_ttl,
                "delivered": status == FLOWLET_DELIVERED,
                "dropped": status == FLOWLET_DROPPED,
                "drop_time": flowlets.drop_time,
                "drop_reason": flowlets.drop_reason,
                "total_queue_cost": flowlets.total_queue_cost,
                "first_access_delay": flowlets.first_access_delay,
                "final_access_delay": flowlets.final_access_delay,
            }
        )

    def save_flowlets_to_csv(self, file_path: str):
        self.get_flowlet_dataframe().to_csv(file_path, index=False)
