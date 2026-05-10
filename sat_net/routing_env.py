from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from sat_net.geometric import LIGHT_SPEED_MS
from sat_net.network import SatelliteNetwork
from sat_net.solver.base_solver import ACTION_COUNT, BaseSolver, RoutingBatch
from sat_net.stats import Metrics
from sat_net.traffic_region import TrafficRegionModel
from sat_net.util import NamedDict, NetworkError, ms2str


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FLOWLET_NOT_STARTED = 0
FLOWLET_AT_NODE = 1
FLOWLET_ON_LINK = 2
FLOWLET_DELIVERED = 3
FLOWLET_DROPPED = 4


@dataclass(slots=True)
class FlowletState:
    """Array state for all generated flowlets."""

    slot_offsets: np.ndarray
    status: np.ndarray
    creation_slot: np.ndarray
    creation_time: np.ndarray
    source_region_id: np.ndarray
    target_region_id: np.ndarray
    source_id: np.ndarray
    current_sat: np.ndarray
    next_sat: np.ndarray
    link_id: np.ndarray
    packet_count: np.ndarray
    packet_size: np.ndarray
    is_normal: np.ndarray
    size: np.ndarray
    ttl: np.ndarray
    hops: np.ndarray
    queue_delay: np.ndarray
    transmission_delay: np.ndarray
    propagation_delay: np.ndarray
    total_queue_cost: np.ndarray
    first_access_delay: np.ndarray
    final_access_delay: np.ndarray
    delivery_time: np.ndarray
    drop_time: np.ndarray
    drop_reason: np.ndarray
    transmit_end_time: np.ndarray
    arrival_time: np.ndarray
    link_released: np.ndarray
    scheduled_prop_delay: np.ndarray
    shortest_gcd: np.ndarray
    initial_gcd: np.ndarray
    last_node1: np.ndarray
    last_node2: np.ndarray

    @property
    def count(self) -> int:
        return len(self.status)

    @classmethod
    def empty(cls) -> "FlowletState":
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_f64 = np.empty(0, dtype=np.float64)
        return cls(
            slot_offsets=np.zeros(1, dtype=np.int64),
            status=np.empty(0, dtype=np.int8),
            creation_slot=empty_i64.copy(),
            creation_time=empty_f64.copy(),
            source_region_id=empty_i64.copy(),
            target_region_id=empty_i64.copy(),
            source_id=empty_i64.copy(),
            current_sat=empty_i64.copy(),
            next_sat=empty_i64.copy(),
            link_id=np.empty(0, dtype=np.int32),
            packet_count=empty_i64.copy(),
            packet_size=empty_f64.copy(),
            is_normal=np.empty(0, dtype=bool),
            size=empty_f64.copy(),
            ttl=np.empty(0, dtype=np.int16),
            hops=np.empty(0, dtype=np.int16),
            queue_delay=empty_f64.copy(),
            transmission_delay=empty_f64.copy(),
            propagation_delay=empty_f64.copy(),
            total_queue_cost=empty_f64.copy(),
            first_access_delay=empty_f64.copy(),
            final_access_delay=empty_f64.copy(),
            delivery_time=empty_f64.copy(),
            drop_time=empty_f64.copy(),
            drop_reason=np.empty(0, dtype=np.int16),
            transmit_end_time=empty_f64.copy(),
            arrival_time=empty_f64.copy(),
            link_released=np.empty(0, dtype=bool),
            scheduled_prop_delay=empty_f64.copy(),
            shortest_gcd=empty_f64.copy(),
            initial_gcd=empty_f64.copy(),
            last_node1=empty_i64.copy(),
            last_node2=empty_i64.copy(),
        )


@dataclass(slots=True)
class LinkState:
    """Runtime link arrays used by the slot scheduler."""

    source_ids: np.ndarray
    sink_ids: np.ndarray
    data_rate: np.ndarray
    capacity: np.ndarray
    connected: np.ndarray
    delay: np.ndarray
    id_by_pair: np.ndarray
    free_time: np.ndarray
    queue_load: np.ndarray

    @property
    def count(self) -> int:
        return len(self.source_ids)


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

        self.start_time = 0.0  # the start timestamp of the simulation. randomize this to init the topology
        self.current_time = 0.0  #  # the current time offset, in milliseconds
        self.topology_update_steps = 0
        self.time_limit = float(self.config.time_limit_seconds * 1000.0)

        self.action_dim = ACTION_COUNT  # N, E, S, W - fixed for satellite routing
        self.obs_dim = 94

        self.current_solver: Optional["BaseSolver"] = None

        self.verbose = self.config.verbose
        self.update_interval_ms = self.config.update_interval_ms

        self.next_flowlet_id = 0
        self._region_positions = np.array(
            [region.position for region in self.traffic_model.regions],
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
        self._region_access_cache_time: float | None = None
        self._array_metrics: Metrics | None = None
        self._flowlets: FlowletState | None = None
        self._links: LinkState | None = None

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

    def reset(self, seed=None, start_time=None):
        """
        Reset the environment to the initial state.

        Returns:
            Initial observations for all agents
        """
        self._set_seed(seed)

        self.network = self._create_network()

        self.topology_update_steps = 0

        self.next_flowlet_id = 0
        self._array_metrics = None
        self._flowlets = None
        self._links = None

        if start_time is None:
            self.start_time = 0
        else:
            self.start_time = start_time
        self.current_time = self.start_time
        self.network.update_topology(self.start_time, None)
        self._refresh_satellite_index_cache()
        self._update_region_access_cache()

    def run(
        self,
        solver: "BaseSolver",
    ):
        return self._run_slot_array_kernel(solver)

    def _run_slot_array_kernel(
        self,
        solver: "BaseSolver",
    ):
        self.current_solver = solver
        self._build_link_array_cache()
        self._generate_flowlet_state()

        num_steps = int(np.ceil(self.time_limit / self.slot_ms))
        next_topology_time = self.start_time + self.update_interval_ms
        end_time = self.start_time + self.time_limit

        if self.verbose:
            print("Env is in slot-array evaluation mode")

        for step in range(num_steps):
            self.current_time = self.start_time + step * self.slot_ms
            if self.current_time >= end_time:
                break

            if self.current_time >= next_topology_time:
                while self.current_time >= next_topology_time:
                    next_topology_time += self.update_interval_ms
                self.network.update_topology(self.current_time, None)
                self.topology_update_steps += 1
                self._update_region_access_cache()
                self._refresh_link_state_arrays()
                self._drop_flowlets_on_disconnected_links()

            self._release_transmitted_flowlets()
            arrived_ids = self._handle_flowlet_arrivals()
            activated_ids = self._activate_flowlets_at_current_slot()
            if len(arrived_ids) > 0 and len(activated_ids) > 0:
                self._route_flowlets_at_nodes(np.sort(np.concatenate((arrived_ids, activated_ids))))
            elif len(arrived_ids) > 0:
                self._route_flowlets_at_nodes(arrived_ids)
            elif len(activated_ids) > 0:
                self._route_flowlets_at_nodes(activated_ids)

        self.current_time = end_time
        self._array_metrics = self._calc_array_metrics()

    def _build_link_array_cache(self):
        source_ids = self.network._link_source_ids
        sink_ids = self.network._link_sink_ids
        num_links = len(source_ids)
        id_by_pair = np.full((self.network.num_nodes, self.network.num_nodes), -1, dtype=np.int32)
        id_by_pair[source_ids, sink_ids] = np.arange(
            num_links,
            dtype=np.int32,
        )
        self._links = LinkState(
            source_ids=source_ids,
            sink_ids=sink_ids,
            data_rate=self.network._link_data_rate_array,
            capacity=self.network._link_capacity_array,
            connected=self.network._link_connected_array,
            delay=self.network._link_delay_array,
            id_by_pair=id_by_pair,
            free_time=np.zeros(num_links, dtype=np.float64),
            queue_load=np.zeros(num_links, dtype=np.float64),
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
        slot_offsets = np.empty(num_slots + 1, dtype=np.int64)
        slot_offsets[0] = 0
        np.cumsum(slot_counts, out=slot_offsets[1:])

        self.next_flowlet_id = num_flowlets

        if num_flowlets == 0:
            self._flowlets = FlowletState.empty()
            return

        creation_slots = np.repeat(np.arange(num_slots, dtype=np.int64), slot_counts)
        creation_times = self.start_time + creation_slots.astype(np.float64) * self.slot_ms
        source_region_ids = self.traffic_model.sample_source_ids(self.np_random, num_flowlets).astype(np.int64)
        target_region_ids = self.traffic_model.sample_target_ids(self.np_random, source_region_ids).astype(np.int64)
        is_normal = self.np_random.uniform(size=num_flowlets) < self.prob_normal_packet
        packet_size = np.where(is_normal, self.normal_packet_size, self.small_packet_size).astype(np.float64)
        packet_count = np.maximum(
            1,
            self.np_random.poisson(lam=self.mean_packets_per_flowlet, size=num_flowlets),
        ).astype(np.int64)
        size = packet_size * packet_count

        self._flowlets = FlowletState(
            slot_offsets=slot_offsets,
            status=np.full(num_flowlets, FLOWLET_NOT_STARTED, dtype=np.int8),
            creation_slot=creation_slots,
            creation_time=creation_times,
            source_region_id=source_region_ids,
            target_region_id=target_region_ids,
            source_id=np.full(num_flowlets, -1, dtype=np.int64),
            current_sat=np.full(num_flowlets, -1, dtype=np.int64),
            next_sat=np.full(num_flowlets, -1, dtype=np.int64),
            link_id=np.full(num_flowlets, -1, dtype=np.int32),
            packet_count=packet_count,
            packet_size=packet_size,
            is_normal=is_normal,
            size=size,
            ttl=np.full(num_flowlets, self.default_ttl, dtype=np.int16),
            hops=np.zeros(num_flowlets, dtype=np.int16),
            queue_delay=np.zeros(num_flowlets, dtype=np.float64),
            transmission_delay=np.zeros(num_flowlets, dtype=np.float64),
            propagation_delay=np.zeros(num_flowlets, dtype=np.float64),
            total_queue_cost=np.zeros(num_flowlets, dtype=np.float64),
            first_access_delay=np.zeros(num_flowlets, dtype=np.float64),
            final_access_delay=np.zeros(num_flowlets, dtype=np.float64),
            delivery_time=np.full(num_flowlets, np.nan, dtype=np.float64),
            drop_time=np.full(num_flowlets, np.nan, dtype=np.float64),
            drop_reason=np.full(num_flowlets, -1, dtype=np.int16),
            transmit_end_time=np.full(num_flowlets, np.inf, dtype=np.float64),
            arrival_time=np.full(num_flowlets, np.inf, dtype=np.float64),
            link_released=np.ones(num_flowlets, dtype=bool),
            scheduled_prop_delay=np.zeros(num_flowlets, dtype=np.float64),
            shortest_gcd=np.full(num_flowlets, np.inf, dtype=np.float64),
            initial_gcd=np.ones(num_flowlets, dtype=np.float64),
            last_node1=np.full(num_flowlets, -1, dtype=np.int64),
            last_node2=np.full(num_flowlets, -1, dtype=np.int64),
        )

    def _release_transmitted_flowlets(self):
        status = self._flowlets.status
        if len(status) == 0:
            return
        release_mask = (
            (status == FLOWLET_ON_LINK)
            & (~self._flowlets.link_released)
            & (self._flowlets.transmit_end_time <= self.current_time)
        )
        release_ids = np.flatnonzero(release_mask)
        if len(release_ids) == 0:
            return
        np.add.at(
            self._links.queue_load,
            self._flowlets.link_id[release_ids],
            -self._flowlets.size[release_ids],
        )
        self._flowlets.link_released[release_ids] = True
        np.maximum(self._links.queue_load, 0.0, out=self._links.queue_load)

    def _handle_flowlet_arrivals(self) -> np.ndarray:
        status = self._flowlets.status
        if len(status) == 0:
            return np.empty(0, dtype=np.int64)
        arrival_mask = (status == FLOWLET_ON_LINK) & (
            self._flowlets.arrival_time <= self.current_time
        )
        flowlet_ids = np.flatnonzero(arrival_mask)
        if len(flowlet_ids) == 0:
            return flowlet_ids

        self._flowlets.current_sat[flowlet_ids] = self._flowlets.next_sat[flowlet_ids]
        self._flowlets.hops[flowlet_ids] += 1
        self._flowlets.ttl[flowlet_ids] -= 1
        self._flowlets.propagation_delay[flowlet_ids] += self._flowlets.scheduled_prop_delay[flowlet_ids]
        self._flowlets.queue_delay[flowlet_ids] += (
            self.current_time - self._flowlets.arrival_time[flowlet_ids]
        )
        self._flowlets.status[flowlet_ids] = FLOWLET_AT_NODE
        self._flowlets.link_id[flowlet_ids] = -1

        expired_mask = self._flowlets.ttl[flowlet_ids] <= 0
        if expired_mask.any():
            expired = flowlet_ids[expired_mask]
            self._drop_flowlet_ids(expired, NetworkError.TTL_EXPIRED)
            return flowlet_ids[~expired_mask]
        return flowlet_ids

    def _activate_flowlets_at_current_slot(self) -> np.ndarray:
        if self._flowlets is None:
            return np.empty(0, dtype=np.int64)

        slot_idx = int(round((self.current_time - self.start_time) / self.slot_ms))
        if slot_idx < 0 or slot_idx + 1 >= len(self._flowlets.slot_offsets):
            return np.empty(0, dtype=np.int64)
        start = int(self._flowlets.slot_offsets[slot_idx])
        end = int(self._flowlets.slot_offsets[slot_idx + 1])
        if end <= start:
            return np.empty(0, dtype=np.int64)
        flowlet_ids = np.arange(start, end, dtype=np.int64)

        source_regions = self._flowlets.source_region_id[flowlet_ids]
        source_sat_ids = self._nearest_region_sat_ids[source_regions]
        visible = source_sat_ids >= 0
        if (~visible).any():
            self._drop_flowlet_ids(flowlet_ids[~visible], NetworkError.NO_AVAILABLE_SAT)

        active_ids = flowlet_ids[visible]
        if len(active_ids) == 0:
            return active_ids

        source_sat_ids = source_sat_ids[visible]
        source_distances = self._nearest_region_sat_distances[source_regions[visible]]
        source_prop_delay = source_distances / LIGHT_SPEED_MS
        source_tx_delay = self._flowlets.size[active_ids] / self.access_data_rate

        self._flowlets.source_id[active_ids] = source_sat_ids
        self._flowlets.current_sat[active_ids] = source_sat_ids
        self._flowlets.status[active_ids] = FLOWLET_AT_NODE
        self._flowlets.first_access_delay[active_ids] = source_prop_delay + source_tx_delay
        self._flowlets.propagation_delay[active_ids] += source_prop_delay
        self._flowlets.transmission_delay[active_ids] += source_tx_delay

        initial_gcd = self._region_distance_matrix[
            self._flowlets.source_region_id[active_ids],
            self._flowlets.target_region_id[active_ids],
        ].copy()
        initial_gcd[initial_gcd <= 0] = 1e-6
        self._flowlets.initial_gcd[active_ids] = initial_gcd
        self._flowlets.shortest_gcd[active_ids] = initial_gcd
        return active_ids

    def _route_flowlets_at_nodes(self, at_node_ids: np.ndarray):
        if len(at_node_ids) == 0:
            return

        delivered_ids, route_ids = self._partition_deliverable_flowlets(at_node_ids)
        if len(delivered_ids) > 0:
            self._deliver_flowlet_ids(delivered_ids)
        if len(route_ids) == 0:
            return

        self._route_flowlets_batch(route_ids)

    def _partition_deliverable_flowlets(self, flowlet_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current_sats = self._flowlets.current_sat[flowlet_ids]
        target_regions = self._flowlets.target_region_id[flowlet_ids]
        sat_cols = self._sat_id_to_col_array[current_sats]
        valid = sat_cols >= 0
        deliverable = np.zeros(len(flowlet_ids), dtype=bool)
        deliverable[valid] = self._region_sat_visible[target_regions[valid], sat_cols[valid]]
        return flowlet_ids[deliverable], flowlet_ids[~deliverable]

    def _deliver_flowlet_ids(self, flowlet_ids: np.ndarray):
        current_sats = self._flowlets.current_sat[flowlet_ids]
        target_regions = self._flowlets.target_region_id[flowlet_ids]
        sat_cols = self._sat_id_to_col_array[current_sats]
        distance2 = self._region_sat_distance2[target_regions, sat_cols]
        final_prop_delay = np.sqrt(distance2) / LIGHT_SPEED_MS
        final_tx_delay = self._flowlets.size[flowlet_ids] / self.access_data_rate
        final_delay = final_prop_delay + final_tx_delay

        self._flowlets.final_access_delay[flowlet_ids] = final_delay
        self._flowlets.propagation_delay[flowlet_ids] += final_prop_delay
        self._flowlets.transmission_delay[flowlet_ids] += final_tx_delay
        self._flowlets.delivery_time[flowlet_ids] = self.current_time + final_delay
        self._flowlets.status[flowlet_ids] = FLOWLET_DELIVERED

    def _route_flowlets_batch(self, flowlet_ids: np.ndarray):
        current_sats = self._flowlets.current_sat[flowlet_ids]
        target_regions = self._flowlets.target_region_id[flowlet_ids]
        target_access_sats = self._nearest_region_sat_ids[target_regions]

        no_access = target_access_sats < 0
        if no_access.any():
            self._drop_flowlet_ids(flowlet_ids[no_access], NetworkError.NO_AVAILABLE_SAT)

        candidate_ids = flowlet_ids[~no_access]
        if len(candidate_ids) == 0:
            return

        current_sats = current_sats[~no_access]
        target_regions = target_regions[~no_access]
        if solver_requires_spf := self.current_solver.requires_shortest_path_table:
            if not self._region_next_hop_eager_ready:
                self._ensure_region_next_hops(target_regions)

        batch = self._build_routing_batch(
            flowlet_ids=candidate_ids,
            current_sats=current_sats,
            target_regions=target_regions,
            target_access_sats=target_access_sats[~no_access],
            include_spf_table=solver_requires_spf,
        )
        decision = self.current_solver.next_hops(batch)
        next_hops = np.asarray(decision.next_hop_sat_ids, dtype=np.int64)
        if len(next_hops) != len(candidate_ids):
            raise ValueError(
                f"Solver {self.current_solver.name} returned {len(next_hops)} next hops "
                f"for {len(candidate_ids)} flowlets."
            )

        no_route = next_hops < 0
        if no_route.any():
            self._drop_flowlet_ids(candidate_ids[no_route], NetworkError.FAILED_TO_FIND_NEXT_HOP)

        routable_ids = candidate_ids[~no_route]
        if len(routable_ids) == 0:
            return

        current_sats = current_sats[~no_route]
        next_hops = next_hops[~no_route]
        link_ids = self._links.id_by_pair[current_sats, next_hops]

        valid_link = link_ids >= 0
        connected = np.zeros(len(link_ids), dtype=bool)
        connected[valid_link] = self._links.connected[link_ids[valid_link]]
        invalid = (~valid_link) | (~connected)
        if invalid.any():
            self._drop_flowlet_ids(routable_ids[invalid], NetworkError.INVALID_NEXT_HOP)

        scheduled_ids = routable_ids[~invalid]
        if len(scheduled_ids) == 0:
            return

        current_sats = current_sats[~invalid]
        next_hops = next_hops[~invalid]
        link_ids = link_ids[~invalid]

        rejected_ids = self._schedule_flowlets_by_link(
            flowlet_ids=scheduled_ids,
            link_ids=link_ids,
            next_hops=next_hops,
        )
        if len(rejected_ids) > 0:
            self._drop_flowlet_ids(rejected_ids, NetworkError.LINK_FULL)

    def _build_routing_batch(
        self,
        flowlet_ids: np.ndarray,
        current_sats: np.ndarray,
        target_regions: np.ndarray,
        target_access_sats: np.ndarray,
        include_spf_table: bool,
    ) -> RoutingBatch:
        neighbor_sat_ids = np.column_stack(
            (
                self.network.isl_n[current_sats],
                self.network.isl_e[current_sats],
                self.network.isl_s[current_sats],
                self.network.isl_w[current_sats],
            )
        ).astype(np.int64, copy=False)

        neighbor_link_ids = np.full(neighbor_sat_ids.shape, -1, dtype=np.int32)
        valid_neighbor = neighbor_sat_ids >= 0
        if valid_neighbor.any():
            source_ids = np.broadcast_to(current_sats[:, None], neighbor_sat_ids.shape)
            neighbor_link_ids[valid_neighbor] = self._links.id_by_pair[
                source_ids[valid_neighbor],
                neighbor_sat_ids[valid_neighbor],
            ]

        valid_link = neighbor_link_ids >= 0
        safe_link_ids = np.where(valid_link, neighbor_link_ids, 0)
        connected = np.zeros(neighbor_link_ids.shape, dtype=bool)
        if valid_link.any():
            connected[valid_link] = self._links.connected[safe_link_ids[valid_link]]
        action_mask = valid_link & connected

        return RoutingBatch(
            flowlet_ids=flowlet_ids,
            current_sat_ids=current_sats,
            target_region_ids=target_regions,
            target_access_sat_ids=target_access_sats,
            neighbor_sat_ids=neighbor_sat_ids,
            neighbor_link_ids=neighbor_link_ids,
            action_mask=action_mask,
            neighbor_queue_load=np.where(valid_link, self._links.queue_load[safe_link_ids], np.inf),
            neighbor_link_capacity=np.where(valid_link, self._links.capacity[safe_link_ids], 0.0),
            neighbor_link_delay=np.where(valid_link, self._links.delay[safe_link_ids], np.inf),
            neighbor_link_free_time=np.where(valid_link, self._links.free_time[safe_link_ids], np.inf),
            flowlet_size=self._flowlets.size[flowlet_ids],
            ttl=self._flowlets.ttl[flowlet_ids],
            current_time=self.current_time,
            region_next_hop_table=self._region_next_hop_table if include_spf_table else None,
        )

    def _schedule_flowlets_by_link(
        self,
        flowlet_ids: np.ndarray,
        link_ids: np.ndarray,
        next_hops: np.ndarray,
    ) -> np.ndarray:
        if len(flowlet_ids) == 0:
            return np.empty(0, dtype=np.int64)

        order = np.argsort(link_ids, kind="stable")
        sorted_link_ids = link_ids[order]
        sorted_ids = flowlet_ids[order]
        sorted_next_hops = next_hops[order]
        split_points = np.flatnonzero(np.diff(sorted_link_ids)) + 1
        starts = np.r_[0, split_points]
        ends = np.r_[split_points, len(sorted_link_ids)]
        group_link_ids = sorted_link_ids[starts]
        group_lengths = ends - starts
        group_ids = np.repeat(np.arange(len(starts)), group_lengths)

        sizes = self._flowlets.size[sorted_ids]
        cumulative_size = np.cumsum(sizes)
        size_offsets = np.zeros(len(starts), dtype=np.float64)
        if len(starts) > 1:
            size_offsets[1:] = cumulative_size[starts[1:] - 1]
        group_cumulative_size = cumulative_size - size_offsets[group_ids]
        remaining_capacity = self._links.capacity[group_link_ids] - self._links.queue_load[group_link_ids]
        accepted = group_cumulative_size <= remaining_capacity[group_ids]

        rejected_ids = sorted_ids[~accepted]
        if not accepted.any():
            return rejected_ids

        accepted_ids = sorted_ids[accepted]
        accepted_link_ids = sorted_link_ids[accepted]
        accepted_next_hops = sorted_next_hops[accepted]
        accepted_sizes = sizes[accepted]
        accepted_group_ids = group_ids[accepted]

        accepted_split_points = np.flatnonzero(np.diff(accepted_group_ids)) + 1
        accepted_starts = np.r_[0, accepted_split_points]
        accepted_ends = np.r_[accepted_split_points, len(accepted_ids)]
        accepted_group_numbers = np.repeat(
            np.arange(len(accepted_starts)),
            accepted_ends - accepted_starts,
        )
        accepted_original_group_ids = accepted_group_ids[accepted_starts]

        transmit_times = accepted_sizes / self._links.data_rate[accepted_link_ids]
        cumulative_tx = np.cumsum(transmit_times)
        tx_offsets = np.zeros(len(accepted_starts), dtype=np.float64)
        if len(accepted_starts) > 1:
            tx_offsets[1:] = cumulative_tx[accepted_starts[1:] - 1]
        group_cumulative_tx = cumulative_tx - tx_offsets[accepted_group_numbers]

        accepted_group_link_ids = group_link_ids[accepted_original_group_ids]
        start_base_by_group = np.maximum(
            self.current_time,
            self._links.free_time[accepted_group_link_ids],
        )
        start_base = start_base_by_group[accepted_group_numbers]
        transmit_end_times = start_base + group_cumulative_tx
        wait_times = transmit_end_times - transmit_times - self.current_time
        propagation_delays = self._links.delay[accepted_link_ids]

        self._links.free_time[accepted_group_link_ids] = transmit_end_times[accepted_ends - 1]
        self._links.queue_load[accepted_group_link_ids] += np.add.reduceat(
            accepted_sizes,
            accepted_starts,
        )

        self._flowlets.queue_delay[accepted_ids] += wait_times
        self._flowlets.transmission_delay[accepted_ids] += transmit_times
        self._flowlets.total_queue_cost[accepted_ids] += wait_times
        self._flowlets.link_id[accepted_ids] = accepted_link_ids
        self._flowlets.next_sat[accepted_ids] = accepted_next_hops
        self._flowlets.transmit_end_time[accepted_ids] = transmit_end_times
        self._flowlets.arrival_time[accepted_ids] = transmit_end_times + propagation_delays
        self._flowlets.scheduled_prop_delay[accepted_ids] = propagation_delays
        self._flowlets.link_released[accepted_ids] = False
        self._flowlets.status[accepted_ids] = FLOWLET_ON_LINK

        self._flowlets.last_node2[accepted_ids] = self._flowlets.last_node1[accepted_ids]
        self._flowlets.last_node1[accepted_ids] = self._flowlets.current_sat[accepted_ids]
        return rejected_ids

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

    def _drop_flowlets_on_disconnected_links(self):
        status = self._flowlets.status
        if len(status) == 0:
            return
        on_link = np.flatnonzero(status == FLOWLET_ON_LINK)
        if len(on_link) == 0:
            return
        link_ids = self._flowlets.link_id[on_link]
        dropped = on_link[~self._links.connected[link_ids]]
        if len(dropped) == 0:
            return
        unreleased = dropped[~self._flowlets.link_released[dropped]]
        if len(unreleased) > 0:
            np.add.at(
                self._links.queue_load,
                self._flowlets.link_id[unreleased],
                -self._flowlets.size[unreleased],
            )
            self._flowlets.link_released[unreleased] = True
        self._links.queue_load[~self._links.connected] = 0.0
        self._links.free_time[~self._links.connected] = self.current_time
        self._drop_flowlet_ids(dropped, NetworkError.LINK_DISCONNECTED)

    def _drop_flowlet_ids(self, flowlet_ids: np.ndarray, reason: NetworkError):
        if len(flowlet_ids) == 0:
            return
        self._flowlets.status[flowlet_ids] = FLOWLET_DROPPED
        self._flowlets.drop_time[flowlet_ids] = self.current_time
        self._flowlets.drop_reason[flowlet_ids] = int(reason)

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

        throughput = float(flowlets.size[delivered_mask].sum()) / max(self.time_limit / 1000.0, 1e-12)

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
            service_rate=delivered / max(self.time_limit / 1000.0, 1e-12),
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

    def _print(self, line: str, end=None):
        print(
            f"{ms2str(self.start_time)}+{ms2str(self.current_time-self.start_time)}: {line}",
            end=end,
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
