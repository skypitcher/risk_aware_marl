from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from sat_net.geometric import (
    EARTH_R_KM,
    GM_EARTH,
    LIGHT_SPEED_MS,
    calculate_maximum_inter_satellite_range,
    calculate_orbital_period,
    calculate_slant_range,
    distance_between,
    get_projected_position,
)


DISCONNECTED_LINK_WEIGHT = 9999.0


class SatelliteNetwork:
    """Array-oriented Walker satellite network used by the slot routing kernel."""

    def __init__(
        self,
        altitude: int = 550,
        inclination: int = 53,
        num_orbits: int = 24,
        num_sats_per_orbit: int = 24,
        phasing: int = 3,
        min_elevation_angle_deg: int = 15,
        link_buffer_size: float = 10.0,
        isl_data_rate: float = 1.0,
    ):
        self.altitude = altitude
        self.inclination = inclination
        self.num_orbits = num_orbits
        self.num_sats_per_orbit = num_sats_per_orbit
        self.phasing = phasing
        self.min_elevation_angle_deg = min_elevation_angle_deg
        self.link_buffer_size = link_buffer_size
        self.isl_data_rate = isl_data_rate

        self.orbit_radius = EARTH_R_KM + self.altitude
        self.num_satellites = self.num_orbits * self.num_sats_per_orbit
        self.num_nodes = self.num_satellites
        self.angular_shift = self.phasing * (360 / self.num_satellites) if self.num_satellites else 0.0
        self.max_access_range = calculate_slant_range(self.min_elevation_angle_deg, self.altitude)
        self.max_isl_range = calculate_maximum_inter_satellite_range(self.altitude)
        self.orbit_cycle = calculate_orbital_period(self.altitude)

        self.topology_version = 0
        self._shortest_act = np.full((self.num_nodes, self.num_nodes), -1, dtype=np.int64)
        self._shortest_next_hop_ready = np.zeros(self.num_nodes, dtype=bool)
        self._shortest_reverse_csr: csr_matrix | None = None
        self._shortest_reverse_csr_version = -1

        self._init_node_arrays()
        self._init_isl_arrays()
        self._update_satellite_positions_vectorized(0.0)
        self._refresh_link_geometry()

    @property
    def node_positions(self) -> np.ndarray:
        return self._node_positions_by_id

    @property
    def satellite_ids(self) -> np.ndarray:
        return self._satellite_ids

    def _init_node_arrays(self):
        self._node_positions_by_id = np.empty((self.num_nodes, 3), dtype=np.float64)

        self._satellite_ids = np.arange(self.num_satellites, dtype=np.int64)
        self._satellite_orbits = np.repeat(
            np.arange(self.num_orbits, dtype=np.int64),
            self.num_sats_per_orbit,
        )
        self._satellite_index_in_orbit = np.tile(
            np.arange(self.num_sats_per_orbit, dtype=np.int64),
            self.num_orbits,
        )
        if 80 <= self.inclination <= 100:
            raan_by_orbit = np.arange(self.num_orbits, dtype=np.float64) * (180.0 / self.num_orbits)
        else:
            raan_by_orbit = np.arange(self.num_orbits, dtype=np.float64) * (360.0 / self.num_orbits)

        self._satellite_altitudes = np.full(self.num_satellites, self.altitude, dtype=np.float64)
        self._satellite_raan_rad = np.deg2rad(raan_by_orbit[self._satellite_orbits])
        self._satellite_inc_rad = np.full(self.num_satellites, np.deg2rad(self.inclination), dtype=np.float64)
        self._satellite_true_anomaly = (
            self._satellite_index_in_orbit * (360.0 / self.num_sats_per_orbit)
            + self._satellite_orbits * self.angular_shift
        ) % 360.0
        self._satellite_positions_by_id = self._node_positions_by_id

    def _init_isl_arrays(self):
        sources: list[int] = []
        sinks: list[int] = []
        self.isl_n = np.full(self.num_satellites, -1, dtype=np.int64)
        self.isl_s = np.full(self.num_satellites, -1, dtype=np.int64)
        self.isl_e = np.full(self.num_satellites, -1, dtype=np.int64)
        self.isl_w = np.full(self.num_satellites, -1, dtype=np.int64)

        def sat_id(orbit_idx: int, sat_idx: int) -> int:
            return orbit_idx * self.num_sats_per_orbit + sat_idx

        def add_bidirectional(source_id: int, sink_id: int):
            sources.extend([source_id, sink_id])
            sinks.extend([sink_id, source_id])

        for orbit_idx in range(self.num_orbits):
            for sat_idx in range(self.num_sats_per_orbit):
                source_id = sat_id(orbit_idx, sat_idx)
                sink_id = sat_id(orbit_idx, (sat_idx + 1) % self.num_sats_per_orbit)
                add_bidirectional(source_id, sink_id)
                self.isl_n[source_id] = sink_id
                self.isl_s[sink_id] = source_id

        for orbit_idx in range(self.num_orbits):
            neighbor_orbit = (orbit_idx + 1) % self.num_orbits
            for sat_idx in range(self.num_sats_per_orbit):
                shift_per_orbit = self.phasing / self.num_sats_per_orbit
                total_shift_on_idx = (orbit_idx - neighbor_orbit) * shift_per_orbit
                neighbor_idx = int(np.round(sat_idx + total_shift_on_idx) % self.num_sats_per_orbit)
                source_id = sat_id(orbit_idx, sat_idx)
                sink_id = sat_id(neighbor_orbit, neighbor_idx)
                add_bidirectional(source_id, sink_id)
                self.isl_e[source_id] = sink_id
                self.isl_w[sink_id] = source_id

        self._link_source_ids = np.asarray(sources, dtype=np.int64)
        self._link_sink_ids = np.asarray(sinks, dtype=np.int64)
        self._link_capacity_array = np.full(len(sources), self.link_buffer_size, dtype=np.float64)
        self._link_data_rate_array = np.full(len(sources), self.isl_data_rate, dtype=np.float64)
        self._link_connected_array = np.ones(len(sources), dtype=bool)
        self._link_delay_array = np.zeros(len(sources), dtype=np.float64)
        self._link_spf_weights = np.zeros(len(sources), dtype=np.float64)
        self._link_index_by_pair = {
            (int(source), int(sink)): idx
            for idx, (source, sink) in enumerate(zip(self._link_source_ids, self._link_sink_ids))
        }
        self.num_links = len(self._link_source_ids)
        self.neighbor_sat_ids = np.column_stack(
            (self.isl_n, self.isl_e, self.isl_s, self.isl_w)
        ).astype(np.int64, copy=False)

    def update_topology(self, timestamp: float, on_link_disconnected=None):
        if on_link_disconnected is not None:
            raise NotImplementedError("Link-disconnect callbacks were removed with the slot-array kernel.")

        self._update_satellite_positions_vectorized(timestamp)
        self._refresh_link_geometry()
        self.topology_version += 1
        self._reset_shortest_next_hop_cache()
        self._shortest_reverse_csr = None
        self._shortest_reverse_csr_version = -1

    def _reset_shortest_next_hop_cache(self):
        self._shortest_act.fill(-1)
        self._shortest_next_hop_ready.fill(False)

    def _update_satellite_positions_vectorized(self, timestamp: float):
        semi_major_axis_m = (EARTH_R_KM + self._satellite_altitudes) * 1000.0
        orbit_cycles = 2 * np.pi * np.sqrt(np.power(semi_major_axis_m, 3) / GM_EARTH)
        theta = (
            self._satellite_true_anomaly
            + (360.0 / orbit_cycles) * timestamp / 1000.0
        ) % 360.0
        theta_rad = np.deg2rad(theta)

        orbit_radius = EARTH_R_KM + self._satellite_altitudes
        cos_raan = np.cos(self._satellite_raan_rad)
        sin_raan = np.sin(self._satellite_raan_rad)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        cos_inc = np.cos(self._satellite_inc_rad)
        sin_inc = np.sin(self._satellite_inc_rad)

        x_eci = orbit_radius * (cos_raan * cos_theta - sin_raan * sin_theta * cos_inc)
        y_eci = orbit_radius * (sin_raan * cos_theta + cos_raan * sin_theta * cos_inc)
        z_eci = orbit_radius * sin_theta * sin_inc

        theta_earth = 7.2921150e-5 * timestamp / 1000.0
        cos_earth = np.cos(theta_earth)
        sin_earth = np.sin(theta_earth)
        x_ecef = x_eci * cos_earth + y_eci * sin_earth
        y_ecef = -x_eci * sin_earth + y_eci * cos_earth
        self._satellite_positions_by_id[self._satellite_ids] = np.column_stack((x_ecef, y_ecef, z_eci))

    def _refresh_link_geometry(self):
        source_pos = self._satellite_positions_by_id[self._link_source_ids]
        sink_pos = self._satellite_positions_by_id[self._link_sink_ids]
        distances = np.linalg.norm(source_pos - sink_pos, axis=1)
        self._link_connected_array = distances <= self.max_isl_range
        self._link_delay_array = distances / LIGHT_SPEED_MS
        self._link_spf_weights = np.where(
            self._link_connected_array,
            self._link_delay_array,
            DISCONNECTED_LINK_WEIGHT,
        )

    def is_satellite(self, node_id: int) -> bool:
        return 0 <= node_id < self.num_satellites

    def _get_shortest_edge_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._link_source_ids, self._link_sink_ids, self._link_spf_weights

    def _get_shortest_reverse_csr(self) -> csr_matrix:
        if self._shortest_reverse_csr_version != self.topology_version:
            sources, sinks, weights = self._get_shortest_edge_arrays()
            self._shortest_reverse_csr = csr_matrix(
                (weights, (sinks, sources)),
                shape=(self.num_nodes, self.num_nodes),
            )
            self._shortest_reverse_csr_version = self.topology_version
        return self._shortest_reverse_csr

    def get_visible_satellites_for_position(
        self,
        position: np.ndarray,
        max_count: int | None = None,
    ) -> list[tuple[int, float]]:
        sat_positions = self._satellite_positions_by_id[self._satellite_ids]
        distances = np.linalg.norm(sat_positions - position, axis=1)
        visible_idx = np.flatnonzero(distances <= self.max_access_range)
        visible_idx = visible_idx[np.argsort(distances[visible_idx])]
        if max_count is not None:
            visible_idx = visible_idx[:max_count]
        return [
            (int(self._satellite_ids[idx]), float(distances[idx]))
            for idx in visible_idx
        ]

    def get_nearest_satellite_for_position(
        self,
        position: np.ndarray,
    ) -> tuple[int, float] | tuple[None, None]:
        candidates = self.get_visible_satellites_for_position(position, max_count=1)
        return candidates[0] if candidates else (None, None)

    def can_satellite_serve_position(self, sat_id: int, position: np.ndarray) -> bool:
        if not self.is_satellite(sat_id):
            return False
        return distance_between(position, self._satellite_positions_by_id[sat_id]) <= self.max_access_range

    def get_satellite_lon_lat(self) -> tuple[np.ndarray, np.ndarray]:
        positions = self._satellite_positions_by_id[self._satellite_ids]
        return get_projected_position(positions[:, 0], positions[:, 1], positions[:, 2])

    def get_shortest_path(self, current: int, sink: int, weight_fn=None):
        if current < 0 or current >= self.num_nodes or sink < 0 or sink >= self.num_nodes:
            return float("inf"), []
        return self._get_shortest_path_from_next_hops(current=current, sink=sink)

    def _get_shortest_path_from_next_hops(self, current: int, sink: int):
        if current == sink:
            return 0.0, [current]

        path = [current]
        path_weight = 0.0
        visited = {current}
        node = current
        while node != sink and len(path) <= self.num_nodes:
            next_hop = self.get_shortest_next_hop(node, sink)
            if next_hop is None or next_hop in visited:
                return float("inf"), []
            link_idx = self._link_index_by_pair.get((node, next_hop))
            if link_idx is None:
                return float("inf"), []
            path_weight += float(self._link_spf_weights[link_idx])
            path.append(next_hop)
            visited.add(next_hop)
            node = next_hop

        if node != sink:
            return float("inf"), []
        return path_weight, path

    def get_shortest_next_hop(self, current: int, sink: int) -> int | None:
        if current < 0 or current >= self.num_nodes or sink < 0 or sink >= self.num_nodes or current == sink:
            return None
        self._ensure_shortest_next_hop_rows(np.array([sink], dtype=np.int64))
        next_hop = int(self._shortest_act[sink, current])
        return next_hop if next_hop >= 0 else None

    def get_shortest_next_hops(self, current: np.ndarray, sink: np.ndarray) -> np.ndarray:
        act = np.full(len(current), -1, dtype=np.int64)
        if len(current) == 0:
            return act

        valid = (
            (current >= 0)
            & (current < self.num_nodes)
            & (sink >= 0)
            & (sink < self.num_nodes)
            & (current != sink)
        )
        if not valid.any():
            return act

        target_sinks = np.unique(sink[valid])
        self._ensure_shortest_next_hop_rows(target_sinks)
        act[valid] = self._shortest_act[sink[valid], current[valid]]
        return act

    def precompute_shortest_next_hops(self, sinks: np.ndarray):
        self._ensure_shortest_next_hop_rows(np.asarray(sinks, dtype=np.int64))

    def shortest_next_hop_rows(self, sinks: np.ndarray) -> np.ndarray:
        sinks = np.asarray(sinks, dtype=np.int64)
        valid_sinks = sinks[(sinks >= 0) & (sinks < self.num_nodes)]
        self._ensure_shortest_next_hop_rows(valid_sinks)
        rows = np.full((len(sinks), self.num_nodes), -1, dtype=np.int64)
        valid = (sinks >= 0) & (sinks < self.num_nodes)
        rows[valid] = self._shortest_act[sinks[valid]]
        return rows

    def _ensure_shortest_next_hop_rows(self, sinks: np.ndarray):
        sinks = np.unique(sinks[(sinks >= 0) & (sinks < self.num_nodes)])
        if len(sinks) == 0:
            return
        missing_sinks = sinks[~self._shortest_next_hop_ready[sinks]].astype(np.int32, copy=False)
        if len(missing_sinks) == 0:
            return

        _distances, predecessors = dijkstra(
            csgraph=self._get_shortest_reverse_csr(),
            directed=True,
            indices=missing_sinks,
            return_predecessors=True,
        )
        if predecessors.ndim == 1:
            predecessors = predecessors[np.newaxis, :]

        next_hop_rows = predecessors.astype(np.int64, copy=False)
        next_hop_rows[next_hop_rows < 0] = -1
        next_hop_rows[np.arange(len(missing_sinks)), missing_sinks] = -1

        self._shortest_act[missing_sinks] = next_hop_rows
        self._shortest_next_hop_ready[missing_sinks] = True
