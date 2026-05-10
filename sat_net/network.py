from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import rustworkx as rx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from sat_net.geometric import (
    EARTH_R_KM,
    GM_EARTH,
    LIGHT_SPEED_MS,
    calculate_delay,
    calculate_maximum_inter_satellite_range,
    calculate_orbital_period,
    calculate_slant_range,
    distance_between,
)
from sat_net.link import Link
from sat_net.node import GroundStation, Node, Satellite


DISCONNECTED_LINK_WEIGHT = 9999.0


def _default_link_weight(link: Link) -> float:
    if link is not None and link.is_connected:
        return link.propagation_delay
    return DISCONNECTED_LINK_WEIGHT


class Network(ABC):
    """Represents a dynamic network where the topology is updated periodically due to the movement of nodes."""

    def __init__(self):
        """Initialize an instance of the DynamicNetwork class."""
        self.nodes: dict[int, Node] = {}  # {node_id: Node}
        self.links: dict[tuple[int, int], Link] = {}  # {(source, destination): Link}
        self.G: rx.PyDiGraph = rx.PyDiGraph()
        self.topology_version = 0
        self._shortest_next_hop_cache: dict[int, np.ndarray] = {}
        self._shortest_reverse_csr: csr_matrix | None = None
        self._shortest_reverse_csr_version = -1

    def _add_node(self, node: Node):
        """Add a node to the network."""
        assert node.id is None
        node_index = self.G.add_node(node)
        node.id = node_index
        self.nodes[node.id] = node
        return node_index

    def _add_link(self, link: Link):
        """Add a link to the network."""
        self.links[link.id] = link
        link.source.add_outgoing_link(link)
        self.G.add_edge(link.source.id, link.sink.id, link)

    def _remove_link(self, link_id: tuple[int, int]):
        """Remove a link from the network."""
        sender_id, receiver_id = link_id
        if link_id in self.links:
            del self.links[link_id]

        sender = self.nodes[sender_id]
        sender.remove_outgoing_link(receiver_id)
        if self.G.has_edge(sender_id, receiver_id):
            self.G.remove_edge(sender_id, receiver_id)

    def update_topology(self, timestamp: float, on_link_disconnected: Callable[[Link], None] = None):
        """
        Update the network topology based on node positions.

        Args:
            timestamp: Current simulation time
            on_link_disconnected: Callback for handling link unavailability.
        """
        self._change_topology(timestamp, on_link_disconnected)
        self.topology_version += 1
        self._shortest_next_hop_cache.clear()
        self._shortest_reverse_csr = None
        self._shortest_reverse_csr_version = -1
        # self._update_routing_table()

    @abstractmethod
    def _change_topology(self, timestamp: float, on_link_disconnected: Callable[[Link], None]):
        """
        Update the network topology based on node positions.

        Args:
            timestamp: Current simulation time
            on_link_disconnected: Callback for handling link unavailability.
        """
        pass

    # def _update_routing_table(self):
    #     """Update the routing table after topology changes, storing path lengths and paths."""
    #     # self.all_pairs_shortest_paths: dict[int, dict[int, list[int]]] = {}
    #     # self.all_pairs_shortest_path_lengths: dict[int, dict[int, float]] = {}
    #     all_pairs_shortest_paths = rx.all_pairs_dijkstra_shortest_paths(self.G, edge_cost_fn=_weight_func)
    #     all_pairs_shortest_path_lengths = rx.all_pairs_dijkstra_path_lengths(self.G, edge_cost_fn=_weight_func)

    def get_link(self, u: int, v: int) -> Optional[Link]:
        """Get the link between two nodes, or None if no such link exists."""
        return self.links.get((u, v), None)
    
    def get_backpressure_next_hop(self, current: int, destination: int) -> Optional[int]:
        """
        Get the next hop using backpressure algorithm.
        
        Args:
            current: Current node ID
            destination: Destination node ID
            
        Returns:
            The next hop node ID that maximizes backpressure, or None if no valid next hop
        """
        if current not in self.nodes:
            return None
        
        current_node = self.nodes[current]
        return current_node.get_backpressure_next_hop(destination)

    def get_shortest_path(self, current: int, sink: int) -> list[int]:
        """
        Find the shortest path between two nodes, relying on the precomputed routing table.
        Returns:
            path_length: The length of the shortest path.
            path: The shortest path between the two nodes.
            If no path exists, returns (float('inf'), []).
        """
        raise NotImplementedError("get_shortest_path not implemented.")

    def average_link_load(self) -> float:
        """Calculate the average network link load."""
        return sum(link.get_load_factor() for link in self.links.values()) / len(self.links)

    def highest_link_load(self) -> float:
        """Calculate the highest network link load."""
        return max(link.get_load_factor() for link in self.links.values())

    def count_hotspot_links(self, threshold: float = 0.8) -> int:
        """Count the number of links whose load exceeds a given threshold."""
        if not self.links:  # Avoid issues if there are no links
            return 0
        return sum(1 for link in self.links.values() if link.is_connected and link.get_load_factor() >= threshold)

    def _get_shortest_edge_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        link_items = list(self.links.items())
        sources = np.fromiter((source for (source, _sink), _link in link_items), dtype=np.int64)
        sinks = np.fromiter((sink for (_source, sink), _link in link_items), dtype=np.int64)
        weights = np.fromiter(
            (_default_link_weight(link) for _key, link in link_items),
            dtype=np.float64,
            count=len(link_items),
        )
        return sources, sinks, weights

    def _get_shortest_reverse_csr(self) -> csr_matrix:
        if self._shortest_reverse_csr_version != self.topology_version:
            sources, sinks, weights = self._get_shortest_edge_arrays()
            self._shortest_reverse_csr = csr_matrix(
                (weights, (sinks, sources)),
                shape=(self.num_nodes, self.num_nodes),
            )
            self._shortest_reverse_csr_version = self.topology_version
        return self._shortest_reverse_csr


class SatelliteNetwork(Network):
    """The integrated terrestrial-satellite network."""

    def __init__(
        self,
        ground_stations: list[dict[str, Any]],
        altitude: int = 550,  # km
        inclination: int = 53,  # degrees
        num_orbits: int = 24,
        num_sats_per_orbit: int = 24,
        phasing: int = 3,
        min_elevation_angle_deg: int = 15,
        max_gsl_per_gs: int = 2,
        max_gsl_per_sat: int = 2,
        node_buffer_size: float = 10.0,
        link_buffer_size: float = 10.0,
        gsl_data_rate: float = 10.0,
        isl_data_rate: float = 1.0,
    ):
        """Initialize the satellite network.
        Args:
            ground_stations: List of ground station configurations.
            altitude: Altitude of the satellites.
            inclination: Inclination of the satellites.
            num_orbits: Number of orbital planes.
            num_sats_per_orbit: Number of satellites per orbit.
            phasing: Phasing of the satellites.
            min_elevation_angle_deg: Minimum elevation angle for GSLs.
            max_gsl_per_gs: Maximum number of GSLs per ground station.
            max_gsl_per_sat: Maximum number of GSLs per satellite.
            node_buffer_size: Buffer size for nodes, in Megabits.
            link_buffer_size: Buffer size for links, in Megabits.
            gsl_data_rate: Data rate for GSLs, in Megabits per millisecond.
            isl_data_rate: Data rate for ISLs, in Megabits per millisecond.
        """
        super().__init__()

        self.altitude = altitude
        self.inclination = inclination
        self.num_orbits = num_orbits
        self.num_sats_per_orbit = num_sats_per_orbit
        self.phasing = phasing
        self.min_elevation_angle_deg = min_elevation_angle_deg
        self.mas_gsl_per_gs = max_gsl_per_gs
        self.max_gsl_per_sat = max_gsl_per_sat
        self.ground_station_raw_data = ground_stations

        self.node_buffer_size = node_buffer_size
        self.link_buffer_size = link_buffer_size
        self.gsl_data_rate = gsl_data_rate
        self.isl_data_rate = isl_data_rate

        # derived parameters
        self.orbit_radius = EARTH_R_KM + self.altitude
        self.num_satellites = self.num_orbits * self.num_sats_per_orbit
        self.num_ground_stations = len(self.ground_station_raw_data)
        self.num_nodes = self.num_ground_stations + self.num_satellites
        self._sync_object_state = self.num_ground_stations > 0
        if self.num_satellites > 0:
            self.angular_shift = self.phasing * (360 / self.num_satellites)
        else:
            self.angular_shift = 0.0
        self.max_gsl_range = calculate_slant_range(self.min_elevation_angle_deg, self.altitude)
        self.max_gsl_delay = self.max_gsl_range / LIGHT_SPEED_MS
        self.max_isl_range = calculate_maximum_inter_satellite_range(self.altitude)
        self.orbit_cycle = calculate_orbital_period(self.altitude)

        # Additional stores for the network
        self.orbits: list[list[Satellite]] = []
        self.satellites: dict[int, Satellite] = {}  # {node_id: Satellite}
        self.ground_stations: dict[int, GroundStation] = {}  # {node_id: GroundStation}
        self.ground_station_idx_list = []

        # Direction-based ISL topology
        self.ISL_N: dict[int, int] = {}  # {sat_id: sat_id}
        self.ISL_S: dict[int, int] = {}  # {sat_id: sat_id}
        self.ISL_E: dict[int, int] = {}  # {sat_id: sat_id}
        self.ISL_W: dict[int, int] = {}  # {sat_id: sat_id}

        # record the number of ground stations connected to a satellite
        self.S2G: dict[int, int] = {}  # {sat_id: num_gs_connected}

        self._initialize_network()
        self._rebuild_vector_update_cache()

    def _initialize_network(self):
        """Initialize the network."""
        self._create_satellites()
        self._create_ground_stations()
        self._setup_ISLs()
        self._setup_GSLs()

    def _create_ground_stations(self):
        """Create ground stations based on the city list."""
        for gs_config in self.ground_station_raw_data:
            node = GroundStation(
                name=gs_config["name"],
                buffer_size=self.node_buffer_size,
                longitude=gs_config["longitude"],
                latitude=gs_config["latitude"],
                altitude=gs_config.get("altitude", 0),
                population=gs_config.get("population", 1.0),
            )
            self._add_node(node)
            self.ground_station_idx_list.append(node.id)

    def _create_satellites(self):
        """Create satellites based on the constellation parameters."""
        for i in range(self.num_orbits):
            # Calculate RAAN (Right Ascension of Ascending Node)
            if 80 <= self.inclination <= 100:
                raan = i * (180.0 / self.num_orbits)  # Walker-Star
            else:
                raan = i * (360.0 / self.num_orbits)  # Walker-Delta

            # Create satellites in this orbit
            orbit_sats = []
            for j in range(self.num_sats_per_orbit):
                # Calculate true anomaly
                mean_anomaly = j * (360.0 / self.num_sats_per_orbit)
                true_anomaly = (mean_anomaly + i * self.angular_shift) % 360

                # Create satellite
                node = Satellite(
                    name=f"Sat({i},{j})",
                    orbit=i,
                    index_in_orbit=j,
                    altitude=self.altitude,
                    inclination=self.inclination,
                    raan=raan,
                    true_anomaly=true_anomaly,
                    eccentricity=0.0,
                    arg_perigee=0.0,
                    buffer_size=self.node_buffer_size,
                )

                self._add_node(node)
                orbit_sats.append(node)
                self.S2G[node.id] = 0

            self.orbits.append(orbit_sats)

    def _setup_ISLs(self):
        """Set up the initial inter-satellite links based on the constellation topology."""
        num_orbits = self.num_orbits
        num_sats_per_orbit = self.num_sats_per_orbit

        def _add_link_between_sats(sat1, sat2):
            delay = calculate_delay(sat1.position, sat2.position)
            self._add_link(
                Link(
                    source=sat1,
                    sink=sat2,
                    capacity=self.link_buffer_size,
                    propagation_delay=delay,
                    data_rate=self.isl_data_rate,
                )
            )
            self._add_link(
                Link(
                    source=sat2,
                    sink=sat1,
                    capacity=self.link_buffer_size,
                    propagation_delay=delay,
                    data_rate=self.isl_data_rate,
                )
            )

        # Create intra-plane links (connecting satellites within the same orbit)
        for i, orbit_i in enumerate(self.orbits):
            num_sats = len(orbit_i)
            for j in range(num_sats):
                sat_i_j = orbit_i[j]
                sat_i_k = orbit_i[(j + 1) % num_sats]
                _add_link_between_sats(sat_i_j, sat_i_k)
                # Record the ISLs topology
                self.ISL_N[sat_i_j.id] = sat_i_k.id
                self.ISL_S[sat_i_k.id] = sat_i_j.id

        # Create inter-plane links (connecting satellites in adjacent orbital planes)
        # ------------------------------------------------------------------------------
        # We find the "deterministic" nearest inter-plane satellite to preserve stable "+" topology (mesh network).
        # True Anomaly Matching:
        # For (i,j) the j-th satellite at the i-th circular orbital plane, we find the nearest inter-plane
        # neighbor (p,q) by solving q from $nu_{i,j} \approx nu_{p, q}$, where p=(i+1)%num_orbits is the neighbor orbit.
        # -----------------------------------------------------------------------------------
        # Rather than using physical distance proximity that would change constantly, this approach creates fixed
        # topological connections that persist even when some links become temporarily invalid due to movement.
        # That is, this ISL always connects a "deterministic" neighbor (p,q)
        for i in range(num_orbits):
            p = (i + 1) % num_orbits

            # Get satellites in current and next orbit
            orbit_i = self.orbits[i]
            orbit_p = self.orbits[p]
            for j in range(num_sats_per_orbit):
                # The sat_idx $q$ of the nearest satellite shifts ``shift_per_orbit`` per orbit
                # If F=0 then we have q=j meaning there is no shift on sat_idx
                shift_per_orbit = self.phasing / num_sats_per_orbit
                total_shift_on_idx = (i - p) * shift_per_orbit
                target_q = np.round(j + total_shift_on_idx)  # Round to the nearest
                q = int(target_q % num_sats_per_orbit)  # Wrap-around to get a valid sat_idx

                sat_i_j = orbit_i[j]
                sat_p_q = orbit_p[q]
                _add_link_between_sats(sat_i_j, sat_p_q)

                # Record the ISLs topology
                self.ISL_E[sat_i_j.id] = sat_p_q.id
                self.ISL_W[sat_p_q.id] = sat_i_j.id

    def _setup_GSLs(self):
        """Set up the initial ground-satellite links by finding the nearest visible satellite for each ground station."""
        for gs in self.ground_stations.values():
            if len(gs.outgoing_links) >= self.mas_gsl_per_gs:
                continue  # already connected

            # Find the closest visible satellite that is not already connected
            candidates = []
            for sat in self.satellites.values():
                if self._is_satellite_visible(gs, sat) and self.S2G[sat.id] < self.max_gsl_per_sat and gs.id not in sat.outgoing_links:
                    distance = np.linalg.norm(gs.position - sat.position)
                    candidates.append((sat, distance))

            # Select the closest satellite(s) for uplink
            candidates.sort(key=lambda x: x[1])
            for sat, distance in candidates:
                # Create uplink
                delay = calculate_delay(gs.position, sat.position)
                uplink = Link(
                    source=gs,
                    sink=sat,
                    capacity=self.link_buffer_size,
                    propagation_delay=delay,
                    data_rate=self.gsl_data_rate,
                )
                self._add_link(uplink)

                # This should never happen
                assert (sat.id, gs.id) not in self.links, f"Downlink already exists for {sat.id} to {gs.id}"

                # Create downlink
                downlink = Link(
                    source=sat,
                    sink=gs,
                    capacity=self.link_buffer_size,
                    propagation_delay=delay,
                    data_rate=self.gsl_data_rate,
                )
                self._add_link(downlink)

                # Record the num ground stations connected to that satellite
                self.S2G[sat.id] += 1

                if len(gs.outgoing_links) >= self.mas_gsl_per_gs:
                    break

    def _add_node(self, node: Node):
        """Add a node to the network."""
        super()._add_node(node)
        if isinstance(node, Satellite):
            self.satellites[node.id] = node
        elif isinstance(node, GroundStation):
            self.ground_stations[node.id] = node

    def _change_topology(self, timestamp: float, on_link_disconnected: Callable[[Link], None]):
        """
        Update the network topology based on satellite positions.

        Args:
            timestamp: Current simulation time
            on_link_disconnected: Callback for handling link unavailability.
        """
        if not self.ground_stations:
            self._change_satellite_only_topology(timestamp, on_link_disconnected)
            return

        # Update node positions
        for node in self.nodes.values():
            node.update(timestamp)

        # Check existing links and remove those that are no longer valid.
        links_unavailable: list[Link] = []
        for link in self.links.values():
            self._update_link(link)
            if not link.is_connected:
                links_unavailable.append(link)

        # Remove invalid links
        for link in links_unavailable:
            # Handle the link unavailable events
            if on_link_disconnected is not None:
                on_link_disconnected(link)

            # We have two kinds of topology changes: link handovers and temporary link unavailabile events
            # We should only remove links for handovers (GSLs in this case)
            if not self.is_ISL(link):
                # NESW ISLs could be temporarily unavailable, but will be re-established when the satellites move closer
                # We maintain a stable mesh topology so we never remove NESW ISLs

                # TIPS: It is possible to have more than 4 ISLs such that we can dynamically connect unconnected neighboring
                # satellites to establish connections crossing parts in the network (those parts moving in different directions)
                (s, r) = link.id
                self._remove_link((s, r))
                # self._remove_link((r, s))

                # Update the number of ground stations connected to the satellite
                if link.source.id in self.S2G:
                    self.S2G[link.source.id] -= 1
                if link.sink.id in self.S2G:
                    self.S2G[link.sink.id] -= 1

        # Add new ground-satellite links as needed
        self._setup_GSLs()

    def _rebuild_vector_update_cache(self):
        self._satellite_update_list = list(self.satellites.values())
        self._satellite_ids = np.array(
            [sat.id for sat in self._satellite_update_list],
            dtype=np.int64,
        )
        self._satellite_altitudes = np.array(
            [sat.altitude for sat in self._satellite_update_list],
            dtype=np.float64,
        )
        self._satellite_raan_rad = np.deg2rad(
            [sat.raan for sat in self._satellite_update_list]
        )
        self._satellite_inc_rad = np.deg2rad(
            [sat.inclination for sat in self._satellite_update_list]
        )
        self._satellite_true_anomaly = np.array(
            [sat.true_anomaly for sat in self._satellite_update_list],
            dtype=np.float64,
        )
        self._satellite_positions_by_id = np.empty((self.num_nodes, 3), dtype=np.float64)
        self._satellite_positions_by_id[self._satellite_ids] = np.array(
            [sat.position for sat in self._satellite_update_list],
            dtype=np.float64,
        )

        self._link_update_list = list(self.links.values())
        self._link_source_ids = np.array(
            [link.source.id for link in self._link_update_list],
            dtype=np.int64,
        )
        self._link_sink_ids = np.array(
            [link.sink.id for link in self._link_update_list],
            dtype=np.int64,
        )
        self._link_index_by_pair = {
            (int(source), int(sink)): idx
            for idx, (source, sink) in enumerate(zip(self._link_source_ids, self._link_sink_ids))
        }
        self._link_spf_weights = np.array(
            [_default_link_weight(link) for link in self._link_update_list],
            dtype=np.float64,
        )
        self._link_connected_array = np.array(
            [link.is_connected for link in self._link_update_list],
            dtype=bool,
        )
        self._link_delay_array = np.array(
            [link.propagation_delay for link in self._link_update_list],
            dtype=np.float64,
        )

    def _change_satellite_only_topology(
        self,
        timestamp: float,
        on_link_disconnected: Callable[[Link], None],
    ):
        self._update_satellite_positions_vectorized(timestamp)

        source_pos = self._satellite_positions_by_id[self._link_source_ids]
        sink_pos = self._satellite_positions_by_id[self._link_sink_ids]
        distances = np.linalg.norm(source_pos - sink_pos, axis=1)
        connected = distances <= self.max_isl_range
        delays = distances / LIGHT_SPEED_MS
        self._link_connected_array = connected
        self._link_delay_array = delays
        self._link_spf_weights = np.where(connected, delays, DISCONNECTED_LINK_WEIGHT)

        if on_link_disconnected is None and not self._sync_object_state:
            return

        links_unavailable: list[Link] = []
        for link, is_connected, delay in zip(self._link_update_list, connected, delays):
            link.is_connected = bool(is_connected)
            link.propagation_delay = float(delay)
            if not link.is_connected:
                links_unavailable.append(link)

        if on_link_disconnected is not None:
            for link in links_unavailable:
                on_link_disconnected(link)

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
        positions = np.column_stack((x_ecef, y_ecef, z_eci))

        self._satellite_positions_by_id[self._satellite_ids] = positions
        if not self._sync_object_state:
            return

        for sat, position in zip(self._satellite_update_list, positions):
            sat.position = position
            sat._projected_pos = None
            sat._great_circle_distances_cache.clear()

    def _get_shortest_edge_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.ground_stations and hasattr(self, "_link_spf_weights"):
            return self._link_source_ids, self._link_sink_ids, self._link_spf_weights
        return super()._get_shortest_edge_arrays()

    def _update_link(self, link: Link):
        """Check if a link is valid."""
        if self.is_ISL(link):  # Inter-satellite link
            distance = np.linalg.norm(link.source.position - link.sink.position)
            link.is_connected = distance <= self.max_isl_range
            # Update delay based on current distance
            link.propagation_delay = calculate_delay(link.source.position, link.sink.position)
        else:
            # Ground-to-satellite or satellite-to-ground link
            link.is_connected = self._is_satellite_visible(link.sink, link.source)
            # Update delay based on current distance
            link.propagation_delay = calculate_delay(link.source.position, link.sink.position)

    def _is_satellite_visible(self, gs: Node, sat: Node) -> bool:
        """Check if a satellite is visible from a ground station."""
        slant_range = distance_between(gs.position, sat.position)
        return slant_range <= self.max_gsl_range

    def get_visible_satellites_for_position(
        self,
        position: np.ndarray,
        max_count: int | None = None,
    ) -> list[tuple[Satellite, float]]:
        """Return visible satellites for an arbitrary terrestrial position."""
        positions_by_id = getattr(self, "_satellite_positions_by_id", None)
        sat_ids = getattr(self, "_satellite_ids", None)
        if positions_by_id is not None and sat_ids is not None and len(sat_ids) > 0:
            sat_positions = positions_by_id[sat_ids]
            distances = np.linalg.norm(sat_positions - position, axis=1)
            visible_idx = np.flatnonzero(distances <= self.max_gsl_range)
            visible_idx = visible_idx[np.argsort(distances[visible_idx])]
            if max_count is not None:
                visible_idx = visible_idx[:max_count]
            return [
                (self.satellites[int(sat_ids[idx])], float(distances[idx]))
                for idx in visible_idx
            ]

        candidates = []
        for sat in self.satellites.values():
            distance = distance_between(position, sat.position)
            if distance <= self.max_gsl_range:
                candidates.append((sat, distance))
        candidates.sort(key=lambda item: item[1])
        if max_count is not None:
            return candidates[:max_count]
        return candidates

    def get_nearest_satellite_for_position(
        self, position: np.ndarray
    ) -> tuple[Satellite, float] | tuple[None, None]:
        """Return the nearest currently visible satellite for an arbitrary terrestrial position."""
        candidates = self.get_visible_satellites_for_position(position, max_count=1)
        return candidates[0] if candidates else (None, None)

    def can_satellite_serve_position(self, sat_id: int, position: np.ndarray) -> bool:
        """Check whether a satellite can currently serve an arbitrary terrestrial position."""
        if sat_id not in self.satellites:
            return False

        positions_by_id = getattr(self, "_satellite_positions_by_id", None)
        if positions_by_id is not None and 0 <= sat_id < len(positions_by_id):
            return distance_between(position, positions_by_id[sat_id]) <= self.max_gsl_range

        return distance_between(position, self.satellites[sat_id].position) <= self.max_gsl_range

    def is_satellite(self, node_id: int) -> bool:
        """Check if a node is a satellite."""
        return node_id in self.satellites

    def is_ISL(self, link: Link) -> bool:
        """Check if the link is an inter-satellite link."""
        return self.is_satellite(node_id=link.source.id) and self.is_satellite(node_id=link.sink.id)

    def get_shortest_path(self, current: int, sink: int, weight_fn=None):
        """
        Find the shortest path between two nodes, relying on the precomputed routing table.
        Returns:
            (weight, path): The shortest path between the two nodes.
            If no path exists, returns [].
        """
        if current not in self.nodes or sink not in self.nodes:
            return float('inf'), []

        if weight_fn is None:
            weight_fn = _default_link_weight
            if not self.ground_stations:
                return self._get_shortest_path_from_next_hops(current=current, sink=sink)

        if self.ground_stations:
            G = self.G.copy()
            for gs in self.ground_stations.values():
                if gs.id != sink:
                    G.remove_node(gs.id)
        else:
            G = self.G

        paths_from_source = rx.dijkstra_shortest_paths(G, current, sink, weight_fn=weight_fn)
        if paths_from_source:
            path_data = paths_from_source[sink]
            if path_data is not None:
                path_weight = self._calculate_path_weight(path_data, weight_fn=weight_fn)
                return path_weight, path_data

        return float('inf'), []

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
                return float('inf'), []
            link_idx = self._link_index_by_pair.get((node, next_hop))
            if link_idx is None:
                return float('inf'), []
            path_weight += float(self._link_spf_weights[link_idx])
            path.append(next_hop)
            visited.add(next_hop)
            node = next_hop

        if node != sink:
            return float('inf'), []
        return path_weight, path

    def get_shortest_next_hop(self, current: int, sink: int) -> int | None:
        if current not in self.nodes or sink not in self.nodes or current == sink:
            return None

        if self.ground_stations:
            _path_weight, path = self.get_shortest_path(current=current, sink=sink)
            return path[1] if path and len(path) > 1 else None

        self._ensure_shortest_next_hop_rows(np.array([sink], dtype=np.int64))
        next_hops = self._shortest_next_hop_cache[sink]
        next_hop = int(next_hops[current])
        return next_hop if next_hop >= 0 else None

    def get_shortest_next_hops(self, current: np.ndarray, sink: np.ndarray) -> np.ndarray:
        next_hops = np.full(len(current), -1, dtype=np.int64)
        if len(current) == 0:
            return next_hops

        target_sinks = np.unique(sink[sink >= 0])
        self._ensure_shortest_next_hop_rows(target_sinks)
        for target_sink in target_sinks:
            target_sink = int(target_sink)
            table = self._shortest_next_hop_cache[target_sink]
            mask = sink == target_sink
            next_hops[mask] = table[current[mask]]
        return next_hops

    def precompute_shortest_next_hops(self, sinks: np.ndarray):
        if self.ground_stations:
            return
        self._ensure_shortest_next_hop_rows(np.asarray(sinks, dtype=np.int64))

    def _ensure_shortest_next_hop_rows(self, sinks: np.ndarray):
        sinks = np.unique(sinks[sinks >= 0])
        if len(sinks) == 0:
            return
        missing_sinks = np.array(
            [int(sink) for sink in sinks if int(sink) not in self._shortest_next_hop_cache],
            dtype=np.int32,
        )
        if len(missing_sinks) == 0:
            return

        reverse_csr = self._get_shortest_reverse_csr()
        _distances, predecessors = dijkstra(
            csgraph=reverse_csr,
            directed=True,
            indices=missing_sinks,
            return_predecessors=True,
        )
        if predecessors.ndim == 1:
            predecessors = predecessors[np.newaxis, :]

        next_hop_rows = predecessors.astype(np.int64, copy=False)
        next_hop_rows[next_hop_rows < 0] = -1
        next_hop_rows[np.arange(len(missing_sinks)), missing_sinks] = -1

        for sink, row in zip(missing_sinks, next_hop_rows):
            self._shortest_next_hop_cache[int(sink)] = row.copy()

    def _calculate_path_weight(self, path, weight_fn: Callable[[Link], float]) -> float:
        """
        Calculate the total weight of a path using the given weight function.
        
        Args:
            path: List of node IDs representing the path
            weight_fn: Weight function to apply to each link
            
        Returns:
            Total weight of the path
        """
        if len(path) < 2:
            return 0.0
        
        total_weight = 0.0
        for i in range(len(path) - 1):
            link = self.get_link(path[i], path[i+1])
            total_weight += weight_fn(link)
        
        return total_weight
