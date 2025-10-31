from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import numpy as np

from sat_net.buffer import DictBuffer
from sat_net.datablock import DataBlock
from sat_net.geometric import (
    EARTH_R_KM,
    geo_to_ecef_position,
    get_position_ecef,
    get_projected_position,
    great_circle_distance,
)
from sat_net.util import NetworkError

if TYPE_CHECKING:
    from .link import Link


class Node:
    """
    Represents a node (potentially movable) in the network with queueing buffers for receiving and sending data blocks.
    """

    def __init__(
        self,
        name: str,
        buffer_size: float,
        node_id: Optional[int] = None,
        position: Optional[np.ndarray] = None,
    ):
        """
        Initialize a new node.

        Args:
            name: Human-readable name for the node
            buffer_size: buffer capacity for the recv buffer
            node_id: Unique identifier for the node
            position: Initial 3D position (x, y, z) in km
        """
        self.name = name
        self.id = node_id
        self.position = position
        self.recv_buffer = DictBuffer(capacity=buffer_size)
        self.outgoing_links: OrderedDict[int, "Link"] = OrderedDict()  # Dict of outgoing links: {destination_id: Link}

        self._projected_pos = None
        self._great_circle_distances_cache = {}

        self.num_packet_recv = 0
        self.num_packet_sent = 0
        self.num_packet_dropped = 0
        self.max_load_factor = 0.0

    def reset_stats(self):
        self.num_packet_recv = 0
        self.num_packet_sent = 0
        self.num_packet_dropped = 0
        self.max_load_factor = 0.0

    def is_satellite(self):
        raise NotImplementedError("Not implemented yet")

    def is_ground_station(self):
        return not self.is_satellite()

    def __eq__(self, other: "Node"):
        return self.id == other.id

    def __repr__(self):
        return f"Node({self.id}, {self.name})"

    def get_load_factor(self) -> float:
        """Get the load of the node."""
        return self.recv_buffer.get_load_factor()

    def can_receive(self, data_size: float) -> bool:
        """Check if the node can receive a DataBlock."""
        return self.recv_buffer.get_remaining_capacity() >= data_size

    def add_outgoing_link(self, link: "Link"):
        """Add an outgoing link from this node."""
        assert link.source.id == self.id
        dest_id = link.sink.id
        self.outgoing_links[dest_id] = link

    def remove_outgoing_link(self, destination: int):
        """Remove an outgoing link from this node."""
        if destination in self.outgoing_links:
            del self.outgoing_links[destination]

    def update(self, timestamp: float):
        """Update the node's position to the given timestamp."""
        # By default, this is a fixed node. Use a subclass to define the movement logic
        self.position = self.predict_position(timestamp)
        self._projected_pos = None
        self._great_circle_distances_cache.clear()

    def predict_position(self, timestamp: float | np.ndarray) -> np.ndarray:
        """Predict the position of the node at the given timestamp."""
        if np.size(timestamp) == 1:
            return self.position
        else:
            return np.tile(self.position, (np.size(timestamp), 1))

    def get_projected_position(self):
        """Get the projected position of the node on the surface of the Earth, in the form of (longitude, latitude)."""
        if self._projected_pos is None:
            longitude, latitude = get_projected_position(self.position[0], self.position[1], self.position[2])
            self._projected_pos = (longitude, latitude)
        return self._projected_pos

    def get_great_circle_distance_to(self, target_node: "Node"):
        """Get the great circle distance to a target node, using a cache if available."""
        if target_node.id not in self._great_circle_distances_cache:
            lon1, lat1 = self.get_projected_position()
            lon2, lat2 = target_node.get_projected_position()
            distance = great_circle_distance(lon1, lat1, lon2, lat2)
            self.update_great_circle_distances_cache(target_node.id, distance)
            target_node.update_great_circle_distances_cache(self.id, distance)
            return distance
        else:
            return self._great_circle_distances_cache[target_node.id]
        
    def update_great_circle_distances_cache(self, target_node_id: int, distance: float):
        self._great_circle_distances_cache[target_node_id] = distance

    def receive(self, data_block: "DataBlock") -> tuple[bool, Optional["NetworkError"]]:
        """
        Attempt to receive a DataBlock. Return True if successfully buffered or delivered, False if dropped.

        Returns:
            success: bool - True if successfully buffered or delivered, False if dropped
            reason_if_failed: ReasonFailed - Reason for failure if not successful, otherwise None
        """
        reason_if_failed = None

        if not self.can_receive(data_size=data_block.size):
            reason_if_failed = NetworkError.NODE_FULL
            self.num_packet_dropped += 1
            return False, reason_if_failed

        self.recv_buffer.add(data_block=data_block)
        self.num_packet_recv += 1

        self.max_load_factor = max(self.max_load_factor, self.get_load_factor())

        return True, reason_if_failed

    def send(
        self,
        data_block: "DataBlock",
        next_hop: int,
        current_time: float,
    ):
        """
        Send a DataBlock to the next hop.

        This is done by conceptually removing a DataBlock from the recv_buffer and adding it to the output channel for the specified next hop.

        Args:
            data_block: The DataBlock to forward (must be at the front of recv_buffer)
            next_hop: The ID of the next hop node
            current_time: The current time in milliseconds

        Returns:
            success: Whether the DataBlock was successfully moved to the send buffer
            wait_time: The time needed to wait in the queue for transmission
            reason_if_failed: Reason for failure if not successful, otherwise None
        """
        assert data_block.id in self.recv_buffer, f"DataBlock {data_block.id} not in receive buffer"

        # Check if we have a link to the next hop
        if next_hop not in self.outgoing_links:
            reason_if_failed = NetworkError.INVALID_NEXT_HOP
            self.num_packet_dropped += 1
            return False, None, reason_if_failed

        # Check if the link is currently unavailable
        link = self.outgoing_links[next_hop]
        if not link.is_connected:
            reason_if_failed = NetworkError.INVALID_NEXT_HOP
            self.num_packet_dropped += 1
            return False, None, reason_if_failed

        if link.get_remaining_capacity() < data_block.size:
            reason_if_failed = NetworkError.LINK_FULL
            return False, None, reason_if_failed

        success, (wait_time, transmit_time), reason_if_failed = link.transmit(packet=data_block, current_time=current_time)
        if not success:
            self.num_packet_dropped += 1
            return False, None, reason_if_failed

        self.recv_buffer.remove(data_block.id)
        self.num_packet_sent += 1

        return True, (wait_time, transmit_time), reason_if_failed


class GroundStation(Node):
    """Represents a ground station in the network."""

    def __init__(
        self,
        name: str,
        buffer_size: float,
        latitude: float,
        longitude: float,
        altitude: float = 0.0,
        population: float = 0.0,
        node_id: Optional[int] = None,
    ):
        """
        Initialize a new ground station.

        Args:
            name: Human-readable name for the ground station
            buffer_size: buffer capacity of the recv buffer
            latitude: Latitude of the ground station in degrees
            longitude: Longitude of the ground station in degrees
            altitude: Altitude of the ground station in kilometers
            population: Population of the ground station in millions
            node_id: Unique identifier for the ground station
        """
        super().__init__(name=name, buffer_size=buffer_size, node_id=node_id)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.population = population
        self.timezone_offset = self.longitude / 15.0

        # Calculate position
        self.position = geo_to_ecef_position(lat=self.latitude, lon=self.longitude, alt=self.altitude)


    def is_satellite(self):
        return False


class Satellite(Node):
    """Represents a satellite in the network."""

    def __init__(
        self,
        name: str,
        orbit: int,
        index_in_orbit: int,
        altitude: float,
        inclination: float,
        raan: float,
        true_anomaly: float,
        eccentricity: float,
        arg_perigee: float,
        buffer_size: float,
        node_id: Optional[int] = None,
    ):
        """
        Initialize a new satellite.

        Args:
            name: Human-readable name for the satellite
            orbit: Orbit number
            index_in_orbit: Index of the satellite in the orbit
            altitude: Altitude of the satellite in kilometers
            inclination: Inclination of the orbit in degrees
            raan: Right ascension of the ascending node in degrees
            true_anomaly: True anomaly (angular position) in degrees
            eccentricity: Eccentricity of the orbit
            arg_perigee: Argument of perigee in degrees
            buffer_size: buffer capacity of the recv buffer
            node_id: Unique identifier for the satellite
        """
        super().__init__(name=name, buffer_size=buffer_size, node_id=node_id)
        self.orbit = orbit
        self.index_in_orbit = index_in_orbit

        # Orbital parameters
        self.altitude = altitude
        self.inclination = inclination
        self.raan = raan
        self.true_anomaly = true_anomaly
        self.orbit_radius = EARTH_R_KM + altitude
        self.eccentricity = eccentricity
        self.arg_perigee = arg_perigee

        # Calculate initial position
        self.position = self.predict_position(0)

    def is_satellite(self):
        return True

    def predict_position(self, timestamp: float | np.ndarray) -> np.ndarray:
        """Predicts the satellite's ECEF position at a given timestamp."""
        return get_position_ecef(
            altitude=self.altitude,
            raan=self.raan,
            inclination=self.inclination,
            true_anomaly=self.true_anomaly,
            timestamp=timestamp,
        )

    def update(self, timestamp: float):
        """Update the satellite's position based on orbital parameters and time."""
        self.position = self.predict_position(timestamp)
        self._projected_pos = None
        self._great_circle_distances_cache.clear()
