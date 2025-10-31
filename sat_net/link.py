from collections import deque
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sat_net.datablock import DataBlock
    from sat_net.node import Node

from sat_net.util import NetworkError


class Link:
    """Represents a directed link between two nodes."""

    def __init__(self, source: "Node", sink: "Node", capacity: float, propagation_delay: float, data_rate: float):
        """
        Initialize a new link.

        Args:
            source: Source node ID
            sink: Destination node ID
            capacity: Buffer capacity in bytes
            propagation_delay: Propagation delay in milliseconds
            data_rate: Data rate in bytes per second
        """
        assert source != sink, "Source and sink cannot be the same"

        self.id = (source.id, sink.id)
        self.source = source
        self.sink = sink
        self.load = 0
        self.propagation_delay = propagation_delay  # ms
        self.capacity = capacity
        self.data_rate = data_rate  # 1.0 defaults to ISL capacity
        self.is_connected = True  # Whether the link is currently connected

        self.link_free_time: float = 0.0
        self.queue = deque()
        self.temp_buffer: dict[int, "DataBlock"] = {}
        self.size = 0.0

        self.num_packet_recv = 0
        self.num_packet_sent = 0
        self.num_packet_dropped = 0
        self.max_load_factor = 0.0
        self.max_queueing_delay = 0

    def reset_stats(self):
        self.num_packet_recv = 0
        self.num_packet_sent = 0
        self.num_packet_dropped = 0
        self.max_load_factor = 0.0
        self.max_queueing_delay = 0

    def get_queue_length_to_destination(self, destination: int) -> float:
        """Get the total size of packets in queue destined for a specific destination."""
        return sum(p.size for t, p in self.queue if p.target_id == destination)

    def get_busy_time_remaining(self, current_time: float) -> float:
        if self.link_free_time < current_time:
            return 0
        return self.link_free_time - current_time

    def transmit(self, packet: "DataBlock", current_time: float):
        """
        Add a DataBlock to the buffer if capacity allows.

        Returns:
            bool: True if the block was added, False if the buffer is full.
            wait_time: Queueing time needed in milliseconds.
            error: Error information if failed
        """

        self.num_packet_recv += 1
        remaining = self.get_remaining_capacity()
        if remaining < packet.size:
            self.num_packet_dropped += 1
            return False, None, NetworkError.LINK_FULL

        # update the link free time if the current time is greater than the link free time
        if self.link_free_time < current_time:
            self.link_free_time = current_time
            wait_time = 0
        else:
            wait_time = self.link_free_time - current_time  # queueing delay
            
        self.max_queueing_delay = max(self.max_queueing_delay, wait_time)

        transmit_time = packet.size / self.data_rate
        self.link_free_time += transmit_time

        self.queue.append((current_time, packet))

        self.size += packet.size
        self.max_load_factor = max(self.max_load_factor, self.get_load_factor())

        return True, (wait_time, transmit_time), NetworkError.SUCCESS

    def start_propagate(self, packet_id: int):
        """The packet is on-the-flight as all the bits of a packet have been put onto the link."""
        queue_time, packet = self.peek()
        if packet.id != packet_id:
            raise RuntimeError("Packet ID does not match")

        self.queue.popleft()
        self.temp_buffer[packet.id] = packet # move to buffer for counting propagation delays
        self.size -= packet.size

    def finish_propagate(self, packet_id: int):
        """A packet has been forwarded to the sink node of this link"""
        packet = self.temp_buffer[packet_id]
        del self.temp_buffer[packet.id]
        self.num_packet_sent += 1

    def drop_all(self):
        """Drop all data blocks in the buffer."""
        while not self.is_empty():
            queue_time, packet = self.queue.popleft()
            self.num_packet_dropped += 1
            yield packet
        for packet in self.temp_buffer.values():
            self.num_packet_dropped += 1
            yield packet
        self.temp_buffer.clear()
        self.size = 0
        self.link_free_time = 0

    def peek(self) -> "DataBlock":
        """Return the next DataBlock without removing it."""
        return self.queue[0]

    def get_data_size(self) -> float:
        """Return the current size of all data in the buffer."""
        return self.size

    def get_remaining_capacity(self) -> float:
        """Get the remaining capacity of the buffer."""
        return self.capacity - self.size

    def get_load_factor(self):
        """Return the load of the buffer."""
        return self.size / self.capacity

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self.queue) == 0

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return (
            f"Link({self.id}, {self.source.name}->{self.sink.name}, Load={self.get_load_factor()}, PropDelay={self.propagation_delay}ms)"
        )
