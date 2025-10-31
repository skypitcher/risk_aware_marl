from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sat_net.event import Event

from sat_net.util import NamedDict, NetworkError


class DataBlock:
    """
    Represents a data block, the collection of packets with the same source and destination.
    """

    def __init__(
        self,
        block_id: int,
        source: int,
        target: int,
        is_normal: bool,
        size: float,
        delay_limit: float,
        creation_time: float,
        ttl: int = 64,
    ):
        """
        Initialize a new DataBlock.

        Args:
            block_id: Unique identifier for the DataBlock.
            source: Node ID where the DataBlock originates.
            target: Node ID where the DataBlock is heading.
            is_normal: Whether the DataBlock is an normal sized one.
            size: Size of the DataBlock, in Megabits.
            creation_time: Time when the DataBlock was created.
            ttl: Time to live (max hop count before DataBlock is dropped).
            delay_limit: QoS delay requirement in milliseconds.
            ttl: Time to live (max hop count before DataBlock is dropped).
        """
        assert source != target, "Source and target cannot be the same"

        self.id = block_id
        self.source_id = source
        self.target_id = target
        self.is_normal_packet = is_normal
        self.size = size #Mbits
        self.creation_time = creation_time  # Timestamp in milliseconds
        self.delivery_time: float | None = None  # Timestamp in milliseconds
        self.ttl_max = ttl
        self.delay_limit = delay_limit  # QoS delay requirement in ms， if used

        self.ttl = ttl
        self.path: list[int] = []  # Track the path taken
        self.hops: int = 0

        self.current_location: int | None = None  # Should be set when the DataBlock is received at the source node
        self.delivered: bool = False
        self.dropped: bool = False
        self.drop_time: float | None = None
        self.drop_reason: NetworkError | None = None  # Reason for dropping the DataBlock


        # great circle distance to the target ground station in degrees
        self.initial_gcd: float = 0.0
        self.shortest_gcd: float = 0.0

        self.first_gsl_delay: float = 0  # in ms
        self.final_gsl_delay: float = 0  # in ms

        # delay distribution
        self.e2e_delay: float = 0
        self.queue_delay: float = 0  # queueing delay
        self.transmission_delay: float = 0  # transmission delay
        self.propagation_delay: float = 0  # propagation delay

        self.total_queue_cost: float = 0.0 # indicates the network congestion level, measured in ms

        # For canceling the event associated with the DataBlock when it is dropped
        self.last_event: Optional[Event] = None  # Event that associated with the DataBlock

        # For tracking the outcome of the last action
        self.last_action: Optional[NamedDict] = None
        self.last_action_time = None
        self.trajectory: list[NamedDict] = []

    def reset(self):
        self.delivery_time = None
        self.ttl = self.ttl_max
        self.path = []
        self.hops = 0

        self.current_location = None
        self.delivered = False
        self.dropped = False
        self.drop_reason = None

        self.shortest_gcd = self.initial_gcd

        self.first_gsl_delay = 0  # in ms
        self.final_gsl_delay = 0  # in ms

        # delay distribution
        self.e2e_delay = 0
        self.queue_delay = 0
        self.transmission_delay = 0
        self.propagation_delay = 0

        self.total_queue_cost = 0.0

        self.last_event= None

        self.last_action = None
        self.last_action_time = None
        self.trajectory = []

    @property
    def total_delay(self) -> float | None:
        """
        Calculate the total delay of the DataBlock.
        """
        if self.delivered:
            return self.delivery_time - self.creation_time
        else:
            return None

    def cancel_event(self):
        """
        Cancel the event associated with this DataBlock.
        """
        if self.last_event is not None:
            self.last_event.is_cancelled = True
            self.last_event = None

    def to_dict(self):
        return {
            "packet_id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "is_normal_packet": self.is_normal_packet,
            "size": self.size,
            "creation_time": self.creation_time,
            "delivery_time": self.delivery_time,
            "total_delay": self.total_delay,
            "queue_delay": self.queue_delay,
            "transmission_delay": self.transmission_delay,
            "propagation_delay": self.propagation_delay,
            "hops": self.hops,
            "ttl": self.ttl,
            "ttl_max": self.ttl_max,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "drop_time": self.drop_time,
            "drop_reason": str(self.drop_reason) if self.drop_reason else None,
            "total_queue_cost": self.total_queue_cost,
            "first_gsl_delay": self.first_gsl_delay,
            "final_gsl_delay": self.final_gsl_delay,
        }

    def __repr__(self) -> str:
        return f"DataBlock(id={self.id}, src={self.source_id}, dst={self.target_id}, cur_loc={self.current_location}, size={self.size}, ttl={self.ttl})"

    def __eq__(self, other: "DataBlock") -> bool:
        """
        Check if two DataBlocks are the same. Only the id is considered.

        Args:
            other: The other data block to compare with

        Returns:
            True if the two DataBlocks are the same, False otherwise
        """
        return self.id == other.id
