import heapq
from enum import Enum
from typing import Any, Optional

from sat_net.util import ms2str


class EventType(Enum):
    """Types of events in the simulation."""

    DATA_GENERATED = 0  # New DataBlock is generated at a source node
    TRANSMIT_END = 1  # A DataBlock has been fully serialized onto the link
    DATA_FORWARDED = 2  # A DataBlock arrived at the receiver
    TOPOLOGY_CHANGE = 3  # Network topology changes due to satellite movement
    TIME_LIMIT_REACHED = 4  # Simulation ends due to time limit
    TRAIN_EVENT = 5  # Training models
    DEBUG_CALLBACK_EVENT = 6  # Callback event


class Event:
    """Event in the simulation."""

    def __init__(self, time: float, event_type: EventType, data: Optional[Any] = None):
        self.time = time  # When the event occurs (milliseconds)
        self.event_type = event_type  # Type of event
        self.data = data
        self.is_cancelled = False  # Flag indicating if the event has been cancelled

    def __str__(self):
        return f"Event(time={ms2str(self.time)}, type={self.event_type.name})"

    def __lt__(self, other: "Event"):
        return self.time < other.time


class EventScheduler:
    def __init__(self):
        self.event_queue = []  # min-heap

    def push_event(self, event_type: EventType, time: float, data: Optional[Any] = None) -> Event:
        """
        Schedule an event to occur at the specified time.

        Args:
            event_type: Type of event
            time: When the event should occur
            data: Additional data for the event
        """
        if data is None:
            data = {}

        event = Event(time=time, event_type=event_type, data=data)
        heapq.heappush(self.event_queue, event)
        return event

    def pop_event(self) -> Event:
        """
        Pop the next event from the event queue.
        """
        return heapq.heappop(self.event_queue)

    def peek_event(self) -> Event:
        """
        Peek at the next event from the event queue.
        """
        return self.event_queue[0]

    def __len__(self):
        """
        Get the number of events in the event queue.
        """
        return len(self.event_queue)

    def is_empty(self) -> bool:
        """
        Check if the event queue is empty.
        """
        return len(self.event_queue) == 0

    def reset(self):
        """
        Reset the event scheduler.
        """
        self.event_queue = []
