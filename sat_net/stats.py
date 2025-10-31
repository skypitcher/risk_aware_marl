import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sat_net.util import NetworkError

if TYPE_CHECKING:
    from sat_net.datablock import DataBlock


def _safe_div(numerator: float, denominator: float) -> float:
    """Safely divides two numbers, returning 0.0 if the denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0


@dataclass
class TimeStats:
    """Holds time-related statistics."""

    elapsed: float = 0.0  # in ms

    def reset(self):
        """Resets time statistics."""
        self.elapsed = 0.0

    def update(self, delta_time: float):
        """Updates the elapsed time."""
        self.elapsed += delta_time

    @property
    def seconds(self) -> float:
        """Returns the elapsed time in seconds."""
        return self.elapsed / 1000.0


@dataclass
class DelayStats:
    """Holds delay statistics components."""

    total: float = 0.0
    queue: float = 0.0
    transmission: float = 0.0
    propagation: float = 0.0

    def reset(self):
        """Resets all delay statistics to their initial values."""
        self.total = 0.0
        self.queue = 0.0
        self.transmission = 0.0
        self.propagation = 0.0

    def update(self, packet: "DataBlock"):
        """Updates the delay statistics from a DataBlock."""
        total_delay = packet.total_delay
        if total_delay is None:
            return

        self.total += total_delay
        self.queue += packet.queue_delay
        self.transmission += packet.transmission_delay
        self.propagation += packet.propagation_delay


@dataclass
class Metrics:
    """A dataclass to hold all calculated metrics."""

    generated: int = 0
    generated_normal_packet: int = 0
    generated_small_packet: int = 0
    delivered: int = 0
    delivered_normal_packet: int = 0
    delivered_small_packet: int = 0
    dropped: int = 0
    dropped_by_ttl: int = 0
    dropped_normal_packet: int = 0
    dropped_small_packet: int = 0
    throughput: float = 0.0
    service_rate: float = 0.0
    delivery_rate: float = 0.0
    drop_rate: float = 0.0
    normal_packet_delivery_rate: float = 0.0
    normal_packet_drop_rate: float = 0.0
    small_packet_delivery_rate: float = 0.0
    small_packet_drop_rate: float = 0.0
    e2e_delay_mean: float = 0.0
    queue_delay_mean: float = 0.0
    transmission_delay_mean: float = 0.0
    propagation_delay_mean: float = 0.0
    normal_packet_e2e_delay_mean: float = 0.0
    normal_packet_queue_delay_mean: float = 0.0
    normal_packet_transmission_delay_mean: float = 0.0
    normal_packet_propagation_delay_mean: float = 0.0
    small_packet_e2e_delay_mean: float = 0.0
    small_packet_queue_delay_mean: float = 0.0
    small_packet_transmission_delay_mean: float = 0.0
    small_packet_propagation_delay_mean: float = 0.0
    cost_mean: float = 0.0
    cost_small_packet_mean: float = 0.0
    cost_normal_packet_mean: float = 0.0

    def to_json(self, pretty: bool = False) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, indent=4 if pretty else None)

    def get_summary(self) -> str:
        info_text = (
            f"TOT: {self.generated:<4} | "
            f"OK: {self.delivered:<4}({self.delivery_rate * 100:5.4f}%) | "
            f"DROP: {self.dropped:<4}({self.drop_rate * 100:5.4f}%) TTL={self.dropped_by_ttl:<4}| "
            f"TH: {self.throughput:6.2f} | "
            f"SR: {self.service_rate:6.2f} | "
            f"DELAY(QPT) {self.e2e_delay_mean:.1f}({self.queue_delay_mean:.1f}+{self.propagation_delay_mean:.1f}+{self.transmission_delay_mean:.1f})  "
            f"S:{self.small_packet_e2e_delay_mean:.1f}({self.small_packet_queue_delay_mean:.1f}+{self.small_packet_propagation_delay_mean:.1f}+{self.small_packet_transmission_delay_mean:.1f}) "
            f"N:{self.normal_packet_e2e_delay_mean:.1f}({self.normal_packet_queue_delay_mean:.1f}+{self.normal_packet_propagation_delay_mean:.1f}+{self.normal_packet_transmission_delay_mean:.1f}) | "
            f"C: {self.cost_mean:.2f}(S:{self.cost_small_packet_mean:.2f}|N:{self.cost_normal_packet_mean:.2f})"
        )
        return info_text


@dataclass
class Stats:
    """A class to hold statistics for the routing environment."""

    time: TimeStats = field(default_factory=TimeStats)

    num_packets_generated: int = 0
    num_packets_generated_normal_packet: int = 0
    num_packets_generated_small_packet: int = 0

    num_packets_delivered: int = 0
    num_normal_packet_packets_delivered: int = 0
    num_small_packet_packets_delivered: int = 0

    num_packets_dropped: int = 0
    num_normal_packet_packets_dropped: int = 0
    num_small_packet_packets_dropped: int = 0
    num_packets_dropped_by_ttl: int = 0

    total_throughput: float = 0.0

    all_packet_delay: DelayStats = field(default_factory=DelayStats)
    normal_packet_packet_delay: DelayStats = field(default_factory=DelayStats)
    small_packet_packet_delay: DelayStats = field(default_factory=DelayStats)

    total_cost: float = 0.0
    total_cost_small_packet: float = 0.0
    total_cost_normal_packet: float = 0.0

    def reset(self):
        """Resets all statistics to their initial values."""
        self.time.reset()
        self.num_packets_generated = 0
        self.num_packets_generated_normal_packet = 0
        self.num_packets_generated_small_packet = 0
        self.num_packets_delivered = 0
        self.num_normal_packet_packets_delivered = 0
        self.num_small_packet_packets_delivered = 0
        self.num_packets_dropped = 0
        self.num_normal_packet_packets_dropped = 0
        self.num_small_packet_packets_dropped = 0
        self.num_packets_dropped_by_ttl = 0
        self.total_throughput = 0.0

        self.all_packet_delay.reset()
        self.normal_packet_packet_delay.reset()
        self.small_packet_packet_delay.reset()

        self.total_cost: float = 0.0
        self.total_cost_small_packet: float = 0.0
        self.total_cost_normal_packet: float = 0.0

    def on_packet_generated(self, packet: "DataBlock"):
        """Updates the statistics when a packet is generated."""
        self.num_packets_generated += 1
        if packet.is_normal_packet:
            self.num_packets_generated_normal_packet += 1
        else:
            self.num_packets_generated_small_packet += 1

    def on_packet_finished(self, packet: "DataBlock"):
        """Updates the statistics when a packet is finished (delivered or dropped)."""
        if packet.delivered:
            self._on_packet_delivered(packet)
        elif packet.dropped:
            self._on_packet_dropped(packet)

    def _on_packet_delivered(self, packet: "DataBlock"):
        """Update stats for a delivered packet."""
        self.num_packets_delivered += 1
        if packet.is_normal_packet:
            self.num_normal_packet_packets_delivered += 1
        else:
            self.num_small_packet_packets_delivered += 1

        self.total_throughput += packet.size

        self.all_packet_delay.update(packet)

        if packet.is_normal_packet:
            self.normal_packet_packet_delay.update(packet)
        else:
            self.small_packet_packet_delay.update(packet)

        self.total_cost += packet.total_queue_cost
        if packet.is_normal_packet:
            self.total_cost_normal_packet += packet.total_queue_cost
        else:
            self.total_cost_small_packet += packet.total_queue_cost

    def _on_packet_dropped(self, packet: "DataBlock"):
        """Update stats for a dropped packet."""
        self.num_packets_dropped += 1
        if packet.is_normal_packet:
            self.num_normal_packet_packets_dropped += 1
        else:
            self.num_small_packet_packets_dropped += 1

        if packet.drop_reason == NetworkError.TTL_EXPIRED:
            self.num_packets_dropped_by_ttl += 1

    def calc_metrics(self) -> Metrics:
        """Calculates and returns a dictionary of performance metrics."""
        return Metrics(
            generated=self.num_packets_generated,
            generated_normal_packet=self.num_packets_generated_normal_packet,
            generated_small_packet=self.num_packets_generated_small_packet,
            delivered=self.num_packets_delivered,
            delivered_normal_packet=self.num_normal_packet_packets_delivered,
            delivered_small_packet=self.num_small_packet_packets_delivered,
            dropped=self.num_packets_dropped,
            dropped_by_ttl=self.num_packets_dropped_by_ttl,
            dropped_normal_packet=self.num_normal_packet_packets_dropped,
            dropped_small_packet=self.num_small_packet_packets_dropped,
            throughput=_safe_div(self.total_throughput, self.time.seconds),
            service_rate=_safe_div(self.num_packets_delivered, self.time.seconds),
            delivery_rate=_safe_div(self.num_packets_delivered, self.num_packets_generated),
            drop_rate=_safe_div(self.num_packets_dropped, self.num_packets_generated),
            normal_packet_delivery_rate=_safe_div(self.num_normal_packet_packets_delivered, self.num_packets_generated_normal_packet),
            normal_packet_drop_rate=_safe_div(self.num_normal_packet_packets_dropped, self.num_packets_generated_normal_packet),
            small_packet_delivery_rate=_safe_div(self.num_small_packet_packets_delivered, self.num_packets_generated_small_packet),
            small_packet_drop_rate=_safe_div(self.num_small_packet_packets_dropped, self.num_packets_generated_small_packet),
            e2e_delay_mean=_safe_div(self.all_packet_delay.total, self.num_packets_delivered),
            queue_delay_mean=_safe_div(self.all_packet_delay.queue, self.num_packets_delivered),
            transmission_delay_mean=_safe_div(self.all_packet_delay.transmission, self.num_packets_delivered),
            propagation_delay_mean=_safe_div(self.all_packet_delay.propagation, self.num_packets_delivered),
            normal_packet_e2e_delay_mean=_safe_div(self.normal_packet_packet_delay.total, self.num_normal_packet_packets_delivered),
            normal_packet_queue_delay_mean=_safe_div(self.normal_packet_packet_delay.queue, self.num_normal_packet_packets_delivered),
            normal_packet_transmission_delay_mean=_safe_div(self.normal_packet_packet_delay.transmission, self.num_normal_packet_packets_delivered),
            normal_packet_propagation_delay_mean=_safe_div(self.normal_packet_packet_delay.propagation, self.num_normal_packet_packets_delivered),
            small_packet_e2e_delay_mean=_safe_div(self.small_packet_packet_delay.total, self.num_small_packet_packets_delivered),
            small_packet_queue_delay_mean=_safe_div(self.small_packet_packet_delay.queue, self.num_small_packet_packets_delivered),
            small_packet_transmission_delay_mean=_safe_div(self.small_packet_packet_delay.transmission, self.num_small_packet_packets_delivered),
            small_packet_propagation_delay_mean=_safe_div(self.small_packet_packet_delay.propagation, self.num_small_packet_packets_delivered),
            cost_mean=_safe_div(self.total_cost, self.num_packets_delivered),
            cost_small_packet_mean=_safe_div(self.total_cost_small_packet, self.num_small_packet_packets_delivered),
            cost_normal_packet_mean=_safe_div(self.total_cost_normal_packet, self.num_normal_packet_packets_delivered),
        )
