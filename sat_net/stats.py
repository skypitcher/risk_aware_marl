from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class Metrics:
    """Packet-weighted aggregate metrics produced by the flowlet array kernel."""

    # Counts.
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

    # Rates.
    throughput: float = 0.0
    service_rate: float = 0.0
    delivery_rate: float = 0.0
    drop_rate: float = 0.0
    normal_packet_delivery_rate: float = 0.0
    normal_packet_drop_rate: float = 0.0
    small_packet_delivery_rate: float = 0.0
    small_packet_drop_rate: float = 0.0

    # Mean delays.
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

    # Mean queue-risk costs.
    cost_mean: float = 0.0
    cost_small_packet_mean: float = 0.0
    cost_normal_packet_mean: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def to_json(self, pretty: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=4 if pretty else None)

    def get_summary(self) -> str:
        delay_all = self._format_delay(
            self.e2e_delay_mean,
            self.queue_delay_mean,
            self.propagation_delay_mean,
            self.transmission_delay_mean,
        )
        delay_small = self._format_delay(
            self.small_packet_e2e_delay_mean,
            self.small_packet_queue_delay_mean,
            self.small_packet_propagation_delay_mean,
            self.small_packet_transmission_delay_mean,
        )
        delay_normal = self._format_delay(
            self.normal_packet_e2e_delay_mean,
            self.normal_packet_queue_delay_mean,
            self.normal_packet_propagation_delay_mean,
            self.normal_packet_transmission_delay_mean,
        )
        return (
            f"TOT: {self.generated:<4} | "
            f"OK: {self.delivered:<4}({self.delivery_rate * 100:5.4f}%) | "
            f"DROP: {self.dropped:<4}({self.drop_rate * 100:5.4f}%) TTL={self.dropped_by_ttl:<4}| "
            f"TH: {self.throughput:6.2f} | "
            f"SR: {self.service_rate:6.2f} | "
            f"DELAY(QPT) {delay_all} S:{delay_small} N:{delay_normal} | "
            f"C: {self.cost_mean:.2f}(S:{self.cost_small_packet_mean:.2f}|N:{self.cost_normal_packet_mean:.2f})"
        )

    @staticmethod
    def _format_delay(e2e: float, queue: float, propagation: float, transmission: float) -> str:
        return f"{e2e:.1f}({queue:.1f}+{propagation:.1f}+{transmission:.1f})"
