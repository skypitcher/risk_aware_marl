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


@dataclass(slots=True)
class ContinuingMetricsAccumulator:
    """Accumulates packet-weighted metrics across continuing rollout windows."""

    windows: int = 0
    elapsed_ms: float = 0.0
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
    delivered_mbit: float = 0.0
    e2e_delay_sum: float = 0.0
    queue_delay_sum: float = 0.0
    transmission_delay_sum: float = 0.0
    propagation_delay_sum: float = 0.0
    normal_e2e_delay_sum: float = 0.0
    normal_queue_delay_sum: float = 0.0
    normal_transmission_delay_sum: float = 0.0
    normal_propagation_delay_sum: float = 0.0
    small_e2e_delay_sum: float = 0.0
    small_queue_delay_sum: float = 0.0
    small_transmission_delay_sum: float = 0.0
    small_propagation_delay_sum: float = 0.0
    cost_sum: float = 0.0
    cost_normal_sum: float = 0.0
    cost_small_sum: float = 0.0

    def add(self, metrics: Metrics, duration_ms: float) -> None:
        duration_seconds = max(float(duration_ms), 0.0) / 1000.0
        self.windows += 1
        self.elapsed_ms += max(float(duration_ms), 0.0)
        self.generated += int(metrics.generated)
        self.generated_normal_packet += int(metrics.generated_normal_packet)
        self.generated_small_packet += int(metrics.generated_small_packet)
        self.delivered += int(metrics.delivered)
        self.delivered_normal_packet += int(metrics.delivered_normal_packet)
        self.delivered_small_packet += int(metrics.delivered_small_packet)
        self.dropped += int(metrics.dropped)
        self.dropped_by_ttl += int(metrics.dropped_by_ttl)
        self.dropped_normal_packet += int(metrics.dropped_normal_packet)
        self.dropped_small_packet += int(metrics.dropped_small_packet)
        self.delivered_mbit += float(metrics.throughput) * duration_seconds

        self.e2e_delay_sum += float(metrics.e2e_delay_mean) * metrics.delivered
        self.queue_delay_sum += float(metrics.queue_delay_mean) * metrics.delivered
        self.transmission_delay_sum += float(metrics.transmission_delay_mean) * metrics.delivered
        self.propagation_delay_sum += float(metrics.propagation_delay_mean) * metrics.delivered
        self.normal_e2e_delay_sum += float(metrics.normal_packet_e2e_delay_mean) * metrics.delivered_normal_packet
        self.normal_queue_delay_sum += float(metrics.normal_packet_queue_delay_mean) * metrics.delivered_normal_packet
        self.normal_transmission_delay_sum += (
            float(metrics.normal_packet_transmission_delay_mean) * metrics.delivered_normal_packet
        )
        self.normal_propagation_delay_sum += float(metrics.normal_packet_propagation_delay_mean) * metrics.delivered_normal_packet
        self.small_e2e_delay_sum += float(metrics.small_packet_e2e_delay_mean) * metrics.delivered_small_packet
        self.small_queue_delay_sum += float(metrics.small_packet_queue_delay_mean) * metrics.delivered_small_packet
        self.small_transmission_delay_sum += (
            float(metrics.small_packet_transmission_delay_mean) * metrics.delivered_small_packet
        )
        self.small_propagation_delay_sum += float(metrics.small_packet_propagation_delay_mean) * metrics.delivered_small_packet
        self.cost_sum += float(metrics.cost_mean) * metrics.delivered
        self.cost_normal_sum += float(metrics.cost_normal_packet_mean) * metrics.delivered_normal_packet
        self.cost_small_sum += float(metrics.cost_small_packet_mean) * metrics.delivered_small_packet

    def to_metrics(self) -> Metrics:
        elapsed_seconds = max(self.elapsed_ms / 1000.0, 1e-12)
        return Metrics(
            generated=self.generated,
            generated_normal_packet=self.generated_normal_packet,
            generated_small_packet=self.generated_small_packet,
            delivered=self.delivered,
            delivered_normal_packet=self.delivered_normal_packet,
            delivered_small_packet=self.delivered_small_packet,
            dropped=self.dropped,
            dropped_by_ttl=self.dropped_by_ttl,
            dropped_normal_packet=self.dropped_normal_packet,
            dropped_small_packet=self.dropped_small_packet,
            throughput=self.delivered_mbit / elapsed_seconds,
            service_rate=self.delivered / elapsed_seconds,
            delivery_rate=self.delivered / self.generated if self.generated else 0.0,
            drop_rate=self.dropped / self.generated if self.generated else 0.0,
            normal_packet_delivery_rate=(
                self.delivered_normal_packet / self.generated_normal_packet if self.generated_normal_packet else 0.0
            ),
            normal_packet_drop_rate=(
                self.dropped_normal_packet / self.generated_normal_packet if self.generated_normal_packet else 0.0
            ),
            small_packet_delivery_rate=(
                self.delivered_small_packet / self.generated_small_packet if self.generated_small_packet else 0.0
            ),
            small_packet_drop_rate=(
                self.dropped_small_packet / self.generated_small_packet if self.generated_small_packet else 0.0
            ),
            e2e_delay_mean=self.e2e_delay_sum / self.delivered if self.delivered else 0.0,
            queue_delay_mean=self.queue_delay_sum / self.delivered if self.delivered else 0.0,
            transmission_delay_mean=self.transmission_delay_sum / self.delivered if self.delivered else 0.0,
            propagation_delay_mean=self.propagation_delay_sum / self.delivered if self.delivered else 0.0,
            normal_packet_e2e_delay_mean=(
                self.normal_e2e_delay_sum / self.delivered_normal_packet if self.delivered_normal_packet else 0.0
            ),
            normal_packet_queue_delay_mean=(
                self.normal_queue_delay_sum / self.delivered_normal_packet if self.delivered_normal_packet else 0.0
            ),
            normal_packet_transmission_delay_mean=(
                self.normal_transmission_delay_sum / self.delivered_normal_packet if self.delivered_normal_packet else 0.0
            ),
            normal_packet_propagation_delay_mean=(
                self.normal_propagation_delay_sum / self.delivered_normal_packet if self.delivered_normal_packet else 0.0
            ),
            small_packet_e2e_delay_mean=(
                self.small_e2e_delay_sum / self.delivered_small_packet if self.delivered_small_packet else 0.0
            ),
            small_packet_queue_delay_mean=(
                self.small_queue_delay_sum / self.delivered_small_packet if self.delivered_small_packet else 0.0
            ),
            small_packet_transmission_delay_mean=(
                self.small_transmission_delay_sum / self.delivered_small_packet if self.delivered_small_packet else 0.0
            ),
            small_packet_propagation_delay_mean=(
                self.small_propagation_delay_sum / self.delivered_small_packet if self.delivered_small_packet else 0.0
            ),
            cost_mean=self.cost_sum / self.delivered if self.delivered else 0.0,
            cost_small_packet_mean=self.cost_small_sum / self.delivered_small_packet if self.delivered_small_packet else 0.0,
            cost_normal_packet_mean=self.cost_normal_sum / self.delivered_normal_packet if self.delivered_normal_packet else 0.0,
        )

    def to_record(self) -> dict[str, int | float | dict[str, int | float]]:
        return {
            "windows": self.windows,
            "elapsed_ms": self.elapsed_ms,
            "elapsed_seconds": self.elapsed_ms / 1000.0,
            "metrics": self.to_metrics().to_dict(),
        }

    def state_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_state(cls, state: dict | None) -> "ContinuingMetricsAccumulator":
        if not state:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{key: state[key] for key in valid_keys if key in state})
