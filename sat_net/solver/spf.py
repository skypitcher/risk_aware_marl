from __future__ import annotations

import numpy as np

from sat_net.solver.base_solver import BaseSolver, RoutingBatch, RoutingDecision

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised only when JAX is not installed.
    jax = None
    jnp = None


def spf_next_hops(
    current_sat_ids: np.ndarray,
    target_region_ids: np.ndarray,
    target_access_sat_ids: np.ndarray,
    region_next_hop_table: np.ndarray,
) -> np.ndarray:
    valid = (
        (current_sat_ids >= 0)
        & (target_region_ids >= 0)
        & (target_access_sat_ids >= 0)
    )
    safe_regions = np.where(valid, target_region_ids, 0)
    safe_current = np.where(valid, current_sat_ids, 0)
    table_next_hops = region_next_hop_table[safe_regions, safe_current]
    return np.where(valid, table_next_hops, -1).astype(np.int64, copy=False)


if jax is not None:

    @jax.jit
    def _jax_spf_next_hops(
        current_sat_ids,
        target_region_ids,
        target_access_sat_ids,
        region_next_hop_table,
    ):
        valid = (
            (current_sat_ids >= 0)
            & (target_region_ids >= 0)
            & (target_access_sat_ids >= 0)
        )
        safe_regions = jnp.where(valid, target_region_ids, 0)
        safe_current = jnp.where(valid, current_sat_ids, 0)
        table_next_hops = region_next_hop_table[safe_regions, safe_current]
        return jnp.where(valid, table_next_hops, -1)

else:
    _jax_spf_next_hops = None


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def jax_spf_next_hops(
    current_sat_ids: np.ndarray,
    target_region_ids: np.ndarray,
    target_access_sat_ids: np.ndarray,
    region_next_hop_table,
) -> np.ndarray:
    if _jax_spf_next_hops is None:
        raise RuntimeError("JAX is not installed.")

    batch_size = len(current_sat_ids)
    if batch_size == 0:
        return np.empty(0, dtype=np.int64)

    padded_size = _next_power_of_two(batch_size)
    pad_width = padded_size - batch_size
    current = np.pad(current_sat_ids, (0, pad_width), constant_values=-1)
    target_regions = np.pad(target_region_ids, (0, pad_width), constant_values=-1)
    target_access = np.pad(target_access_sat_ids, (0, pad_width), constant_values=-1)

    next_hops = _jax_spf_next_hops(
        jnp.asarray(current),
        jnp.asarray(target_regions),
        jnp.asarray(target_access),
        jnp.asarray(region_next_hop_table),
    )
    return np.asarray(next_hops[:batch_size], dtype=np.int64)


class SPF(BaseSolver):
    """Shortest-path policy backed by a precomputed region-to-next-hop table."""

    requires_shortest_path_table = True

    def __init__(self, use_jax: bool | None = None):
        self.use_jax = _jax_spf_next_hops is not None if use_jax is None else use_jax
        if self.use_jax and _jax_spf_next_hops is None:
            raise RuntimeError("SPF was configured to use JAX, but JAX is not installed.")
        self._jax_table_key = None
        self._jax_region_next_hop_table = None

    @property
    def name(self) -> str:
        return "SPF"

    def get_stats(self) -> str | None:
        return f"backend={'jax' if self.use_jax else 'numpy'}"

    def next_hops(self, batch: RoutingBatch) -> RoutingDecision:
        if batch.region_next_hop_table is None:
            raise ValueError("SPF requires region_next_hop_table in RoutingBatch.")

        if self.use_jax:
            next_hops = jax_spf_next_hops(
                current_sat_ids=batch.current_sat_ids,
                target_region_ids=batch.target_region_ids,
                target_access_sat_ids=batch.target_access_sat_ids,
                region_next_hop_table=self._get_jax_region_next_hop_table(batch),
            )
            return RoutingDecision(next_hop_sat_ids=next_hops)

        return RoutingDecision(
            next_hop_sat_ids=spf_next_hops(
                current_sat_ids=batch.current_sat_ids,
                target_region_ids=batch.target_region_ids,
                target_access_sat_ids=batch.target_access_sat_ids,
                region_next_hop_table=batch.region_next_hop_table,
            )
        )

    def _get_jax_region_next_hop_table(self, batch: RoutingBatch):
        table = batch.region_next_hop_table
        table_key = (id(table), table.shape, batch.region_next_hop_version)
        if table_key != self._jax_table_key:
            self._jax_region_next_hop_table = jnp.asarray(table)
            self._jax_table_key = table_key
        return self._jax_region_next_hop_table
