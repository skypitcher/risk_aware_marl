from __future__ import annotations

import numpy as np

from sat_net.solver.base_solver import BaseSolver, RoutingBatch, RoutingDecision


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


class SPF(BaseSolver):
    """Shortest-path policy backed by a precomputed region-to-next-hop table."""

    requires_shortest_path_table = True

    @property
    def name(self) -> str:
        return "SPF"

    def next_hops(self, batch: RoutingBatch) -> RoutingDecision:
        if batch.region_next_hop_table is None:
            raise ValueError("SPF requires region_next_hop_table in RoutingBatch.")

        return RoutingDecision(
            next_hop_sat_ids=spf_next_hops(
                current_sat_ids=batch.current_sat_ids,
                target_region_ids=batch.target_region_ids,
                target_access_sat_ids=batch.target_access_sat_ids,
                region_next_hop_table=batch.region_next_hop_table,
            )
        )
