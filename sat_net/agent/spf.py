from __future__ import annotations

import numpy as np

from sat_net.agent.base_agent import BaseAgent, RoutingBatch, RoutingDecision


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


class SPFAgent(BaseAgent):
    """Shortest-path routing agent backed by a precomputed region-to-next-hop table."""

    requires_shortest_path_table = True

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "SPF"

    def get_stats(self) -> str | None:
        return "backend=numpy"

    def act(self, batch: RoutingBatch) -> RoutingDecision:
        if batch.region_next_hop_tables is not None:
            return RoutingDecision(next_hop_sat_ids=self._act_vector_batch(batch))
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

    def _act_vector_batch(self, batch: RoutingBatch) -> np.ndarray:
        actions = np.full(batch.decision_count, -1, dtype=np.int64)
        env_ids = batch.row_env_ids
        for env_id in np.unique(env_ids):
            env_id = int(env_id)
            if env_id < 0 or env_id >= len(batch.region_next_hop_tables):
                raise ValueError(f"SPF received out-of-range env_id={env_id}.")
            table = batch.region_next_hop_tables[env_id]
            if table is None:
                raise ValueError(f"SPF requires region_next_hop_table for env_id={env_id}.")
            rows = np.flatnonzero(env_ids == env_id)
            actions[rows] = spf_next_hops(
                current_sat_ids=batch.current_sat_ids[rows],
                target_region_ids=batch.target_region_ids[rows],
                target_access_sat_ids=batch.target_access_sat_ids[rows],
                region_next_hop_table=table,
            )
        return actions
