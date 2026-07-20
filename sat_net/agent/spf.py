from __future__ import annotations

import numpy as np
import torch

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
        return_tensor = isinstance(batch.current_sat_ids, torch.Tensor)
        output_device = batch.current_sat_ids.device if return_tensor else None
        if batch.region_next_hop_tables is not None:
            actions = self._act_vector_batch(batch)
            if return_tensor:
                actions = torch.as_tensor(actions, dtype=torch.long, device=output_device)
            return RoutingDecision(next_hop_sat_ids=actions)
        if batch.region_next_hop_table is None:
            raise ValueError("SPF requires region_next_hop_table in RoutingBatch.")

        actions = spf_next_hops(
            current_sat_ids=self._to_numpy(batch.current_sat_ids),
            target_region_ids=self._to_numpy(batch.target_region_ids),
            target_access_sat_ids=self._to_numpy(batch.target_access_sat_ids),
            region_next_hop_table=batch.region_next_hop_table,
        )
        if return_tensor:
            actions = torch.as_tensor(actions, dtype=torch.long, device=output_device)
        return RoutingDecision(next_hop_sat_ids=actions)

    def _act_vector_batch(self, batch: RoutingBatch) -> np.ndarray:
        actions = np.full(batch.batch_size, -1, dtype=np.int64)
        env_ids = self._to_numpy(batch.row_env_ids)
        current_sat_ids = self._to_numpy(batch.current_sat_ids)
        target_region_ids = self._to_numpy(batch.target_region_ids)
        target_access_sat_ids = self._to_numpy(batch.target_access_sat_ids)
        decision_mask = (
            np.ones(batch.batch_size, dtype=bool)
            if batch.decision_mask is None
            else self._to_numpy(batch.decision_mask).astype(bool, copy=False)
        )
        decision_rows_all = (
            self._to_numpy(batch.decision_rows).astype(np.int64, copy=False)
            if batch.decision_rows is not None
            else np.flatnonzero(decision_mask)
        )
        if len(decision_rows_all) == 0:
            return actions
        for env_id in np.unique(env_ids[decision_rows_all]):
            env_id = int(env_id)
            if env_id < 0 or env_id >= len(batch.region_next_hop_tables):
                raise ValueError(f"SPF received out-of-range env_id={env_id}.")
            table = batch.region_next_hop_tables[env_id]
            if table is None:
                raise ValueError(f"SPF requires region_next_hop_table for env_id={env_id}.")
            rows = decision_rows_all[env_ids[decision_rows_all] == env_id]
            if len(rows) == 0:
                continue
            actions[rows] = spf_next_hops(
                current_sat_ids=current_sat_ids[rows],
                target_region_ids=target_region_ids[rows],
                target_access_sat_ids=target_access_sat_ids[rows],
                region_next_hop_table=table,
            )
        return actions

    @staticmethod
    def _to_numpy(values) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)
