from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


ACTION_N = 0
ACTION_E = 1
ACTION_S = 2
ACTION_W = 3
ACTION_COUNT = 4


@dataclass(slots=True)
class RoutingBatch:
    """Multi-agent routing decisions for flowlets currently resident at satellites."""

    flowlet_ids: Any
    current_sat_ids: Any
    source_region_ids: Any
    target_region_ids: Any
    target_access_sat_ids: Any
    neighbor_sat_ids: Any
    neighbor_link_ids: Any
    action_mask: Any
    neighbor_queue_load: Any
    neighbor_link_capacity: Any
    neighbor_link_delay: Any
    neighbor_link_free_time: Any
    flowlet_size: Any
    packet_count: Any
    is_normal: Any
    creation_time: Any
    ttl: Any
    current_time: float
    region_next_hop_table: np.ndarray | None = None
    region_next_hop_version: int = 0
    region_next_hop_tables: Any | None = None
    region_next_hop_versions: np.ndarray | None = None
    node_state: Any | None = None
    link_state: Any | None = None
    observations: Any | None = None
    hops: Any | None = None
    queue_delay: Any | None = None
    transmission_delay: Any | None = None
    propagation_delay: Any | None = None
    total_queue_cost: Any | None = None
    remaining_gcd: Any | None = None
    shortest_gcd: Any | None = None
    initial_gcd: Any | None = None
    last_action1: Any | None = None
    last_action2: Any | None = None
    last_node1: Any | None = None
    last_node2: Any | None = None
    env_ids: Any | None = None
    current_times: Any | None = None
    decision_mask: Any | None = None
    decision_rows: Any | None = None

    @property
    def agent_ids(self) -> Any:
        """Satellite-agent id for each row in this decision batch."""
        return self.current_sat_ids

    @property
    def row_env_ids(self) -> Any:
        if self.env_ids is None:
            if isinstance(self.flowlet_ids, torch.Tensor):
                return torch.zeros(len(self.flowlet_ids), dtype=torch.long, device=self.flowlet_ids.device)
            return np.zeros(len(self.flowlet_ids), dtype=np.int64)
        return self.env_ids

    @property
    def row_current_times(self) -> Any:
        if self.current_times is None:
            if isinstance(self.flowlet_ids, torch.Tensor):
                return torch.full(
                    (len(self.flowlet_ids),),
                    float(self.current_time),
                    dtype=torch.float32,
                    device=self.flowlet_ids.device,
                )
            return np.full(len(self.flowlet_ids), float(self.current_time), dtype=np.float64)
        return self.current_times

    @property
    def decision_count(self) -> int:
        return self.active_decision_count

    @property
    def batch_size(self) -> int:
        return len(self.flowlet_ids)

    @property
    def active_decision_count(self) -> int:
        if self.decision_rows is not None:
            return len(self.decision_rows)
        if self.decision_mask is None:
            return len(self.flowlet_ids)
        if isinstance(self.decision_mask, torch.Tensor):
            return int(self.decision_mask.sum().detach().cpu().item())
        return int(np.asarray(self.decision_mask, dtype=bool).sum())

    @property
    def active_agent_ids(self) -> Any:
        if len(self.current_sat_ids) == 0:
            if isinstance(self.current_sat_ids, torch.Tensor):
                return torch.empty(0, dtype=torch.long, device=self.current_sat_ids.device)
            return np.empty(0, dtype=np.int64)
        if isinstance(self.current_sat_ids, torch.Tensor):
            # Avoid torch.unique(..., dim=0): it is unsupported on Apple MPS and
            # this property is only used for lightweight progress statistics.
            return self.current_sat_ids
        current_sat_ids = self.current_sat_ids
        if self.decision_mask is not None:
            current_sat_ids = current_sat_ids[np.asarray(self.decision_mask, dtype=bool)]
        if self.env_ids is not None:
            env_ids = self.env_ids
            if self.decision_mask is not None:
                env_ids = env_ids[np.asarray(self.decision_mask, dtype=bool)]
            env_agent_pairs = np.column_stack((env_ids, current_sat_ids))
            return np.unique(env_agent_pairs, axis=0)
        return np.unique(current_sat_ids)


@dataclass(slots=True)
class RoutingDecision:
    """Next-hop satellite ids selected by a batched routing policy."""

    next_hop_sat_ids: Any


class BaseAgent(ABC):
    """Batched MARL routing-agent interface used by the slot-array simulator."""

    requires_shortest_path_table = False

    def __init__(self):
        self._is_eval = True

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def act(self, batch: RoutingBatch) -> RoutingDecision:
        raise NotImplementedError

    def set_train(self):
        self._is_eval = False

    def set_eval(self):
        self._is_eval = True

    def is_train(self) -> bool:
        return not getattr(self, "_is_eval", True)

    def is_eval(self) -> bool:
        return getattr(self, "_is_eval", True)

    def on_train_signal(self, force: bool = False, steps: int = 1):
        pass

    def observe_flowlet_outcomes(self, flowlets, current_time: float):
        pass

    def on_rollout_start(self):
        pass

    def on_rollout_end(self, flowlets, current_time: float):
        pass

    def save_models(self, model_dir_path: str):
        pass

    def load_models(self, model_dir_path: str):
        pass

    def get_stats(self) -> str | None:
        return None

    def get_train_stats(self) -> dict:
        return {}
