from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


ACTION_N = 0
ACTION_E = 1
ACTION_S = 2
ACTION_W = 3
ACTION_COUNT = 4


@dataclass(slots=True)
class RoutingBatch:
    """Multi-agent routing decisions for flowlets currently resident at satellites."""

    flowlet_ids: np.ndarray
    current_sat_ids: np.ndarray
    source_region_ids: np.ndarray
    target_region_ids: np.ndarray
    target_access_sat_ids: np.ndarray
    neighbor_sat_ids: np.ndarray
    neighbor_link_ids: np.ndarray
    action_mask: np.ndarray
    neighbor_queue_load: np.ndarray
    neighbor_link_capacity: np.ndarray
    neighbor_link_delay: np.ndarray
    neighbor_link_free_time: np.ndarray
    flowlet_size: np.ndarray
    packet_count: np.ndarray
    is_normal: np.ndarray
    creation_time: np.ndarray
    ttl: np.ndarray
    current_time: float
    region_next_hop_table: np.ndarray | None = None
    region_next_hop_version: int = 0
    region_next_hop_tables: tuple[np.ndarray | None, ...] | None = None
    region_next_hop_versions: np.ndarray | None = None
    observations: np.ndarray | None = None
    hops: np.ndarray | None = None
    queue_delay: np.ndarray | None = None
    transmission_delay: np.ndarray | None = None
    propagation_delay: np.ndarray | None = None
    total_queue_cost: np.ndarray | None = None
    remaining_gcd: np.ndarray | None = None
    shortest_gcd: np.ndarray | None = None
    initial_gcd: np.ndarray | None = None
    last_action1: np.ndarray | None = None
    last_action2: np.ndarray | None = None
    last_node1: np.ndarray | None = None
    last_node2: np.ndarray | None = None
    env_ids: np.ndarray | None = None
    current_times: np.ndarray | None = None

    @property
    def agent_ids(self) -> np.ndarray:
        """Satellite-agent id for each row in this decision batch."""
        return self.current_sat_ids

    @property
    def row_env_ids(self) -> np.ndarray:
        if self.env_ids is None:
            return np.zeros(len(self.flowlet_ids), dtype=np.int64)
        return self.env_ids

    @property
    def row_current_times(self) -> np.ndarray:
        if self.current_times is None:
            return np.full(len(self.flowlet_ids), float(self.current_time), dtype=np.float64)
        return self.current_times

    @property
    def decision_count(self) -> int:
        return len(self.flowlet_ids)

    @property
    def active_agent_ids(self) -> np.ndarray:
        if len(self.current_sat_ids) == 0:
            return np.empty(0, dtype=np.int64)
        if self.env_ids is not None:
            env_agent_pairs = np.column_stack((self.env_ids, self.current_sat_ids))
            return np.unique(env_agent_pairs, axis=0)
        return np.unique(self.current_sat_ids)


@dataclass(slots=True)
class RoutingDecision:
    """Next-hop satellite ids selected by a batched routing policy."""

    next_hop_sat_ids: np.ndarray


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
