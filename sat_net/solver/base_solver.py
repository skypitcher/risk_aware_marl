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
    """Batched routing inputs for flowlets currently resident at satellites."""

    flowlet_ids: np.ndarray
    current_sat_ids: np.ndarray
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
    ttl: np.ndarray
    current_time: float
    region_next_hop_table: np.ndarray | None = None
    region_next_hop_version: int = 0
    hops: np.ndarray | None = None
    queue_delay: np.ndarray | None = None
    transmission_delay: np.ndarray | None = None
    propagation_delay: np.ndarray | None = None
    total_queue_cost: np.ndarray | None = None
    shortest_gcd: np.ndarray | None = None
    initial_gcd: np.ndarray | None = None


@dataclass(slots=True)
class RoutingDecision:
    """Next-hop satellite ids selected by a batched routing policy."""

    next_hop_sat_ids: np.ndarray


class BaseSolver(ABC):
    """Batched routing policy interface used by the slot-array simulator."""

    requires_shortest_path_table = False

    def __init__(self):
        self._is_eval = True

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def next_hops(self, batch: RoutingBatch) -> RoutingDecision:
        raise NotImplementedError

    def set_train(self):
        self._is_eval = False

    def set_eval(self):
        self._is_eval = True

    def is_train(self) -> bool:
        return not getattr(self, "_is_eval", True)

    def is_eval(self) -> bool:
        return getattr(self, "_is_eval", True)

    def on_train_signal(self):
        pass

    def observe_flowlet_outcomes(self, flowlets, current_time: float):
        pass

    def save_models(self, model_dir_path: str):
        pass

    def load_models(self, model_dir_path: str):
        pass

    def get_stats(self) -> str | None:
        return None
