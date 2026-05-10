from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sat_net.sim_kernel import FLOWLET_DELIVERED, FLOWLET_DROPPED
from sat_net.solver.base_solver import ACTION_COUNT, BaseSolver, RoutingBatch, RoutingDecision
from sat_net.util import NamedDict


def resolve_device(config: NamedDict) -> torch.device:
    configured = str(config.get("device", "auto")).lower()
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(configured)


@dataclass(slots=True)
class PendingTransition:
    state: np.ndarray
    action: int
    action_mask: np.ndarray
    total_delay: float
    queue_cost: float
    shortest_gcd: float
    initial_gcd: float
    delay_norm: float
    target_cost: float


class BatchedRLSolver(BaseSolver):
    """Shared RoutingBatch adapter for RL policies."""

    requires_shortest_path_table = False

    def __init__(self, config: NamedDict, obs_dim: int = 94, action_dim: int = ACTION_COUNT, tf_writer: Any = None):
        super().__init__()
        self.config = config
        self.obs_dim = int(config.get("obs_dim", obs_dim))
        self.action_dim = int(action_dim)
        self._tf_writer = tf_writer
        self.device = resolve_device(config)
        self.delay_norm = float(config.get("delay_norm", 1000.0))
        self.cost_limit = float(config.get("cost_limit", 10.0))
        self.node_norm = float(config.get("node_norm", 2048.0))
        self.region_norm = float(config.get("region_norm", 512.0))
        self.queue_norm = float(config.get("queue_norm", 32.0))
        self.delay_feature_norm = float(config.get("delay_feature_norm", 1000.0))
        self.size_norm = float(config.get("size_norm", 8.0))
        self.max_ttl = float(config.get("max_ttl", 64.0))
        self.delivered_bonus = float(config.get("delivered_bonus", 1.0))
        self.dropped_penalty = float(config.get("dropped_penalty", 1.0))
        self._pending: dict[int, PendingTransition] = {}

    def next_hops(self, batch: RoutingBatch) -> RoutingDecision:
        states = self.build_states(batch)
        action_masks = batch.action_mask.astype(bool, copy=False)
        self._finalize_revisited(batch, states, action_masks)
        actions = self.select_actions(states, action_masks)
        rows = np.arange(len(actions))
        valid = (actions >= 0) & action_masks[rows, np.maximum(actions, 0)]
        next_hops = np.full(len(actions), -1, dtype=np.int64)
        if valid.any():
            next_hops[valid] = batch.neighbor_sat_ids[rows[valid], actions[valid]]
        self._remember_pending(batch, states, action_masks, actions)
        return RoutingDecision(next_hop_sat_ids=next_hops)

    def select_actions(self, states: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def add_transition(
        self,
        state: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        reward: float,
        cost: float | None,
        done: bool,
        truncated: bool,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        target_cost: float | None,
    ) -> None:
        raise NotImplementedError

    def learn(self) -> None:
        pass

    def on_train_signal(self) -> None:
        if self.is_train():
            self.learn()

    def build_states(self, batch: RoutingBatch) -> np.ndarray:
        n = len(batch.flowlet_ids)
        states = np.zeros((n, self.obs_dim), dtype=np.float32)
        if n == 0:
            return states

        total_delay = self._total_delay(batch)
        queue_cost = self._optional(batch.total_queue_cost, n)
        hops = self._optional(batch.hops, n)
        shortest_gcd = self._optional(batch.shortest_gcd, n)
        initial_gcd = np.maximum(self._optional(batch.initial_gcd, n), 1e-6)
        progress = 1.0 - shortest_gcd / initial_gcd
        target_access_valid = (batch.target_access_sat_ids >= 0).astype(np.float32)

        base = np.column_stack(
            (
                batch.current_sat_ids / self.node_norm,
                batch.target_region_ids / self.region_norm,
                np.maximum(batch.target_access_sat_ids, 0) / self.node_norm,
                batch.flowlet_size / self.size_norm,
                batch.ttl / self.max_ttl,
                total_delay / self.delay_norm,
                queue_cost / self.delay_norm,
                hops / self.max_ttl,
                progress,
                target_access_valid,
            )
        ).astype(np.float32, copy=False)
        width = min(base.shape[1], self.obs_dim)
        states[:, :width] = base[:, :width]

        cursor = width
        per_action = (
            batch.action_mask.astype(np.float32),
            np.maximum(batch.neighbor_sat_ids, 0) / self.node_norm,
            np.maximum(batch.neighbor_queue_load, 0.0) / self.queue_norm,
            batch.neighbor_link_capacity / self.queue_norm,
            np.minimum(batch.neighbor_link_delay, self.delay_norm) / self.delay_feature_norm,
            np.maximum(batch.neighbor_link_free_time - batch.current_time, 0.0) / self.delay_feature_norm,
            np.maximum(batch.neighbor_link_capacity - batch.neighbor_queue_load, 0.0) / self.queue_norm,
        )
        for feature in per_action:
            if cursor >= self.obs_dim:
                break
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            width = min(feature.shape[1], self.obs_dim - cursor)
            states[:, cursor : cursor + width] = feature[:, :width]
            cursor += width
        return states

    def observe_flowlet_outcomes(self, flowlets, _current_time: float) -> None:
        if not self._pending:
            return
        pending_ids = np.fromiter(self._pending.keys(), dtype=np.int64)
        pending_ids = pending_ids[pending_ids < flowlets.count]
        if len(pending_ids) == 0:
            return
        terminal_mask = (flowlets.status[pending_ids] == FLOWLET_DELIVERED) | (
            flowlets.status[pending_ids] == FLOWLET_DROPPED
        )
        for flowlet_id in pending_ids[terminal_mask]:
            pending = self._pending.pop(int(flowlet_id), None)
            if pending is None:
                continue
            total_delay = (
                flowlets.queue_delay[flowlet_id]
                + flowlets.transmission_delay[flowlet_id]
                + flowlets.propagation_delay[flowlet_id]
            )
            delta_delay = max(0.0, float(total_delay - pending.total_delay))
            delta_queue = max(0.0, float(flowlets.total_queue_cost[flowlet_id] - pending.queue_cost))
            delivered = flowlets.status[flowlet_id] == FLOWLET_DELIVERED
            terminal_bonus = self.delivered_bonus if delivered else -self.dropped_penalty
            reward = terminal_bonus - delta_delay / pending.delay_norm
            cost = delta_queue / pending.delay_norm
            self.add_transition(
                state=pending.state,
                action=pending.action,
                action_mask=pending.action_mask,
                reward=reward,
                cost=cost,
                done=True,
                truncated=False,
                next_state=np.zeros(self.obs_dim, dtype=np.float32),
                next_action_mask=np.zeros(self.action_dim, dtype=bool),
                target_cost=pending.target_cost,
            )

    def _finalize_revisited(self, batch: RoutingBatch, states: np.ndarray, action_masks: np.ndarray) -> None:
        total_delay = self._total_delay(batch)
        queue_cost = self._optional(batch.total_queue_cost, len(batch.flowlet_ids))
        shortest_gcd = self._optional(batch.shortest_gcd, len(batch.flowlet_ids))
        for row, flowlet_id in enumerate(batch.flowlet_ids):
            pending = self._pending.pop(int(flowlet_id), None)
            if pending is None:
                continue
            delta_delay = max(0.0, float(total_delay[row] - pending.total_delay))
            delta_queue = max(0.0, float(queue_cost[row] - pending.queue_cost))
            progress = (pending.shortest_gcd - float(shortest_gcd[row])) / max(pending.initial_gcd, 1e-6)
            reward = progress - delta_delay / pending.delay_norm
            cost = delta_queue / pending.delay_norm
            self.add_transition(
                state=pending.state,
                action=pending.action,
                action_mask=pending.action_mask,
                reward=reward,
                cost=cost,
                done=False,
                truncated=False,
                next_state=states[row],
                next_action_mask=action_masks[row],
                target_cost=pending.target_cost,
            )

    def _remember_pending(
        self,
        batch: RoutingBatch,
        states: np.ndarray,
        action_masks: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        total_delay = self._total_delay(batch)
        queue_cost = self._optional(batch.total_queue_cost, len(batch.flowlet_ids))
        shortest_gcd = self._optional(batch.shortest_gcd, len(batch.flowlet_ids))
        initial_gcd = np.maximum(self._optional(batch.initial_gcd, len(batch.flowlet_ids)), 1e-6)
        for row, action in enumerate(actions):
            if action < 0 or not action_masks[row, action]:
                continue
            flowlet_id = int(batch.flowlet_ids[row])
            delay_norm = self.delay_norm
            self._pending[flowlet_id] = PendingTransition(
                state=states[row].copy(),
                action=int(action),
                action_mask=action_masks[row].copy(),
                total_delay=float(total_delay[row]),
                queue_cost=float(queue_cost[row]),
                shortest_gcd=float(shortest_gcd[row]),
                initial_gcd=float(initial_gcd[row]),
                delay_norm=delay_norm,
                target_cost=self.cost_limit / delay_norm,
            )

    @staticmethod
    def _optional(values: np.ndarray | None, length: int) -> np.ndarray:
        if values is None:
            return np.zeros(length, dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    def _total_delay(self, batch: RoutingBatch) -> np.ndarray:
        return (
            self._optional(batch.queue_delay, len(batch.flowlet_ids))
            + self._optional(batch.transmission_delay, len(batch.flowlet_ids))
            + self._optional(batch.propagation_delay, len(batch.flowlet_ids))
        )

    def _valid_random_actions(self, action_masks: np.ndarray) -> np.ndarray:
        actions = np.full(len(action_masks), -1, dtype=np.int64)
        for row, mask in enumerate(action_masks):
            valid_actions = np.flatnonzero(mask)
            if len(valid_actions) > 0:
                actions[row] = int(np.random.choice(valid_actions))
        return actions

    def _masked_argmax(self, values: torch.Tensor, action_masks: np.ndarray) -> np.ndarray:
        mask = torch.as_tensor(action_masks, dtype=torch.bool, device=values.device)
        masked = values.masked_fill(~mask, -1e9)
        actions = torch.argmax(masked, dim=1).cpu().numpy().astype(np.int64)
        has_action = action_masks.any(axis=1)
        actions[~has_action] = -1
        return actions
