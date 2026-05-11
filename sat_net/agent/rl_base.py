from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sat_net.sim_kernel import FLOWLET_DELIVERED, FLOWLET_DROPPED
from sat_net.agent.base_agent import ACTION_COUNT, BaseAgent, RoutingBatch, RoutingDecision
from sat_net.reward import RewardConfig, RewardStats, compute_transition_reward
from sat_net.util import NamedDict


def resolve_device(config: NamedDict) -> torch.device:
    configured = str(config.get("device", "auto")).lower()
    if configured == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if configured == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if configured in {"mps", "metal"}:
        return torch.device("mps" if _mps_available() else "cpu")
    return torch.device(configured)


def resolve_inference_device(config: NamedDict, training_device: torch.device) -> torch.device:
    configured = str(config.get("inference_device", "auto")).lower()
    if configured == "auto":
        return torch.device("cpu" if training_device.type == "mps" else training_device.type)
    if configured in {"mps", "metal"}:
        return torch.device("mps" if _mps_available() else "cpu")
    if configured == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(configured)


def sync_module_to_device(target: torch.nn.Module, source: torch.nn.Module, device: torch.device) -> None:
    state = {key: value.detach().to(device) for key, value in source.state_dict().items()}
    target.load_state_dict(state)


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())


@dataclass(slots=True)
class PendingTransition:
    flowlet_id: int
    agent_id: int
    state: np.ndarray
    action: int
    action_mask: np.ndarray
    total_delay: float
    queue_cost: float
    shortest_gcd: float
    initial_gcd: float
    delay_norm: float
    target_cost: float
    weight: float


class BatchedRLAgent(BaseAgent):
    """Shared RoutingBatch adapter for RL routing agents."""

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
        self.learn_interval = max(int(config.get("learn_interval", 1)), 1)
        self._train_signal_count = 0
        self.reward_config = RewardConfig.from_config(config)
        self.delay_norm = self.reward_config.delay_norm
        self.cost_limit = self.reward_config.cost_limit
        self._pending: dict[int, PendingTransition] = {}
        self._reward_stats = RewardStats()

    def set_eval(self):
        super().set_eval()
        sync_inference = getattr(getattr(self, "global_agent", None), "sync_inference", None)
        if callable(sync_inference):
            sync_inference(force=True)

    def act(self, batch: RoutingBatch) -> RoutingDecision:
        states = self.build_states(batch)
        action_masks = batch.action_mask.astype(bool, copy=False)
        if self.is_train():
            self._finalize_revisited(batch, states, action_masks)
        actions = self.select_actions(states, action_masks)
        rows = np.arange(len(actions))
        valid = (actions >= 0) & action_masks[rows, np.maximum(actions, 0)]
        act = np.full(len(actions), -1, dtype=np.int64)
        if valid.any():
            act[valid] = batch.neighbor_sat_ids[rows[valid], actions[valid]]
        if self.is_train():
            self._remember_pending(batch, states, action_masks, actions)
        return RoutingDecision(next_hop_sat_ids=act)

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
        weight: float = 1.0,
        flowlet_id: int = -1,
        agent_id: int = -1,
        next_agent_id: int = -1,
    ) -> None:
        raise NotImplementedError

    def learn(self) -> None:
        pass

    def on_train_signal(self, force: bool = False) -> None:
        if not self.is_train():
            return
        self._train_signal_count += 1
        if force or self._train_signal_count % self.learn_interval == 0:
            self.learn()

    def on_episode_start(self) -> None:
        self._pending.clear()
        self._reward_stats = RewardStats()
        self._train_signal_count = 0

    def on_episode_end(self, flowlets, current_time: float) -> None:
        if not self.is_train():
            self._pending.clear()
            return
        self.observe_flowlet_outcomes(flowlets, current_time)
        if not self._pending:
            return
        pending_ids = list(self._pending.keys())
        for flowlet_id in pending_ids:
            if flowlet_id >= flowlets.count:
                self._pending.pop(flowlet_id, None)
                continue
            pending = self._pending.pop(flowlet_id, None)
            if pending is None:
                continue
            transition_reward = self._compute_reward(
                pending=pending,
                next_remaining_distance=float(flowlets.shortest_gcd[flowlet_id]),
                total_delay=float(
                    flowlets.queue_delay[flowlet_id]
                    + flowlets.transmission_delay[flowlet_id]
                    + flowlets.propagation_delay[flowlet_id]
                ),
                queue_cost=float(flowlets.total_queue_cost[flowlet_id]),
                terminal_status=None,
                truncated=True,
            )
            self.add_transition(
                state=pending.state,
                action=pending.action,
                action_mask=pending.action_mask,
                reward=transition_reward.reward,
                cost=transition_reward.cost,
                done=False,
                truncated=True,
                next_state=np.zeros(self.obs_dim, dtype=np.float32),
                next_action_mask=np.zeros(self.action_dim, dtype=bool),
                target_cost=pending.target_cost,
                weight=pending.weight,
                flowlet_id=pending.flowlet_id,
                agent_id=pending.agent_id,
                next_agent_id=-1,
            )

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
        progress_norm = np.maximum.reduce((initial_gcd, np.abs(shortest_gcd), np.ones(n, dtype=np.float64) * 1e-6))
        progress = (initial_gcd - shortest_gcd) / progress_norm
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
        if not self.is_train():
            return
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
            terminal_status = int(flowlets.status[flowlet_id])
            transition_reward = self._compute_reward(
                pending=pending,
                next_remaining_distance=float(flowlets.shortest_gcd[flowlet_id]),
                total_delay=float(
                    flowlets.queue_delay[flowlet_id]
                    + flowlets.transmission_delay[flowlet_id]
                    + flowlets.propagation_delay[flowlet_id]
                ),
                queue_cost=float(flowlets.total_queue_cost[flowlet_id]),
                terminal_status=terminal_status,
                truncated=False,
            )
            self.add_transition(
                state=pending.state,
                action=pending.action,
                action_mask=pending.action_mask,
                reward=transition_reward.reward,
                cost=transition_reward.cost,
                done=True,
                truncated=False,
                next_state=np.zeros(self.obs_dim, dtype=np.float32),
                next_action_mask=np.zeros(self.action_dim, dtype=bool),
                target_cost=pending.target_cost,
                weight=pending.weight,
                flowlet_id=pending.flowlet_id,
                agent_id=pending.agent_id,
                next_agent_id=-1,
            )

    def _finalize_revisited(self, batch: RoutingBatch, states: np.ndarray, action_masks: np.ndarray) -> None:
        total_delay = self._total_delay(batch)
        queue_cost = self._optional(batch.total_queue_cost, len(batch.flowlet_ids))
        shortest_gcd = self._optional(batch.shortest_gcd, len(batch.flowlet_ids))
        for row, flowlet_id in enumerate(batch.flowlet_ids):
            pending = self._pending.pop(int(flowlet_id), None)
            if pending is None:
                continue
            transition_reward = self._compute_reward(
                pending=pending,
                next_remaining_distance=float(shortest_gcd[row]),
                total_delay=float(total_delay[row]),
                queue_cost=float(queue_cost[row]),
                terminal_status=None,
                truncated=False,
            )
            self.add_transition(
                state=pending.state,
                action=pending.action,
                action_mask=pending.action_mask,
                reward=transition_reward.reward,
                cost=transition_reward.cost,
                done=False,
                truncated=False,
                next_state=states[row],
                next_action_mask=action_masks[row],
                target_cost=pending.target_cost,
                weight=pending.weight,
                flowlet_id=pending.flowlet_id,
                agent_id=pending.agent_id,
                next_agent_id=int(batch.current_sat_ids[row]),
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
        packet_count = self._optional(batch.packet_count, len(batch.flowlet_ids))
        for row, action in enumerate(actions):
            if action < 0 or not action_masks[row, action]:
                continue
            flowlet_id = int(batch.flowlet_ids[row])
            self._pending[flowlet_id] = PendingTransition(
                flowlet_id=flowlet_id,
                agent_id=int(batch.agent_ids[row]),
                state=states[row].copy(),
                action=int(action),
                action_mask=action_masks[row].copy(),
                total_delay=float(total_delay[row]),
                queue_cost=float(queue_cost[row]),
                shortest_gcd=float(shortest_gcd[row]),
                initial_gcd=float(initial_gcd[row]),
                delay_norm=self.reward_config.delay_norm,
                target_cost=self.reward_config.cost_limit / max(self.reward_config.cost_norm, 1e-6),
                weight=max(float(packet_count[row]), 1.0),
            )

    def _compute_reward(
        self,
        pending: PendingTransition,
        next_remaining_distance: float,
        total_delay: float,
        queue_cost: float,
        terminal_status: int | None,
        truncated: bool,
    ):
        delta_delay = max(0.0, float(total_delay - pending.total_delay))
        delta_queue = max(0.0, float(queue_cost - pending.queue_cost))
        reward = compute_transition_reward(
            config=self.reward_config,
            previous_remaining_distance=pending.shortest_gcd,
            next_remaining_distance=next_remaining_distance,
            initial_distance=pending.initial_gcd,
            delta_delay=delta_delay,
            delta_queue_cost=delta_queue,
            terminal_status=terminal_status,
            truncated=truncated,
        )
        self._reward_stats.add(reward, terminal_status=terminal_status, truncated=truncated)
        return reward

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

    def get_train_stats(self) -> dict:
        stats = self._reward_stats.to_dict()
        stats["pending_transitions"] = len(self._pending)
        stats["learn_interval"] = self.learn_interval
        stats["train_signals"] = self._train_signal_count
        stats["reward_config"] = self.reward_config.to_dict()
        replay_buffer = getattr(getattr(self, "global_agent", None), "replay_buffer", None)
        if replay_buffer is not None and hasattr(replay_buffer, "metadata_summary"):
            stats["replay"] = replay_buffer.metadata_summary()
        return stats

    def _masked_argmax(self, values: torch.Tensor, action_masks: np.ndarray) -> np.ndarray:
        mask = torch.as_tensor(action_masks, dtype=torch.bool, device=values.device)
        masked = values.masked_fill(~mask, -1e9)
        actions = torch.argmax(masked, dim=1).cpu().numpy().astype(np.int64)
        has_action = action_masks.any(axis=1)
        actions[~has_action] = -1
        return actions
