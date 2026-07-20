from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sat_net.flowlet_status import FLOWLET_DELIVERED, FLOWLET_DROPPED
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
    env_id: int
    flowlet_id: int
    agent_id: int
    state: np.ndarray
    action: int
    action_mask: np.ndarray
    total_delay: float
    queue_cost: float
    shortest_gcd: float
    initial_gcd: float
    flowlet_size: float
    ttl: float
    delay_norm: float
    target_cost: float | None
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
        self.delay_norm = float(config.get("delay_norm", 100.0))
        self.cost_limit = float(config.get("cost_limit", 10.0))
        self.max_ttl = float(config.get("max_ttl", 64.0))
        self.discount_cost = float(config.get("discount_cost", 1.0))
        self.train_freq = max(int(config.get("train_freq", 1)), 1)
        self.utd = max(float(config.get("utd", 1.0)), 0.0)
        self.max_updates_per_train_signal = max(int(config.get("max_updates_per_train_signal", 1024)), 1)
        self._train_signal_count = 0
        self._transitions_since_update_credit = 0
        self._transitions_added_total = 0
        self._update_credit = 0.0
        self._optimizer_update_steps_at_rollout_start = 0
        self.reward_config = RewardConfig.from_config(config)
        self.delay_norm = self.reward_config.delay_norm
        self.cost_limit = self.reward_config.cost_limit
        self._pending: dict[tuple[int, int], PendingTransition] = {}
        self._pending_by_env: dict[int, set[int]] = {}
        self._updated_shortest_gcd: dict[tuple[int, int], float] = {}
        self._reward_stats = RewardStats()

    def set_eval(self):
        super().set_eval()
        sync_inference = getattr(getattr(self, "global_agent", None), "sync_inference", None)
        if callable(sync_inference):
            sync_inference(force=True)

    def act(self, batch: RoutingBatch) -> RoutingDecision:
        states = self.build_states(batch)
        action_masks = self._as_bool_mask(batch.action_mask, states)
        decision_mask = self._as_decision_mask(batch, action_masks)
        action_masks = action_masks & decision_mask[:, None]
        decision_rows = self._decision_rows_numpy(batch, len(batch.flowlet_ids))
        if self.is_train():
            self._finalize_revisited(batch, self._to_numpy(states, np.float32), self._to_numpy(action_masks, bool))
        actions = self._empty_action_indices(len(batch.flowlet_ids), states, action_masks)
        if len(decision_rows) > 0:
            selected = self.select_actions(
                self._take_rows(states, decision_rows),
                self._take_rows(action_masks, decision_rows),
            )
            actions = self._put_rows(actions, decision_rows, selected)
        act = self._action_indices_to_next_hops(batch, actions, action_masks)
        if self.is_train():
            self._remember_pending(
                batch,
                self._to_numpy(states, np.float32),
                self._to_numpy(action_masks, bool),
                self._to_numpy(actions, np.int64),
            )
        return RoutingDecision(next_hop_sat_ids=act)

    def select_actions(self, states, action_masks):
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

    def learn(self, updates: int = 1) -> int:
        return 0

    def on_train_signal(self, force: bool = False, steps: int = 1) -> None:
        if not self.is_train():
            return
        previous_count = self._train_signal_count
        if not force:
            self._train_signal_count += max(int(steps), 1)
            if self._train_signal_count // self.train_freq == previous_count // self.train_freq:
                return
        self._schedule_update_credit()
        self._run_update_budget()

    def _schedule_update_credit(self) -> None:
        if self._transitions_since_update_credit <= 0:
            return
        batch_size = max(self._batch_size(), 1)
        self._update_credit += float(self._transitions_since_update_credit) * self.utd / float(batch_size)
        self._transitions_since_update_credit = 0

    def _run_update_budget(self) -> None:
        update_budget = int(self._update_credit)
        if update_budget <= 0:
            return
        requested_updates = min(update_budget, self.max_updates_per_train_signal)
        completed_updates = int(self.learn(updates=requested_updates) or 0)
        if completed_updates <= 0:
            return
        self._update_credit = max(self._update_credit - float(completed_updates), 0.0)

    def _record_transition_added(self, count: int = 1) -> None:
        count = max(int(count), 0)
        self._transitions_since_update_credit += count
        self._transitions_added_total += count

    def _batch_size(self) -> int:
        return int(getattr(getattr(self, "global_agent", None), "batch_size", self.config.get("batch_size", 1)))

    def _training_steps(self) -> int:
        return int(getattr(getattr(self, "global_agent", None), "training_steps", 0))

    def on_rollout_start(self) -> None:
        self._pending.clear()
        self._pending_by_env.clear()
        self._updated_shortest_gcd.clear()
        self._reward_stats = RewardStats()
        self._train_signal_count = 0
        self._transitions_since_update_credit = 0
        self._transitions_added_total = 0
        self._update_credit = 0.0
        self._optimizer_update_steps_at_rollout_start = self._training_steps()

    def on_rollout_end(self, flowlets, current_time: float) -> None:
        if not self.is_train():
            self._pending.clear()
            self._pending_by_env.clear()
            return
        self.observe_flowlet_outcomes(flowlets, current_time)
        if not self._pending:
            return
        if self._has_vector_flowlet_arrays(flowlets):
            self._finalize_vector_pending(flowlets)
            return
        raise RuntimeError("BatchedRLAgent expects vector flowlet arrays with shape [num_envs, capacity].")

    def build_states(self, batch: RoutingBatch):
        n = len(batch.flowlet_ids)
        if batch.observations is None:
            raise RuntimeError("RoutingBatch is missing the legacy 94-dimensional RL observation.")
        if isinstance(batch.observations, torch.Tensor):
            observations = batch.observations.to(dtype=torch.float32)
        else:
            observations = np.asarray(batch.observations, dtype=np.float32)
        expected_shape = (n, self.obs_dim)
        if tuple(observations.shape) != expected_shape:
            raise RuntimeError(f"Expected observation shape {expected_shape}, got {observations.shape}.")
        return observations

    def observe_flowlet_outcomes(self, flowlets, _current_time: float) -> None:
        if not self.is_train():
            return
        if not self._pending:
            return
        if self._has_vector_flowlet_arrays(flowlets):
            self._observe_vector_flowlet_outcomes(flowlets)
            return
        raise RuntimeError("BatchedRLAgent expects vector flowlet arrays with shape [num_envs, capacity].")

    def _observe_vector_flowlet_outcomes(self, flowlets) -> None:
        status = np.asarray(flowlets.status)
        if status.ndim != 2:
            return
        capacity = status.shape[1]
        for env_id in list(self._pending_by_env.keys()):
            env_pending = self._pending_by_env.get(int(env_id))
            if not env_pending or env_id < 0 or env_id >= status.shape[0]:
                continue
            pending_ids = np.fromiter(env_pending, dtype=np.int64, count=len(env_pending))
            pending_ids = pending_ids[pending_ids < capacity]
            if len(pending_ids) == 0:
                continue
            terminal_mask = (status[env_id, pending_ids] == FLOWLET_DELIVERED) | (
                status[env_id, pending_ids] == FLOWLET_DROPPED
            )
            for flowlet_id in pending_ids[terminal_mask]:
                key = (int(env_id), int(flowlet_id))
                pending = self._pop_pending(key)
                if pending is None:
                    continue
                terminal_status = int(status[env_id, flowlet_id])
                transition_reward = self._compute_reward(
                    pending=pending,
                    current_distance=float(flowlets.remaining_gcd[env_id, flowlet_id]),
                    total_delay=float(
                        flowlets.queue_delay[env_id, flowlet_id]
                        + flowlets.transmission_delay[env_id, flowlet_id]
                        + flowlets.propagation_delay[env_id, flowlet_id]
                    ),
                    queue_cost=float(flowlets.total_queue_cost[env_id, flowlet_id]),
                    ttl_remaining=float(flowlets.ttl[env_id, flowlet_id]),
                    terminal_status=terminal_status,
                    truncated=False,
                )
                self.add_transition(
                    state=pending.state,
                    action=pending.action,
                    action_mask=pending.action_mask,
                    reward=transition_reward.reward,
                    cost=self._buffer_cost(transition_reward.cost),
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
                self._record_transition_added()

    def _finalize_vector_pending(self, flowlets) -> None:
        status = np.asarray(flowlets.status)
        if status.ndim != 2:
            return
        capacity = status.shape[1]
        pending_keys = list(self._pending.keys())
        for key in pending_keys:
            env_id, flowlet_id = int(key[0]), int(key[1])
            if env_id < 0 or env_id >= status.shape[0] or flowlet_id >= capacity:
                self._pop_pending(key)
                continue
            pending = self._pop_pending(key)
            if pending is None:
                continue
            transition_reward = self._compute_reward(
                pending=pending,
                current_distance=float(flowlets.remaining_gcd[env_id, flowlet_id]),
                total_delay=float(
                    flowlets.queue_delay[env_id, flowlet_id]
                    + flowlets.transmission_delay[env_id, flowlet_id]
                    + flowlets.propagation_delay[env_id, flowlet_id]
                ),
                queue_cost=float(flowlets.total_queue_cost[env_id, flowlet_id]),
                ttl_remaining=float(flowlets.ttl[env_id, flowlet_id]),
                terminal_status=None,
                truncated=True,
            )
            self.add_transition(
                state=pending.state,
                action=pending.action,
                action_mask=pending.action_mask,
                reward=transition_reward.reward,
                cost=self._buffer_cost(transition_reward.cost),
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
            self._record_transition_added()

    def _finalize_revisited(self, batch: RoutingBatch, states: np.ndarray, action_masks: np.ndarray) -> None:
        total_delay = self._total_delay(batch)
        queue_cost = self._optional(batch.total_queue_cost, len(batch.flowlet_ids))
        remaining_gcd = self._optional(batch.remaining_gcd, len(batch.flowlet_ids))
        env_ids = self._to_numpy(batch.row_env_ids, np.int64)
        flowlet_ids = self._to_numpy(batch.flowlet_ids, np.int64)
        current_sat_ids = self._to_numpy(batch.current_sat_ids, np.int64)
        ttl = self._optional(batch.ttl, len(batch.flowlet_ids))
        decision_rows = self._decision_rows_numpy(batch, len(flowlet_ids))
        for row in decision_rows:
            flowlet_id = flowlet_ids[row]
            key = (int(env_ids[row]), int(flowlet_id))
            pending = self._pop_pending(key)
            if pending is None:
                continue
            transition_reward = self._compute_reward(
                pending=pending,
                current_distance=float(remaining_gcd[row]),
                total_delay=float(total_delay[row]),
                queue_cost=float(queue_cost[row]),
                ttl_remaining=float(ttl[row]),
                terminal_status=None,
                truncated=False,
            )
            self._updated_shortest_gcd[key] = min(
                pending.shortest_gcd,
                float(remaining_gcd[row]),
            )
            self.add_transition(
                state=pending.state,
                action=pending.action,
                action_mask=pending.action_mask,
                reward=transition_reward.reward,
                cost=self._buffer_cost(transition_reward.cost),
                done=False,
                truncated=False,
                next_state=states[row],
                next_action_mask=action_masks[row],
                target_cost=pending.target_cost,
                weight=pending.weight,
                flowlet_id=pending.flowlet_id,
                agent_id=pending.agent_id,
                next_agent_id=int(current_sat_ids[row]),
            )
            self._record_transition_added()

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
        env_ids = self._to_numpy(batch.row_env_ids, np.int64)
        flowlet_ids = self._to_numpy(batch.flowlet_ids, np.int64)
        agent_ids = self._to_numpy(batch.agent_ids, np.int64)
        flowlet_size = self._optional(batch.flowlet_size, len(batch.flowlet_ids))
        ttl = self._optional(batch.ttl, len(batch.flowlet_ids))
        decision_rows = self._decision_rows_numpy(batch, len(flowlet_ids))
        for row in decision_rows:
            action = actions[row]
            if action < 0 or not action_masks[row, action]:
                continue
            env_id = int(env_ids[row])
            flowlet_id = int(flowlet_ids[row])
            key = (env_id, flowlet_id)
            updated_shortest = self._updated_shortest_gcd.pop(key, float(shortest_gcd[row]))
            self._set_pending(
                key,
                PendingTransition(
                    env_id=env_id,
                    flowlet_id=flowlet_id,
                    agent_id=int(agent_ids[row]),
                    state=states[row].copy(),
                    action=int(action),
                    action_mask=action_masks[row].copy(),
                    total_delay=float(total_delay[row]),
                    queue_cost=float(queue_cost[row]),
                    shortest_gcd=float(updated_shortest),
                    initial_gcd=float(initial_gcd[row]),
                    flowlet_size=float(flowlet_size[row]),
                    ttl=float(ttl[row]),
                    delay_norm=self.reward_config.delay_norm,
                    target_cost=self._target_cost(),
                    weight=max(float(packet_count[row]), 1.0),
                ),
            )

    def _set_pending(self, key: tuple[int, int], pending: PendingTransition) -> None:
        env_id, flowlet_id = int(key[0]), int(key[1])
        self._pending[(env_id, flowlet_id)] = pending
        self._pending_by_env.setdefault(env_id, set()).add(flowlet_id)

    def _pop_pending(self, key: tuple[int, int]) -> PendingTransition | None:
        env_id, flowlet_id = int(key[0]), int(key[1])
        pending = self._pending.pop((env_id, flowlet_id), None)
        env_pending = self._pending_by_env.get(env_id)
        if env_pending is not None:
            env_pending.discard(flowlet_id)
            if not env_pending:
                self._pending_by_env.pop(env_id, None)
        return pending

    def _compute_reward(
        self,
        pending: PendingTransition,
        current_distance: float,
        total_delay: float,
        queue_cost: float,
        ttl_remaining: float,
        terminal_status: int | None,
        truncated: bool,
    ):
        delta_delay = max(0.0, float(total_delay - pending.total_delay))
        delta_queue = max(0.0, float(queue_cost - pending.queue_cost))
        reward = compute_transition_reward(
            config=self.reward_config,
            previous_best_distance=pending.shortest_gcd,
            current_distance=current_distance,
            initial_distance=pending.initial_gcd,
            delta_delay=delta_delay,
            delta_queue_cost=delta_queue,
            flowlet_size=pending.flowlet_size,
            ttl_remaining=ttl_remaining,
            terminal_status=terminal_status,
            truncated=truncated,
            queue_delay_in_reward=self._queue_delay_in_reward(),
        )
        self._reward_stats.add(reward, terminal_status=terminal_status, truncated=truncated)
        return reward

    def _queue_delay_in_reward(self) -> bool:
        return False

    def _uses_cost_constraint(self) -> bool:
        return True

    def _buffer_cost(self, cost: float) -> float | None:
        return cost if self._uses_cost_constraint() else None

    def _target_cost(self) -> float | None:
        if not self._uses_cost_constraint():
            return None
        cost_limit = self.cost_limit / max(self.reward_config.delay_norm, 1e-6)
        if self.discount_cost < 1.0:
            max_episode_len = max(float(self.max_ttl), 1.0)
            return (
                cost_limit
                / max_episode_len
                * (1.0 - self.discount_cost**max_episode_len)
                / (1.0 - self.discount_cost)
            )
        return cost_limit

    @staticmethod
    def _optional(values: np.ndarray | None, length: int) -> np.ndarray:
        if values is None:
            return np.zeros(length, dtype=np.float64)
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(np.float64, copy=False)
        return np.asarray(values, dtype=np.float64)

    def _total_delay(self, batch: RoutingBatch) -> np.ndarray:
        return (
            self._optional(batch.queue_delay, len(batch.flowlet_ids))
            + self._optional(batch.transmission_delay, len(batch.flowlet_ids))
            + self._optional(batch.propagation_delay, len(batch.flowlet_ids))
        )

    @staticmethod
    def _has_vector_flowlet_arrays(flowlets) -> bool:
        status = getattr(flowlets, "status", None)
        if status is None:
            return False
        return np.asarray(status).ndim == 2

    def _valid_random_actions(self, action_masks):
        if isinstance(action_masks, torch.Tensor):
            scores = torch.rand(action_masks.shape, device=action_masks.device)
            scores = scores.masked_fill(~action_masks, -1.0)
            actions = torch.argmax(scores, dim=1)
            actions = torch.where(action_masks.any(dim=1), actions, torch.full_like(actions, -1))
            return actions
        actions = np.full(len(action_masks), -1, dtype=np.int64)
        for row, mask in enumerate(action_masks):
            valid_actions = np.flatnonzero(mask)
            if len(valid_actions) > 0:
                actions[row] = int(np.random.choice(valid_actions))
        return actions

    def get_train_stats(self) -> dict:
        stats = self._reward_stats.to_dict()
        stats["pending_transitions"] = len(self._pending)
        stats["train_freq"] = self.train_freq
        stats["train_signals"] = self._train_signal_count
        stats["utd"] = self.utd
        stats["utd_definition"] = "replay_samples_per_transition"
        optimizer_update_steps = self._training_steps()
        optimizer_update_steps_since_rollout_start = max(
            optimizer_update_steps - self._optimizer_update_steps_at_rollout_start,
            0,
        )
        stats["optimizer_update_steps"] = optimizer_update_steps
        stats["optimizer_update_steps_since_rollout_start"] = optimizer_update_steps_since_rollout_start
        stats["max_updates_per_train_signal"] = self.max_updates_per_train_signal
        stats["update_credit"] = self._update_credit
        stats["transitions_added_total"] = self._transitions_added_total
        stats["transitions_since_update_credit"] = self._transitions_since_update_credit
        stats["effective_utd"] = (
            optimizer_update_steps_since_rollout_start * self._batch_size() / self._transitions_added_total
            if self._transitions_added_total
            else 0.0
        )
        stats["reward_config"] = self.reward_config.to_dict()
        replay_buffer = getattr(getattr(self, "global_agent", None), "replay_buffer", None)
        if replay_buffer is not None and hasattr(replay_buffer, "metadata_summary"):
            stats["replay"] = replay_buffer.metadata_summary()
        return stats

    def _masked_argmax(self, values: torch.Tensor, action_masks):
        mask = torch.as_tensor(action_masks, dtype=torch.bool, device=values.device)
        masked = values.masked_fill(~mask, -1e9)
        actions = torch.argmax(masked, dim=1)
        has_action = mask.any(dim=1)
        actions = torch.where(has_action, actions, torch.full_like(actions, -1))
        if isinstance(action_masks, torch.Tensor):
            return actions.to(action_masks.device)
        actions = actions.cpu().numpy().astype(np.int64)
        return actions

    @staticmethod
    def _to_numpy(values, dtype=None) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            array = values.detach().cpu().numpy()
        else:
            array = np.asarray(values)
        if dtype is not None:
            return array.astype(dtype, copy=False)
        return array

    @staticmethod
    def _as_bool_mask(values, like):
        if isinstance(values, torch.Tensor):
            device = like.device if isinstance(like, torch.Tensor) else values.device
            return values.to(device=device, dtype=torch.bool)
        return np.asarray(values, dtype=bool)

    @staticmethod
    def _as_decision_mask(batch: RoutingBatch, action_masks):
        if batch.decision_mask is None:
            if isinstance(action_masks, torch.Tensor):
                return torch.ones(len(action_masks), dtype=torch.bool, device=action_masks.device)
            return np.ones(len(action_masks), dtype=bool)
        if isinstance(action_masks, torch.Tensor):
            return torch.as_tensor(batch.decision_mask, dtype=torch.bool, device=action_masks.device)
        return np.asarray(batch.decision_mask, dtype=bool)

    def _decision_rows_numpy(self, batch: RoutingBatch, length: int) -> np.ndarray:
        if batch.decision_rows is not None:
            return self._to_numpy(batch.decision_rows, np.int64)
        if batch.decision_mask is None:
            return np.arange(length, dtype=np.int64)
        return np.flatnonzero(self._to_numpy(batch.decision_mask, bool))

    @staticmethod
    def _empty_action_indices(length: int, states, action_masks):
        if isinstance(states, torch.Tensor):
            return torch.full((length,), -1, dtype=torch.long, device=states.device)
        if isinstance(action_masks, torch.Tensor):
            return torch.full((length,), -1, dtype=torch.long, device=action_masks.device)
        return np.full(length, -1, dtype=np.int64)

    @staticmethod
    def _take_rows(values, rows: np.ndarray):
        if isinstance(values, torch.Tensor):
            return values[torch.as_tensor(rows, dtype=torch.long, device=values.device)]
        return values[rows]

    @staticmethod
    def _put_rows(actions, rows: np.ndarray, selected):
        if isinstance(actions, torch.Tensor):
            actions[torch.as_tensor(rows, dtype=torch.long, device=actions.device)] = (
                selected.to(device=actions.device, dtype=torch.long)
                if isinstance(selected, torch.Tensor)
                else torch.as_tensor(selected, dtype=torch.long, device=actions.device)
            )
            return actions
        actions[rows] = selected.detach().cpu().numpy().astype(np.int64, copy=False) if isinstance(selected, torch.Tensor) else np.asarray(selected, dtype=np.int64)
        return actions

    @staticmethod
    def _action_indices_to_next_hops(batch: RoutingBatch, actions, action_masks):
        if isinstance(actions, torch.Tensor) or isinstance(action_masks, torch.Tensor):
            mask = action_masks if isinstance(action_masks, torch.Tensor) else torch.as_tensor(action_masks, dtype=torch.bool)
            device = mask.device
            act_idx = actions.to(device=device, dtype=torch.long) if isinstance(actions, torch.Tensor) else torch.as_tensor(actions, dtype=torch.long, device=device)
            rows = torch.arange(len(act_idx), dtype=torch.long, device=device)
            safe_actions = torch.clamp(act_idx, min=0)
            valid = (act_idx >= 0) & mask[rows, safe_actions]
            next_hops = torch.full((len(act_idx),), -1, dtype=torch.long, device=device)
            neighbor_sat_ids = batch.neighbor_sat_ids.to(device=device, dtype=torch.long)
            if len(act_idx) > 0:
                next_hops = torch.where(valid, neighbor_sat_ids[rows, safe_actions], next_hops)
            return next_hops

        rows = np.arange(len(actions))
        valid = (actions >= 0) & action_masks[rows, np.maximum(actions, 0)]
        act = np.full(len(actions), -1, dtype=np.int64)
        if valid.any():
            act[valid] = batch.neighbor_sat_ids[rows[valid], actions[valid]]
        return act
