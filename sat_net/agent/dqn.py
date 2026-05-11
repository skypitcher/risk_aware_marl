from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sat_net.nn import MLP, ReplayBuffer, hard_update, soft_update, weighted_mse_loss
from sat_net.agent.rl_base import BatchedRLAgent, resolve_inference_device, sync_module_to_device
from sat_net.util import NamedDict


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: NamedDict, device: torch.device, tf_writer: Any = None):
        self.config = config
        self.device = device
        self._tf_writer = tf_writer
        self.state_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = int(config.get("hidden_dim", 256))
        self.num_hidden_layers = int(config.get("num_hidden_layers", 2))
        self.gamma = float(config.get("gamma", 0.99))
        self.epsilon_start = float(config.get("epsilon_start", 0.99))
        self.epsilon_end = float(config.get("epsilon_end", 0.01))
        self.epsilon_train = self.epsilon_start
        self.epsilon_decay_method = str(config.get("epsilon_decay_method", "exponential"))
        self.epsilon_decay_steps = int(config.get("epsilon_decay_steps", 200000))
        self.epsilon_step_count = 0
        self.batch_size = int(config.get("batch_size", 2048))
        self.train_start_size = int(config.get("train_start_size", 1000))
        self.train_steps_per_update = int(config.get("train_steps_per_update", 1))
        self.update_method = str(config.get("update_method", "soft"))
        self.soft_update_tau = float(config.get("soft_update_tau", 0.05))
        self.hard_update_interval = int(config.get("hard_update_interval", 10))
        self.clip_grad_norm = float(config.get("clip_grad_norm", 0.5))
        self.inference_device = resolve_inference_device(config, device)
        self.inference_sync_interval = max(int(config.get("inference_sync_interval", 64)), 1)
        self._last_q_sync_step = -1
        self.training_steps = 0
        use_layer_norm = bool(config.get("use_layer_norm", True))
        init_method = str(config.get("weight_init", "orthogonal"))

        self.Q = MLP(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            use_layer_norm=use_layer_norm,
            init_method=init_method,
        ).to(device)
        self.Q_target = MLP(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            use_layer_norm=use_layer_norm,
            init_method=init_method,
        ).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        if self.inference_device == self.device:
            self.Q_inference = self.Q
        else:
            self.Q_inference = MLP(
                input_dim=obs_dim,
                output_dim=action_dim,
                hidden_dim=self.hidden_dim,
                num_hidden_layers=self.num_hidden_layers,
                use_layer_norm=use_layer_norm,
                init_method=init_method,
            ).to(self.inference_device)
            self._sync_q_inference(force=True)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=float(config.get("learning_rate", 1e-4)))
        self.replay_buffer = ReplayBuffer(
            state_dim=obs_dim,
            action_dim=action_dim,
            buffer_size=int(config.get("buffer_size", 300000)),
            device=device,
        )

    def act(self, states: np.ndarray, action_masks: np.ndarray, eval_mode: bool) -> np.ndarray:
        if len(states) == 0:
            return np.empty(0, dtype=np.int64)
        if not eval_mode:
            explore = np.random.random(len(states)) < self.epsilon_train
            random_actions = self._random_actions(action_masks)
        else:
            explore = np.zeros(len(states), dtype=bool)
            random_actions = np.full(len(states), -1, dtype=np.int64)

        with torch.inference_mode():
            self.Q_inference.eval()
            state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.inference_device)
            q_values = self.Q_inference(state_tensor)
            mask = torch.as_tensor(action_masks, dtype=torch.bool, device=self.inference_device)
            greedy_actions = torch.argmax(q_values.masked_fill(~mask, -1e9), dim=1).cpu().numpy()

        actions = np.where(explore, random_actions, greedy_actions).astype(np.int64, copy=False)
        actions[~action_masks.any(axis=1)] = -1
        if not eval_mode:
            self._update_epsilon(len(states))
        return actions

    def add_transition(self, **kwargs) -> None:
        self.replay_buffer.add(**kwargs)

    def learn(self) -> None:
        if len(self.replay_buffer) < max(self.batch_size, self.train_start_size):
            return
        for _ in range(self.train_steps_per_update):
            self._train_step()

    def _train_step(self) -> None:
        batch = self.replay_buffer.sample(self.batch_size)
        bootstrap_mask = ~(batch.dones & ~batch.truncateds)
        current_q = self.Q(batch.states).gather(1, batch.actions)
        with torch.no_grad():
            next_q_main = self.Q(batch.next_states).masked_fill(~batch.next_action_masks, -1e9)
            next_actions = next_q_main.argmax(1, keepdim=True)
            next_q = self.Q_target(batch.next_states).gather(1, next_actions)
            target_q = batch.rewards + bootstrap_mask.float() * self.gamma * next_q
        loss = weighted_mse_loss(current_q, target_q, batch.weights)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=self.clip_grad_norm)
        self.optimizer.step()
        self.training_steps += 1
        self._update_target_network()
        self._sync_q_inference()
        if self._tf_writer is not None:
            self._tf_writer.add_scalar("madqn/loss", loss.item(), global_step=self.training_steps)
            self._tf_writer.add_scalar("madqn/q", current_q.mean().item(), global_step=self.training_steps)

    def _update_epsilon(self, steps: int) -> None:
        self.epsilon_step_count += steps
        progress = min(self.epsilon_step_count / max(self.epsilon_decay_steps, 1), 1.0)
        if self.epsilon_decay_method == "exponential":
            self.epsilon_train = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-5.0 * progress)
        elif self.epsilon_decay_method == "cosine":
            self.epsilon_train = self.epsilon_end + 0.5 * (self.epsilon_start - self.epsilon_end) * (
                1.0 + math.cos(math.pi * progress)
            )
        else:
            self.epsilon_train = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        self.epsilon_train = max(self.epsilon_train, self.epsilon_end)

    def _update_target_network(self) -> None:
        if self.update_method == "soft":
            soft_update(self.Q_target, self.Q, self.soft_update_tau)
        elif self.training_steps % self.hard_update_interval == 0:
            hard_update(self.Q_target, self.Q)

    def _sync_q_inference(self, force: bool = False) -> None:
        if self.Q_inference is self.Q:
            self._last_q_sync_step = self.training_steps
            return
        if not force and self.training_steps - self._last_q_sync_step < self.inference_sync_interval:
            return
        sync_module_to_device(self.Q_inference, self.Q, self.inference_device)
        self.Q_inference.eval()
        self._last_q_sync_step = self.training_steps

    def sync_inference(self, force: bool = True) -> None:
        self._sync_q_inference(force=force)

    @staticmethod
    def _random_actions(action_masks: np.ndarray) -> np.ndarray:
        actions = np.full(len(action_masks), -1, dtype=np.int64)
        for row, mask in enumerate(action_masks):
            valid = np.flatnonzero(mask)
            if len(valid) > 0:
                actions[row] = int(np.random.choice(valid))
        return actions

    def get_model_dict(self) -> dict[str, Any]:
        return {
            "qnet_state_dict": self.Q.state_dict(),
            "target_qnet_state_dict": self.Q_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "epsilon": self.epsilon_train,
            "epsilon_step_count": self.epsilon_step_count,
        }

    def load_model_dict(self, checkpoint: dict[str, Any]) -> None:
        self.Q.load_state_dict(checkpoint["qnet_state_dict"])
        self.Q_target.load_state_dict(checkpoint["target_qnet_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = int(checkpoint.get("training_steps", 0))
        self.epsilon_train = float(checkpoint.get("epsilon", self.epsilon_train))
        self.epsilon_step_count = int(checkpoint.get("epsilon_step_count", 0))
        self._sync_q_inference(force=True)


class MaDQN(BatchedRLAgent):
    def __init__(self, config: NamedDict, obs_dim: int = 94, action_dim: int = 4, tf_writer: Any = None):
        super().__init__(config=config, obs_dim=obs_dim, action_dim=action_dim, tf_writer=tf_writer)
        self.global_agent = DQNAgent(self.obs_dim, self.action_dim, config, self.device, tf_writer)

    @property
    def name(self) -> str:
        return "MaDQN"

    def select_actions(self, states: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        return self.global_agent.act(states, action_masks, eval_mode=self.is_eval())

    def add_transition(self, **kwargs) -> None:
        self.global_agent.add_transition(**kwargs)

    def learn(self) -> None:
        self.global_agent.learn()

    def get_stats(self) -> str:
        return (
            f"epsilon={self.global_agent.epsilon_train:.4f} "
            f"buffer={len(self.global_agent.replay_buffer)} "
            f"training_steps={self.global_agent.training_steps} device={self.device} "
            f"inference_device={self.global_agent.inference_device}"
        )

    def save_models(self, model_dir_path: str) -> None:
        os.makedirs(model_dir_path, exist_ok=True)
        torch.save(self.global_agent.get_model_dict(), f"{model_dir_path}/{self.name}.pth")

    def load_models(self, model_dir_path: str) -> None:
        model_path = Path(model_dir_path) / f"{self.name}.pth"
        if not model_path.exists():
            raise RuntimeError(f"Model file not found at {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.global_agent.load_model_dict(checkpoint)
