import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sat_net.nn import IQN, hard_update, quantile_huber_loss, sample_taus, soft_update, ReplayBuffer
from sat_net.solver.base_solver import BaseSolver
from sat_net.util import NamedDict

if TYPE_CHECKING:
    from sat_net.datablock import DataBlock


class MaIQN(BaseSolver):
    """
    Multi-agent Independent IQN solver using a global model (CTDE).
    """

    def __init__(self, obs_dim: int, action_dim: int, config: "NamedDict", tf_writer: Optional[SummaryWriter] = None):
        super().__init__(tf_writer=tf_writer)
        self.config = config
        
        # Dimensions
        self.state_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_dim = config.embedding_dim
        self.weight_init = config.weight_init

        assert self.weight_init in ["orthogonal", "xavier", "he"]
        assert config.epsilon_decay_method in ["linear", "exponential", "polynomial", "cosine"]

        # Quantile parameters
        self.num_quantiles = config.num_quantiles
        self.policy_tau_range = config.policy_tau_range
        assert 0.0 <= self.policy_tau_range[0] <= self.policy_tau_range[1] <= 1.0, "Policy tau range must be in [0, 1]"

        # Training parameters
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_train = self.epsilon_start
        self.epsilon_decay_method = config.epsilon_decay_method
        self.epsilon_decay_steps = config.epsilon_decay_steps
        self.epsilon_step_count = 0  # Track steps for epsilon decay

        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.train_steps_per_update = config.train_steps_per_update

        self.learning_rate = config.learning_rate
        self.device = config.device

        self.train_start_size = config.train_start_size
        self.update_method = config.update_method
        self.soft_update_tau = config.soft_update_tau
        self.hard_update_interval = config.hard_update_interval
        self.clip_grad_norm = config.clip_grad_norm

        self.training_steps = 0

        self.safty_factor_start = 100
        self.safty_factor_end = 1
        self.safty_factor = self.safty_factor_start

        # Create Q networks
        self.Q = IQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            embedding_dim=self.embedding_dim,
            init_method=self.weight_init,
        ).to(self.device)

        self.Q_target = IQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            embedding_dim=self.embedding_dim,
            init_method=self.weight_init,
        ).to(self.device)

        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.95)

        self.replay_buffers = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            buffer_size=self.buffer_size,
            device=self.device
        )

    @property
    def name(self):
        return "MaIQN"

    def route(self, obs: np.ndarray, info: dict):
        """Select action using epsilon-greedy policy."""
        action_mask: np.ndarray = info["action_mask"]
        
        chosen_action = None
        epsilon = self.epsilon_train
        if self.is_train() and np.random.rand() < epsilon:
            # Exploration during training: random action from valid actions
            valid_actions = np.where(action_mask)[0]  # Get indices where mask is 1
            chosen_action = np.random.choice(valid_actions)
        else:
            # Exploitation: greedy action based on Q-values
            self.Q.eval()
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.device)
                action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
                taus = sample_taus(
                    batch_size=1,
                    n_quantiles=self.num_quantiles,
                    device=self.device,
                    min_tau=self.policy_tau_range[0],
                    max_tau=self.policy_tau_range[1],
                )
                q_values_for_quantiles = self.Q(obs_tensor, taus)  # (1, n_quantiles, action_dim)
                q_values = q_values_for_quantiles.mean(dim=1)  # (1, action_dim)
                masked_q_values = q_values.masked_fill(~action_mask_tensor, -1e8)  # mask out invalid actions
                chosen_action = masked_q_values.argmax().item()

        if self.is_train():
            self._update_epsilon()

        return chosen_action, None

    def on_action_over(self, packet: "DataBlock"):
        """Store experience in replay buffer."""
        if self.is_eval():
            return

        last_action = packet.last_action

        state = last_action.state
        action = last_action.action
        action_mask = last_action.action_mask
        baseline_reward = last_action.baseline_reward
        done = last_action.done
        truncated = last_action.truncated
        next_state = last_action.next_state
        next_action_mask = last_action.next_action_mask

        self.replay_buffers.add(
            state=state,
            action=action,
            action_mask=action_mask,
            reward=baseline_reward,
            cost=None,
            done=done,
            truncated=truncated,
            next_state=next_state,
            next_action_mask=next_action_mask,
            target_cost=None,
        )

    def learn(self):
        """Train the networks if we have enough experiences in the replay buffers."""
        # Check if we have enough experiences to train
        if len(self.replay_buffers) < self.batch_size:
            return

        for _ in range(self.train_steps_per_update):
            self._train_step()

    def _train_step(self):
        if len(self.replay_buffers) < max(self.train_start_size, self.batch_size):
            return

        self.Q.train()

        # Sample a batch of experiences
        batch = self.replay_buffers.sample(self.batch_size)

        # Only stop bootstrapping if the episode is done and not just truncated
        bootstrap_mask = ~(batch.dones & ~batch.truncateds).reshape(-1, 1, 1)  # (batch_size, 1, 1)
        rewards = batch.rewards.reshape(-1, 1, 1)  # (batch_size, 1, 1)
        actions = batch.actions.reshape(-1, 1, 1).expand(-1, self.num_quantiles, 1)  # (batch_size, n_quantiles, 1)

        # Sample quantiles
        taus = torch.rand(self.batch_size, self.num_quantiles).to(self.device)
        target_taus = torch.rand(self.batch_size, self.num_quantiles).to(self.device)

        # Current quantile values
        quantile_values = self.Q.forward(batch.states, taus)  # (batch_size, n_quantiles, action_dim)
        quantile_values = quantile_values.gather(-1, actions)  # (batch_size, n_quantiles, 1)

        # Get next Q-values from target network
        with torch.no_grad():
            # Use main network for action selection (Double DQN)
            next_taus = torch.rand(self.batch_size, self.num_quantiles).to(self.device)
            next_quantile_values = self.Q.forward(batch.next_states, next_taus)
            next_q_values = next_quantile_values.mean(dim=1, keepdim=True)  # (batch_size, 1, action_dim)
            next_actions = next_q_values.argmax(dim=-1, keepdim=True)  # (batch_size, 1, 1)
            next_actions = next_actions.expand(-1, self.num_quantiles, 1)  # (batch_size, n_quantiles, 1)

            # Use target network for value estimation
            target_quantile_values = self.Q_target.forward(batch.next_states, target_taus)  # (batch_size, n_quantiles, action_dim)
            target_quantile_values = target_quantile_values.gather(2, next_actions)  # (batch_size, n_quantiles, 1)

            # Compute targets
            targets = rewards + self.gamma * target_quantile_values * bootstrap_mask  # (batch_size, n_quantiles, 1)

        loss = quantile_huber_loss(quantile_values, targets, taus.unsqueeze(-1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=self.clip_grad_norm)
        self.optimizer.step()

        # Step the learning rate scheduler
        self.lr_scheduler.step()

        self.training_steps += 1
        if self._tf_writer is not None:
            self._tf_writer.add_scalar("qval", quantile_values.mean().item(), global_step=self.training_steps)

        # Use soft target network updates every step (more stable than hard updates)
        self._update_target_network()

    def _update_target_network(self):
        """update of target network parameters"""
        if self.update_method == "soft":
            soft_update(target=self.Q_target, source=self.Q, tau=self.soft_update_tau)
        elif self.training_steps % self.hard_update_interval == 0:
            hard_update(target=self.Q_target, source=self.Q)

    def _update_epsilon(self):
        """Update epsilon using the specified decay strategy."""
        if self.epsilon_train <= self.epsilon_end:
            self.epsilon_train = self.epsilon_end
            return

        self.epsilon_step_count += 1
        progress = min(self.epsilon_step_count / self.epsilon_decay_steps, 1.0)

        if self.epsilon_decay_method == "exponential":
            # Exponential decay: ε = ε_end + (ε_start - ε_end) * exp(-5 * progress)
            decay_rate = 5.0  # Controls decay speed
            self.epsilon_train = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-decay_rate * progress)

        elif self.epsilon_decay_method == "polynomial":
            # Polynomial decay: ε = ε_end + (ε_start - ε_end) * (1 - progress)^power
            power = 2.0  # Higher power = slower initial decay, faster later
            self.epsilon_train = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * ((1 - progress) ** power)

        elif self.epsilon_decay_method == "cosine":
            # Cosine annealing: ε = ε_end + 0.5 * (ε_start - ε_end) * (1 + cos(π * progress))
            self.epsilon_train = self.epsilon_end + 0.5 * (self.epsilon_start - self.epsilon_end) * (1 + math.cos(math.pi * progress))
        else:
            # Linear decay: ε = ε_start - (ε_start - ε_end) * progress
            self.epsilon_train = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress

        safty_progress = min(self.epsilon_step_count / 100000, 1.0)
        self.safty_factor = self.epsilon_start - (self.safty_factor_start - self.safty_factor_end) * safty_progress
        self.safty_factor = max(self.safty_factor, self.safty_factor_end)

        self.epsilon_train = max(self.epsilon_train, self.epsilon_end)

    def reset_epsilon(self, epsilon: float = None):
        """Reset epsilon and epsilon decay schedule."""
        self.epsilon_step_count = 0
        self.epsilon_train = epsilon if epsilon is not None else self.epsilon_start

    def on_train_signal(self):
        """Train the networks if we have enough experiences in the replay buffers."""
        if self.is_train():
            self.learn()

    def get_model_dict(self):
        return {
            "qnet_state_dict": self.Q.state_dict(),
            "target_qnet_state_dict": self.Q_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "epsilon": self.epsilon_train,
            "epsilon_step_count": self.epsilon_step_count,
        }

    def save_models(self, model_dir_path: str):
        """Save model to file."""
        model_path = f"{model_dir_path}/{self.name}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.get_model_dict(), model_path)
        print(f"Saved {self.name} model to: {model_dir_path}")

    def load_models(self, model_dir_path: str):
        """Load model from file."""
        model_path = f"{model_dir_path}/{self.name}.pth"
        if not Path(model_path).exists():
            raise RuntimeError(f"Model file not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.Q.load_state_dict(checkpoint["qnet_state_dict"])
        self.Q_target.load_state_dict(checkpoint["target_qnet_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = checkpoint.get("training_steps", 0)
        self.epsilon_train = checkpoint.get("epsilon", self.epsilon_train)
        self.epsilon_step_count = checkpoint.get("epsilon_step_count", 0)
        print(f"Loaded {self.name} model from: {model_dir_path}")
