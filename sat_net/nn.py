import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_data(data: np.ndarray | torch.Tensor) -> bool:
    if isinstance(data, torch.Tensor):
        return data is not None and not data.isinf().any() and not data.isnan().any()
    else:
        return data is not None and not np.isinf(data).any() and not np.isnan(data).any()


@torch.no_grad()
def init_weights(network: nn.Module, init_method: str = "xavier"):
    """
    Initialize network weights using specified initialization method.

    This utility function can be applied to any PyTorch network to initialize
    its Linear layer weights and biases.

    Args:
        network: PyTorch network module to initialize
        init_method: Initialization method - "xavier", "he", or "orthogonal"

    Weight Initialization Guidelines:
    - "xavier": Good for networks with sigmoid/tanh activations (balanced variance)
    - "he(kaiming)": Best for ReLU networks (accounts for ReLU's zero-negative property)
    - "orthogonal": Good for deep networks, helps with gradient flow and stability
    """
    for module in network.modules():
        if isinstance(module, nn.Linear):
            if init_method.lower() == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif init_method.lower() == "he":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif init_method.lower() == "orthogonal":
                nn.init.orthogonal_(module.weight)
            else:
                raise ValueError(
                    f"Unknown initialization method: {init_method}. Choose from 'xavier', 'he', or 'orthogonal'")

            # Initialize bias to zero for all methods
            nn.init.constant_(module.bias, 0.0)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    Soft update: θ_target = τ*θ_local + (1-τ)*θ_target
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: nn.Module, source: nn.Module):
    """
    hard update: θ_target = θ_local
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def take_grad_step(network: nn.Module, opt: torch.optim.Optimizer, loss: torch.Tensor, clip_grad_norm: float):
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=clip_grad_norm)
    opt.step()


def get_activation_func(func_name: str):
    if func_name == "none":
        return None
    if func_name == "tanh":
        return F.tanh
    if func_name == "softplus":
        return F.softplus
    if func_name == "sigmoid":
        return F.sigmoid
    if func_name == "relu":
        return F.relu
    if func_name == "silu":
        return F.silu

    raise ValueError(
        f"Unknown output activation: {func_name}. Choose from 'tanh', 'softplus', 'sigmoid', 'relu', or 'silu'")


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_hidden_layers: int,
            use_layer_norm: bool = True,
            init_method: str = "orthogonal",
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        init_weights(self.net, init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Multiplier(nn.Module):
    def __init__(
            self,
            state_dim: int,
            hidden_dim: int,
            num_hidden_layers: int,
            use_layer_norm: bool = True,
            init_method: str = "orthogonal",
    ):
        super().__init__()
        self.fc_net = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=use_layer_norm,
            init_method=init_method,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(self.fc_net(state)) * 1e4


class RunningMeanStd:
    """Computes a running mean and standard deviation for normalization trick."""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)


class DiscretePolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            num_hidden_layers: int,
            use_layer_norm: bool = True,
            temperature: float = 1.0,
            init_method: str = "orthogonal",
    ):
        super().__init__()
        layers = []
        in_dim = state_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)

        self.output_head = nn.Linear(hidden_dim, action_dim)

        init_weights(self.feature_net, init_method)
        init_weights(self.output_head, init_method)

        self.temperature = temperature

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        x = state
        x = self.feature_net(x)
        x = self.output_head(x) / self.temperature
        if action_mask is not None:
            logits = x.masked_fill(~action_mask, -1e-8)
        else:
            logits = x
        return logits


class PIDController:
    """A Proportional-Integral-Derivative (PID) feedback-based controller for updating lagrangian multiplier."""

    def __init__(
            self, Kp: float, Ki: float, Kd: float, device: str, max_multiplier: float = 1000.0,
            default_multiplier: float = 0.0
    ):
        self.device = device
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_integral = torch.zeros(1, device=self.device)
        self.error_last = torch.zeros(1, device=self.device)
        self.multiplier = torch.ones(1, device=self.device) * default_multiplier
        self.max_multiplier = torch.ones(1, device=self.device) * max_multiplier

    def update(self, violations: torch.Tensor):
        error_new = violations.mean()
        error_diff = torch.clamp(error_new - self.error_last, min=0.0)
        self.error_integral = torch.clamp(self.error_integral + error_new, min=0.0)
        self.error_last = error_new
        # Calculate PID output for lagrangian multiplier
        new_multiplier = self.Kp * error_new + self.Ki * self.error_integral + self.Kd * error_diff
        self.multiplier = torch.clamp(new_multiplier, min=torch.zeros_like(new_multiplier), max=self.max_multiplier)


class IQN(nn.Module):
    """
    Implicit Quantile Network (IQN) for distributional RL
    See: https://arxiv.org/abs/1806.06923
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            feature_dim: int,
            hidden_dim: int,
            num_hidden_layers: int,
            embedding_dim: int = 64,
            use_layer_norm: bool = True,
            init_method: str = "orthogonal",
    ):
        super(IQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # State embedding network
        self.state_net = MLP(
            input_dim=state_dim,
            output_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            init_method=init_method,
            use_layer_norm=use_layer_norm,
        )

        # Quantile embedding network
        self.quantile_net = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            nn.ReLU(),
        )

        # Combined network to produce quantile values
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim),
        )

        # Cosine embedding constants
        self.register_buffer("cos_embedding", torch.arange(1, embedding_dim + 1).float() * math.pi)

    def forward(self, states: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim)
            taus: (batch_size, n_quantiles)
        Returns:
            quantile_values: (batch_size, n_quantiles, action_dim)
        """
        # batch_size = states.shape[0]
        # n_quantiles = taus.shape[1]

        # State embedding
        state_embed = torch.relu(self.state_net(states))  # (batch_size, hidden_dim)

        # Quantile embedding using cosine basis functions
        # taus: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
        taus_expanded = taus.unsqueeze(-1)

        # Compute cosine embeddings: cos(i * pi * tau) for i in [1, embedding_dim]
        cos_embed = torch.cos(taus_expanded * self.cos_embedding)  # (batch_size, n_quantiles, embedding_dim)

        # Apply quantile network
        quantile_embed = self.quantile_net(cos_embed)  # (batch_size, n_quantiles, hidden_dim)

        # Element-wise product and sum
        # state_embed: (batch_size, 1, hidden_dim), quantile_embed: (batch_size, n_quantiles, hidden_dim)
        combined = state_embed.unsqueeze(1) * quantile_embed  # (batch_size, n_quantiles, hidden_dim)

        # Generate quantile values
        quantile_values = self.value_net(combined)  # (batch_size, n_quantiles, action_dim)

        return quantile_values


def sample_taus(batch_size: int, n_quantiles: int, device: str, min_tau: float = 0.0,
                max_tau: float = 1.0) -> torch.Tensor:
    """Sample random quantiles uniformly from [min_tau, max_tau], default is [0, 1]"""
    assert 0.0 <= min_tau <= max_tau <= 1.0, "Min and max tau must be in [0, 1]"
    return torch.rand(batch_size, n_quantiles, device=device) * (max_tau - min_tau) + min_tau


def quantile_huber_loss(
        quantile_values: torch.Tensor,
        target_values: torch.Tensor,
        taus: torch.Tensor,
        kappa: float = 1.0,
) -> torch.Tensor:
    """
    Compute the quantile Huber loss
    Args:
        quantile_values: (batch_size, n_quantiles, 1) - predicted quantile values
        target_values: (batch_size, n_target_quantiles, 1) - target quantile values
        taus: (batch_size, n_quantiles, 1) - quantile fractions
        kappa: Huber loss threshold
    """
    # Compute pairwise differences
    # quantile_values: (batch_size, n_quantiles, 1)
    # target_values: (batch_size, 1, n_target_quantiles)
    diff = target_values.unsqueeze(1) - quantile_values.unsqueeze(2)  # (batch_size, n_quantiles, n_target_quantiles)

    # Huber loss
    huber_loss = torch.where(diff.abs() <= kappa, 0.5 * diff.pow(2), kappa * (diff.abs() - 0.5 * kappa))

    # Quantile loss weighting
    # taus: (batch_size, n_quantiles, 1, 1)
    quantile_weight = torch.abs(taus.unsqueeze(-1) - (diff < 0).float())

    loss = quantile_weight * huber_loss

    return loss.mean()


class TwinCritic(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            num_hidden_layers: int,
            use_layer_norm: bool = True,
            init_method: str = "orthogonal",
    ):
        super().__init__()
        self.q1 = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=use_layer_norm,
            init_method=init_method,
        )
        self.q2 = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=use_layer_norm,
            init_method=init_method,
        )

    def forward(self, state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1 = self.q1.forward(state)
        q2 = self.q2.forward(state)
        q = torch.min(q1, q2)  # Use min to stabilize training
        return q, q1, q2


class GaussianCritic(nn.Module):
    """Gaussian Critic Network for estimating the distribution of Long Term Returns (Mean and Variance)"""

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            num_hidden_layers: int,
            use_layer_norm: bool = True,
            init_method: str = "orthogonal",
    ):
        super().__init__()
        self.mean_head = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=use_layer_norm,
            init_method=init_method,
        )
        self.var_head = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=use_layer_norm,
            init_method=init_method,
        )

    def forward(self, state):
        qc_vec = self.mean_head.forward(state)

        var_vec = F.softplus(self.var_head.forward(state)) + 1e-8

        return qc_vec, var_vec


def calc_heuristic_entropy(action_dim: int, max_action_prob: float):
    rest_prob = (1 - max_action_prob) / (action_dim - 1)
    action_probs = rest_prob * np.ones(action_dim)
    action_probs[0] = max_action_prob
    return -np.sum(action_probs * np.log(action_probs))


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor
    truncateds: torch.Tensor
    next_states: torch.Tensor
    next_action_masks: torch.Tensor
    target_costs: torch.Tensor


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str | torch.device):
        """
        Replay Buffer with pre-allocated memory on a specific device.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space (for action_mask).
            buffer_size (int): The maximum size of the buffer.
            device (str | torch.device): The device (e.g., 'cuda' or 'cpu') to store the tensors on.
        """
        self.buffer_size = buffer_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # Pre-allocate memory for all experience components
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 1), dtype=np.long)
        self.action_masks = np.zeros((buffer_size, action_dim), dtype=np.bool)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.costs = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.bool)
        self.truncateds = np.zeros((buffer_size, 1), dtype=np.bool)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_action_masks = np.zeros((buffer_size, action_dim), dtype=np.bool)
        self.target_costs = np.zeros((buffer_size, 1), dtype=np.float32)

        # Pointers for managing the circular buffer
        self.ptr = 0
        self.current_size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        reward: float,
        cost: float|None,
        done: bool,
        truncated: bool,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        target_cost: float|None
    ):
        """
        Add a new experience to the buffer.
        Inputs are expected to be NumPy arrays or Python scalars.
        """
        # Use the pointer to determine the index to write to
        idx = self.ptr

        self.states[idx] = state
        self.actions[idx] = action
        self.action_masks[idx] = action_mask
        self.rewards[idx] = reward
        if cost is not None:
            self.costs[idx] = cost
        self.dones[idx] = done
        self.truncateds[idx] = truncated
        self.next_states[idx] = next_state
        self.next_action_masks[idx] = next_action_mask
        if target_cost is not None:
            self.target_costs[idx] = target_cost

        # Update the pointer and size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a batch of experiences from the buffer.
        """
        assert batch_size <= self.current_size, "Not enough samples in the buffer to sample the requested batch size."

        # Generate random indices from the valid range of the buffer
        indices = np.random.choice(self.current_size, batch_size, replace=False)

        # Use the indices to retrieve the batch from pre-allocated tensors. This is very fast.
        states = torch.tensor(self.states[indices], dtype=torch.float, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.long, device=self.device)
        action_masks = torch.tensor(self.action_masks[indices], dtype=torch.bool, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        costs = torch.tensor(self.costs[indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.bool, device=self.device)
        truncated = torch.tensor(self.truncateds[indices], dtype=torch.bool, device=self.device)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device)
        next_action_masks = torch.tensor(self.next_action_masks[indices], dtype=torch.bool, device=self.device)
        target_costs = torch.tensor(self.target_costs[indices], dtype=torch.float32, device=self.device)

        return Batch(
            states=states,
            actions=actions,
            action_masks=action_masks,
            rewards=rewards,
            costs=costs,
            dones=dones,
            truncateds=truncated,
            next_states=next_states,
            next_action_masks=next_action_masks,
            target_costs=target_costs
        )

    def __len__(self):
        """Return the current number of items in the buffer."""
        return self.current_size
