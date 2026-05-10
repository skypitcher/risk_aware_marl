from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def init_weights(network: nn.Module, init_method: str = "orthogonal") -> None:
    for module in network.modules():
        if isinstance(module, nn.Linear):
            method = init_method.lower()
            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif method in {"he", "kaiming"}:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif method == "orthogonal":
                nn.init.orthogonal_(module.weight)
            else:
                raise ValueError(f"Unknown initialization method: {init_method}")
            nn.init.constant_(module.bias, 0.0)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


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
        layers: list[nn.Module] = []
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
        self.net = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=use_layer_norm,
            init_method=init_method,
        )
        self.temperature = temperature

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(state) / self.temperature
        if action_mask is None:
            return logits
        return logits.masked_fill(~action_mask, -1e9)


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
        self.q1 = MLP(state_dim, action_dim, hidden_dim, num_hidden_layers, use_layer_norm, init_method)
        self.q2 = MLP(state_dim, action_dim, hidden_dim, num_hidden_layers, use_layer_norm, init_method)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1 = self.q1(state)
        q2 = self.q2(state)
        return torch.minimum(q1, q2), q1, q2


class IQN(nn.Module):
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
        super().__init__()
        self.state_net = MLP(state_dim, feature_dim, hidden_dim, num_hidden_layers, use_layer_norm, init_method)
        self.quantile_net = nn.Sequential(nn.Linear(embedding_dim, feature_dim), nn.ReLU())
        self.value_net = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, action_dim))
        self.register_buffer("cos_embedding", torch.arange(1, embedding_dim + 1).float() * math.pi)
        init_weights(self.quantile_net, init_method)
        init_weights(self.value_net, init_method)

    def forward(self, states: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        state_embed = torch.relu(self.state_net(states))
        cos_embed = torch.cos(taus.unsqueeze(-1) * self.cos_embedding)
        quantile_embed = self.quantile_net(cos_embed)
        return self.value_net(state_embed.unsqueeze(1) * quantile_embed)


def sample_taus(
    batch_size: int,
    n_quantiles: int,
    device: str | torch.device,
    min_tau: float = 0.0,
    max_tau: float = 1.0,
) -> torch.Tensor:
    return torch.rand(batch_size, n_quantiles, device=device) * (max_tau - min_tau) + min_tau


def quantile_huber_loss(
    quantile_values: torch.Tensor,
    target_values: torch.Tensor,
    taus: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    kappa: float = 1.0,
) -> torch.Tensor:
    diff = target_values.unsqueeze(1) - quantile_values.unsqueeze(2)
    huber = torch.where(diff.abs() <= kappa, 0.5 * diff.square(), kappa * (diff.abs() - 0.5 * kappa))
    weight = torch.abs(taus.unsqueeze(-1) - (diff < 0).float())
    loss = weight * huber
    if sample_weights is None:
        return loss.mean()
    sample_loss = loss.flatten(start_dim=1).mean(dim=1, keepdim=True)
    return weighted_mean(sample_loss, sample_weights)


def normalized_sample_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float()
    return weights / weights.mean().clamp_min(1e-6)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = normalized_sample_weights(weights)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).mean()


def weighted_mse_loss(input_values: torch.Tensor, target_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return weighted_mean((input_values - target_values).square(), weights)


def calc_heuristic_entropy(action_dim: int, max_action_prob: float) -> float:
    rest_prob = (1.0 - max_action_prob) / max(action_dim - 1, 1)
    probs = np.full(action_dim, rest_prob, dtype=np.float64)
    probs[0] = max_action_prob
    return float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))


@dataclass(slots=True)
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
    weights: torch.Tensor
    flowlet_ids: torch.Tensor
    agent_ids: torch.Tensor
    next_agent_ids: torch.Tensor


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str | torch.device):
        self.buffer_size = int(buffer_size)
        self.device = torch.device(device)
        self.states = np.zeros((self.buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, 1), dtype=np.int64)
        self.action_masks = np.zeros((self.buffer_size, action_dim), dtype=bool)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.costs = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=bool)
        self.truncateds = np.zeros((self.buffer_size, 1), dtype=bool)
        self.next_states = np.zeros((self.buffer_size, state_dim), dtype=np.float32)
        self.next_action_masks = np.zeros((self.buffer_size, action_dim), dtype=bool)
        self.target_costs = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.weights = np.ones((self.buffer_size, 1), dtype=np.float32)
        self.flowlet_ids = np.full((self.buffer_size, 1), -1, dtype=np.int64)
        self.agent_ids = np.full((self.buffer_size, 1), -1, dtype=np.int64)
        self.next_agent_ids = np.full((self.buffer_size, 1), -1, dtype=np.int64)
        self.ptr = 0
        self.current_size = 0

    def add(
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
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx, 0] = action
        self.action_masks[idx] = action_mask
        self.rewards[idx, 0] = reward
        self.costs[idx, 0] = 0.0 if cost is None else cost
        self.dones[idx, 0] = done
        self.truncateds[idx, 0] = truncated
        self.next_states[idx] = next_state
        self.next_action_masks[idx] = next_action_mask
        self.target_costs[idx, 0] = 0.0 if target_cost is None else target_cost
        self.weights[idx, 0] = max(float(weight), 1.0)
        self.flowlet_ids[idx, 0] = int(flowlet_id)
        self.agent_ids[idx, 0] = int(agent_id)
        self.next_agent_ids[idx, 0] = int(next_agent_id)
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Batch:
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        return Batch(
            states=torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(self.actions[indices], dtype=torch.long, device=self.device),
            action_masks=torch.as_tensor(self.action_masks[indices], dtype=torch.bool, device=self.device),
            rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            costs=torch.as_tensor(self.costs[indices], dtype=torch.float32, device=self.device),
            dones=torch.as_tensor(self.dones[indices], dtype=torch.bool, device=self.device),
            truncateds=torch.as_tensor(self.truncateds[indices], dtype=torch.bool, device=self.device),
            next_states=torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            next_action_masks=torch.as_tensor(self.next_action_masks[indices], dtype=torch.bool, device=self.device),
            target_costs=torch.as_tensor(self.target_costs[indices], dtype=torch.float32, device=self.device),
            weights=torch.as_tensor(self.weights[indices], dtype=torch.float32, device=self.device),
            flowlet_ids=torch.as_tensor(self.flowlet_ids[indices], dtype=torch.long, device=self.device),
            agent_ids=torch.as_tensor(self.agent_ids[indices], dtype=torch.long, device=self.device),
            next_agent_ids=torch.as_tensor(self.next_agent_ids[indices], dtype=torch.long, device=self.device),
        )

    def metadata_summary(self) -> dict[str, float | int]:
        if self.current_size == 0:
            return {
                "replay_size": 0,
                "sample_weight_mean": 0.0,
                "sample_weight_max": 0.0,
                "unique_agent_count": 0,
                "unique_next_agent_count": 0,
            }

        weights = self.weights[: self.current_size, 0]
        agent_ids = self.agent_ids[: self.current_size, 0]
        next_agent_ids = self.next_agent_ids[: self.current_size, 0]
        return {
            "replay_size": int(self.current_size),
            "sample_weight_mean": float(weights.mean()),
            "sample_weight_max": float(weights.max()),
            "unique_agent_count": int(len(np.unique(agent_ids[agent_ids >= 0]))),
            "unique_next_agent_count": int(len(np.unique(next_agent_ids[next_agent_ids >= 0]))),
        }

    def __len__(self) -> int:
        return self.current_size
