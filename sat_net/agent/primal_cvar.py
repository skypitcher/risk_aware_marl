from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sat_net.nn import (
    DiscretePolicy,
    IQN,
    ReplayBuffer,
    TwinCritic,
    calc_heuristic_entropy,
    hard_update,
    quantile_huber_loss,
    sample_taus,
    soft_update,
    weighted_mean,
    weighted_mse_loss,
)
from sat_net.agent.rl_base import BatchedRLAgent
from sat_net.util import NamedDict


class PrimalCVaRAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: NamedDict, device: torch.device, tf_writer: Any = None):
        self.config = config
        self.device = device
        self._tf_writer = tf_writer
        self.state_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = int(config.get("hidden_dim", 256))
        self.feature_dim = int(config.get("feature_dim", 256))
        self.embedding_dim = int(config.get("embedding_dim", 64))
        self.num_quantiles = int(config.get("num_quantiles", 64))
        self.num_hidden_layers = int(config.get("num_hidden_layers", 2))
        self.risk_level = float(config.get("risk_level", 0.25))
        self.discount_reward = float(config.get("discount_reward", 0.99))
        self.discount_cost = float(config.get("discount_cost", 0.97))
        self.batch_size = int(config.get("batch_size", 2048))
        self.train_start_size = int(config.get("train_start_size", 10000))
        self.train_steps_per_update = int(config.get("train_steps_per_update", 1))
        self.actor_update_freq = int(config.get("actor_update_freq", 2))
        self.cost_multiplier_update_freq = int(config.get("cost_multiplier_update_freq", 2))
        self.update_lambda_after_step = int(config.get("update_lambda_after_step", 300000))
        self.consider_risk_after_step = int(config.get("consider_risk_after_step", 300000))
        self.update_method = str(config.get("update_method", "soft"))
        self.soft_update_tau = float(config.get("soft_update_tau", 0.05))
        self.hard_update_interval = int(config.get("hard_update_interval", 1))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))
        self.training_steps = 0

        init_method = str(config.get("weight_init", "orthogonal"))
        actor_ln = bool(config.get("actor_use_layer_norm", True))
        critic_ln = bool(config.get("critic_use_layer_norm", True))
        lr = float(config.get("learning_rate", 1e-4))
        self.Qr = TwinCritic(obs_dim, action_dim, self.hidden_dim, self.num_hidden_layers, critic_ln, init_method).to(device)
        self.Qr_target = TwinCritic(obs_dim, action_dim, self.hidden_dim, self.num_hidden_layers, critic_ln, init_method).to(device)
        self.Qr_target.load_state_dict(self.Qr.state_dict())
        self.opt_Qr = torch.optim.Adam(self.Qr.parameters(), lr=lr)

        self.Qc = IQN(
            obs_dim,
            action_dim,
            self.feature_dim,
            self.hidden_dim,
            self.num_hidden_layers,
            self.embedding_dim,
            critic_ln,
            init_method,
        ).to(device)
        self.Qc_target = IQN(
            obs_dim,
            action_dim,
            self.feature_dim,
            self.hidden_dim,
            self.num_hidden_layers,
            self.embedding_dim,
            critic_ln,
            init_method,
        ).to(device)
        self.Qc_target.load_state_dict(self.Qc.state_dict())
        self.opt_Qc = torch.optim.Adam(self.Qc.parameters(), lr=lr)

        self.actor = DiscretePolicy(
            obs_dim,
            action_dim,
            self.hidden_dim,
            self.num_hidden_layers,
            actor_ln,
            float(config.get("softmax_temperature", 1.0)),
            init_method,
        ).to(device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.target_entropy = calc_heuristic_entropy(action_dim, float(config.get("max_action_prob", 0.99)))
        self.log_alpha = torch.tensor(np.log(1.0), dtype=torch.float32, requires_grad=True, device=device)
        self.log_lambda = torch.tensor(np.log(1.0), dtype=torch.float32, requires_grad=True, device=device)
        self.opt_log_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.opt_log_lambda = torch.optim.Adam([self.log_lambda], lr=lr)
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, int(config.get("buffer_size", 300000)), device)

    def alpha(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)

    def lambdar(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def act(self, states: np.ndarray, action_masks: np.ndarray, eval_mode: bool) -> np.ndarray:
        if len(states) == 0:
            return np.empty(0, dtype=np.int64)
        with torch.no_grad():
            state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
            logits = self.actor(state_tensor, mask_tensor)
            if eval_mode:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        out = actions.cpu().numpy().astype(np.int64)
        out[~action_masks.any(axis=1)] = -1
        return out

    def add_transition(self, **kwargs) -> None:
        self.replay_buffer.add(**kwargs)

    def learn(self) -> None:
        if len(self.replay_buffer) < max(self.batch_size, self.train_start_size):
            return
        for _ in range(self.train_steps_per_update):
            self.training_steps += 1
            self._train_step()

    def _train_step(self) -> None:
        batch = self.replay_buffer.sample(self.batch_size)
        bootstrap = (~(batch.dones & ~batch.truncateds)).float()
        with torch.no_grad():
            next_logits = self.actor(batch.next_states, batch.next_action_masks)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_qr, _, _ = self.Qr_target(batch.next_states)
            next_vr = torch.sum(next_probs * (next_qr - self.alpha() * next_log_probs), dim=-1, keepdim=True)
            target_reward = batch.rewards + self.discount_reward * bootstrap * next_vr

        qr, qr1, qr2 = self.Qr(batch.states)
        reward_loss = weighted_mse_loss(qr1.gather(-1, batch.actions), target_reward, batch.weights) + weighted_mse_loss(
            qr2.gather(-1, batch.actions), target_reward, batch.weights
        )
        self.opt_Qr.zero_grad()
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Qr.parameters(), self.max_grad_norm)
        self.opt_Qr.step()
        self._update_target(self.Qr_target, self.Qr)

        with torch.no_grad():
            next_taus = sample_taus(self.batch_size, self.num_quantiles, self.device)
            next_quantiles = self.Qc_target(batch.next_states, next_taus)
            next_vc = torch.sum(next_probs.unsqueeze(1) * next_quantiles, dim=-1, keepdim=True)
            target_cost = batch.costs.unsqueeze(1) + self.discount_cost * bootstrap.unsqueeze(1) * next_vc

        taus = sample_taus(self.batch_size, self.num_quantiles, self.device)
        qc = self.Qc(batch.states, taus)
        qc_a = qc.gather(-1, batch.actions.unsqueeze(1).expand(-1, self.num_quantiles, 1))
        cost_loss = quantile_huber_loss(qc_a, target_cost, taus.unsqueeze(-1), sample_weights=batch.weights)
        self.opt_Qc.zero_grad()
        cost_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Qc.parameters(), self.max_grad_norm)
        self.opt_Qc.step()
        self._update_target(self.Qc_target, self.Qc)

        if self.training_steps % self.actor_update_freq == 0:
            logits = self.actor(batch.states, batch.action_masks)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            min_tau = 1.0 - self.risk_level if self.training_steps >= self.consider_risk_after_step else 0.0
            with torch.no_grad():
                cvar_taus = sample_taus(self.batch_size, self.num_quantiles, self.device, min_tau=min_tau, max_tau=1.0)
                cvar = self.Qc(batch.states, cvar_taus).mean(dim=1)
            actor_objective = (
                probs
                * (self.alpha().detach() * log_probs - qr.detach() + self.lambdar().detach() * cvar.detach())
            ).sum(dim=-1, keepdim=True)
            actor_loss = weighted_mean(actor_objective, batch.weights)
            self.opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.opt_actor.step()

            entropy = (-probs * log_probs).sum(dim=-1).detach()
            alpha_loss = weighted_mean(self.alpha() * (entropy.unsqueeze(-1) - self.target_entropy), batch.weights)
            self.opt_log_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_log_alpha.step()

            if self.training_steps >= self.update_lambda_after_step:
                policy_cvar = (probs * cvar).sum(dim=-1, keepdim=True).detach()
                if self.training_steps % (self.cost_multiplier_update_freq * self.actor_update_freq) == 0:
                    lambda_loss = weighted_mean(self.lambdar() * (batch.target_costs - policy_cvar), batch.weights)
                    self.opt_log_lambda.zero_grad()
                    lambda_loss.backward()
                    self.opt_log_lambda.step()

        if self._tf_writer is not None:
            self._tf_writer.add_scalar("primal_cvar/reward_loss", reward_loss.item(), self.training_steps)
            self._tf_writer.add_scalar("primal_cvar/cost_loss", cost_loss.item(), self.training_steps)

    def _update_target(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        if self.update_method == "soft":
            soft_update(target, source, self.soft_update_tau)
        elif self.training_steps % self.hard_update_interval == 0:
            hard_update(target, source)

    def get_model_dict(self) -> dict[str, Any]:
        return {
            "training_steps": self.training_steps,
            "Qr": self.Qr.state_dict(),
            "Qr_target": self.Qr_target.state_dict(),
            "Qc": self.Qc.state_dict(),
            "Qc_target": self.Qc_target.state_dict(),
            "actor": self.actor.state_dict(),
            "log_alpha": float(self.log_alpha.detach().cpu()),
            "log_lambda": float(self.log_lambda.detach().cpu()),
        }

    def load_model_dict(self, checkpoint: dict[str, Any]) -> None:
        self.training_steps = int(checkpoint.get("training_steps", 0))
        self.Qr.load_state_dict(checkpoint["Qr"])
        self.Qr_target.load_state_dict(checkpoint["Qr_target"])
        self.Qc.load_state_dict(checkpoint["Qc"])
        self.Qc_target.load_state_dict(checkpoint["Qc_target"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.log_alpha.data.fill_(float(checkpoint.get("log_alpha", 0.0)))
        self.log_lambda.data.fill_(float(checkpoint.get("log_lambda", 0.0)))

    def get_stats(self) -> str:
        return (
            f"alpha={self.alpha().item():.4f} lambda={self.lambdar().item():.4f} "
            f"buffer={len(self.replay_buffer)} training_steps={self.training_steps}"
        )


class PrimalCVaR(BatchedRLAgent):
    def __init__(self, config: NamedDict, obs_dim: int = 94, action_dim: int = 4, tf_writer: Any = None):
        super().__init__(config=config, obs_dim=obs_dim, action_dim=action_dim, tf_writer=tf_writer)
        self.global_agent = PrimalCVaRAgent(self.obs_dim, self.action_dim, config, self.device, tf_writer)

    @property
    def name(self) -> str:
        return "PrimalCVaR"

    def select_actions(self, states: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        return self.global_agent.act(states, action_masks, eval_mode=self.is_eval())

    def add_transition(self, **kwargs) -> None:
        self.global_agent.add_transition(**kwargs)

    def learn(self) -> None:
        self.global_agent.learn()

    def get_stats(self) -> str:
        return f"{self.global_agent.get_stats()} device={self.device}"

    def save_models(self, model_dir_path: str) -> None:
        os.makedirs(model_dir_path, exist_ok=True)
        torch.save(self.global_agent.get_model_dict(), f"{model_dir_path}/{self.name}.pth")

    def load_models(self, model_dir_path: str) -> None:
        model_path = Path(model_dir_path) / f"{self.name}.pth"
        if not model_path.exists():
            raise RuntimeError(f"Model file not found at {model_path}")
        self.global_agent.load_model_dict(torch.load(model_path, map_location=self.device, weights_only=False))
