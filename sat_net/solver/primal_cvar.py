import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sat_net.nn import (
    ReplayBuffer,
    DiscretePolicy,
    TwinCritic,
    calc_heuristic_entropy,
    hard_update,
    soft_update,
    IQN,
    quantile_huber_loss,
    sample_taus, )
from sat_net.solver.base_solver import BaseSolver
from sat_net.util import NamedDict

if TYPE_CHECKING:
    from sat_net.datablock import DataBlock


class PrimalCVaR(BaseSolver):
    """
    Principled RIsk-aware Multi-Agent Learning (PRIMAL) with conditional-value-at-risk(CVaR) cost constraint
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: "NamedDict",
        tf_writer: Optional[SummaryWriter] = None,
    ):
        super().__init__(tf_writer=tf_writer)
        self.config = config
        
        # Dimensions
        self.state_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.hidden_dim
        self.feature_dim = config.feature_dim
        self.embedding_dim = config.embedding_dim
        self.num_quantiles = config.num_quantiles
        self.num_hidden_layers = config.num_hidden_layers
        self.weight_init = config.weight_init
        self.actor_use_layer_norm = config.actor_use_layer_norm
        self.critic_use_layer_norm = config.critic_use_layer_norm
        self.softmax_temperature = config.softmax_temperature

        assert self.weight_init in ["orthogonal", "xavier", "he"]

        # Risk parameters
        self.risk_level = config.risk_level
        self.cost_limit = config.cost_limit

        # Training parameters
        self.discount_reward = config.discount_reward
        self.discount_cost = config.discount_cost
        self.max_action_prob = config.max_action_prob

        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size

        # The Multi-timescale Alternating Update (adapted from TD3's delayed policy update trick)
        self.train_steps_per_update = config.train_steps_per_update
        self.actor_update_freq = config.actor_update_freq
        self.cost_multiplier_update_freq = config.cost_multiplier_update_freq
        self.update_lambda_after_step = config.update_lambda_after_step

        self.learning_rate = config.learning_rate
        self.device = config.device

        self.train_start_size = config.train_start_size

        self.update_method = config.update_method
        self.soft_update_tau = config.soft_update_tau
        self.hard_update_interval = config.hard_update_interval

        self.max_grad_norm = config.max_grad_norm

        self.training_steps = 0

        self.consider_risk_after_step = config.consider_risk_after_step

        # Create networks
        self.Qr, self.Qr_target, self.opt_Qr, self.lr_scheduler_Qr = self._create_reward_critics()
        self.Qc, self.Qc_target, self.opt_Qc, self.lr_scheduler_Qc = self._create_cost_critics()
        self.actor, self.opt_actor, self.lr_scheduler_actor = self._create_actor()

        self.target_entropy = calc_heuristic_entropy(self.action_dim, self.max_action_prob)

        self.log_alpha = torch.tensor(np.log(1.0), requires_grad=True, device=self.device)
        self.opt_log_alpha = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.log_lambda = torch.tensor(np.log(1.0), requires_grad=True, device=self.device)
        self.opt_log_lambda = torch.optim.Adam([self.log_lambda], lr=self.learning_rate)

        self.replay_buffers = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            buffer_size=self.buffer_size,
            device=self.device
        )

    @property
    def name(self):
        return "PrimalCVaR"

    def alpha(self):
        return F.softplus(self.log_alpha)

    def lambdar(self):
        return F.softplus(self.log_lambda)

    def _create_reward_critics(self):
        main_network = TwinCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            use_layer_norm=self.critic_use_layer_norm,
            init_method=self.weight_init,
        ).to(self.device)

        target_network = TwinCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            use_layer_norm=self.critic_use_layer_norm,
            init_method=self.weight_init,
        ).to(self.device)

        target_network.load_state_dict(main_network.state_dict())
        target_network.eval()

        optimizer = torch.optim.Adam(main_network.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.95)
        return main_network, target_network, optimizer, lr_scheduler

    def _create_cost_critics(self):
        main_network = IQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            feature_dim=self.feature_dim,
            num_hidden_layers=self.num_hidden_layers,
            embedding_dim=self.embedding_dim,
            use_layer_norm=self.critic_use_layer_norm,
            init_method=self.weight_init,
        ).to(self.device)

        target_network = IQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            feature_dim=self.feature_dim,
            num_hidden_layers=self.num_hidden_layers,
            embedding_dim=self.embedding_dim,
            use_layer_norm=self.critic_use_layer_norm,
            init_method=self.weight_init,
        ).to(self.device)

        target_network.load_state_dict(main_network.state_dict())
        target_network.eval()

        optimizer = torch.optim.Adam(main_network.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.95)
        return main_network, target_network, optimizer, lr_scheduler

    def _create_actor(self):
        actor = DiscretePolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            use_layer_norm=self.actor_use_layer_norm,
            temperature=self.softmax_temperature,
            init_method=self.weight_init,
        ).to(self.device)

        opt_actor = torch.optim.Adam(actor.parameters(), lr=self.learning_rate)
        lr_scheduler_actor = torch.optim.lr_scheduler.StepLR(opt_actor, step_size=10000, gamma=0.95)
        return actor, opt_actor, lr_scheduler_actor

    def route(self, obs: np.ndarray, info: dict) -> tuple[Optional[int], Optional[dict]]:
        """
        Select action using actor policy.
        """
        state_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.device)

        if "action_mask" in info:
            action_mask = info["action_mask"]
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        else:
            action_mask_tensor = None

        with torch.no_grad():
            self.actor.eval()
            logits = self.actor.forward(state_tensor, action_mask_tensor)
            probs = F.softmax(logits, dim=-1)
            chosen_action = torch.multinomial(probs, 1).item()

            return chosen_action, None

    def on_action_over(self, packet: "DataBlock"):
        """
        Store experience in replay buffer and train the model.
        """
        if self.is_eval():
            return

        # basic transition information
        state = packet.last_action.state
        action = packet.last_action.action
        action_mask = packet.last_action.action_mask
        queue_delay = packet.last_action.queue_delay
        delay_norm = packet.last_action.delay_norm
        next_state = packet.last_action.next_state
        next_action_mask = packet.last_action.next_action_mask
        done = packet.last_action.done
        truncated = packet.last_action.truncated

        progress_gain = packet.last_action.progress_gain
        reached_goal = packet.last_action.reached_goal
        current_progress = packet.last_action.current_progress
        action_delay = packet.last_action.action_delay

        # consider adding a terminal bonus/penalty, depending on the event type
        reward = progress_gain + reached_goal * (1 + packet.size)
        reward -= (action_delay - queue_delay) / delay_norm
        if reached_goal == -1:  # dropped penalties
            reward = -current_progress  # all previous efforts made on approaching the target are lost
            reward -= packet.ttl * 5 / delay_norm

        cost = queue_delay / delay_norm

        cost_limit = self.cost_limit / delay_norm  # normalized delay
        if self.discount_cost < 1:
            max_episode_len = packet.ttl_max
            target_cost = cost_limit / max_episode_len * (1 - self.discount_cost**max_episode_len) / (1 - self.discount_cost)
        else:
            target_cost = cost_limit

        self.replay_buffers.add(
            state=state,
            action=action,
            action_mask=action_mask,
            reward=reward,
            cost=cost,
            done=done,
            truncated=truncated,
            next_state=next_state,
            next_action_mask=next_action_mask,
            target_cost=target_cost,
        )

    def on_episode_over(self, packet: "DataBlock"):
        pass

    def learn(self):
        """
        Train the networks if we have enough experiences in the replay buffers.
        """
        if self.is_eval():
            return

        # Check if any agent has enough experiences to train
        agents_ready_to_train = len(self.replay_buffers) >= self.batch_size

        if not agents_ready_to_train:
            return

        for _ in range(self.train_steps_per_update):
            self.training_steps += 1
            self._train_step()

    def _train_step(self):
        if len(self.replay_buffers) < max(self.train_start_size, self.batch_size):
            return

        self.Qr.train()
        self.actor.train()

        # Sample a batch of experiences
        batch = self.replay_buffers.sample(self.batch_size)
        bootstrap_mask = ~(batch.dones & ~batch.truncateds)

        # ========= reward critic loss =========
        with torch.no_grad():
            next_logits = self.actor.forward(batch.next_states, batch.next_action_masks)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            next_qr, _, _ = self.Qr_target.forward(batch.next_states)
            next_vr = torch.sum(
                next_probs * (next_qr - self.alpha() * next_log_probs),
                dim=-1,
                keepdim=True,
            )
            yr_a = batch.rewards + self.discount_reward * bootstrap_mask * next_vr

        qr, qr1, qr2 = self.Qr.forward(batch.states)
        qr_a = qr.gather(1, batch.actions)
        qr1_a = qr1.gather(-1, batch.actions)
        qr2_a = qr2.gather(-1, batch.actions)
        reward_critic_loss = F.mse_loss(qr1_a, yr_a) + F.mse_loss(qr2_a, yr_a)

        self.opt_Qr.zero_grad()
        reward_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Qr.parameters(), max_norm=self.max_grad_norm)
        self.opt_Qr.step()
        self._update_target_network(target=self.Qr_target, source=self.Qr)

        if self._tf_writer is not None:
            self._tf_writer.add_scalar("primal_cvar/qr", qr_a.mean().item(), global_step=self.training_steps)
            self._tf_writer.add_scalar("primal_cvar/reward_critic_loss", reward_critic_loss, global_step=self.training_steps)

        # ========= cost critic loss =========
        with torch.no_grad():
            next_taus = torch.rand(self.batch_size, self.num_quantiles).to(self.device)
            next_quantiles = self.Qc_target.forward(batch.next_states, next_taus)  # (batch_size, n_quantiles, action_dim)
            next_vc = torch.sum(next_probs.unsqueeze(1) * next_quantiles, dim=-1, keepdim=True)  # (batch_size, n_quantiles, 1)
            yc_a = batch.costs.unsqueeze(1) + self.discount_cost * (bootstrap_mask.unsqueeze(1)) * next_vc  # (batch_size, n_quantiles, 1)

        taus = torch.rand(self.batch_size, self.num_quantiles).to(self.device)  # (batch_size, n_quantiles)
        qc = self.Qc.forward(batch.states, taus)  # (batch_size, n_quantiles, action_dim)
        qc_a = qc.gather(-1, batch.actions.unsqueeze(1).expand(-1, self.num_quantiles, 1))  # (batch_size, n_quantiles, 1)
        cost_critic_loss = quantile_huber_loss(qc_a, yc_a, taus.unsqueeze(-1))

        self.opt_Qc.zero_grad()
        cost_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Qc.parameters(), max_norm=self.max_grad_norm)
        self.opt_Qc.step()
        self._update_target_network(target=self.Qc_target, source=self.Qc)

        if self._tf_writer is not None:
            qc_mean = qc_a.mean(dim=1).mean().item()
            self._tf_writer.add_scalar("primal_cvar/qc", qc_mean, global_step=self.training_steps)
            self._tf_writer.add_scalar("primal_cvar/cost_critic_loss", cost_critic_loss, global_step=self.training_steps)

        # ========= actor loss =========
        if self.training_steps % self.actor_update_freq == 0:  # delayed policy update
            logits = self.actor.forward(batch.states, batch.action_masks)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            with torch.no_grad():
                if self.training_steps >= self.consider_risk_after_step:
                    taus_CVaR = sample_taus(
                        batch_size=self.batch_size,
                        n_quantiles=self.num_quantiles,
                        device=self.device,
                        min_tau=1 - self.risk_level,
                        max_tau=1.0,
                    )
                else:
                    taus_CVaR = sample_taus(
                        batch_size=self.batch_size,
                        n_quantiles=self.num_quantiles,
                        device=self.device,
                        min_tau=0.0,
                        max_tau=1.0,
                    )
                qc_all = self.Qc.forward(batch.states, taus_CVaR)  # (batch_size, n_quantiles, action_dim)
                CVaR = qc_all.mean(dim=-2)  # (batch_size, action_dim)

            lam = self.lambdar()
            alpha = self.alpha()
            actor_loss = (probs * (alpha.detach() * log_probs - qr.detach() + lam.detach() * CVaR)).sum(dim=-1).mean()

            self.opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
            self.opt_actor.step()

            policy_entropy = (-probs * log_probs).sum(dim=-1).detach()
            alpha_loss = (alpha * (policy_entropy - self.target_entropy)).mean()

            self.opt_log_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_log_alpha.step()
            max_probs = probs.max(dim=-1).values.mean().item()
            if self._tf_writer is not None:
                self._tf_writer.add_scalar("primal_cvar/actor_loss", actor_loss.item(), global_step=self.training_steps)
                self._tf_writer.add_scalar("primal_cvar/max_probs", max_probs, global_step=self.training_steps)
                self._tf_writer.add_scalar("primal_cvar/policy_entropy", policy_entropy.mean().item(),global_step=self.training_steps)
                self._tf_writer.add_scalar("primal_cvar/target_entropy",self.target_entropy,global_step=self.training_steps)
                self._tf_writer.add_scalar("primal_cvar/alpha", alpha.item(), global_step=self.training_steps)

            policy_CVaR = (probs * CVaR).sum(dim=-1, keepdim=True).detach()

            if self.training_steps >= self.update_lambda_after_step:
                if self.training_steps % (self.cost_multiplier_update_freq * self.actor_update_freq) == 0:
                    lambda_loss = (lam * (batch.target_costs - policy_CVaR)).mean()
                    self.opt_log_lambda.zero_grad()
                    lambda_loss.backward()
                    self.opt_log_lambda.step()

            if self._tf_writer is not None:
                self._tf_writer.add_scalar("primal_cvar/off_policy_CVaR", policy_CVaR.mean().item(), global_step=self.training_steps)
                self._tf_writer.add_scalar("primal_cvar/off_policy_lambda", lam.item(), global_step=self.training_steps)
                self._tf_writer.add_scalar("primal_cvar/off_policy_target_cost", batch.target_costs.mean().item(), global_step=self.training_steps)

    def get_stats(self) -> str:
        actor_training_steps = self.training_steps // self.actor_update_freq
        info_text = f"α:{self.alpha().item()} λ:{self.lambdar().item()} training_steps={self.training_steps} actor_training_steps:{actor_training_steps}"
        return info_text

    def _update_target_network(self, target: nn.Module, source: nn.Module):
        """update of target network parameters"""
        if self.update_method == "soft":
            soft_update(target=target, source=source, tau=self.soft_update_tau)
        elif (self.training_steps + 1) % self.hard_update_interval == 0:
            hard_update(target=target, source=source)

    def on_train_signal(self):
        """Train the networks if we have enough experiences in the replay buffers."""
        if self.is_train():
            self.learn()

    def get_model_dict(self):
        state_dict = {
            "training_steps": self.training_steps,
            "Qr": self.Qr.state_dict(),
            "Qr_target": self.Qr_target.state_dict(),
            "Qc": self.Qc.state_dict(),
            "Qc_target": self.Qc_target.state_dict(),
            "actor": self.actor.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "log_lambda": self.log_lambda.item(),
        }
        return state_dict

    def save_models(self, model_dir_path: str):
        """
        Save model to file.

        Args:
            model_dir_path: Directory path for the saved model files
        """
        model_path = f"{model_dir_path}/{self.name}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.get_model_dict(), model_path)
        print(f"Saved {self.name} model: {model_dir_path}")

    def load_models(self, model_dir_path: str):
        """
        Load model from file.

        Args:
            model_dir_path: Directory path for the saved model files
        """
        model_path = f"{model_dir_path}/{self.name}.pth"
        if not Path(model_path).exists():
            raise RuntimeError(f"Model file not found at {model_path}")

        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)

        self.training_steps = checkpoint.get("training_steps", 0)
        self.Qr.load_state_dict(checkpoint["Qr"])
        self.Qr_target.load_state_dict(checkpoint["Qr_target"])
        self.Qc.load_state_dict(checkpoint["Qc"])
        self.Qc_target.load_state_dict(checkpoint["Qc_target"])
        self.actor.load_state_dict(checkpoint["actor"])
        if "log_alpha" in checkpoint:
            self.log_alpha = torch.tensor(checkpoint["log_alpha"], requires_grad=True, device=self.device)
            self.opt_log_alpha = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        if "log_lambda" in checkpoint:
            self.log_lambda = torch.tensor(checkpoint["log_lambda"], requires_grad=True, device=self.device)
            self.opt_log_lambda = torch.optim.Adam([self.log_lambda], lr=self.learning_rate)
        print(f"Loaded {self.name} model: {model_dir_path}")
