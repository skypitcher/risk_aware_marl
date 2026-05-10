import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sat_net.nn import (
    DiscretePolicy,
    TwinCritic,
    calc_heuristic_entropy,
    hard_update,
    soft_update,
    ReplayBuffer,
)
from sat_net.solver.base_solver import BaseSolver
from sat_net.util import NamedDict


class SACAgent:
    """
    SAC Agent encapsulating the Soft Actor-Critic logic.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: "NamedDict", tf_writer: Optional[SummaryWriter] = None):
        self.config = config
        self._tf_writer = tf_writer
        self.device = config.device

        # Dimensions
        self.state_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config.hidden_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.weight_init = config.weight_init
        self.actor_use_layer_norm = config.actor_use_layer_norm
        self.critic_use_layer_norm = config.critic_use_layer_norm
        self.softmax_temperature = config.softmax_temperature

        # Training parameters
        self.discount = config.discount
        self.max_action_prob = config.max_action_prob
        self.is_eval = False

        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.train_steps_per_update = config.train_steps_per_update

        self.learning_rate = config.learning_rate

        self.train_start_size = config.train_start_size
        self.actor_update_freq = config.actor_update_freq
        self.update_method = config.update_method
        self.soft_update_tau = config.soft_update_tau
        self.hard_update_interval = config.hard_update_interval

        self.max_grad_norm = config.max_grad_norm

        self.training_steps = 0

        # Create networks
        self.Qr, self.Qr_target, self.opt_Qr, self.lr_scheduler_Qr = self._create_critics()
        self.actor, self.opt_actor, self.lr_scheduler_actor = self._create_actor()

        self.target_entropy = calc_heuristic_entropy(self.action_dim, self.max_action_prob)

        self.log_alpha = torch.tensor(np.log(1.0), requires_grad=True, device=self.device)
        self.opt_log_alpha = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(state_dim=self.state_dim, action_dim=self.action_dim, buffer_size=self.buffer_size, device=self.device)

    def alpha(self):
        return F.softplus(self.log_alpha)

    def _create_critics(self):
        """Create Twin-Q networks and its optimizer for the clipped double-Q learning trick"""
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

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None, eval_mode: bool = False):
        """Select action using actor policy."""
        state_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.device)

        if action_mask is not None:
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        else:
            action_mask_tensor = None

        with torch.no_grad():
            self.actor.eval()
            logits = self.actor.forward(state_tensor, action_mask_tensor)
            if eval_mode:
                chosen_action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                chosen_action = torch.multinomial(probs, 1).item()

            return chosen_action

    def store_experience(self, packet):
        """Store experience in replay buffer."""
        last_action = packet.last_action

        self.replay_buffer.add(
            state=last_action.state,
            action=last_action.action,
            action_mask=last_action.action_mask,
            reward=last_action.baseline_reward,
            cost=None,
            done=last_action.done,
            truncated=last_action.truncated,
            next_state=last_action.next_state,
            next_action_mask=last_action.next_action_mask,
            target_cost=None,
        )

    def learn(self):
        """Train the networks if we have enough experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.train_steps_per_update):
            self.training_steps += 1
            self._train_step()

    def _train_step(self):
        if len(self.replay_buffer) < max(self.train_start_size, self.batch_size):
            return

        self.Qr.train()
        self.actor.train()

        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        bootstrap_mask = ~(batch.dones & ~batch.truncateds)

        # ========= critic loss =========
        with torch.no_grad():
            next_logits = self.actor.forward(batch.next_states, batch.next_action_masks)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            qval_next, _, _ = self.Qr_target.forward(batch.next_states)
            v_next = torch.sum(next_probs * (qval_next - self.alpha() * next_log_probs), dim=-1, keepdim=True)
            y = batch.rewards + self.discount * bootstrap_mask * v_next

        q_val, q1_val, q2_val = self.Qr.forward(batch.states)
        q1 = q1_val.gather(-1, batch.actions)
        q2 = q2_val.gather(-1, batch.actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt_Qr.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Qr.parameters(), max_norm=self.max_grad_norm)
        self.opt_Qr.step()
        self._update_target_network(target=self.Qr_target, source=self.Qr)

        if self._tf_writer is not None:
            self._tf_writer.add_scalar("sac/soft_qval", q_val.mean().item(), global_step=self.training_steps)
            self._tf_writer.add_scalar("sac/critic_loss", critic_loss, global_step=self.training_steps)

        # ========= actor loss =========
        if self.training_steps % self.actor_update_freq == 0:  # delayed policy update
            logits = self.actor.forward(batch.states, batch.action_masks)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            alpha = self.alpha()
            actor_loss = (probs * (alpha.detach() * log_probs - q_val.detach())).sum(dim=-1).mean()

            policy_entropy = (-probs * log_probs).sum(dim=-1).detach()
            alpha_loss = (self.log_alpha * (policy_entropy - self.target_entropy)).mean()

            self.opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
            self.opt_actor.step()

            self.opt_log_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_log_alpha.step()

            max_probs = probs.max(dim=-1).values.mean().item()
            if self._tf_writer is not None:
                self._tf_writer.add_scalar("sac/actor_loss", actor_loss.item(), global_step=self.training_steps)
                self._tf_writer.add_scalar("sac/max_probs", max_probs, global_step=self.training_steps)
                self._tf_writer.add_scalar("sac/policy_entropy", policy_entropy.mean().item(), global_step=self.training_steps)
                self._tf_writer.add_scalar("sac/target_entropy", self.target_entropy, global_step=self.training_steps)
                self._tf_writer.add_scalar("sac/alpha", alpha.item(), global_step=self.training_steps)

    def _update_target_network(self, target: nn.Module, source: nn.Module):
        """update of target network parameters"""
        if self.update_method == "soft":
            soft_update(target=target, source=source, tau=self.soft_update_tau)
        elif (self.training_steps + 1) % self.hard_update_interval == 0:
            hard_update(target=target, source=source)

    def get_model_dict(self):
        state_dict = {
            "agent_id": 0,  # For backward compatibility
            "training_steps": self.training_steps,
            "Qr": self.Qr.state_dict(),
            "Qr_target": self.Qr_target.state_dict(),
            "actor": self.actor.state_dict(),
            "log_alpha": self.log_alpha.item(),
        }
        return state_dict

    def load_model_dict(self, checkpoint):
        self.training_steps = checkpoint.get("training_steps", 0)
        self.Qr.load_state_dict(checkpoint["Qr"])
        self.Qr_target.load_state_dict(checkpoint["Qr_target"])
        self.actor.load_state_dict(checkpoint["actor"])
        if "log_alpha" in checkpoint:
            self.log_alpha = torch.tensor(checkpoint["log_alpha"], requires_grad=True, device=self.device)
            self.opt_log_alpha = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

    def load_from_other(self, other: "SACAgent"):
        self.Qr.load_state_dict(other.Qr.state_dict())
        self.Qr_target.load_state_dict(other.Qr_target.state_dict())
        self.actor.load_state_dict(other.actor.state_dict())
        self.log_alpha = other.log_alpha.clone().detach().requires_grad_(True)
        self.opt_log_alpha = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)
        self.training_steps = other.training_steps


class MaSAC(BaseSolver):
    """
    Multi-agent soft actor-critic solver using a global model (CTDE).
    Supports switching to online mode where each agent has its own model.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: "NamedDict", tf_writer: Optional[SummaryWriter] = None):
        super().__init__(tf_writer=tf_writer)
        self.config = config
        self.state_dim = obs_dim
        self.action_dim = action_dim

        # Global agent for offline training
        self.global_agent = SACAgent(obs_dim, action_dim, config, tf_writer)
        self.local_agents: Dict[int, SACAgent] = {}

    @property
    def name(self):
        return "MaSAC"

    def _get_agent(self, node_id: int) -> SACAgent:
        """Get the appropriate agent for the given node."""
        if not self.online_mode:
            return self.global_agent

        if node_id not in self.local_agents:
            # Initialize new agent from global agent
            agent = SACAgent(self.state_dim, self.action_dim, self.config, self._tf_writer)
            agent.load_from_other(self.global_agent)
            self.local_agents[node_id] = agent

        return self.local_agents[node_id]

    def route(self, obs: np.ndarray, info: dict) -> tuple[Optional[int], Optional[dict]]:
        """Select action using actor policy."""
        node_id = int(info["node_id"])
        agent = self._get_agent(node_id)

        action_mask = info.get("action_mask")
        chosen_action = agent.act(obs, action_mask, eval_mode=self.is_eval())

        return chosen_action, None

    def on_action_over(self, packet):
        """Store experience in replay buffer."""
        if self.is_eval():
            return

        node_id = packet.last_action.node_id
        agent = self._get_agent(node_id)
        agent.store_experience(packet)

    def learn(self):
        """Train the networks."""
        if self.online_mode:
            for agent in self.local_agents.values():
                agent.learn()
        else:
            self.global_agent.learn()

    def on_train_signal(self):
        """Train the networks if we have enough experiences in the replay buffers."""
        if self.is_train():
            self.learn()

    def save_models(self, model_dir_path: str):
        """Save model to file."""
        model_path = f"{model_dir_path}/{self.name}_agent_0.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.global_agent.get_model_dict(), model_path)
        print(f"Saved {self.name} model: {model_dir_path}")

    def load_models(self, model_dir_path: str):
        """Load model from file."""
        model_path = f"{model_dir_path}/{self.name}.pth"
        if not Path(model_path).exists():
            raise RuntimeError(f"Model file not found at {model_path}")

        checkpoint = torch.load(model_path, weights_only=False, map_location=self.global_agent.device)
        self.global_agent.load_model_dict(checkpoint)
        print(f"Loaded {self.name} model: {model_dir_path}")

        # If in online mode, update all initialized agents
        if self.online_mode:
            for agent in self.local_agents.values():
                agent.load_model_dict(checkpoint)
