"""
Multi-agent routing policy implementations.
"""

from sat_net.agent.base_agent import (
    ACTION_COUNT,
    ACTION_E,
    ACTION_N,
    ACTION_S,
    ACTION_W,
    BaseAgent,
    RoutingBatch,
    RoutingDecision,
)
from sat_net.agent.dqn import MaDQN
from sat_net.agent.primal_avg import PrimalAvg
from sat_net.agent.primal_cvar import PrimalCVaR
from sat_net.agent.spf import SPFAgent, spf_next_hops
from sat_net.util import NamedDict


def create_agent(
    agent_config: NamedDict,
    obs_dim: int = 94,
    action_dim: int = ACTION_COUNT,
    tf_writer=None,
    **_kwargs,
) -> BaseAgent:
    if agent_config.name == "SPF":
        return SPFAgent()
    if agent_config.name == "MaDQN":
        return MaDQN(config=agent_config, obs_dim=obs_dim, action_dim=action_dim, tf_writer=tf_writer)
    if agent_config.name == "PrimalAvg":
        return PrimalAvg(config=agent_config, obs_dim=obs_dim, action_dim=action_dim, tf_writer=tf_writer)
    if agent_config.name == "PrimalCVaR":
        return PrimalCVaR(config=agent_config, obs_dim=obs_dim, action_dim=action_dim, tf_writer=tf_writer)
    raise RuntimeError(
        f"Unknown or retired agent type: {agent_config.name}. "
        "The slot-array kernel expects a batched MARL routing agent."
    )


__all__ = [
    "ACTION_COUNT",
    "ACTION_E",
    "ACTION_N",
    "ACTION_S",
    "ACTION_W",
    "BaseAgent",
    "MaDQN",
    "PrimalAvg",
    "PrimalCVaR",
    "RoutingBatch",
    "RoutingDecision",
    "SPFAgent",
    "create_agent",
    "spf_next_hops",
]
