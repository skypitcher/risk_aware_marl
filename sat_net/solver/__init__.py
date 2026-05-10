"""
Batched routing policy implementations.
"""

from sat_net.solver.base_solver import (
    ACTION_COUNT,
    ACTION_E,
    ACTION_N,
    ACTION_S,
    ACTION_W,
    BaseSolver,
    RoutingBatch,
    RoutingDecision,
)
from sat_net.solver.dqn import MaDQN
from sat_net.solver.primal_avg import PrimalAvg
from sat_net.solver.primal_cvar import PrimalCVaR
from sat_net.solver.spf import SPF, spf_next_hops
from sat_net.util import NamedDict


def create_solver(
    solver_config: NamedDict,
    obs_dim: int = 94,
    action_dim: int = ACTION_COUNT,
    tf_writer=None,
    **_kwargs,
) -> BaseSolver:
    if solver_config.name == "SPF":
        return SPF()
    if solver_config.name == "MaDQN":
        return MaDQN(config=solver_config, obs_dim=obs_dim, action_dim=action_dim, tf_writer=tf_writer)
    if solver_config.name == "PrimalAvg":
        return PrimalAvg(config=solver_config, obs_dim=obs_dim, action_dim=action_dim, tf_writer=tf_writer)
    if solver_config.name == "PrimalCVaR":
        return PrimalCVaR(config=solver_config, obs_dim=obs_dim, action_dim=action_dim, tf_writer=tf_writer)
    raise RuntimeError(
        f"Unknown or retired solver type: {solver_config.name}. "
        "The slot-array kernel now expects a batched RoutingBatch policy."
    )


__all__ = [
    "ACTION_COUNT",
    "ACTION_E",
    "ACTION_N",
    "ACTION_S",
    "ACTION_W",
    "BaseSolver",
    "MaDQN",
    "PrimalAvg",
    "PrimalCVaR",
    "RoutingBatch",
    "RoutingDecision",
    "SPF",
    "create_solver",
    "spf_next_hops",
]
