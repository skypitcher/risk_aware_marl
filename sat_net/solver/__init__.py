"""
Routing agents implementations.
"""

from sat_net.solver.base_solver import BaseSolver
from sat_net.solver.dqn import MaDQN
from sat_net.solver.iqn import MaIQN
from sat_net.solver.primal_avg import PrimalAvg
from sat_net.solver.primal_cvar import PrimalCVaR
from sat_net.solver.sac import MaSAC
from sat_net.solver.spf import SPF


def create_solver(obs_dim, action_dim, solver_config, tf_writer):
    if solver_config.name == "PrimalCVaR":
        solver = PrimalCVaR(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=solver_config,
            tf_writer=tf_writer,
        )
    elif solver_config.name == "PrimalAvg":
        solver = PrimalAvg(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=solver_config,
            tf_writer=tf_writer,
        )
    elif solver_config.name == "MaSAC":
        solver = MaSAC(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=solver_config,
            tf_writer=tf_writer,
        )
    elif solver_config.name == "MaIQN":
        solver = MaIQN(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=solver_config,
            tf_writer=tf_writer,
        )
    elif solver_config.name == "MaDQN":
        solver = MaDQN(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=solver_config,
            tf_writer=tf_writer,
        )
    elif solver_config.name == "SPF":
        solver = SPF()
    else:
        raise RuntimeError(f"Unknown solver type: {solver_config.name}")

    return solver


__all__ = [
    "BaseSolver",
    "SPF",
    "MaDQN",
    "MaIQN",
    "MaSAC",
    "PrimalCVaR",
    "PrimalAvg",
    "BackpressureSolver",
    "create_solver",
]
