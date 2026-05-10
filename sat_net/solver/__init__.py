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
from sat_net.solver.spf import SPF, jax_spf_next_hops, spf_next_hops
from sat_net.util import NamedDict


def create_solver(solver_config: NamedDict, **_kwargs) -> BaseSolver:
    if solver_config.name == "SPF":
        return SPF(use_jax=solver_config.get("use_jax", None))
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
    "RoutingBatch",
    "RoutingDecision",
    "SPF",
    "create_solver",
    "jax_spf_next_hops",
    "spf_next_hops",
]
