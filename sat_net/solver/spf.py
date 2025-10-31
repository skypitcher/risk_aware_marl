from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sat_net.datablock import DataBlock
    from sat_net.network import SatelliteNetwork
    from sat_net.node import Node

from sat_net.solver.base_solver import BaseSolver


class SPF(BaseSolver):
    """
    Shortest Path First (SPF) Solver.

    This is a simple solver that is used to compare the performance of the RL-based solvers.
    """

    @property
    def name(self):
        return "SPF"

    def route(self, obs: np.ndarray, info: dict):
        packet: "DataBlock" = info["packet"]
        network: "SatelliteNetwork" = info["network"]
        node: "Node" = info["node"]
        action_list: list[int] = info["action_list"]
        _action_mask: np.ndarray = info["action_mask"]

        current_node_id = packet.current_location
        target_id = packet.target_id

        if current_node_id == target_id:
            # The DataBlock is already at its destination
            return None

        path_weight, path = network.get_shortest_path(current=current_node_id, sink=target_id)

        # If a path exists and has more than one node (i.e., not just the source)
        if path and len(path) > 1:
            # The next hop is the second node in the path
            next_hop = path[1]
            for i, neighbor_id in enumerate(action_list):
                if neighbor_id == next_hop:
                    return i, None

            raise ValueError(f"Invalid next hop: {next_hop} for node {node.id} {node.name}, action_list: {action_list}")
        else:
            # No path found or path is just the source node
            return None, None
    
    def is_train(self):
        """Check if the solver is in training mode."""
        return False
