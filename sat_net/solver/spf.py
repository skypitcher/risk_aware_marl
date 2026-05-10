from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sat_net.datablock import DataBlock
    from sat_net.network import SatelliteNetwork
    from sat_net.node import Node
    from sat_net.traffic_region import TrafficRegion

from sat_net.solver.base_solver import BaseSolver


class SPF(BaseSolver):
    """
    Shortest Path First (SPF) Solver.

    This is a simple solver that is used to compare the performance of the RL-based solvers.
    """

    def __init__(self, tf_writer=None):
        super().__init__(tf_writer=tf_writer)

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
        target_access_sat_id = self._resolve_target_access_satellite(
            network=network,
            info=info,
        )
        if target_access_sat_id is None:
            return None, None

        if current_node_id == target_access_sat_id:
            # The DataBlock is already at its destination
            return None, None

        next_hop = network.get_shortest_next_hop(
            current=current_node_id,
            sink=target_access_sat_id,
        )
        if next_hop is None:
            return None, None

        for i, neighbor_id in enumerate(action_list):
            if neighbor_id == next_hop:
                return i, None

        raise ValueError(
            f"Invalid next hop: {next_hop} for node {node.id} {node.name}, "
            f"action_list: {action_list}"
        )

    def _resolve_target_access_satellite(
        self,
        network: "SatelliteNetwork",
        info: dict,
    ) -> int | None:
        if "target_access_sat_id" in info:
            return info["target_access_sat_id"]

        target_region: "TrafficRegion" = info["target_region"]
        target_sat, _distance = network.get_nearest_satellite_for_position(
            target_region.position
        )
        return target_sat.id if target_sat is not None else None

    def is_train(self):
        """Check if the solver is in training mode."""
        return False
