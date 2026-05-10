import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from sat_net.datablock import DataBlock
from sat_net.event import Event, EventScheduler, EventType
from sat_net.geometric import LIGHT_SPEED_MS, great_circle_distance
from sat_net.link import Link
from sat_net.network import SatelliteNetwork
from sat_net.node import Node
from sat_net.solver.base_solver import BaseSolver
from sat_net.stats import Metrics, Stats
from sat_net.traffic_region import TrafficRegion, TrafficRegionModel
from sat_net.util import NamedDict, NetworkError, ms2str


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RoutingEnvAsync:
    """
    Asynchronous Environment for Network Routing in Dynamic Networks.
    """

    def __init__(self, config: NamedDict, tf_writer: Optional[SummaryWriter] = None):
        """
        Initialize the environment.

        Args:
            config: The configuration used for this environment.
        """
        self.config = config
        self.network_config: NamedDict = self.config.network
        self.tf_writer = tf_writer

        self.np_random = None
        self.py_random: Optional[random.Random] = None
        self._set_seed(self.config.get("seed", default=None))

        self.network = self._create_network()
        self.ground_stations = list(self.network.ground_stations.values())
        self.traffic_config: NamedDict = self.config.traffic
        self.traffic_model = TrafficRegionModel.from_config(self.traffic_config, PROJECT_ROOT)
        self.packet_rate_per_ms: float = self.traffic_config.get("packet_rate_per_ms", 1.0)
        self.flowlet_interval_ms: float = self.traffic_config.get("flowlet_interval_ms", 10.0)
        self.mean_packets_per_flowlet: float = self.traffic_config.get("mean_packets_per_flowlet", 16.0)
        self.access_data_rate: float = self.traffic_config.get(
            "access_data_rate", self.network_config.gsl_data_rate
        )
        self.prob_normal_packet: float = self.config.prob_normal_packet
        self.normal_packet_size = self.config.normal_packet_size
        self.small_packet_size = self.config.small_packet_size

        self.default_ttl: int = self.config.default_ttl
        self.small_packet_delay_limit: float = self.config.small_packet_delay_limit
        self.normal_packet_delay_limit: float = self.config.normal_packet_delay_limit

        self.delay_norm = 100.0

        # Initialize time and scheduler first since they're needed for dimension calculation
        self.start_time = 0.0  # the start timestamp of the simulation. randomize this to init the topology
        self.current_time = 0.0  #  # the current time offset, in milliseconds
        self.topology_update_steps = 0
        self.time_limit = float(self.config.time_limit_seconds * 1000.0)
        self.scheduler = EventScheduler()

        self.action_dim = 4  # N, E, S, W - fixed for satellite routing
        self.obs_dim = 94

        self.current_solver: Optional["BaseSolver"] = None

        self.verbose = self.config.verbose
        self.train_interval_ms = self.config.train_interval_ms
        self.update_interval_ms = self.config.update_interval_ms

        self.stats = Stats()
        self.next_packet_id = 0
        self.generated_packets: list[DataBlock] = []
        self.delivered_packets: list[DataBlock] = []
        self.dropped_packets: list[DataBlock] = []

    def _create_network(self):
        """Create the network based on the configuration."""
        return SatelliteNetwork(
            ground_stations=self.network_config.ground_stations,
            altitude=self.network_config.altitude,
            inclination=self.network_config.inclination,
            num_orbits=self.network_config.num_orbits,
            num_sats_per_orbit=self.network_config.num_sats_per_orbit,
            phasing=self.network_config.phasing,
            min_elevation_angle_deg=self.network_config.min_elevation_angle_deg,
            max_gsl_per_gs=self.network_config.max_gsl_per_gs,
            max_gsl_per_sat=self.network_config.max_gsl_per_sat,
            node_buffer_size=self.network_config.node_buffer_size,
            link_buffer_size=self.network_config.link_buffer_size,
            gsl_data_rate=self.network_config.gsl_data_rate,
            isl_data_rate=self.network_config.isl_data_rate,
        )

    def _set_seed(self, seed):
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
            self.py_random = random.Random(seed)
        else:
            self.np_random = np.random.default_rng()
            self.py_random = random.Random()

    def reset(self, seed=None, start_time=None):
        """
        Reset the environment to the initial state.

        Returns:
            Initial observations for all agents
        """
        self._set_seed(seed)

        self.network = self._create_network()

        self.ground_stations = list(self.network.ground_stations.values())

        self.topology_update_steps = 0
        self.scheduler.reset()

        self.next_packet_id = 0
        self.stats.reset()
        self.generated_packets.clear()
        self.delivered_packets.clear()
        self.dropped_packets.clear()

        if start_time is None:
            self.start_time = 0
        else:
            self.start_time = start_time
        self.current_time = self.start_time
        self.network.update_topology(self.start_time, None)

    def run(
        self,
        solver: "BaseSolver",
        debug_callback: Optional[Callable] = None,
        callback_interval_ms: int = 0,
    ):
        """
        Run the simulation until the termination condition is met.

        Event flow in the simulation:
        1. DATA_GENERATED: A new DataBlock is generated at a source node.
        2. TRANSMIT_END: A DataBlock has been fully serialized onto the link.
        3. DATA_FORWARDED: A DataBlock arrived at the receiver.
        4. TOPOLOGY_CHANGE: The network topology changes due to satellite movement.
        5. TIME_LIMIT_REACHED: The simulation ends due to time limit.
        6. TRAIN_EVENT: A training event is triggered.

        Args:
            solver: The solver to use for the making routing decisions

        Returns:
            Statistics from the simulation
        """
        self.current_solver = solver

        self.scheduler.push_event(
            event_type=EventType.TIME_LIMIT_REACHED,
            time=self.start_time + self.time_limit,
        )
        self.scheduler.push_event(
            event_type=EventType.TOPOLOGY_CHANGE,
            time=self.current_time + self.update_interval_ms,
        )
        
        # train event is only pushed if the solver is in training mode
        if self.current_solver is not None and self.current_solver.is_train():
            self.scheduler.push_event(
                event_type=EventType.TRAIN_EVENT,
                time=self.current_time + self.train_interval_ms,
            )
        else:
            print("Env is in evaluation mode")
        
        if debug_callback is not None:
            self.scheduler.push_event(
                event_type=EventType.DEBUG_CALLBACK_EVENT,
                time=self.current_time + callback_interval_ms,
            )
        self._schedule_region_flowlet_traffic(interval_ms=self.time_limit)

        event_count = 0
        while not self.scheduler.is_empty():
            event = self.scheduler.pop_event()
            if event.is_cancelled:
                continue

            delta_time = event.time - self.current_time
            self.current_time = event.time
            self.stats.time.update(delta_time)

            if event.event_type == EventType.TIME_LIMIT_REACHED:
                break
            elif event.event_type == EventType.TOPOLOGY_CHANGE:
                self._handle_topology_change(event)
            elif event.event_type == EventType.DATA_GENERATED:
                self._handle_data_generated(event)
            elif event.event_type == EventType.TRANSMIT_END:
                self._handle_transmit_end(event)
            elif event.event_type == EventType.DATA_FORWARDED:
                self._handle_data_fowarded(event)
            elif event.event_type == EventType.TRAIN_EVENT:
                if self.current_solver is not None and self.current_solver.is_train():
                    self.current_solver.on_train_signal()
                    self.scheduler.push_event(
                        event_type=EventType.TRAIN_EVENT,
                        time=self.current_time + self.train_interval_ms,
                    )
            elif event.event_type == EventType.DEBUG_CALLBACK_EVENT:
                if debug_callback is not None:
                    debug_callback(self)
                    self.scheduler.push_event(
                        event_type=EventType.DEBUG_CALLBACK_EVENT,
                        time=self.current_time + callback_interval_ms,
                    )
            event_count += 1

        if self.verbose:
            self._print_current_metrics()
            print("")

    def _handle_topology_change(self, event: "Event"):
        """
        Handle a topology change event.
        """
        self.network.update_topology(
            timestamp=event.time, on_link_disconnected=self._on_link_disconnected
        )

        self.scheduler.push_event(
            event_type=EventType.TOPOLOGY_CHANGE,
            time=self.current_time + self.update_interval_ms,
            data=event.data,
        )
        self.topology_update_steps += 1

        if self.verbose:
            self._print_current_metrics()

    def _on_link_disconnected(self, link: "Link"):
        """
        Callback when a link is disconnected due to topology change.
        """
        for packet in link.drop_all():
            self._drop_data_block(
                packet, error=NetworkError.LINK_DISCONNECTED, current_node=link.source
            )

    def _handle_data_generated(self, event: "Event"):
        """
        Handle a DataBlock arrival event.

        This occurs when a new DataBlock enters the network at its source node.
        """
        packet: DataBlock = event.data
        packet.last_event = None

        if packet.source_id is None:
            if not self._attach_flowlet_to_access_satellite(packet):
                self._drop_data_block(packet, NetworkError.NO_AVAIABLE_SAT, current_node=None)
                return

        receiver = self.network.nodes[packet.source_id]

        packet.current_location = packet.source_id
        packet.path.append(receiver.id)

        success, reason_if_failed = receiver.receive(packet)
        if not success:
            self._drop_data_block(packet, reason_if_failed, current_node=receiver)
            return

        self._process_data_block(receiver, packet)

    def _handle_transmit_end(self, event: "Event"):
        """
        Handle a DataBlock transmission end event.
        """
        packet, sender, next_hop, (wait_time, transmit_time) = event.data

        # update packet metadata
        packet.last_event = None
        packet.e2e_delay += wait_time + transmit_time
        packet.queue_delay += wait_time
        packet.transmission_delay += transmit_time

        # record the cost of load-imbalancing
        if packet.last_action is not None:
            if "queue_delay" not in packet.last_action:
                packet.last_action.queue_delay = wait_time
            else:
                packet.last_action.queue_delay += wait_time

        link_used = sender.outgoing_links[next_hop]
        link_used.start_propagate(packet_id=packet.id)

        propagation_delay = link_used.propagation_delay
        packet.last_event = self.scheduler.push_event(
            event_type=EventType.DATA_FORWARDED,
            time=self.current_time + propagation_delay,
            data=(packet, link_used, propagation_delay),
        )

    def _handle_data_fowarded(self, event: "Event"):
        """
        Handle a DataBlock reception event.

        This event occurs when a DataBlock arrives at a node after being transmitted over a link.
        """
        packet, link_used, propagation_delay = event.data

        # update packet metadata
        packet.last_event = None
        packet.hops += 1
        packet.ttl -= 1
        packet.e2e_delay += propagation_delay
        packet.propagation_delay += propagation_delay
        packet.current_location = link_used.sink.id
        packet.path.append(link_used.sink.id)

        link_used.finish_propagate(packet_id=packet.id)

        receiver = link_used.sink
        success, reason_if_failed = receiver.receive(packet)
        if not success:
            self._drop_data_block(packet, error=reason_if_failed, current_node=receiver)
            return

        if packet.ttl <= 0:
            self._drop_data_block(
                packet, error=NetworkError.TTL_EXPIRED, current_node=receiver
            )
            return

        self._process_data_block(receiver, packet)

    def _process_data_block(self, current_node: "Node", packet: "DataBlock"):
        """
        Process a DataBlock in a node's receive buffer using the agent's routing decisions.
        """
        assert not packet.dropped
        if not packet.delivered:
            assert packet.id in current_node.recv_buffer

        assert packet.current_location == current_node.id

        assert (
            current_node.id in self.network.satellites
        ), f"Node {current_node.id} is not a satellite"

        current_obs = self._get_observation(current_node, packet)
        action_mask = self._get_action_mask(current_node, packet)
        target_region = self._target_region(packet)

        if self._can_satellite_deliver_to_target_region(current_node.id, packet):
            packet.final_gsl_delay = self._region_access_delay(
                current_node, target_region, packet
            )
            self._finalize_action(
                packet=packet,
                done=True,
                truncated=False,
                next_obs=current_obs,
                next_action_mask=action_mask,
                reached_goal=1,
            )
            self._deliver_data_block(current_node, packet)
            return

        # Finalize the pending transition, if any
        self._finalize_action(
            packet=packet,
            done=False,
            truncated=False,
            next_obs=current_obs,
            next_action_mask=action_mask,
            reached_goal=0,
        )

        # If no action is available, reroute the packet
        if action_mask.sum() == 0:
            self._drop_data_block(
                packet,
                error=NetworkError.FAILED_TO_FIND_NEXT_HOP,
                current_node=current_node,
            )
            return

        action_list = self._get_action_list(current_node)

        info = {
            "packet": packet,
            "node": current_node,
            "network": self.network,
            "action_mask": action_mask,
            "action_list": action_list,
            "target_region": target_region,
        }

        # Forward to the next hop, determined by the solver
        chosen_action, solver_data = self.current_solver.route(
            obs=current_obs, info=info
        )

        if chosen_action is None:
            self._drop_data_block(
                packet,
                error=NetworkError.FAILED_TO_FIND_NEXT_HOP,
                current_node=current_node,
            )
            return

        assert packet.last_action is None

        packet.last_action_time = self.current_time
        packet.last_action = NamedDict(
            {
                "node_id": current_node.id,
                "state": current_obs,
                "action": chosen_action,
                "action_mask": action_mask,
                "congestion_cost": 0.0,
                "queue_delay": 0.0,
            }
        )

        if solver_data is not None:
            packet.last_action.update(solver_data)

        next_hop = action_list[chosen_action]
        self._forward_data_block(current_node, packet, next_hop)

    def _forward_data_block(
        self,
        sender: "Node",
        packet: "DataBlock",
        next_hop: int,
        drop_on_failure: bool = True,
    ):
        """
        Forward a DataBlock to the next hop.
        """
        success, time_info, reason_if_failed = sender.send(
            packet, next_hop, self.current_time
        )
        if success:
            wait_time, transmit_time = time_info
            packet.last_event = self.scheduler.push_event(
                event_type=EventType.TRANSMIT_END,
                time=self.current_time + wait_time + transmit_time,
                data=(packet, sender, next_hop, (wait_time, transmit_time)),
            )
            if packet.last_action is not None:
                packet.last_action.queue_delay += wait_time
                packet.total_queue_cost += wait_time
        else:
            if drop_on_failure:
                self._drop_data_block(
                    packet, error=reason_if_failed, current_node=sender
                )

        return success

    def _drop_data_block(
        self,
        packet: "DataBlock",
        error: Optional[NetworkError],
        current_node: Optional["Node"] = None,
        current_obs: Optional[np.ndarray] = None,
    ):
        """
        Process pending actions for a dropped DataBlock.
        Assigns penalties to agents that handled this DataBlock.

        Args:
            packet: DataBlock to drop.
            error: Reason for dropping.
        """
        packet.cancel_event()
        packet.dropped = True
        packet.drop_time = self.current_time
        packet.drop_reason = error

        if current_node is not None and packet.id in current_node.recv_buffer:
            current_node.recv_buffer.remove(packet.id)

        self.dropped_packets.append(packet)
        self.stats.on_packet_finished(packet)

        if packet.last_action is not None:
            if current_obs is None:
                if current_node is None:
                    if packet.current_location is None:
                        return
                    current_node = self.network.satellites[packet.current_location]
                current_obs = self._get_observation(current_node, packet)

            action_mask = self._get_action_mask(current_node, packet)
            self._finalize_action(
                packet=packet,
                done=True,
                truncated=False,
                next_obs=current_obs,
                next_action_mask=action_mask,
                reached_goal=-1,
            )

    def _finalize_action(
        self,
        packet: "DataBlock",
        done: bool,
        truncated: bool,
        next_obs: np.ndarray,
        next_action_mask: np.ndarray,
        reached_goal: float,
    ):
        """
        Finalize the pending transition for a DataBlock.

        Args:
            packet: DataBlock to finalize the transition for.
            done: Whether the transition is done.
            truncated: Whether the transition is truncated.
            next_obs: The next observation after the transition.
            next_action_mask: The next action mask after the transition.
            reached_goal: Whether the target satelite is reached or not.
        """
        if packet.last_action is None:
            return

        current_node = self.network.nodes[packet.current_location]
        target_region = self._target_region(packet)

        current_gcd = self._great_circle_distance_to_region(current_node, target_region)
        current_progress = current_gcd / packet.initial_gcd
        progress_gain = max(0, packet.shortest_gcd - current_gcd)
        # 0=no (normalized) progress gain, (0,1]=progress gain achieved by approaching the goal
        progress_gain = progress_gain / packet.initial_gcd
        if current_gcd < packet.shortest_gcd:
            packet.shortest_gcd = current_gcd

        # basic timing penalty
        action_delay = (
            self.current_time - packet.last_action_time + packet.final_gsl_delay
        )

        # consider adding a terminal bonus/penalty, depending on the event type
        baseline_reward = progress_gain + reached_goal * (1 + packet.size)
        baseline_reward -= action_delay / self.delay_norm
        if reached_goal == -1:  # dropped penalties
            baseline_reward = (
                -current_progress
            )  # all previous efforts made on approaching the target are lost
            baseline_reward -= packet.ttl * 5 / self.delay_norm

        # hand-crafted reward baseline, which also minimizes the delay of all packets
        packet.last_action.baseline_reward = baseline_reward

        # basic transition information
        packet.last_action.next_state = next_obs
        packet.last_action.next_action_mask = next_action_mask
        packet.last_action.done = done
        packet.last_action.truncated = truncated

        # metadata for reward shaping
        packet.last_action.current_progress = current_progress  # [0, 1]
        packet.last_action.reached_goal = (
            reached_goal  # 1=reached goal, 0=pending, -1=dropped
        )
        packet.last_action.progress_gain = progress_gain  # [0, 1]
        packet.last_action.action_delay = action_delay  # in ms
        packet.last_action.delay_norm = self.delay_norm  # in ms

        packet.trajectory.append(packet.last_action)

        self.current_solver.on_action_over(packet)
        if done or truncated:
            self.current_solver.on_episode_over(packet)

        packet.last_action = None
        packet.last_action_time = None

    def _get_action_mask(self, node: "Node", packet: "DataBlock"):
        action_map = self._get_action_list(node)

        # action_mask is a list of 0s and 1s, 1s mean the action is enabled
        action_mask = np.zeros(len(action_map), dtype=np.int8)
        for idx in range(4):
            sink_id = action_map[idx]
            link = self.network.get_link(node.id, sink_id)
            if link is None or not link.is_connected:
                continue

            if not self._can_satellite_deliver_to_target_region(sink_id, packet):
                if len(packet.path) >= 2 and packet.path[-2] == sink_id:
                    continue  # avoid direct loop back unless it is the target

            action_mask[idx] = 1  # enable the action

        return action_mask

    def _get_action_list(self, node: "Node"):
        assert node.id in self.network.satellites, f"Node {node.id} is not a satellite"
        return [
            self.network.ISL_N[node.id],
            self.network.ISL_E[node.id],
            self.network.ISL_S[node.id],
            self.network.ISL_W[node.id],
        ]

    def _get_observation(self, current_node: "Node", packet: "DataBlock") -> np.ndarray:
        assert current_node.id in self.network.satellites

        target_region = self._target_region(packet)

        current_pos = current_node.position / self.network.orbit_radius
        target_pos = target_region.position / self.network.orbit_radius

        relative_pos = current_pos - target_pos
        relative_distance = float(np.linalg.norm(relative_pos))
        current_delay = float(self.current_time - packet.creation_time)

        orbit_cycle = int(self.network.orbit_cycle * 1000)
        time_prog = (self.current_time % orbit_cycle) / orbit_cycle

        current_gcd = self._great_circle_distance_to_region(current_node, target_region)
        current_progress = current_gcd / packet.initial_gcd

        # knowledge on the action history
        last_action1 = -1
        last_node1 = -1
        last_action2 = -1
        last_node2 = -1
        if len(packet.trajectory) >= 1:
            trans = packet.trajectory[-1]
            last_action1 = trans.action
            last_node1 = trans.node_id
        if len(packet.trajectory) >= 2:
            trans = packet.trajectory[-2]
            last_action2 = trans.action
            last_node2 = trans.node_id

        obs = (
            time_prog,
            float(current_pos[0]),
            float(current_pos[1]),
            float(current_pos[2]),
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]),
            float(relative_pos[0]),
            float(relative_pos[1]),
            float(relative_pos[2]),
            # 10
            relative_distance,
            current_progress,
            current_node.get_load_factor(),
            current_node.recv_buffer.get_remaining_capacity(),
            current_delay / self.delay_norm,
            float(packet.is_normal_packet),
            packet.size,
            packet.ttl,
            self.default_ttl - packet.ttl,
            packet.ttl / self.default_ttl,
            # 20
            packet.e2e_delay / self.delay_norm,
            packet.queue_delay / self.delay_norm,
            last_action1,
            last_node1,
            last_action2,
            last_node2,
            # 26
        )

        neighbors = [
            self.network.ISL_N[current_node.id],
            self.network.ISL_E[current_node.id],
            self.network.ISL_S[current_node.id],
            self.network.ISL_W[current_node.id],
        ]

        for next_hop in neighbors:
            link = current_node.outgoing_links[next_hop]
            normalized_propagation_delay = link.propagation_delay / self.delay_norm
            normalized_transmit_time = (packet.size / link.data_rate) / self.delay_norm
            normalized_queue_delay = (
                link.get_busy_time_remaining(self.current_time) / self.delay_norm
            )
            link_remaining_capacity = link.get_remaining_capacity()
            sink_load_factor = link.sink.get_load_factor()
            sink_remaining_capacity = link.sink.recv_buffer.get_remaining_capacity()
            sink_pos = link.sink.position / self.network.orbit_radius
            sink_relative_pos = sink_pos - target_pos
            sink_relative_distance = float(np.linalg.norm(sink_relative_pos))

            sink_gcd = self._great_circle_distance_to_region(link.sink, target_region)
            progress_sink = sink_gcd / packet.initial_gcd

            has_enough_capacity = 0 if link_remaining_capacity < packet.size else 1
            is_target_access_sat = (
                1
                if self._can_satellite_deliver_to_target_region(link.sink.id, packet)
                else 0
            )
            possibly_looped = (
                1 if next_hop == last_node1 or next_hop == last_node2 else 0
            )

            obs += (
                float(sink_pos[0]),
                float(sink_pos[1]),
                float(sink_pos[2]),
                float(sink_relative_pos[0]),
                float(sink_relative_pos[1]),
                float(sink_relative_pos[2]),
                sink_relative_distance,
                progress_sink,
                normalized_queue_delay,
                normalized_transmit_time,
                # 10
                normalized_propagation_delay,
                sink_load_factor,
                sink_remaining_capacity,
                link_remaining_capacity,
                has_enough_capacity,
                possibly_looped,
                is_target_access_sat,
                # 17
            )  # 17*4=68

        assert len(obs) == self.obs_dim

        return np.array(obs)

    def calc_metrics(self) -> Metrics:
        return self.stats.calc_metrics()

    def _print_current_metrics(self):
        metrics = self.stats.calc_metrics()
        self._print(metrics.get_summary() + " " * 4, end="\r")

    def _print(self, line: str, end=None):
        print(
            f"{ms2str(self.start_time)}+{ms2str(self.current_time-self.start_time)}: {line}",
            end=end,
        )

    def _target_region(self, packet: "DataBlock") -> TrafficRegion:
        return self.traffic_model.get(packet.target_region_id)

    def _source_region(self, packet: "DataBlock") -> TrafficRegion:
        return self.traffic_model.get(packet.source_region_id)

    def _great_circle_distance_to_region(self, node: "Node", region: TrafficRegion) -> float:
        lon1, lat1 = node.get_projected_position()
        return great_circle_distance(lon1, lat1, region.longitude, region.latitude)

    def _region_to_region_distance(self, source: TrafficRegion, target: TrafficRegion) -> float:
        return great_circle_distance(source.longitude, source.latitude, target.longitude, target.latitude)

    def _can_satellite_deliver_to_target_region(self, sat_id: int, packet: "DataBlock") -> bool:
        return self.network.can_satellite_serve_position(sat_id, self._target_region(packet).position)

    def _region_access_delay(self, sat: "Node", region: TrafficRegion, packet: "DataBlock") -> float:
        propagation_delay, transmit_delay = self._region_access_delay_components(sat, region, packet)
        return propagation_delay + transmit_delay

    def _region_access_delay_components(
        self, sat: "Node", region: TrafficRegion, packet: "DataBlock"
    ) -> tuple[float, float]:
        distance = float(np.linalg.norm(sat.position - region.position))
        propagation_delay = distance / LIGHT_SPEED_MS
        transmit_delay = packet.size / self.access_data_rate if self.access_data_rate > 0 else 0.0
        return propagation_delay, transmit_delay

    def _attach_flowlet_to_access_satellite(self, packet: "DataBlock") -> bool:
        source_region = self._source_region(packet)
        source_sat, _source_distance = self.network.get_nearest_satellite_for_position(source_region.position)
        if source_sat is None:
            return False

        target_region = self._target_region(packet)

        packet.source_id = source_sat.id
        source_prop_delay = (_source_distance / LIGHT_SPEED_MS) if _source_distance else 0.0
        source_tx_delay = packet.size / self.access_data_rate if self.access_data_rate > 0 else 0.0
        packet.first_gsl_delay = source_prop_delay + source_tx_delay
        packet.e2e_delay += packet.first_gsl_delay
        packet.propagation_delay += source_prop_delay
        packet.transmission_delay += source_tx_delay
        packet.initial_gcd = self._region_to_region_distance(source_region, target_region)
        if packet.initial_gcd <= 0:
            packet.initial_gcd = 1e-6
        packet.shortest_gcd = packet.initial_gcd
        return True

    def _deliver_data_block(self, receiver: "Node", packet: "DataBlock"):
        assert receiver.recv_buffer.remove(packet.id)
        final_prop_delay, final_tx_delay = self._region_access_delay_components(
            receiver, self._target_region(packet), packet
        )
        packet.final_gsl_delay = final_prop_delay + final_tx_delay
        packet.delivered = True
        packet.delivery_time = self.current_time + packet.final_gsl_delay
        packet.e2e_delay += packet.final_gsl_delay
        packet.propagation_delay += final_prop_delay
        packet.transmission_delay += final_tx_delay

        self.delivered_packets.append(packet)
        self.stats.on_packet_finished(packet)

    def _schedule_region_flowlet_traffic(self, interval_ms: float):
        """
        Schedule population-driven flowlets over the full simulation window.

        Each DataBlock represents a batch of packets with the same source region,
        destination region, traffic class, and generation time bin.
        """
        num_flowlets_generated = 0
        num_bins = int(np.ceil(interval_ms / self.flowlet_interval_ms))
        expected_flowlets_per_bin = (
            self.packet_rate_per_ms
            * self.flowlet_interval_ms
            / max(self.mean_packets_per_flowlet, 1.0)
        )

        for bin_idx in range(num_bins):
            bin_start = self.current_time + bin_idx * self.flowlet_interval_ms
            bin_end = min(
                self.current_time + interval_ms,
                bin_start + self.flowlet_interval_ms,
            )
            if bin_end <= bin_start:
                continue

            num_flowlets = int(self.np_random.poisson(lam=expected_flowlets_per_bin))
            if num_flowlets <= 0:
                continue

            source_region_ids = self.traffic_model.sample_source_ids(self.np_random, num_flowlets)
            target_region_ids = self.traffic_model.sample_target_ids(self.np_random, source_region_ids)
            is_normal_array = self.np_random.uniform(size=num_flowlets) < self.prob_normal_packet
            creation_times = self.np_random.uniform(low=bin_start, high=bin_end, size=num_flowlets)
            packet_counts = np.maximum(
                1,
                self.np_random.poisson(lam=self.mean_packets_per_flowlet, size=num_flowlets),
            )

            for i in range(num_flowlets):
                is_normal = bool(is_normal_array[i])
                packet_size = self.normal_packet_size if is_normal else self.small_packet_size
                delay_tolerance = self.normal_packet_delay_limit if is_normal else self.small_packet_delay_limit
                packet_count = int(packet_counts[i])

                packet = DataBlock(
                    block_id=self.next_packet_id,
                    source=None,
                    source_region_id=int(source_region_ids[i]),
                    target_region_id=int(target_region_ids[i]),
                    packet_count=packet_count,
                    packet_size=packet_size,
                    is_normal=is_normal,
                    size=packet_size * packet_count,
                    delay_limit=delay_tolerance,
                    creation_time=float(creation_times[i]),
                    ttl=self.default_ttl,
                )

                packet.last_event = self.scheduler.push_event(
                    event_type=EventType.DATA_GENERATED,
                    time=packet.creation_time,
                    data=packet,
                )
                self.generated_packets.append(packet)
                self.next_packet_id += 1
                self.stats.on_packet_generated(packet)
                num_flowlets_generated += 1

        return num_flowlets_generated

    def save_packets_to_csv(self, file_path: str):
        generated_data = []
        for block in self.generated_packets:
            generated_data.append(block.to_dict())
        generated_df = pd.DataFrame(generated_data)
        generated_df.to_csv(file_path, index=False)
