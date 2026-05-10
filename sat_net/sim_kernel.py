from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sat_net.geometric import LIGHT_SPEED_MS
from sat_net.solver.base_solver import RoutingBatch


FLOWLET_NOT_STARTED = 0
FLOWLET_AT_NODE = 1
FLOWLET_ON_LINK = 2
FLOWLET_DELIVERED = 3
FLOWLET_DROPPED = 4


@dataclass(slots=True)
class FlowletState:
    """Array state for all generated flowlets."""

    slot_offsets: np.ndarray
    status: np.ndarray
    creation_slot: np.ndarray
    creation_time: np.ndarray
    source_region_id: np.ndarray
    target_region_id: np.ndarray
    source_id: np.ndarray
    current_sat: np.ndarray
    next_sat: np.ndarray
    link_id: np.ndarray
    packet_count: np.ndarray
    packet_size: np.ndarray
    is_normal: np.ndarray
    size: np.ndarray
    ttl: np.ndarray
    hops: np.ndarray
    queue_delay: np.ndarray
    transmission_delay: np.ndarray
    propagation_delay: np.ndarray
    total_queue_cost: np.ndarray
    first_access_delay: np.ndarray
    final_access_delay: np.ndarray
    delivery_time: np.ndarray
    drop_time: np.ndarray
    drop_reason: np.ndarray
    transmit_end_time: np.ndarray
    arrival_time: np.ndarray
    link_released: np.ndarray
    scheduled_prop_delay: np.ndarray
    shortest_gcd: np.ndarray
    initial_gcd: np.ndarray
    last_node1: np.ndarray
    last_node2: np.ndarray

    @property
    def count(self) -> int:
        return len(self.status)

    @classmethod
    def empty(cls, slot_offsets: np.ndarray | None = None) -> "FlowletState":
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_f64 = np.empty(0, dtype=np.float64)
        return cls(
            slot_offsets=np.zeros(1, dtype=np.int64) if slot_offsets is None else slot_offsets,
            status=np.empty(0, dtype=np.int8),
            creation_slot=empty_i64.copy(),
            creation_time=empty_f64.copy(),
            source_region_id=empty_i64.copy(),
            target_region_id=empty_i64.copy(),
            source_id=empty_i64.copy(),
            current_sat=empty_i64.copy(),
            next_sat=empty_i64.copy(),
            link_id=np.empty(0, dtype=np.int32),
            packet_count=empty_i64.copy(),
            packet_size=empty_f64.copy(),
            is_normal=np.empty(0, dtype=bool),
            size=empty_f64.copy(),
            ttl=np.empty(0, dtype=np.int16),
            hops=np.empty(0, dtype=np.int16),
            queue_delay=empty_f64.copy(),
            transmission_delay=empty_f64.copy(),
            propagation_delay=empty_f64.copy(),
            total_queue_cost=empty_f64.copy(),
            first_access_delay=empty_f64.copy(),
            final_access_delay=empty_f64.copy(),
            delivery_time=empty_f64.copy(),
            drop_time=empty_f64.copy(),
            drop_reason=np.empty(0, dtype=np.int16),
            transmit_end_time=empty_f64.copy(),
            arrival_time=empty_f64.copy(),
            link_released=np.empty(0, dtype=bool),
            scheduled_prop_delay=empty_f64.copy(),
            shortest_gcd=empty_f64.copy(),
            initial_gcd=empty_f64.copy(),
            last_node1=empty_i64.copy(),
            last_node2=empty_i64.copy(),
        )


def create_flowlet_state(
    slot_counts: np.ndarray,
    source_region_ids: np.ndarray,
    target_region_ids: np.ndarray,
    is_normal: np.ndarray,
    packet_size: np.ndarray,
    packet_count: np.ndarray,
    default_ttl: int,
    start_time: float,
    slot_ms: float,
) -> FlowletState:
    num_slots = len(slot_counts)
    num_flowlets = int(slot_counts.sum())
    slot_offsets = np.empty(num_slots + 1, dtype=np.int64)
    slot_offsets[0] = 0
    np.cumsum(slot_counts, out=slot_offsets[1:])

    if num_flowlets == 0:
        return FlowletState.empty(slot_offsets=slot_offsets)

    creation_slots = np.repeat(np.arange(num_slots, dtype=np.int64), slot_counts)
    creation_times = start_time + creation_slots.astype(np.float64) * slot_ms
    size = packet_size * packet_count

    return FlowletState(
        slot_offsets=slot_offsets,
        status=np.full(num_flowlets, FLOWLET_NOT_STARTED, dtype=np.int8),
        creation_slot=creation_slots,
        creation_time=creation_times,
        source_region_id=source_region_ids.astype(np.int64, copy=False),
        target_region_id=target_region_ids.astype(np.int64, copy=False),
        source_id=np.full(num_flowlets, -1, dtype=np.int64),
        current_sat=np.full(num_flowlets, -1, dtype=np.int64),
        next_sat=np.full(num_flowlets, -1, dtype=np.int64),
        link_id=np.full(num_flowlets, -1, dtype=np.int32),
        packet_count=packet_count.astype(np.int64, copy=False),
        packet_size=packet_size.astype(np.float64, copy=False),
        is_normal=is_normal.astype(bool, copy=False),
        size=size.astype(np.float64, copy=False),
        ttl=np.full(num_flowlets, default_ttl, dtype=np.int16),
        hops=np.zeros(num_flowlets, dtype=np.int16),
        queue_delay=np.zeros(num_flowlets, dtype=np.float64),
        transmission_delay=np.zeros(num_flowlets, dtype=np.float64),
        propagation_delay=np.zeros(num_flowlets, dtype=np.float64),
        total_queue_cost=np.zeros(num_flowlets, dtype=np.float64),
        first_access_delay=np.zeros(num_flowlets, dtype=np.float64),
        final_access_delay=np.zeros(num_flowlets, dtype=np.float64),
        delivery_time=np.full(num_flowlets, np.nan, dtype=np.float64),
        drop_time=np.full(num_flowlets, np.nan, dtype=np.float64),
        drop_reason=np.full(num_flowlets, -1, dtype=np.int16),
        transmit_end_time=np.full(num_flowlets, np.inf, dtype=np.float64),
        arrival_time=np.full(num_flowlets, np.inf, dtype=np.float64),
        link_released=np.ones(num_flowlets, dtype=bool),
        scheduled_prop_delay=np.zeros(num_flowlets, dtype=np.float64),
        shortest_gcd=np.full(num_flowlets, np.inf, dtype=np.float64),
        initial_gcd=np.ones(num_flowlets, dtype=np.float64),
        last_node1=np.full(num_flowlets, -1, dtype=np.int64),
        last_node2=np.full(num_flowlets, -1, dtype=np.int64),
    )


@dataclass(slots=True)
class LinkState:
    """Runtime link arrays used by the slot scheduler."""

    source_ids: np.ndarray
    sink_ids: np.ndarray
    data_rate: np.ndarray
    capacity: np.ndarray
    connected: np.ndarray
    delay: np.ndarray
    id_by_pair: np.ndarray
    neighbor_link_ids: np.ndarray
    free_time: np.ndarray
    queue_load: np.ndarray

    @property
    def count(self) -> int:
        return len(self.source_ids)


def create_link_state(
    source_ids: np.ndarray,
    sink_ids: np.ndarray,
    data_rate: np.ndarray,
    capacity: np.ndarray,
    connected: np.ndarray,
    delay: np.ndarray,
    num_nodes: int,
    neighbor_sat_ids: np.ndarray,
) -> LinkState:
    num_links = len(source_ids)
    id_by_pair = np.full((num_nodes, num_nodes), -1, dtype=np.int32)
    id_by_pair[source_ids, sink_ids] = np.arange(num_links, dtype=np.int32)

    neighbor_link_ids = np.full(neighbor_sat_ids.shape, -1, dtype=np.int32)
    valid_neighbor = neighbor_sat_ids >= 0
    if valid_neighbor.any():
        source_cols = np.broadcast_to(np.arange(num_nodes, dtype=np.int64)[:, None], neighbor_sat_ids.shape)
        neighbor_link_ids[valid_neighbor] = id_by_pair[
            source_cols[valid_neighbor],
            neighbor_sat_ids[valid_neighbor],
        ]

    return LinkState(
        source_ids=source_ids,
        sink_ids=sink_ids,
        data_rate=data_rate,
        capacity=capacity,
        connected=connected,
        delay=delay,
        id_by_pair=id_by_pair,
        neighbor_link_ids=neighbor_link_ids,
        free_time=np.zeros(num_links, dtype=np.float64),
        queue_load=np.zeros(num_links, dtype=np.float64),
    )


def drop_flowlet_ids(
    flowlets: FlowletState,
    flowlet_ids: np.ndarray,
    current_time: float,
    reason: int,
) -> None:
    if len(flowlet_ids) == 0:
        return
    flowlets.status[flowlet_ids] = FLOWLET_DROPPED
    flowlets.drop_time[flowlet_ids] = current_time
    flowlets.drop_reason[flowlet_ids] = reason


def flowlet_mask_to_ids(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def empty_flowlet_mask(flowlets: FlowletState) -> np.ndarray:
    return np.zeros(flowlets.count, dtype=bool)


def drop_flowlet_mask(
    flowlets: FlowletState,
    flowlet_mask: np.ndarray,
    current_time: float,
    reason: int,
) -> None:
    if not flowlet_mask.any():
        return
    flowlets.status[flowlet_mask] = FLOWLET_DROPPED
    flowlets.drop_time[flowlet_mask] = current_time
    flowlets.drop_reason[flowlet_mask] = reason


def release_transmitted_flowlets(
    flowlets: FlowletState,
    links: LinkState,
    current_time: float,
) -> None:
    status = flowlets.status
    if len(status) == 0:
        return
    release_mask = (
        (status == FLOWLET_ON_LINK)
        & (~flowlets.link_released)
        & (flowlets.transmit_end_time <= current_time)
    )
    release_ids = np.flatnonzero(release_mask)
    if len(release_ids) == 0:
        return
    np.add.at(links.queue_load, flowlets.link_id[release_ids], -flowlets.size[release_ids])
    flowlets.link_released[release_ids] = True
    np.maximum(links.queue_load, 0.0, out=links.queue_load)


def handle_flowlet_arrivals(
    flowlets: FlowletState,
    current_time: float,
    ttl_expired_reason: int,
) -> np.ndarray:
    return flowlet_mask_to_ids(
        handle_flowlet_arrivals_mask(
            flowlets=flowlets,
            current_time=current_time,
            ttl_expired_reason=ttl_expired_reason,
        )
    )


def handle_flowlet_arrivals_mask(
    flowlets: FlowletState,
    current_time: float,
    ttl_expired_reason: int,
) -> np.ndarray:
    status = flowlets.status
    if len(status) == 0:
        return empty_flowlet_mask(flowlets)
    arrival_mask = (status == FLOWLET_ON_LINK) & (flowlets.arrival_time <= current_time)
    if not arrival_mask.any():
        return arrival_mask

    flowlets.current_sat[arrival_mask] = flowlets.next_sat[arrival_mask]
    flowlets.hops[arrival_mask] += 1
    flowlets.ttl[arrival_mask] -= 1
    flowlets.propagation_delay[arrival_mask] += flowlets.scheduled_prop_delay[arrival_mask]
    flowlets.queue_delay[arrival_mask] += current_time - flowlets.arrival_time[arrival_mask]
    flowlets.status[arrival_mask] = FLOWLET_AT_NODE
    flowlets.link_id[arrival_mask] = -1

    expired_mask = arrival_mask & (flowlets.ttl <= 0)
    if expired_mask.any():
        drop_flowlet_mask(flowlets, expired_mask, current_time, ttl_expired_reason)
    return arrival_mask & (~expired_mask)


def activate_flowlets_at_slot(
    flowlets: FlowletState,
    slot_idx: int,
    current_time: float,
    nearest_region_sat_ids: np.ndarray,
    nearest_region_sat_distances: np.ndarray,
    region_distance_matrix: np.ndarray,
    access_data_rate: float,
    no_available_sat_reason: int,
) -> np.ndarray:
    return flowlet_mask_to_ids(
        activate_flowlets_at_slot_mask(
            flowlets=flowlets,
            slot_idx=slot_idx,
            current_time=current_time,
            nearest_region_sat_ids=nearest_region_sat_ids,
            nearest_region_sat_distances=nearest_region_sat_distances,
            region_distance_matrix=region_distance_matrix,
            access_data_rate=access_data_rate,
            no_available_sat_reason=no_available_sat_reason,
        )
    )


def activate_flowlets_at_slot_mask(
    flowlets: FlowletState,
    slot_idx: int,
    current_time: float,
    nearest_region_sat_ids: np.ndarray,
    nearest_region_sat_distances: np.ndarray,
    region_distance_matrix: np.ndarray,
    access_data_rate: float,
    no_available_sat_reason: int,
) -> np.ndarray:
    active_mask = empty_flowlet_mask(flowlets)
    if slot_idx < 0 or slot_idx + 1 >= len(flowlets.slot_offsets):
        return active_mask
    start = int(flowlets.slot_offsets[slot_idx])
    end = int(flowlets.slot_offsets[slot_idx + 1])
    if end <= start:
        return active_mask

    flowlet_ids = np.arange(start, end, dtype=np.int64)
    source_regions = flowlets.source_region_id[flowlet_ids]
    source_sat_ids = nearest_region_sat_ids[source_regions]
    visible = source_sat_ids >= 0
    if (~visible).any():
        drop_flowlet_ids(flowlets, flowlet_ids[~visible], current_time, no_available_sat_reason)

    active_ids = flowlet_ids[visible]
    if len(active_ids) == 0:
        return active_mask

    source_sat_ids = source_sat_ids[visible]
    source_distances = nearest_region_sat_distances[source_regions[visible]]
    source_prop_delay = source_distances / LIGHT_SPEED_MS
    source_tx_delay = flowlets.size[active_ids] / access_data_rate

    flowlets.source_id[active_ids] = source_sat_ids
    flowlets.current_sat[active_ids] = source_sat_ids
    flowlets.status[active_ids] = FLOWLET_AT_NODE
    flowlets.first_access_delay[active_ids] = source_prop_delay + source_tx_delay
    flowlets.propagation_delay[active_ids] += source_prop_delay
    flowlets.transmission_delay[active_ids] += source_tx_delay

    initial_gcd = region_distance_matrix[
        flowlets.source_region_id[active_ids],
        flowlets.target_region_id[active_ids],
    ].copy()
    initial_gcd[initial_gcd <= 0] = 1e-6
    flowlets.initial_gcd[active_ids] = initial_gcd
    flowlets.shortest_gcd[active_ids] = initial_gcd
    active_mask[active_ids] = True
    return active_mask


def partition_deliverable_flowlets(
    flowlets: FlowletState,
    flowlet_ids: np.ndarray,
    sat_id_to_col_array: np.ndarray,
    region_sat_visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current_sats = flowlets.current_sat[flowlet_ids]
    target_regions = flowlets.target_region_id[flowlet_ids]
    sat_cols = sat_id_to_col_array[current_sats]
    valid = sat_cols >= 0
    deliverable = np.zeros(len(flowlet_ids), dtype=bool)
    deliverable[valid] = region_sat_visible[target_regions[valid], sat_cols[valid]]
    return flowlet_ids[deliverable], flowlet_ids[~deliverable]


def partition_deliverable_flowlet_mask(
    flowlets: FlowletState,
    flowlet_mask: np.ndarray,
    sat_id_to_col_array: np.ndarray,
    region_sat_visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    deliverable_mask = empty_flowlet_mask(flowlets)
    route_mask = empty_flowlet_mask(flowlets)
    flowlet_ids = flowlet_mask_to_ids(flowlet_mask)
    if len(flowlet_ids) == 0:
        return deliverable_mask, route_mask

    delivered_ids, route_ids = partition_deliverable_flowlets(
        flowlets=flowlets,
        flowlet_ids=flowlet_ids,
        sat_id_to_col_array=sat_id_to_col_array,
        region_sat_visible=region_sat_visible,
    )
    deliverable_mask[delivered_ids] = True
    route_mask[route_ids] = True
    return deliverable_mask, route_mask


def deliver_flowlet_ids(
    flowlets: FlowletState,
    flowlet_ids: np.ndarray,
    current_time: float,
    sat_id_to_col_array: np.ndarray,
    region_sat_distance2: np.ndarray,
    access_data_rate: float,
) -> None:
    current_sats = flowlets.current_sat[flowlet_ids]
    target_regions = flowlets.target_region_id[flowlet_ids]
    sat_cols = sat_id_to_col_array[current_sats]
    distance2 = region_sat_distance2[target_regions, sat_cols]
    final_prop_delay = np.sqrt(distance2) / LIGHT_SPEED_MS
    final_tx_delay = flowlets.size[flowlet_ids] / access_data_rate
    final_delay = final_prop_delay + final_tx_delay

    flowlets.final_access_delay[flowlet_ids] = final_delay
    flowlets.propagation_delay[flowlet_ids] += final_prop_delay
    flowlets.transmission_delay[flowlet_ids] += final_tx_delay
    flowlets.delivery_time[flowlet_ids] = current_time + final_delay
    flowlets.status[flowlet_ids] = FLOWLET_DELIVERED


def build_routing_batch(
    flowlets: FlowletState,
    links: LinkState,
    flowlet_ids: np.ndarray,
    current_sats: np.ndarray,
    target_regions: np.ndarray,
    target_access_sats: np.ndarray,
    neighbor_sat_ids_by_node: np.ndarray,
    current_time: float,
    region_next_hop_table: np.ndarray | None,
    region_next_hop_version: int = 0,
) -> RoutingBatch:
    neighbor_sat_ids = neighbor_sat_ids_by_node[current_sats]
    neighbor_link_ids = links.neighbor_link_ids[current_sats]

    valid_link = neighbor_link_ids >= 0
    safe_link_ids = np.where(valid_link, neighbor_link_ids, 0)
    connected = np.zeros(neighbor_link_ids.shape, dtype=bool)
    if valid_link.any():
        connected[valid_link] = links.connected[safe_link_ids[valid_link]]
    action_mask = valid_link & connected

    return RoutingBatch(
        flowlet_ids=flowlet_ids,
        current_sat_ids=current_sats,
        target_region_ids=target_regions,
        target_access_sat_ids=target_access_sats,
        neighbor_sat_ids=neighbor_sat_ids,
        neighbor_link_ids=neighbor_link_ids,
        action_mask=action_mask,
        neighbor_queue_load=np.where(valid_link, links.queue_load[safe_link_ids], np.inf),
        neighbor_link_capacity=np.where(valid_link, links.capacity[safe_link_ids], 0.0),
        neighbor_link_delay=np.where(valid_link, links.delay[safe_link_ids], np.inf),
        neighbor_link_free_time=np.where(valid_link, links.free_time[safe_link_ids], np.inf),
        flowlet_size=flowlets.size[flowlet_ids],
        ttl=flowlets.ttl[flowlet_ids],
        current_time=current_time,
        region_next_hop_table=region_next_hop_table,
        region_next_hop_version=region_next_hop_version,
        hops=flowlets.hops[flowlet_ids],
        queue_delay=flowlets.queue_delay[flowlet_ids],
        transmission_delay=flowlets.transmission_delay[flowlet_ids],
        propagation_delay=flowlets.propagation_delay[flowlet_ids],
        total_queue_cost=flowlets.total_queue_cost[flowlet_ids],
        shortest_gcd=flowlets.shortest_gcd[flowlet_ids],
        initial_gcd=flowlets.initial_gcd[flowlet_ids],
    )


def prepare_route_candidate_mask(
    flowlets: FlowletState,
    route_flowlet_ids: np.ndarray,
    nearest_region_sat_ids: np.ndarray,
    current_time: float,
    no_available_sat_reason: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_regions = flowlets.target_region_id[route_flowlet_ids]
    target_access_sats = nearest_region_sat_ids[target_regions]
    candidate_local_mask = target_access_sats >= 0
    if (~candidate_local_mask).any():
        drop_flowlet_ids(
            flowlets=flowlets,
            flowlet_ids=route_flowlet_ids[~candidate_local_mask],
            current_time=current_time,
            reason=no_available_sat_reason,
        )
    return candidate_local_mask, target_access_sats


def apply_no_route_mask(
    flowlets: FlowletState,
    candidate_ids: np.ndarray,
    next_hops: np.ndarray,
    current_time: float,
    failed_to_find_next_hop_reason: int,
) -> np.ndarray:
    routable_local_mask = next_hops >= 0
    if (~routable_local_mask).any():
        drop_flowlet_ids(
            flowlets=flowlets,
            flowlet_ids=candidate_ids[~routable_local_mask],
            current_time=current_time,
            reason=failed_to_find_next_hop_reason,
        )
    return routable_local_mask


def prepare_link_schedule_inputs(
    flowlets: FlowletState,
    links: LinkState,
    routable_ids: np.ndarray,
    current_sats: np.ndarray,
    next_hops: np.ndarray,
    neighbor_sat_ids_by_node: np.ndarray,
    current_time: float,
    invalid_next_hop_reason: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbor_matches = neighbor_sat_ids_by_node[current_sats] == next_hops[:, None]
    direction_ids = np.argmax(neighbor_matches, axis=1)
    valid_neighbor = neighbor_matches.any(axis=1)
    link_ids = links.neighbor_link_ids[current_sats, direction_ids]
    schedulable_mask = valid_neighbor & (link_ids >= 0)
    schedulable_mask[schedulable_mask] &= links.connected[link_ids[schedulable_mask]]
    if (~schedulable_mask).any():
        drop_flowlet_ids(
            flowlets=flowlets,
            flowlet_ids=routable_ids[~schedulable_mask],
            current_time=current_time,
            reason=invalid_next_hop_reason,
        )
    return routable_ids[schedulable_mask], link_ids[schedulable_mask], next_hops[schedulable_mask]


def schedule_flowlets_by_link(
    flowlets: FlowletState,
    links: LinkState,
    flowlet_ids: np.ndarray,
    link_ids: np.ndarray,
    next_hops: np.ndarray,
    current_time: float,
) -> np.ndarray:
    if len(flowlet_ids) == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(link_ids, kind="stable")
    sorted_link_ids = link_ids[order]
    sorted_ids = flowlet_ids[order]
    sorted_next_hops = next_hops[order]

    sizes = flowlets.size[sorted_ids]
    cumulative_size = np.cumsum(sizes)
    group_start_mask = np.empty(len(sorted_link_ids), dtype=bool)
    group_start_mask[0] = True
    group_start_mask[1:] = sorted_link_ids[1:] != sorted_link_ids[:-1]
    group_start_indices = np.maximum.accumulate(
        np.where(group_start_mask, np.arange(len(sorted_link_ids)), 0)
    )
    size_offsets = np.where(
        group_start_indices == 0,
        0.0,
        cumulative_size[group_start_indices - 1],
    )
    group_cumulative_size = cumulative_size - size_offsets
    remaining_capacity = links.capacity[sorted_link_ids] - links.queue_load[sorted_link_ids]
    accepted = group_cumulative_size <= remaining_capacity

    rejected_ids = sorted_ids[~accepted]
    if not accepted.any():
        return rejected_ids

    accepted_ids = sorted_ids[accepted]
    accepted_link_ids = sorted_link_ids[accepted]
    accepted_next_hops = sorted_next_hops[accepted]
    accepted_sizes = sizes[accepted]

    transmit_times = accepted_sizes / links.data_rate[accepted_link_ids]
    cumulative_tx = np.cumsum(transmit_times)
    tx_group_start_mask = np.empty(len(accepted_link_ids), dtype=bool)
    tx_group_start_mask[0] = True
    tx_group_start_mask[1:] = accepted_link_ids[1:] != accepted_link_ids[:-1]
    tx_group_start_indices = np.maximum.accumulate(
        np.where(tx_group_start_mask, np.arange(len(accepted_link_ids)), 0)
    )
    tx_offsets = np.where(
        tx_group_start_indices == 0,
        0.0,
        cumulative_tx[tx_group_start_indices - 1],
    )
    group_cumulative_tx = cumulative_tx - tx_offsets

    start_base = np.maximum(current_time, links.free_time[accepted_link_ids])
    transmit_end_times = start_base + group_cumulative_tx
    wait_times = transmit_end_times - transmit_times - current_time
    propagation_delays = links.delay[accepted_link_ids]

    accepted_group_starts = np.flatnonzero(tx_group_start_mask)
    accepted_group_link_ids = accepted_link_ids[accepted_group_starts]
    # Preserve the old segment summation order; tiny FP drift changes later capacity decisions.
    links.queue_load[accepted_group_link_ids] += np.add.reduceat(
        accepted_sizes,
        accepted_group_starts,
    )

    free_time_updates = np.full(links.count, -np.inf, dtype=np.float64)
    np.maximum.at(free_time_updates, accepted_link_ids, transmit_end_times)
    updated_links = np.isfinite(free_time_updates)
    links.free_time[updated_links] = free_time_updates[updated_links]

    flowlets.queue_delay[accepted_ids] += wait_times
    flowlets.transmission_delay[accepted_ids] += transmit_times
    flowlets.total_queue_cost[accepted_ids] += wait_times
    flowlets.link_id[accepted_ids] = accepted_link_ids
    flowlets.next_sat[accepted_ids] = accepted_next_hops
    flowlets.transmit_end_time[accepted_ids] = transmit_end_times
    flowlets.arrival_time[accepted_ids] = transmit_end_times + propagation_delays
    flowlets.scheduled_prop_delay[accepted_ids] = propagation_delays
    flowlets.link_released[accepted_ids] = False
    flowlets.status[accepted_ids] = FLOWLET_ON_LINK

    flowlets.last_node2[accepted_ids] = flowlets.last_node1[accepted_ids]
    flowlets.last_node1[accepted_ids] = flowlets.current_sat[accepted_ids]
    return rejected_ids


def drop_flowlets_on_disconnected_links(
    flowlets: FlowletState,
    links: LinkState,
    current_time: float,
    link_disconnected_reason: int,
) -> None:
    status = flowlets.status
    if len(status) == 0:
        return
    on_link = np.flatnonzero(status == FLOWLET_ON_LINK)
    if len(on_link) == 0:
        return
    link_ids = flowlets.link_id[on_link]
    dropped = on_link[~links.connected[link_ids]]
    if len(dropped) == 0:
        return
    unreleased = dropped[~flowlets.link_released[dropped]]
    if len(unreleased) > 0:
        np.add.at(links.queue_load, flowlets.link_id[unreleased], -flowlets.size[unreleased])
        flowlets.link_released[unreleased] = True
    links.queue_load[~links.connected] = 0.0
    links.free_time[~links.connected] = current_time
    drop_flowlet_ids(flowlets, dropped, current_time, link_disconnected_reason)
