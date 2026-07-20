from __future__ import annotations

import math

import numpy as np
from numba import njit


ACTION_COUNT = 4

FLOWLET_AT_NODE = 1
FLOWLET_ON_LINK = 2
FLOWLET_DELIVERED = 3
FLOWLET_DROPPED = 4

ERR_LINK_FULL = 2
ERR_INVALID_NEXT_HOP = 3
ERR_FAILED_TO_FIND_NEXT_HOP = 4
ERR_TTL_EXPIRED = 1
ERR_LINK_DISCONNECTED = 5

NODE_POS_X = 0
NODE_POS_Y = 1
NODE_POS_Z = 2
NODE_QUEUE_LOAD = 3
NODE_QUEUE_REMAINING = 4

LINK_CONNECTED = 0
LINK_DELAY = 1
LINK_QUEUE_LOAD = 2
LINK_QUEUE_REMAINING = 3
LINK_FREE_TIME_DELTA = 4
LINK_CAPACITY = 5
LINK_DATA_RATE = 6


@njit(cache=True)
def _drop_one(
    status: np.ndarray,
    packet_count: np.ndarray,
    is_normal: np.ndarray,
    drop_time: np.ndarray,
    drop_reason: np.ndarray,
    dropped: np.ndarray,
    dropped_normal: np.ndarray,
    dropped_small: np.ndarray,
    env_id: int,
    flowlet_id: int,
    current_time: float,
    reason: int,
) -> int:
    current_status = int(status[env_id, flowlet_id])
    if current_status == FLOWLET_DROPPED or current_status == FLOWLET_DELIVERED:
        return 0
    weight = int(packet_count[env_id, flowlet_id])
    dropped[env_id] += weight
    if is_normal[env_id, flowlet_id]:
        dropped_normal[env_id] += weight
    else:
        dropped_small[env_id] += weight
    status[env_id, flowlet_id] = FLOWLET_DROPPED
    drop_time[env_id, flowlet_id] = current_time
    drop_reason[env_id, flowlet_id] = reason
    return 1


@njit(cache=True)
def apply_routing_actions_kernel(
    env_ids: np.ndarray,
    flowlet_ids: np.ndarray,
    actions: np.ndarray,
    decision_mask: np.ndarray,
    current_times: np.ndarray,
    status: np.ndarray,
    current_sat: np.ndarray,
    neighbor_sat_ids: np.ndarray,
    neighbor_link_ids: np.ndarray,
    direction_by_link_id: np.ndarray,
    link_connected: np.ndarray,
    link_capacity: np.ndarray,
    link_data_rate: np.ndarray,
    link_delay: np.ndarray,
    size: np.ndarray,
    schedule_tx_delta: np.ndarray,
    schedule_base_time: np.ndarray,
    schedule_token: np.ndarray,
    schedule_token_value: int,
    packet_count: np.ndarray,
    is_normal: np.ndarray,
    queue_load: np.ndarray,
    free_time: np.ndarray,
    queue_delay: np.ndarray,
    transmission_delay: np.ndarray,
    total_queue_cost: np.ndarray,
    link_id: np.ndarray,
    next_sat: np.ndarray,
    transmit_end_time: np.ndarray,
    arrival_time: np.ndarray,
    scheduled_prop_delay: np.ndarray,
    link_released: np.ndarray,
    last_action1: np.ndarray,
    last_action2: np.ndarray,
    last_node1: np.ndarray,
    last_node2: np.ndarray,
    drop_time: np.ndarray,
    drop_reason: np.ndarray,
    dropped: np.ndarray,
    dropped_normal: np.ndarray,
    dropped_small: np.ndarray,
) -> tuple[int, int, int, int]:
    scheduled = 0
    no_route_dropped = 0
    invalid_dropped = 0
    link_full_dropped = 0
    for row in range(actions.shape[0]):
        if not decision_mask[row]:
            continue
        env_id = int(env_ids[row])
        flowlet_id = int(flowlet_ids[row])
        current_time = float(current_times[env_id])
        action = int(actions[row])
        if action < 0:
            no_route_dropped += _drop_one(
                status,
                packet_count,
                is_normal,
                drop_time,
                drop_reason,
                dropped,
                dropped_normal,
                dropped_small,
                env_id,
                flowlet_id,
                current_time,
                ERR_FAILED_TO_FIND_NEXT_HOP,
            )
            continue
        if status[env_id, flowlet_id] != FLOWLET_AT_NODE:
            continue
        current = int(current_sat[env_id, flowlet_id])
        if current < 0:
            invalid_dropped += _drop_one(
                status,
                packet_count,
                is_normal,
                drop_time,
                drop_reason,
                dropped,
                dropped_normal,
                dropped_small,
                env_id,
                flowlet_id,
                current_time,
                ERR_INVALID_NEXT_HOP,
            )
            continue

        direction_id = -1
        for action_idx in range(ACTION_COUNT):
            if int(neighbor_sat_ids[current, action_idx]) == action:
                direction_id = action_idx
                break
        if direction_id < 0:
            invalid_dropped += _drop_one(
                status,
                packet_count,
                is_normal,
                drop_time,
                drop_reason,
                dropped,
                dropped_normal,
                dropped_small,
                env_id,
                flowlet_id,
                current_time,
                ERR_INVALID_NEXT_HOP,
            )
            continue

        lid = int(neighbor_link_ids[current, direction_id])
        if lid < 0 or not link_connected[env_id, lid]:
            invalid_dropped += _drop_one(
                status,
                packet_count,
                is_normal,
                drop_time,
                drop_reason,
                dropped,
                dropped_normal,
                dropped_small,
                env_id,
                flowlet_id,
                current_time,
                ERR_INVALID_NEXT_HOP,
            )
            continue

        flowlet_size = float(size[env_id, flowlet_id])
        remaining_capacity = float(link_capacity[lid]) - float(queue_load[env_id, lid])
        if flowlet_size > remaining_capacity:
            link_full_dropped += _drop_one(
                status,
                packet_count,
                is_normal,
                drop_time,
                drop_reason,
                dropped,
                dropped_normal,
                dropped_small,
                env_id,
                flowlet_id,
                current_time,
                ERR_LINK_FULL,
            )
            continue

        data_rate = float(link_data_rate[lid])
        if data_rate < 1e-9:
            data_rate = 1e-9
        transmit_time = np.float32(flowlet_size / data_rate)
        if schedule_token[env_id, lid] != schedule_token_value:
            schedule_token[env_id, lid] = schedule_token_value
            schedule_tx_delta[env_id, lid] = 0.0
            base_time = float(free_time[env_id, lid])
            if base_time < current_time:
                base_time = current_time
            schedule_base_time[env_id, lid] = base_time
        schedule_tx_delta[env_id, lid] = np.float32(schedule_tx_delta[env_id, lid] + transmit_time)
        transmit_end = float(schedule_base_time[env_id, lid]) + float(schedule_tx_delta[env_id, lid])
        wait_time = transmit_end - float(transmit_time) - current_time
        prop_delay = float(link_delay[env_id, lid])

        queue_load[env_id, lid] += flowlet_size
        free_time[env_id, lid] = transmit_end
        queue_delay[env_id, flowlet_id] += wait_time
        transmission_delay[env_id, flowlet_id] += transmit_time
        total_queue_cost[env_id, flowlet_id] += wait_time
        link_id[env_id, flowlet_id] = lid
        next_sat[env_id, flowlet_id] = action
        transmit_end_time[env_id, flowlet_id] = transmit_end
        arrival_time[env_id, flowlet_id] = transmit_end + prop_delay
        scheduled_prop_delay[env_id, flowlet_id] = prop_delay
        link_released[env_id, flowlet_id] = False
        status[env_id, flowlet_id] = FLOWLET_ON_LINK
        last_action2[env_id, flowlet_id] = last_action1[env_id, flowlet_id]
        last_action1[env_id, flowlet_id] = direction_by_link_id[lid]
        last_node2[env_id, flowlet_id] = last_node1[env_id, flowlet_id]
        last_node1[env_id, flowlet_id] = current
        scheduled += 1
    return scheduled, no_route_dropped, invalid_dropped, link_full_dropped


@njit(cache=True)
def refresh_link_state_kernel(
    link_state: np.ndarray,
    link_connected: np.ndarray,
    link_delay: np.ndarray,
    queue_load: np.ndarray,
    free_time: np.ndarray,
    current_times: np.ndarray,
    link_capacity: np.ndarray,
    link_data_rate: np.ndarray,
) -> None:
    num_envs = link_state.shape[0]
    num_links = link_state.shape[1]
    for env_id in range(num_envs):
        current_time = float(current_times[env_id])
        for link_id in range(num_links):
            load = float(queue_load[env_id, link_id])
            remaining = float(link_capacity[link_id]) - load
            if remaining < 0.0:
                remaining = 0.0
            free_delta = float(free_time[env_id, link_id]) - current_time
            if free_delta < 0.0:
                free_delta = 0.0
            link_state[env_id, link_id, LINK_CONNECTED] = 1.0 if link_connected[env_id, link_id] else 0.0
            link_state[env_id, link_id, LINK_DELAY] = link_delay[env_id, link_id]
            link_state[env_id, link_id, LINK_QUEUE_LOAD] = queue_load[env_id, link_id]
            link_state[env_id, link_id, LINK_QUEUE_REMAINING] = remaining
            link_state[env_id, link_id, LINK_FREE_TIME_DELTA] = free_delta
            link_state[env_id, link_id, LINK_CAPACITY] = link_capacity[link_id]
            link_state[env_id, link_id, LINK_DATA_RATE] = link_data_rate[link_id]


@njit(cache=True)
def build_action_mask_kernel(
    neighbor_sat_out: np.ndarray,
    neighbor_link_out: np.ndarray,
    action_mask_out: np.ndarray,
    env_ids: np.ndarray,
    flowlet_ids: np.ndarray,
    current_sats: np.ndarray,
    target_regions: np.ndarray,
    decision_mask: np.ndarray,
    neighbor_sat_ids: np.ndarray,
    neighbor_link_ids: np.ndarray,
    link_connected: np.ndarray,
    region_sat_distances: np.ndarray,
    last_node1: np.ndarray,
    max_access_range: float,
) -> None:
    for row in range(current_sats.shape[0]):
        env_id = int(env_ids[row])
        flowlet_id = int(flowlet_ids[row])
        current_sat = int(current_sats[row])
        target_region = int(target_regions[row])
        should_decide = bool(decision_mask[row])
        previous_node = int(last_node1[env_id, flowlet_id])
        for action_idx in range(ACTION_COUNT):
            neighbor_sat = int(neighbor_sat_ids[current_sat, action_idx])
            neighbor_link = int(neighbor_link_ids[current_sat, action_idx])
            neighbor_sat_out[row, action_idx] = neighbor_sat
            neighbor_link_out[row, action_idx] = neighbor_link
            valid = should_decide and neighbor_link >= 0 and link_connected[env_id, neighbor_link]
            if valid and neighbor_sat == previous_node:
                target_access = (
                    neighbor_sat >= 0
                    and target_region >= 0
                    and region_sat_distances[env_id, target_region, neighbor_sat] <= max_access_range
                )
                if not target_access:
                    valid = False
            action_mask_out[row, action_idx] = valid


@njit(cache=True)
def refresh_node_queue_features_kernel(
    node_state: np.ndarray,
    node_queue_load: np.ndarray,
    node_queue_remaining: np.ndarray,
    queue_load: np.ndarray,
    neighbor_link_ids: np.ndarray,
    node_total_link_capacity: np.ndarray,
) -> None:
    num_envs = node_state.shape[0]
    num_nodes = node_state.shape[1]
    for env_id in range(num_envs):
        for node_id in range(num_nodes):
            total_queue = 0.0
            for direction_id in range(ACTION_COUNT):
                link_id = int(neighbor_link_ids[node_id, direction_id])
                if link_id >= 0:
                    total_queue += float(queue_load[env_id, link_id])
            capacity = float(node_total_link_capacity[node_id])
            if capacity > 0.0:
                load = total_queue / capacity
                remaining = capacity - total_queue
                if remaining < 0.0:
                    remaining = 0.0
            else:
                load = 0.0
                remaining = 0.0
            node_queue_load[env_id, node_id] = load
            node_queue_remaining[env_id, node_id] = remaining
            node_state[env_id, node_id, NODE_QUEUE_LOAD] = load
            node_state[env_id, node_id, NODE_QUEUE_REMAINING] = remaining


@njit(cache=True)
def release_transmitted_kernel(
    status: np.ndarray,
    link_released: np.ndarray,
    transmit_end_time: np.ndarray,
    current_times: np.ndarray,
    link_id: np.ndarray,
    size: np.ndarray,
    queue_load: np.ndarray,
) -> int:
    released = 0
    num_envs = status.shape[0]
    capacity = status.shape[1]
    for env_id in range(num_envs):
        current_time = float(current_times[env_id])
        for flowlet_id in range(capacity):
            if (
                status[env_id, flowlet_id] == FLOWLET_ON_LINK
                and not link_released[env_id, flowlet_id]
                and transmit_end_time[env_id, flowlet_id] <= current_time
            ):
                lid = int(link_id[env_id, flowlet_id])
                if lid >= 0:
                    new_load = float(queue_load[env_id, lid]) - float(size[env_id, flowlet_id])
                    queue_load[env_id, lid] = 0.0 if new_load < 0.0 else new_load
                link_released[env_id, flowlet_id] = True
                released += 1
    return released


@njit(cache=True)
def handle_arrivals_kernel(
    status: np.ndarray,
    current_sat: np.ndarray,
    next_sat: np.ndarray,
    link_id: np.ndarray,
    hops: np.ndarray,
    ttl: np.ndarray,
    propagation_delay: np.ndarray,
    queue_delay: np.ndarray,
    scheduled_prop_delay: np.ndarray,
    arrival_time: np.ndarray,
    current_times: np.ndarray,
    packet_count: np.ndarray,
    is_normal: np.ndarray,
    drop_time: np.ndarray,
    drop_reason: np.ndarray,
    dropped: np.ndarray,
    dropped_normal: np.ndarray,
    dropped_small: np.ndarray,
    dropped_by_ttl: np.ndarray,
) -> tuple[int, int]:
    arrived = 0
    ttl_dropped = 0
    num_envs = status.shape[0]
    capacity = status.shape[1]
    for env_id in range(num_envs):
        current_time = float(current_times[env_id])
        for flowlet_id in range(capacity):
            if status[env_id, flowlet_id] == FLOWLET_ON_LINK and arrival_time[env_id, flowlet_id] <= current_time:
                current_sat[env_id, flowlet_id] = next_sat[env_id, flowlet_id]
                hops[env_id, flowlet_id] += 1
                ttl[env_id, flowlet_id] -= 1
                propagation_delay[env_id, flowlet_id] += scheduled_prop_delay[env_id, flowlet_id]
                queue_delay[env_id, flowlet_id] += current_time - arrival_time[env_id, flowlet_id]
                status[env_id, flowlet_id] = FLOWLET_AT_NODE
                link_id[env_id, flowlet_id] = -1
                arrived += 1
                if ttl[env_id, flowlet_id] <= 0:
                    weight = int(packet_count[env_id, flowlet_id])
                    dropped[env_id] += weight
                    if is_normal[env_id, flowlet_id]:
                        dropped_normal[env_id] += weight
                    else:
                        dropped_small[env_id] += weight
                    dropped_by_ttl[env_id] += weight
                    status[env_id, flowlet_id] = FLOWLET_DROPPED
                    drop_time[env_id, flowlet_id] = current_time
                    drop_reason[env_id, flowlet_id] = ERR_TTL_EXPIRED
                    ttl_dropped += 1
    return arrived, ttl_dropped


@njit(cache=True)
def drop_disconnected_kernel(
    status: np.ndarray,
    link_released: np.ndarray,
    link_id: np.ndarray,
    current_times: np.ndarray,
    size: np.ndarray,
    queue_load: np.ndarray,
    link_connected: np.ndarray,
    packet_count: np.ndarray,
    is_normal: np.ndarray,
    drop_time: np.ndarray,
    drop_reason: np.ndarray,
    dropped: np.ndarray,
    dropped_normal: np.ndarray,
    dropped_small: np.ndarray,
) -> int:
    dropped_count = 0
    num_envs = status.shape[0]
    capacity = status.shape[1]
    for env_id in range(num_envs):
        current_time = float(current_times[env_id])
        for flowlet_id in range(capacity):
            if status[env_id, flowlet_id] == FLOWLET_ON_LINK:
                lid = int(link_id[env_id, flowlet_id])
                if lid >= 0 and not link_connected[env_id, lid]:
                    if not link_released[env_id, flowlet_id]:
                        new_load = float(queue_load[env_id, lid]) - float(size[env_id, flowlet_id])
                        queue_load[env_id, lid] = 0.0 if new_load < 0.0 else new_load
                        link_released[env_id, flowlet_id] = True
                    weight = int(packet_count[env_id, flowlet_id])
                    dropped[env_id] += weight
                    if is_normal[env_id, flowlet_id]:
                        dropped_normal[env_id] += weight
                    else:
                        dropped_small[env_id] += weight
                    status[env_id, flowlet_id] = FLOWLET_DROPPED
                    drop_time[env_id, flowlet_id] = current_time
                    drop_reason[env_id, flowlet_id] = ERR_LINK_DISCONNECTED
                    dropped_count += 1
    return dropped_count


@njit(cache=True)
def deliver_visible_kernel(
    deliverable: np.ndarray,
    env_ids: np.ndarray,
    flowlet_ids: np.ndarray,
    current_times: np.ndarray,
    status: np.ndarray,
    current_sat: np.ndarray,
    target_region_id: np.ndarray,
    region_sat_distances: np.ndarray,
    size: np.ndarray,
    packet_count: np.ndarray,
    is_normal: np.ndarray,
    queue_delay: np.ndarray,
    transmission_delay: np.ndarray,
    propagation_delay: np.ndarray,
    total_queue_cost: np.ndarray,
    final_access_delay: np.ndarray,
    delivery_time: np.ndarray,
    delivered: np.ndarray,
    delivered_normal: np.ndarray,
    delivered_small: np.ndarray,
    delivered_mbit: np.ndarray,
    e2e_delay_sum: np.ndarray,
    queue_delay_sum: np.ndarray,
    transmission_delay_sum: np.ndarray,
    propagation_delay_sum: np.ndarray,
    normal_e2e_delay_sum: np.ndarray,
    normal_queue_delay_sum: np.ndarray,
    normal_transmission_delay_sum: np.ndarray,
    normal_propagation_delay_sum: np.ndarray,
    small_e2e_delay_sum: np.ndarray,
    small_queue_delay_sum: np.ndarray,
    small_transmission_delay_sum: np.ndarray,
    small_propagation_delay_sum: np.ndarray,
    cost_sum: np.ndarray,
    cost_normal_sum: np.ndarray,
    cost_small_sum: np.ndarray,
    access_data_rate: float,
    light_speed_ms: float,
) -> int:
    delivered_count = 0
    safe_access_rate = access_data_rate if access_data_rate > 1e-9 else 1e-9
    safe_light_speed = light_speed_ms if light_speed_ms > 1e-9 else 1e-9
    for row in range(deliverable.shape[0]):
        if not deliverable[row]:
            continue
        env_id = int(env_ids[row])
        flowlet_id = int(flowlet_ids[row])
        if status[env_id, flowlet_id] == FLOWLET_DELIVERED or status[env_id, flowlet_id] == FLOWLET_DROPPED:
            continue
        sat_id = int(current_sat[env_id, flowlet_id])
        region_id = int(target_region_id[env_id, flowlet_id])
        distance = float(region_sat_distances[env_id, region_id, sat_id])
        final_prop_delay = distance / safe_light_speed
        final_tx_delay = float(size[env_id, flowlet_id]) / safe_access_rate
        final_delay = final_prop_delay + final_tx_delay
        final_access_delay[env_id, flowlet_id] = final_delay
        propagation_delay[env_id, flowlet_id] += final_prop_delay
        transmission_delay[env_id, flowlet_id] += final_tx_delay
        delivery_time[env_id, flowlet_id] = float(current_times[env_id]) + final_delay

        weight = int(packet_count[env_id, flowlet_id])
        weight_f = float(weight)
        total_delay = (
            float(queue_delay[env_id, flowlet_id])
            + float(transmission_delay[env_id, flowlet_id])
            + float(propagation_delay[env_id, flowlet_id])
        )
        q_delay = float(queue_delay[env_id, flowlet_id])
        tx_delay = float(transmission_delay[env_id, flowlet_id])
        prop_delay = float(propagation_delay[env_id, flowlet_id])
        cost = float(total_queue_cost[env_id, flowlet_id])

        delivered[env_id] += weight
        delivered_mbit[env_id] += float(size[env_id, flowlet_id])
        e2e_delay_sum[env_id] += total_delay * weight_f
        queue_delay_sum[env_id] += q_delay * weight_f
        transmission_delay_sum[env_id] += tx_delay * weight_f
        propagation_delay_sum[env_id] += prop_delay * weight_f
        cost_sum[env_id] += cost * weight_f
        if is_normal[env_id, flowlet_id]:
            delivered_normal[env_id] += weight
            normal_e2e_delay_sum[env_id] += total_delay * weight_f
            normal_queue_delay_sum[env_id] += q_delay * weight_f
            normal_transmission_delay_sum[env_id] += tx_delay * weight_f
            normal_propagation_delay_sum[env_id] += prop_delay * weight_f
            cost_normal_sum[env_id] += cost * weight_f
        else:
            delivered_small[env_id] += weight
            small_e2e_delay_sum[env_id] += total_delay * weight_f
            small_queue_delay_sum[env_id] += q_delay * weight_f
            small_transmission_delay_sum[env_id] += tx_delay * weight_f
            small_propagation_delay_sum[env_id] += prop_delay * weight_f
            cost_small_sum[env_id] += cost * weight_f
        status[env_id, flowlet_id] = FLOWLET_DELIVERED
        delivered_count += 1
    return delivered_count


@njit(cache=True)
def build_legacy_observations_kernel(
    obs: np.ndarray,
    rows: np.ndarray,
    env_ids: np.ndarray,
    flowlet_ids: np.ndarray,
    current_sats: np.ndarray,
    target_regions: np.ndarray,
    neighbor_sat_ids: np.ndarray,
    neighbor_link_ids: np.ndarray,
    remaining_gcd: np.ndarray,
    node_state: np.ndarray,
    link_state: np.ndarray,
    region_positions: np.ndarray,
    region_sat_distances: np.ndarray,
    region_sat_gcd_degrees: np.ndarray,
    current_times: np.ndarray,
    creation_time: np.ndarray,
    is_normal: np.ndarray,
    size: np.ndarray,
    ttl: np.ndarray,
    queue_delay: np.ndarray,
    transmission_delay: np.ndarray,
    propagation_delay: np.ndarray,
    initial_gcd: np.ndarray,
    last_action1: np.ndarray,
    last_action2: np.ndarray,
    last_node1: np.ndarray,
    last_node2: np.ndarray,
    delay_norm: float,
    default_ttl: float,
    orbit_radius: float,
    orbit_cycle_ms: float,
    max_access_range: float,
) -> None:
    safe_delay_norm = delay_norm if delay_norm > 1e-9 else 1e-9
    safe_default_ttl = default_ttl if default_ttl > 1e-6 else 1e-6
    safe_orbit_cycle_ms = orbit_cycle_ms if orbit_cycle_ms > 1e-9 else 1e-9
    safe_orbit_radius = orbit_radius if orbit_radius > 1e-9 else 1e-9
    for idx in range(rows.shape[0]):
        row = int(rows[idx])
        env_id = int(env_ids[row])
        flowlet_id = int(flowlet_ids[row])
        current_sat = int(current_sats[row])
        target_region = int(target_regions[row])

        current_x = node_state[env_id, current_sat, NODE_POS_X]
        current_y = node_state[env_id, current_sat, NODE_POS_Y]
        current_z = node_state[env_id, current_sat, NODE_POS_Z]
        target_x = region_positions[target_region, 0] / safe_orbit_radius
        target_y = region_positions[target_region, 1] / safe_orbit_radius
        target_z = region_positions[target_region, 2] / safe_orbit_radius
        rel_x = current_x - target_x
        rel_y = current_y - target_y
        rel_z = current_z - target_z
        rel_dist = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
        time_prog = (current_times[env_id] % safe_orbit_cycle_ms) / safe_orbit_cycle_ms
        init_gcd = float(initial_gcd[env_id, flowlet_id])
        if init_gcd < 1e-6:
            init_gcd = 1e-6
        current_progress = remaining_gcd[row] / init_gcd
        current_load = node_state[env_id, current_sat, NODE_QUEUE_LOAD]
        current_remaining = node_state[env_id, current_sat, NODE_QUEUE_REMAINING]
        age = (current_times[env_id] - creation_time[env_id, flowlet_id]) / safe_delay_norm
        ttl_value = float(ttl[env_id, flowlet_id])
        total_delay = (
            float(queue_delay[env_id, flowlet_id])
            + float(transmission_delay[env_id, flowlet_id])
            + float(propagation_delay[env_id, flowlet_id])
        )

        obs[row, 0] = time_prog
        obs[row, 1] = current_x
        obs[row, 2] = current_y
        obs[row, 3] = current_z
        obs[row, 4] = target_x
        obs[row, 5] = target_y
        obs[row, 6] = target_z
        obs[row, 7] = rel_x
        obs[row, 8] = rel_y
        obs[row, 9] = rel_z
        obs[row, 10] = rel_dist
        obs[row, 11] = current_progress
        obs[row, 12] = current_load
        obs[row, 13] = current_remaining
        obs[row, 14] = age
        obs[row, 15] = 1.0 if is_normal[env_id, flowlet_id] else 0.0
        obs[row, 16] = size[env_id, flowlet_id]
        obs[row, 17] = ttl_value
        obs[row, 18] = default_ttl - ttl_value
        obs[row, 19] = ttl_value / safe_default_ttl
        obs[row, 20] = total_delay / safe_delay_norm
        obs[row, 21] = queue_delay[env_id, flowlet_id] / safe_delay_norm
        obs[row, 22] = last_action1[env_id, flowlet_id]
        obs[row, 23] = last_node1[env_id, flowlet_id]
        obs[row, 24] = last_action2[env_id, flowlet_id]
        obs[row, 25] = last_node2[env_id, flowlet_id]

        flowlet_size = float(size[env_id, flowlet_id])
        last_1 = int(last_node1[env_id, flowlet_id])
        last_2 = int(last_node2[env_id, flowlet_id])
        cursor = 26
        for action_idx in range(ACTION_COUNT):
            neighbor_sat = int(neighbor_sat_ids[row, action_idx])
            neighbor_link = int(neighbor_link_ids[row, action_idx])
            if neighbor_sat >= 0:
                n_x = node_state[env_id, neighbor_sat, NODE_POS_X]
                n_y = node_state[env_id, neighbor_sat, NODE_POS_Y]
                n_z = node_state[env_id, neighbor_sat, NODE_POS_Z]
                n_rel_x = n_x - target_x
                n_rel_y = n_y - target_y
                n_rel_z = n_z - target_z
                n_rel_dist = math.sqrt(n_rel_x * n_rel_x + n_rel_y * n_rel_y + n_rel_z * n_rel_z)
                neighbor_gcd = region_sat_gcd_degrees[env_id, target_region, neighbor_sat]
                neighbor_progress = neighbor_gcd / init_gcd
                sink_load = node_state[env_id, neighbor_sat, NODE_QUEUE_LOAD]
                sink_remaining = node_state[env_id, neighbor_sat, NODE_QUEUE_REMAINING]
                looped = 1.0 if neighbor_sat == last_1 or neighbor_sat == last_2 else 0.0
                target_distance = region_sat_distances[env_id, target_region, neighbor_sat]
                target_access = 1.0 if target_distance <= max_access_range else 0.0
            else:
                n_x = 0.0
                n_y = 0.0
                n_z = 0.0
                n_rel_x = 0.0
                n_rel_y = 0.0
                n_rel_z = 0.0
                n_rel_dist = 0.0
                neighbor_progress = 0.0
                sink_load = 0.0
                sink_remaining = 0.0
                looped = 0.0
                target_access = 0.0

            if neighbor_link >= 0:
                link_remaining = link_state[env_id, neighbor_link, LINK_QUEUE_REMAINING]
                link_data_rate = link_state[env_id, neighbor_link, LINK_DATA_RATE]
                if link_data_rate < 1e-9:
                    link_data_rate = 1e-9
                norm_queue_delay = link_state[env_id, neighbor_link, LINK_FREE_TIME_DELTA] / safe_delay_norm
                norm_tx_time = flowlet_size / link_data_rate / safe_delay_norm
                norm_prop_delay = link_state[env_id, neighbor_link, LINK_DELAY] / safe_delay_norm
                enough_capacity = 1.0 if link_remaining >= flowlet_size else 0.0
            else:
                link_remaining = 0.0
                norm_queue_delay = 0.0
                norm_tx_time = 0.0
                norm_prop_delay = 0.0
                enough_capacity = 0.0

            obs[row, cursor] = n_x
            obs[row, cursor + 1] = n_y
            obs[row, cursor + 2] = n_z
            obs[row, cursor + 3] = n_rel_x
            obs[row, cursor + 4] = n_rel_y
            obs[row, cursor + 5] = n_rel_z
            obs[row, cursor + 6] = n_rel_dist
            obs[row, cursor + 7] = neighbor_progress
            obs[row, cursor + 8] = norm_queue_delay
            obs[row, cursor + 9] = norm_tx_time
            obs[row, cursor + 10] = norm_prop_delay
            obs[row, cursor + 11] = sink_load
            obs[row, cursor + 12] = sink_remaining
            obs[row, cursor + 13] = link_remaining
            obs[row, cursor + 14] = enough_capacity
            obs[row, cursor + 15] = looped
            obs[row, cursor + 16] = target_access
            cursor += 17
