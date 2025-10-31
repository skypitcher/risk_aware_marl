import random

import numpy as np

from sat_net import DataBlock, GroundStation


def create_packet_dataset(
        lam: float,
        interval_ms: float,
        default_ttl: int,
        normal_packet_prob: float,
        normal_packet_delay_tolerance: float,
        small_packet_delay_tolerance: float,
        normal_packet_size: (float, float),
        small_packet_size: (float, float),
        ground_stations: list[GroundStation],
        target_ground_station_list: dict[int, list[GroundStation]]):
    packets = []
    num_packets = np.random.poisson(lam=lam * interval_ms)
    timestamp = np.random.uniform(low=0, high=interval_ms, size=num_packets)
    for i in range(num_packets):
        creation_time = float(timestamp[i])
        packet = create_packet(
            packet_id=i,
            creation_time=creation_time,
            default_ttl=default_ttl,
            normal_packet_prob=normal_packet_prob,
            normal_packet_delay_tolerance=normal_packet_delay_tolerance,
            small_packet_delay_tolerance=small_packet_delay_tolerance,
            normal_packet_size=normal_packet_size,
            small_packet_size=small_packet_size,
            ground_stations=ground_stations,
            target_ground_station_list=target_ground_station_list,
        )
        packets.append(packet)
    return packets



def create_packet(
        packet_id: int,
        creation_time: float,
        default_ttl: int,
        normal_packet_prob: float,
        normal_packet_delay_tolerance: float,
        small_packet_delay_tolerance: float,
        normal_packet_size: (float, float),
        small_packet_size: (float, float),
        ground_stations: list[GroundStation],
        target_ground_station_list: dict[int, list[GroundStation]]
):
    if not ground_stations or len(ground_stations) < 2:
        raise RuntimeError("Cannot generate data block - insufficient ground stations or weights.")

    source_gs = random.choice(ground_stations)
    target_gs = random.choice(target_ground_station_list[source_gs.id])

    assert target_gs.id != source_gs.id

    is_elephant = np.random.uniform() < normal_packet_prob
    delay_tolerance = normal_packet_delay_tolerance if is_elephant else small_packet_delay_tolerance
    if is_elephant:
        size = np.random.uniform(low=normal_packet_size[0], high=normal_packet_size[1])
    else:
        size = np.random.uniform(low=small_packet_size[0], high=small_packet_size[1])

    packet = DataBlock(
        block_id=packet_id,
        source=source_gs.id,
        target=target_gs.id,
        is_normal=is_elephant,
        size=size,
        delay_limit=delay_tolerance,
        creation_time=creation_time,
        ttl=default_ttl,
    )
    packet.initial_gcd = source_gs.get_great_circle_distance_to(target_gs)
    packet.shortest_gcd = packet.initial_gcd

    return packet