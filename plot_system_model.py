#!/usr/bin/env python3
"""
Script to generate system model figure using the actual satellite network implementation.
Shows Walker-Delta constellation with 4-directional Inter-Satellite Links (N, S, E, W).
"""

import os
# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

from sat_net.network import SatelliteNetwork
from sat_net.util import NamedDict

# Set matplotlib to use Type 1 fonts for EPS
plt.rcParams["ps.useafm"] = True
plt.rcParams["pdf.use14corefonts"] = True
plt.rcParams["text.usetex"] = False

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return NamedDict(json.load(f))

def create_small_constellation_config():
    """Create a small constellation config for system model visualization."""
    config = {
            "altitude": 600,
            "inclination": 53,
            "num_orbits": 20,  # Small constellation for clarity
            "num_sats_per_orbit": 20,  # Small constellation for clarity
            "phasing": 3,
            "min_elevation_angle_deg": 15,
            "max_gsl_per_gs": 2,
            "max_gsl_per_sat": 2,
            "node_buffer_size": 16.0,
            "link_buffer_size": 16.0,
            "gsl_data_rate": 1.0,
            "isl_data_rate": 0.05,
            "ground_stations": [
                {
                    "name": "Luxembourg",
                    "latitude": 49.6116,
                    "longitude": 6.1319,
                    "population": 10
                },
                {
                    "name": "Dubai",
                    "latitude": 25.2769,
                    "longitude": 55.2962,
                    "population": 10
                },
                {
                    "name": "Beijing",
                    "latitude": 39.9087,
                    "longitude": 116.3975,
                    "population": 10
                }
            ]
        }
    return NamedDict(config)

def create_network(network_config):
    """Create satellite network from config."""
    network = SatelliteNetwork(
        ground_stations=network_config.ground_stations,
        altitude=network_config.altitude,
        inclination=network_config.inclination,
        num_orbits=network_config.num_orbits,
        num_sats_per_orbit=network_config.num_sats_per_orbit,
        phasing=network_config.phasing,
        min_elevation_angle_deg=network_config.min_elevation_angle_deg,
        max_gsl_per_gs=network_config.max_gsl_per_gs,
        max_gsl_per_sat=network_config.max_gsl_per_sat,
        node_buffer_size=network_config.node_buffer_size,
        link_buffer_size=network_config.link_buffer_size,
        gsl_data_rate=network_config.gsl_data_rate,
        isl_data_rate=network_config.isl_data_rate,
    )
    return network

def plot_3d_constellation(network, timestamp=0):
    """Plot 3D constellation showing ISL topology."""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Update network topology
    network.update_topology(timestamp)

    # Get satellite positions
    sat_positions = []
    sat_ids = []
    for sat in network.satellites.values():
        pos = sat.position / 1000  # Convert to Earth radii for visualization
        sat_positions.append(pos)
        sat_ids.append(sat.id)

    sat_positions = np.array(sat_positions)

    # Calculate Earth radius for occlusion
    earth_radius = 6371 / 1000  # Normalized Earth radius

    # Determine which satellites are visible (not occluded by Earth)
    # Calculate distance from Earth center and check if satellite is on the near side
    elev_rad = np.radians(25)
    azim_rad = np.radians(25)
    view_direction = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])

    # Check which satellites are visible (dot product with view direction > 0 means facing viewer)
    visible_mask = np.dot(sat_positions, view_direction) > 0
    visible_positions = sat_positions[visible_mask]
    visible_ids = [sat_ids[i] for i in range(len(sat_ids)) if visible_mask[i]]

    # Plot Earth sphere with solid light cyan color (no mesh lines)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth,
                   alpha=0.9, color='cyan', zorder=1,
                   linewidth=0, antialiased=True, shade=True)

    # Plot only visible satellites
    if len(visible_positions) > 0:
        ax.scatter(visible_positions[:, 0], visible_positions[:, 1], visible_positions[:, 2],
                   c='darkblue', s=50, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=2)

    # Find a visible satellite closest to the viewport center for highlighting
    if len(visible_positions) > 0:
        # Find satellite with maximum projection along view direction (closest to viewport center)
        projections = np.dot(visible_positions, view_direction)
        central_idx = np.argmax(projections)
        central_sat_id = visible_ids[central_idx]
        central_pos = visible_positions[central_idx]

        # Highlight central satellite
        ax.scatter([central_pos[0]], [central_pos[1]], [central_pos[2]],
                   c='red', s=200, alpha=1.0, edgecolors='darkred', linewidth=2, zorder=5)

        print(f"Selected central satellite: {central_sat_id}")
        print(f"Position: ({central_pos[0]:.2f}, {central_pos[1]:.2f}, {central_pos[2]:.2f})")
        print(f"View direction projection: {projections[central_idx]:.2f}")

        # Plot ISL connections from central satellite
        connections_plotted = {'N': False, 'S': False, 'E': False, 'W': False}

        # North connection
        if central_sat_id in network.ISL_N:
            north_sat_id = network.ISL_N[central_sat_id]
            if north_sat_id in network.satellites:
                north_sat = network.satellites[north_sat_id]
                north_pos = north_sat.position / 1000
                # Only show connection if target satellite is also visible
                if np.dot(north_pos, view_direction) > 0:
                    ax.plot([central_pos[0], north_pos[0]],
                            [central_pos[1], north_pos[1]],
                            [central_pos[2], north_pos[2]],
                            'g-', linewidth=6, alpha=1.0, label='North', zorder=4)
                    # Draw connection label at midpoint for better visibility
                    mid_pos = (central_pos + north_pos) / 2
                    ax.text(mid_pos[0]*1.15, mid_pos[1]*1.15, mid_pos[2]*1.15, 'N',
                            fontsize=20, fontweight='bold', color='green',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=6)
                    connections_plotted['N'] = True

        # South connection
        if central_sat_id in network.ISL_S:
            south_sat_id = network.ISL_S[central_sat_id]
            if south_sat_id in network.satellites:
                south_sat = network.satellites[south_sat_id]
                south_pos = south_sat.position / 1000
                # Only show connection if target satellite is also visible
                if np.dot(south_pos, view_direction) > 0:
                    ax.plot([central_pos[0], south_pos[0]],
                            [central_pos[1], south_pos[1]],
                            [central_pos[2], south_pos[2]],
                            'g-', linewidth=6, alpha=1.0, label='South', zorder=4)
                    # Draw connection label at midpoint for better visibility
                    mid_pos = (central_pos + south_pos) / 2
                    ax.text(mid_pos[0]*1.15, mid_pos[1]*1.15, mid_pos[2]*1.15, 'S',
                            fontsize=20, fontweight='bold', color='green',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=6)
                    connections_plotted['S'] = True

        # East connection
        if central_sat_id in network.ISL_E:
            east_sat_id = network.ISL_E[central_sat_id]
            if east_sat_id in network.satellites:
                east_sat = network.satellites[east_sat_id]
                east_pos = east_sat.position / 1000
                # Only show connection if target satellite is also visible
                if np.dot(east_pos, view_direction) > 0:
                    ax.plot([central_pos[0], east_pos[0]],
                            [central_pos[1], east_pos[1]],
                            [central_pos[2], east_pos[2]],
                            'b-', linewidth=6, alpha=1.0, label='East', zorder=4)
                    # Draw connection label at midpoint for better visibility
                    mid_pos = (central_pos + east_pos) / 2
                    ax.text(mid_pos[0]*1.15, mid_pos[1]*1.15, mid_pos[2]*1.15, 'E',
                            fontsize=20, fontweight='bold', color='blue',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=6)
                    connections_plotted['E'] = True

        # West connection
        if central_sat_id in network.ISL_W:
            west_sat_id = network.ISL_W[central_sat_id]
            if west_sat_id in network.satellites:
                west_sat = network.satellites[west_sat_id]
                west_pos = west_sat.position / 1000
                # Only show connection if target satellite is also visible
                if np.dot(west_pos, view_direction) > 0:
                    ax.plot([central_pos[0], west_pos[0]],
                            [central_pos[1], west_pos[1]],
                            [central_pos[2], west_pos[2]],
                            'b-', linewidth=6, alpha=1.0, label='West', zorder=4)
                    # Draw connection label at midpoint for better visibility
                    mid_pos = (central_pos + west_pos) / 2
                    ax.text(mid_pos[0]*1.15, mid_pos[1]*1.15, mid_pos[2]*1.15, 'W',
                            fontsize=20, fontweight='bold', color='blue',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=6)
                    connections_plotted['W'] = True

    # Plot orbital lines using ISL_N connections (lighter) - only visible ones
    for sat_id in network.satellites:
        sat1_pos = network.satellites[sat_id].position / 1000
        # Only show connections if source satellite is visible and not the central satellite
        if np.dot(sat1_pos, view_direction) > 0 and sat_id != central_sat_id:
            # Plot only North connections to show orbital structure
            if sat_id in network.ISL_N:
                next_sat_id = network.ISL_N[sat_id]
                if next_sat_id in network.satellites:
                    sat2_pos = network.satellites[next_sat_id].position / 1000
                    # Only show if target satellite is also visible
                    if np.dot(sat2_pos, view_direction) > 0:
                        ax.plot([sat1_pos[0], sat2_pos[0]],
                                [sat1_pos[1], sat2_pos[1]],
                                [sat1_pos[2], sat2_pos[2]],
                                'black', alpha=0.5, linewidth=1, zorder=3)

    # Set axis properties for optimal viewing
    if len(sat_positions) > 0:
        max_range = np.max(np.abs(sat_positions)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    else:
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
    ax.set_box_aspect([1, 1, 1])

    # Completely remove axes
    ax.set_axis_off()


    # Save figures with tight layout and no padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Default view (tilted for better 3D perception)
    ax.view_init(elev=25, azim=25)
    plt.savefig('figs/system_model.eps', format='eps', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig('figs/system_model.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig('figs/system_model.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)

    print("System model figures saved")

    # Reset to default view for display
    ax.view_init(elev=25, azim=25)
    plt.show()


def print_network_info(network):
    """Print network topology information."""
    print(f"\n=== Network Configuration ===")
    print(f"Constellation: Walker-Delta")
    print(f"Altitude: {network.altitude} km")
    print(f"Inclination: {network.inclination}°")
    print(f"Orbits: {network.num_orbits}")
    print(f"Satellites per orbit: {network.num_sats_per_orbit}")
    print(f"Total satellites: {network.num_satellites}")
    print(f"Ground stations: {len(network.ground_stations)}")

    print(f"\n=== ISL Topology ===")
    print(f"North connections: {len(network.ISL_N)}")
    print(f"South connections: {len(network.ISL_S)}")
    print(f"East connections: {len(network.ISL_E)}")
    print(f"West connections: {len(network.ISL_W)}")

    # Show example connections for first satellite
    if network.satellites:
        first_sat_id = list(network.satellites.keys())[0]
        print(f"\nExample satellite {first_sat_id} connections:")
        if first_sat_id in network.ISL_N:
            print(f"  North -> Satellite {network.ISL_N[first_sat_id]}")
        if first_sat_id in network.ISL_S:
            print(f"  South -> Satellite {network.ISL_S[first_sat_id]}")
        if first_sat_id in network.ISL_E:
            print(f"  East -> Satellite {network.ISL_E[first_sat_id]}")
        if first_sat_id in network.ISL_W:
            print(f"  West -> Satellite {network.ISL_W[first_sat_id]}")

if __name__ == "__main__":
    print("Generating system model using actual satellite network implementation...")

    # Create small constellation for clear visualization
    config = create_small_constellation_config()

    # Create network
    network = create_network(config)

    # Print network info
    print_network_info(network)

    # Generate figures
    print("\n=== Generating 3D System Model ===")
    plot_3d_constellation(network)

    print(f"\nAll figures generated successfully!")
    print(f"Network demonstrates 4-directional ISL topology:")
    print(f"  - Green arrows: North/South (intra-orbit)")
    print(f"  - Blue arrows: East/West (inter-orbit)")