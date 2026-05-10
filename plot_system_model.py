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
            "link_buffer_size": 16.0,
            "isl_data_rate": 0.05,
        }
    return NamedDict(config)

def create_network(network_config):
    """Create satellite network from config."""
    network = SatelliteNetwork(
        altitude=network_config.altitude,
        inclination=network_config.inclination,
        num_orbits=network_config.num_orbits,
        num_sats_per_orbit=network_config.num_sats_per_orbit,
        phasing=network_config.phasing,
        min_elevation_angle_deg=network_config.min_elevation_angle_deg,
        link_buffer_size=network_config.link_buffer_size,
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
    sat_ids = network.satellite_ids.copy()
    sat_positions = network.node_positions[sat_ids] / 1000.0

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
    visible_ids = sat_ids[visible_mask]

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
    central_sat_id = None
    if len(visible_positions) > 0:
        # Find satellite with maximum projection along view direction (closest to viewport center)
        projections = np.dot(visible_positions, view_direction)
        central_idx = np.argmax(projections)
        central_sat_id = int(visible_ids[central_idx])
        central_pos = visible_positions[central_idx]

        # Highlight central satellite
        ax.scatter([central_pos[0]], [central_pos[1]], [central_pos[2]],
                   c='red', s=200, alpha=1.0, edgecolors='darkred', linewidth=2, zorder=5)

        print(f"Selected central satellite: {central_sat_id}")
        print(f"Position: ({central_pos[0]:.2f}, {central_pos[1]:.2f}, {central_pos[2]:.2f})")
        print(f"View direction projection: {projections[central_idx]:.2f}")

        # Plot ISL connections from central satellite
        connections_plotted = {'N': False, 'S': False, 'E': False, 'W': False}

        def plot_connection(label, neighbor_id, color, line_style):
            if neighbor_id < 0:
                return
            neighbor_pos = network.node_positions[int(neighbor_id)] / 1000.0
            # Only show connection if target satellite is also visible
            if np.dot(neighbor_pos, view_direction) <= 0:
                return
            ax.plot([central_pos[0], neighbor_pos[0]],
                    [central_pos[1], neighbor_pos[1]],
                    [central_pos[2], neighbor_pos[2]],
                    line_style, linewidth=6, alpha=1.0, label=label, zorder=4)
            # Draw connection label at midpoint for better visibility
            mid_pos = (central_pos + neighbor_pos) / 2
            ax.text(mid_pos[0]*1.15, mid_pos[1]*1.15, mid_pos[2]*1.15, label[0],
                    fontsize=20, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=6)
            connections_plotted[label[0]] = True

        plot_connection('North', int(network.isl_n[central_sat_id]), 'green', 'g-')
        plot_connection('South', int(network.isl_s[central_sat_id]), 'green', 'g-')
        plot_connection('East', int(network.isl_e[central_sat_id]), 'blue', 'b-')
        plot_connection('West', int(network.isl_w[central_sat_id]), 'blue', 'b-')

    # Plot orbital lines using ISL_N connections (lighter) - only visible ones
    for sat_id in network.satellite_ids:
        sat_id = int(sat_id)
        sat1_pos = network.node_positions[sat_id] / 1000.0
        # Only show connections if source satellite is visible and not the central satellite
        if np.dot(sat1_pos, view_direction) > 0 and sat_id != central_sat_id:
            # Plot only North connections to show orbital structure
            next_sat_id = int(network.isl_n[sat_id])
            if next_sat_id >= 0:
                sat2_pos = network.node_positions[next_sat_id] / 1000.0
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

    print(f"\n=== ISL Topology ===")
    print(f"North connections: {int(np.count_nonzero(network.isl_n >= 0))}")
    print(f"South connections: {int(np.count_nonzero(network.isl_s >= 0))}")
    print(f"East connections: {int(np.count_nonzero(network.isl_e >= 0))}")
    print(f"West connections: {int(np.count_nonzero(network.isl_w >= 0))}")

    # Show example connections for first satellite
    if network.num_satellites:
        first_sat_id = int(network.satellite_ids[0])
        print(f"\nExample satellite {first_sat_id} connections:")
        for label, neighbor_id in (
            ("North", int(network.isl_n[first_sat_id])),
            ("South", int(network.isl_s[first_sat_id])),
            ("East", int(network.isl_e[first_sat_id])),
            ("West", int(network.isl_w[first_sat_id])),
        ):
            if neighbor_id >= 0:
                print(f"  {label} -> Satellite {neighbor_id}")

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
