#!/usr/bin/env python3
"""
Static visualization of global network statistics showing path distribution.
Displays node and link density based on total packets received over entire simulation.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
from datetime import datetime
import numpy as np
import argparse

from sat_net.routing_env import RoutingEnvAsync
from sat_net.solver import BaseSolver, create_solver
from sat_net.util import NamedDict


def load_solver_from(env, saved_path: str):
    """
    Loads a trained solver from a specified path, or creates SPF if path is "SPF".
    """
    # Check if using SPF (Shortest Path First)
    if saved_path.upper() == "SPF":
        from sat_net.solver.spf import SPF
        print("Using SPF (Shortest Path First) solver")
        solver = SPF()
        return solver
    
    # Load trained solver
    solver_config = NamedDict.load(f"{saved_path}/solver_config.json")
    solver = create_solver(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        solver_config=solver_config,
        tf_writer=None,
    )
    solver.load_models(f"{saved_path}/models/best_model")
    solver.set_eval()
    return solver


def plot_global_path_distribution(
    env: RoutingEnvAsync,
    solver: BaseSolver,
    eval_seed: int,
    output_dir: str = "figs/path_distribution",
):
    """
    Create a static visualization showing global path distribution.
    
    Visualizes node and link density as:
    density = max_load_factor (maximum load factor experienced during simulation)
    
    Args:
        env: The routing environment
        solver: The trained solver to evaluate
        eval_seed: Random seed for evaluation
        output_dir: Directory to save the figure
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # For nodes: Always use 1.0 as max (max_load_factor is normalized to [0, 1])
    node_density_max = 1.0
    # For links: Normalize queueing delay using reference value
    max_delay_reference = 20.0  # milliseconds (delays >= 10ms will be capped at 1.0)
    
    # Run simulation
    print("Starting simulation to collect global statistics...")
    env.reset(seed=eval_seed, start_time=0)
    solver.set_eval()
    
    # Run simulation
    env.run(solver)
    
    print("Simulation completed. Processing statistics...")
    
    # Get total packets generated
    total_packets_generated = env.stats.num_packets_generated
    
    if total_packets_generated == 0:
        print("No packets generated. Cannot create visualization.")
        return
    
    print(f"Total packets generated: {total_packets_generated}")
    
    # Collect node statistics (only nodes with max_load_factor > 0)
    node_stats = []
    for node in env.network.nodes.values():
        if node.num_packet_recv > 0:  # Skip unused nodes
            density = node.max_load_factor  # Use max_load_factor as density
            longitude, latitude = node.get_projected_position()
            node_stats.append({
                'id': node.id,
                'name': node.name,
                'is_satellite': node.is_satellite(),
                'longitude': longitude,
                'latitude': latitude,
                'density': density,
                'num_packets_recv': node.num_packet_recv,
                'num_packets_sent': node.num_packet_sent,
                'num_packets_dropped': node.num_packet_dropped,
            })
    
    # Collect link statistics (only links with max_queueing_delay > 0)
    link_stats = []
    for link in env.network.links.values():
        if link.is_connected and link.num_packet_recv > 0:  # Skip unused links
            # Use max_queueing_delay as density metric for links
            source_lon, source_lat = link.source.get_projected_position()
            sink_lon, sink_lat = link.sink.get_projected_position()
            link_stats.append({
                'source_id': link.source.id,
                'sink_id': link.sink.id,
                'source_name': link.source.name,
                'sink_name': link.sink.name,
                'source_lon': source_lon,
                'source_lat': source_lat,
                'sink_lon': sink_lon,
                'sink_lat': sink_lat,
                'max_queueing_delay': link.max_queueing_delay,  # Store raw delay in ms
                'num_packet_recv': link.num_packet_recv,
                'num_packet_sent': link.num_packet_sent,
                'num_packet_dropped': link.num_packet_dropped,
            })
    
    # Calculate density ranges
    node_densities = [n['density'] for n in node_stats]
    link_queueing_delays = [l['max_queueing_delay'] for l in link_stats]
    

    if node_densities:
        actual_node_max = max(node_densities)
        print(f"Actual max node load factor: {actual_node_max:.4f}")
    

    if link_queueing_delays:
        actual_link_max_delay = max(link_queueing_delays)
        print(f"Actual max link queueing delay: {actual_link_max_delay:.2f}ms")
        
        # Normalize delays and cap at 1.0 for consistent colormap with nodes
        link_densities = [min(d / max_delay_reference, 1.0) for d in link_queueing_delays]
        actual_link_density_max = max(link_densities)
        print(f"Max normalized link density: {actual_link_density_max:.2f} (capped at 1.0)")
    else:
        link_densities = []
        actual_link_density_max = 1.0
    
    # Always use 1.0 as max for consistent colormap
    link_density_max = 1.0
    
    # Create visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    # Set fixed geographical extent: 60W-180E, 30S-90N
    ax.set_extent([-60, 180, -15, 90], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Separate satellites and ground stations first
    satellites = [n for n in node_stats if n['is_satellite']]
    ground_stations = [n for n in node_stats if not n['is_satellite']]
    
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    # Use 'YlOrRd' (Yellow-Orange-Red) - light colors for low density, dark for high
    # This is intuitive: light = low traffic, dark/red = high traffic
    # Apply power transform to make transition faster (nonlinear mapping)
    base_cmap = cm.get_cmap('YlOrRd')
    
    # Create a nonlinear colormap that transitions faster from yellow to red
    # Using x^0.5 (square root) makes the transition faster
    colors = base_cmap(np.linspace(0, 1, 256) ** 0.5)
    density_cmap = LinearSegmentedColormap.from_list('YlOrRd_fast', colors)
    
    # LAYER 1: Draw ground stations first (bottom layer)
    print(f"Drawing {len(ground_stations)} ground stations...")
    if ground_stations:
        gs_lons = [gs['longitude'] for gs in ground_stations]
        gs_lats = [gs['latitude'] for gs in ground_stations]
        
        # Ground stations use fixed blue color (not part of routing density)
        ax.scatter(gs_lons, gs_lats,
                  c='dodgerblue',
                  s=50,
                  marker='^',
                  edgecolors='black',
                  linewidth=2.0,
                  alpha=1.0,
                  transform=ccrs.PlateCarree(),
                  zorder=1,
                  label='Ground Stations')
    
    # LAYER 2: Draw satellites (middle layer)
    print(f"Drawing {len(satellites)} satellites...")
    if satellites:
        sat_lons = np.array([s['longitude'] for s in satellites])
        sat_lats = np.array([s['latitude'] for s in satellites])
        sat_densities = np.array([s['density'] for s in satellites])
        
        # Calculate properties (vectorized)
        density_ratios = sat_densities / node_density_max if node_density_max > 0 else np.zeros_like(sat_densities)
        sizes = 30
        
        # Use YlOrRd colormap: light yellow (low) -> dark red (high)
        colors = density_cmap(density_ratios)
        
        # Edge colors: always black with full opacity
        edge_colors = 'black'
        edge_widths = 0.5
        
        # Batch render all satellites
        ax.scatter(sat_lons, sat_lats, 
                  s=sizes,
                  c=colors,
                  marker='o',
                  edgecolors=edge_colors,
                  linewidths=edge_widths,
                  alpha=1.0,  # Fully opaque
                  transform=ccrs.PlateCarree(),
                  zorder=3)
    
    # LAYER 3: Draw undirected links (merge bidirectional pairs)
    print(f"Drawing undirected links from {len(link_stats)} directional links...")
    
    # Group bidirectional links and take max delay
    link_pairs = {}  # key: (min_id, max_id), value: {'delay': max_delay, 'positions': ..., 'stats': ...}
    for i, link in enumerate(link_stats):
        src_id = link['source_id']
        sink_id = link['sink_id']
        pair_key = (min(src_id, sink_id), max(src_id, sink_id))
        
        if pair_key not in link_pairs:
            link_pairs[pair_key] = {
                'max_delay': link['max_queueing_delay'],
                'source_lon': link['source_lon'],
                'source_lat': link['source_lat'],
                'sink_lon': link['sink_lon'],
                'sink_lat': link['sink_lat'],
                'source_name': link['source_name'] if src_id < sink_id else link['sink_name'],
                'sink_name': link['sink_name'] if src_id < sink_id else link['source_name'],
                'num_packet_recv': link['num_packet_recv'],
                'num_packet_sent': link['num_packet_sent'],
                'num_packet_dropped': link['num_packet_dropped'],
            }
        else:
            # Take the maximum delay between both directions
            link_pairs[pair_key]['max_delay'] = max(link_pairs[pair_key]['max_delay'], link['max_queueing_delay'])
            # Accumulate packet counts
            link_pairs[pair_key]['num_packet_recv'] += link['num_packet_recv']
            link_pairs[pair_key]['num_packet_sent'] += link['num_packet_sent']
            link_pairs[pair_key]['num_packet_dropped'] += link['num_packet_dropped']
    
    print(f"Merged into {len(link_pairs)} undirected link pairs")
    
    # Normalize delays for the merged pairs
    merged_delays = [info['max_delay'] for info in link_pairs.values()]
    merged_link_densities = [d / max_delay_reference for d in merged_delays]
    
    # Draw the merged undirected links
    for pair_idx, (pair_key, link_info) in enumerate(link_pairs.items()):
        # Get normalized density
        density_ratio = merged_link_densities[pair_idx]
        
        # Normalize to [0, 1] for colormap
        density_ratio_clamped = min(density_ratio / link_density_max, 1.0) if link_density_max > 0 else 0
        
        # Use YlOrRd colormap
        color = density_cmap(density_ratio_clamped)
        
        # Line width
        linewidth = 2.0
        
        src_lon = link_info['source_lon']
        src_lat = link_info['source_lat']
        sink_lon = link_info['sink_lon']
        sink_lat = link_info['sink_lat']
        
        # Draw gray stroke (outline) first for visibility
        ax.plot([src_lon, sink_lon], 
               [src_lat, sink_lat], 
               color='gray', linewidth=linewidth + 0.5, alpha=0.7,
               transform=ccrs.PlateCarree(), zorder=2)
        
        # Draw colored link on top
        ax.plot([src_lon, sink_lon], 
               [src_lat, sink_lat], 
               color=color, linewidth=linewidth, alpha=1.0,
               transform=ccrs.PlateCarree(), zorder=2)
    
    print(f"Drew {len(link_pairs)} undirected links")
    
    # Add colorbar for visualization
    if satellites or link_stats:
        # Use the max of node and link densities for consistent colormap
        colorbar_max = max(node_density_max, link_density_max)
        dummy_scatter = ax.scatter([], [], c=[], cmap=density_cmap, vmin=0, vmax=colorbar_max)
        cbar = plt.colorbar(dummy_scatter, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        
        # Fixed label with consistent scale
        cbar.set_label(f'Load/Delay Factor (Node: 0-1, Link: 0-{max_delay_reference}ms)', fontsize=10)
    
    # Add legend with proper colors
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='dodgerblue', 
                   markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Ground Stations (n={len(ground_stations)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#fee08b',  # Light yellow
                   markersize=10, markeredgecolor='black', markeredgewidth=0.5,
                   label=f'Satellites (n={len(satellites)})'),
        plt.Line2D([0], [0], color='#fee08b', linewidth=2,  # Light yellow
                   label=f'Links (n={len(link_pairs)} undirected)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Title with statistics
    actual_node_max = max(node_densities) if node_densities else 0.0
    actual_link_max_delay = max(link_queueing_delays) if link_queueing_delays else 0.0
    title = (f"Max Load/Queueing Delay ({solver.name})")
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Save high-resolution figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(output_dir, f"{solver.name}_{timestamp}.png")
    
    print(f"Saving high-resolution figure to {fig_path}...")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("Figure saved successfully!")
    
    plt.close(fig)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total packets generated: {total_packets_generated:,}")
    
    # Top 5 busiest nodes
    sorted_nodes = sorted(node_stats, key=lambda x: x['density'], reverse=True)
    print("\nTop 5 busiest nodes (by max load factor):")
    for i, node in enumerate(sorted_nodes[:5], 1):
        node_type = "Satellite" if node['is_satellite'] else "Ground Station"
        print(f"  {i}. {node['name']} ({node_type})")
        print(f"     Max Load: {node['density']:.4f} | Recv: {node['num_packets_recv']:,} | "
              f"Sent: {node['num_packets_sent']:,} | Dropped: {node['num_packets_dropped']:,}")
    
    # Top 5 busiest links (using merged undirected pairs)
    sorted_link_pairs = sorted(link_pairs.items(), key=lambda x: x[1]['max_delay'], reverse=True)
    print("\nTop 5 busiest links (by max queueing delay, undirected):")
    for i, (pair_key, link_info) in enumerate(sorted_link_pairs[:5], 1):
        print(f"  {i}. {link_info['source_name']} <-> {link_info['sink_name']}")
        print(f"     Max Queue Delay: {link_info['max_delay']:.2f}ms | Recv: {link_info['num_packet_recv']:,} | "
              f"Sent: {link_info['num_packet_sent']:,} | Dropped: {link_info['num_packet_dropped']:,}")
    
    return fig_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot global path distribution statistics for satellite network"
    )
    parser.add_argument(
        "--config",
        default="configs/starlink_dvbs2_test.json",
        help="Environment config file",
    )
    parser.add_argument(
        "--solver_path",
        required=True,
        help="Path to trained solver (or 'SPF' for Shortest Path First)",
    )
    parser.add_argument("--eval_seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument(
        "--output_dir",
        default="figs/path_distribution",
        help="Output directory for figure",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = NamedDict.load(args.config)
    
    # Create environment
    print("Creating environment...")
    env = RoutingEnvAsync(config)
    
    # Load solver
    print("Loading solver...")
    solver = load_solver_from(env, args.solver_path)
    
    # Generate path distribution visualization
    plot_global_path_distribution(env, solver, args.eval_seed, args.output_dir)


if __name__ == "__main__":
    main()

