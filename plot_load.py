import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
from datetime import datetime
from tqdm import tqdm
import io
import numpy as np

from sat_net.routing_env import RoutingEnvAsync
from sat_net.solver import BaseSolver, create_solver
from sat_net.util import NamedDict


def load_solver_from(env, saved_path: str):
    """
    Loads a trained solver from a specified path.
    """
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


def create_load_distribution_gif(df: pd.DataFrame, output_dir: str):
    """
    Create an animated GIF showing load distribution dynamics on a world map.

    Args:
        df: DataFrame containing node data with time, position, and load information
        output_dir: Directory to save the GIF
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get unique time steps
    time_steps = sorted(df["time"].unique())
    if len(time_steps) < 2:
        print("Need at least 2 time steps to create animation")
        return

    # Set up color mapping for load levels
    load_min = df["load"].min()
    load_max = df["load"].max()

    # Create figure and axis
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_global()

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Ground stations scatter (always visible)
    gs_scatter = ax.scatter(
        [],
        [],
        c=[],
        s=120,
        cmap="Reds",
        vmin=load_min,
        vmax=load_max,
        marker="^",
        edgecolors="black",
        linewidth=1.5,
        alpha=1.0,  # Always visible
        transform=ccrs.PlateCarree(),
        label="Ground Stations",
    )

    # Add colorbar (use gs_scatter for reference)
    cbar = plt.colorbar(
        gs_scatter, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8
    )
    cbar.set_label("Load Factor (Red = High Load, Transparent = Low Load)", fontsize=11)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Satellites (transparency=load)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Ground Stations'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Links (transparency=load)')
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Title
    title = ax.set_title(
        "Satellite Network Load Distribution\nTime: 0.0s",
        fontsize=14,
        fontweight="bold",
    )

    def animate(frame):
        """Animation function for each frame."""
        current_time = time_steps[frame]
        current_data = df[df["time"] == current_time]

        # Clear previous drawings
        for artist in ax.collections[:]:
            if artist != gs_scatter:  # Keep ground stations
                artist.remove()

        # Separate satellites and ground stations
        satellites = current_data[current_data["is_satellite"]]
        ground_stations = current_data[~current_data["is_satellite"]]

        # Draw satellites with transparency based on load
        if not satellites.empty:
            for _, sat in satellites.iterrows():
                load_ratio = sat["load"] / load_max if load_max > 0 else 0
                alpha = 0.05 + load_ratio * 0.90
                color_intensity = max(0.3, load_ratio)
                size = 30 + load_ratio * 40
                
                if load_ratio < 0.3:
                    edgecolor = (0.7, 0.7, 0.7, 0.3)
                    edgewidth = 1.5
                else:
                    edgecolor = (0, 0, 0, alpha * 0.8)
                    edgewidth = 0.5
                
                ax.scatter(sat["longitude"], sat["latitude"], 
                          s=size,
                          c=[[color_intensity, 0, 0]],
                          alpha=alpha,
                          marker='o',
                          edgecolors=[edgecolor],
                          linewidth=edgewidth,
                          transform=ccrs.PlateCarree())

        # Update ground station scatter plot
        if not ground_stations.empty:
            gs_scatter.set_offsets(
                list(zip(ground_stations["longitude"], ground_stations["latitude"]))
            )
            gs_scatter.set_array(ground_stations["load"])
        else:
            gs_scatter.set_offsets([])
            gs_scatter.set_array([])

        # Update title
        title.set_text(
            f"Satellite Network Load Distribution\nTime: {current_time:.1f}s"
        )

        return gs_scatter, title

    # Create animation
    print(f"Creating animation with {len(time_steps)} frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(time_steps), interval=500, blit=False, repeat=True
    )

    # Save as GIF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(output_dir, f"load_distribution_{timestamp}.gif")

    print(f"Saving GIF to {gif_path}...")
    anim.save(gif_path, writer="pillow", fps=2, dpi=100)

    print(f"GIF saved successfully to {gif_path}")

    # Also save a static plot of the final frame
    static_path = os.path.join(output_dir, f"load_distribution_final_{timestamp}.png")
    plt.savefig(static_path, dpi=300, bbox_inches="tight")
    print(f"Static plot saved to {static_path}")

    plt.show()


def save_network_snapshots(network_snapshots: list, filepath: str):
    """
    Save network snapshots to a pandas-compatible format (parquet).

    Args:
        network_snapshots: List of network snapshots
        filepath: Path to save the snapshots
    """
    # Convert snapshots to pandas DataFrames
    print("Converting snapshots to pandas format...")
    nodes_data = []
    links_data = []
    
    for snapshot in tqdm(network_snapshots, desc="Processing snapshots", unit="snapshot"):
        time = snapshot["time"]
        
        # Extract node data
        for node_data in snapshot["nodes"].values():
            nodes_data.append({
                'time': time,
                'node_id': node_data['id'],
                'name': node_data['name'],
                'is_satellite': node_data['is_satellite'],
                'longitude': node_data['longitude'],
                'latitude': node_data['latitude'],
                'load': node_data['load']
            })
        
        # Extract link data
        for link_data in snapshot["links"].values():
            links_data.append({
                'time': time,
                'link_id': link_data['id'],
                'source_id': link_data['source_id'],
                'sink_id': link_data['sink_id'],
                'source_name': link_data['source_name'],
                'sink_name': link_data['sink_name'],
                'source_longitude': link_data['source_longitude'],
                'source_latitude': link_data['source_latitude'],
                'sink_longitude': link_data['sink_longitude'],
                'sink_latitude': link_data['sink_latitude'],
                'load': link_data['load'],
                'is_connected': link_data['is_connected']
            })
    
    # Create DataFrames
    print("Creating pandas DataFrames...")
    nodes_df = pd.DataFrame(nodes_data)
    links_df = pd.DataFrame(links_data)
    
    # Save to parquet files
    print("Saving to parquet files...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save nodes and links to separate files
    base_path = filepath.replace('.parquet', '')
    nodes_path = f"{base_path}_nodes.parquet"
    links_path = f"{base_path}_links.parquet"
    
    with tqdm(total=2, desc="Saving files", unit="file") as pbar:
        nodes_df.to_parquet(nodes_path, index=False)
        pbar.update(1)
        links_df.to_parquet(links_path, index=False)
        pbar.update(1)
    
    print("Network snapshots saved to:")
    print(f"  - Nodes: {nodes_path}")
    print(f"  - Links: {links_path}")
    print(f"  - Total records: {len(nodes_df)} nodes, {len(links_df)} links")


def load_network_snapshots(filepath: str) -> list:
    """
    Load network snapshots from pandas parquet files.

    Args:
        filepath: Path to load the snapshots from

    Returns:
        List of network snapshots
    """
    # Load DataFrames
    print("Loading parquet files...")
    base_path = filepath.replace('.parquet', '')
    nodes_path = f"{base_path}_nodes.parquet"
    links_path = f"{base_path}_links.parquet"
    
    with tqdm(total=2, desc="Loading files", unit="file") as pbar:
        nodes_df = pd.read_parquet(nodes_path)
        pbar.update(1)
        links_df = pd.read_parquet(links_path)
        pbar.update(1)
    
    # Convert back to snapshot format
    print("Converting back to snapshot format...")
    snapshots = []
    time_steps = sorted(nodes_df['time'].unique())
    
    for time in tqdm(time_steps, desc="Processing time steps", unit="step"):
        # Get nodes for this time step
        time_nodes = nodes_df[nodes_df['time'] == time]
        nodes_dict = {}
        for _, row in time_nodes.iterrows():
            nodes_dict[row['node_id']] = {
                'id': row['node_id'],
                'name': row['name'],
                'is_satellite': row['is_satellite'],
                'longitude': row['longitude'],
                'latitude': row['latitude'],
                'load': row['load']
            }
        
        # Get links for this time step
        time_links = links_df[links_df['time'] == time]
        links_dict = {}
        for _, row in time_links.iterrows():
            links_dict[row['link_id']] = {
                'id': row['link_id'],
                'source_id': row['source_id'],
                'sink_id': row['sink_id'],
                'source_name': row['source_name'],
                'sink_name': row['sink_name'],
                'source_longitude': row['source_longitude'],
                'source_latitude': row['source_latitude'],
                'sink_longitude': row['sink_longitude'],
                'sink_latitude': row['sink_latitude'],
                'load': row['load'],
                'is_connected': row['is_connected']
            }
        
        snapshots.append({
            'time': time,
            'nodes': nodes_dict,
            'links': links_dict
        })
    
    print("Network snapshots loaded from:")
    print(f"  - Nodes: {nodes_path}")
    print(f"  - Links: {links_path}")
    print(f"  - Total snapshots: {len(snapshots)}")
    
    return snapshots


def analyze_network_snapshots(filepath: str):
    """
    Analyze network snapshots using pandas for statistical insights.
    
    Args:
        filepath: Path to the parquet files (without _nodes/_links suffix)
    """
    # Load DataFrames
    base_path = filepath.replace('.parquet', '')
    nodes_path = f"{base_path}_nodes.parquet"
    links_path = f"{base_path}_links.parquet"
    
    nodes_df = pd.read_parquet(nodes_path)
    links_df = pd.read_parquet(links_path)
    
    print("=== Network Snapshot Analysis ===")
    print(f"Time range: {nodes_df['time'].min():.1f}s - {nodes_df['time'].max():.1f}s")
    print(f"Number of time steps: {len(nodes_df['time'].unique())}")
    print(f"Number of unique nodes: {len(nodes_df['node_id'].unique())}")
    print(f"Number of unique links: {len(links_df['link_id'].unique())}")
    
    print("\n=== Node Load Statistics ===")
    node_stats = nodes_df.groupby('time')['load'].agg(['mean', 'std', 'min', 'max']).round(3)
    print(node_stats.head(10))
    
    print("\n=== Link Load Statistics ===")
    link_stats = links_df.groupby('time')['load'].agg(['mean', 'std', 'min', 'max']).round(3)
    print(link_stats.head(10))
    
    print("\n=== Top 5 Busiest Nodes (Average Load) ===")
    busy_nodes = nodes_df.groupby(['node_id', 'name', 'is_satellite'])['load'].mean().sort_values(ascending=False).head()
    for (_, name, is_sat), load in busy_nodes.items():
        node_type = "Satellite" if is_sat else "Ground Station"
        print(f"  {name} ({node_type}): {load:.3f}")
    
    print("\n=== Top 5 Busiest Links (Average Load) ===")
    busy_links = links_df.groupby(['link_id', 'source_name', 'sink_name'])['load'].mean().sort_values(ascending=False).head()
    for (_, source, sink), load in busy_links.items():
        print(f"  {source} -> {sink}: {load:.3f}")
    
    return nodes_df, links_df


def create_load_distribution_gif_from_snapshots(
    network_snapshots: list, output_dir: str, use_imageio: bool = True
):
    """
    Create an animated GIF showing load distribution dynamics from network snapshots.

    Args:
        network_snapshots: List of network snapshots, each containing time and nodes data
        output_dir: Directory to save the GIF
        use_imageio: If True, use imageio for faster GIF generation (requires imageio package)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if len(network_snapshots) < 2:
        print("Need at least 2 snapshots to create animation")
        return

    # Calculate load range across all snapshots (nodes and links)
    all_loads = []
    for snapshot in network_snapshots:
        # Add node loads
        for node_data in snapshot["nodes"].values():
            all_loads.append(node_data["load"])
        # Add link loads
        for link_data in snapshot["links"].values():
            all_loads.append(link_data["load"])

    load_min = min(all_loads) if all_loads else 0
    load_max = max(all_loads) if all_loads else 1

    # Create figure and axis
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_global()

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Note: We'll manually draw satellites with transparency based on load
    # For dense networks, lightly loaded satellites should be nearly invisible
    
    # Ground stations scatter (always visible)
    gs_scatter = ax.scatter(
        [],
        [],
        c=[],
        s=120,
        cmap="Reds",
        vmin=load_min,
        vmax=load_max,
        marker="^",
        edgecolors="black",
        linewidth=1.5,
        alpha=1.0,  # Always visible
        transform=ccrs.PlateCarree(),
        label="Ground Stations",
    )

    # Add colorbar (use gs_scatter for reference)
    cbar = plt.colorbar(
        gs_scatter, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8
    )
    cbar.set_label("Load Factor (Red = High Load, Transparent = Low Load)", fontsize=11)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Satellites'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Ground Stations'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Links')
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Title
    title = ax.set_title(
        "Satellite Network Load Distribution\nTime: 0.0s",
        fontsize=14,
        fontweight="bold",
    )

    def animate(frame):
        """Animation function for each frame."""
        snapshot = network_snapshots[frame]
        current_time = snapshot["time"]
        nodes = snapshot["nodes"]
        links = snapshot["links"]

        # Clear previous drawings (links and satellite circles)
        for artist in ax.collections[:]:
            if artist != gs_scatter:  # Keep ground stations
                artist.remove()
        for line in ax.lines[:]:
            line.remove()

        # Draw links with transparency based on load (OPTIMIZED - skip very low load links)
        # Filter to only draw links with load > 0.01 (skip idle links for massive speedup)
        active_links = [link for link in links.values() 
                       if link["is_connected"] and link["load"] > 0.01]
        
        if active_links:
            for link_data in active_links:
                source_lon = link_data["source_longitude"]
                source_lat = link_data["source_latitude"]
                sink_lon = link_data["sink_longitude"]
                sink_lat = link_data["sink_latitude"]
                load = link_data["load"]
                
                # Calculate alpha: low load = nearly transparent (0.05), high load = opaque (0.9)
                load_ratio = load / load_max if load_max > 0 else 0
                alpha = 0.05 + load_ratio * 0.85  # Range: 0.05 to 0.9
                
                # Color intensity
                color_intensity = max(0.2, load_ratio)
                color = (color_intensity, 0, 0, alpha)
                
                # Line width: thinner for low load, thicker for high load
                linewidth = 0.5 + load_ratio * 2.5  # Range: 0.5 to 3.0
                
                # Draw link line
                ax.plot([source_lon, sink_lon], [source_lat, sink_lat], 
                       color=color, linewidth=linewidth,
                       transform=ccrs.PlateCarree(), zorder=1)

        # Separate satellites and ground stations
        satellites = []
        ground_stations = []

        for node_data in nodes.values():
            if node_data["is_satellite"]:
                satellites.append(node_data)
            else:
                ground_stations.append(node_data)

        # Draw satellites with transparency based on load (BATCH RENDERING for speed)
        if satellites:
            sat_lons = np.array([sat["longitude"] for sat in satellites])
            sat_lats = np.array([sat["latitude"] for sat in satellites])
            sat_loads = np.array([sat["load"] for sat in satellites])
            
            # Calculate all properties at once (vectorized)
            load_ratios = sat_loads / load_max if load_max > 0 else np.zeros_like(sat_loads)
            alphas = 0.05 + load_ratios * 0.90
            color_intensities = np.maximum(0.3, load_ratios)
            sizes = 30 + load_ratios * 40
            
            # Create RGBA colors for all satellites at once
            colors = np.column_stack([color_intensities, np.zeros_like(color_intensities), 
                                     np.zeros_like(color_intensities), alphas])
            
            # Batch render all satellites in one call (MUCH faster)
            ax.scatter(sat_lons, sat_lats, 
                      s=sizes,
                      c=colors,
                      marker='o',
                      edgecolors='none',  # No edges for speed
                      transform=ccrs.PlateCarree(),
                      zorder=2)

        # Update ground station scatter plot (always visible)
        if ground_stations:
            gs_lons = [gs["longitude"] for gs in ground_stations]
            gs_lats = [gs["latitude"] for gs in ground_stations]
            gs_loads = [gs["load"] for gs in ground_stations]
            gs_scatter.set_offsets(list(zip(gs_lons, gs_lats)))
            gs_scatter.set_array(gs_loads)
        else:
            gs_scatter.set_offsets([])
            gs_scatter.set_array([])

        # Update title
        title.set_text(
            f"Satellite Network Load Distribution\nTime: {current_time:.1f}s"
        )

        return gs_scatter, title

    # Save as GIF with progress bar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(output_dir, f"load_distribution_{timestamp}.gif")

    print(f"Saving GIF to {gif_path}...")
    
    if use_imageio:
        # Use imageio for faster GIF generation
        try:
            import imageio
            from PIL import Image
            
            print("Using imageio for fast GIF generation...")
            print(f"Rendering {len(network_snapshots)} frames...")
            
            frames = []
            with tqdm(total=len(network_snapshots), desc="Rendering frames", unit="frame") as pbar:
                for i in range(len(network_snapshots)):
                    # Call animate function to update plot
                    animate(i)
                    
                    # Render frame to buffer (reduced DPI for speed)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=60, bbox_inches='tight')  # Lower DPI for speed
                    buf.seek(0)
                    
                    # Load as PIL Image and convert to numpy array
                    img = Image.open(buf)
                    frames.append(np.array(img))
                    buf.close()
                    pbar.update(1)
            
            # Save using imageio (much faster)
            print("Saving GIF with imageio...")
            imageio.mimsave(gif_path, frames, fps=10, loop=0)
            print(f"GIF saved successfully to {gif_path}")
            
        except ImportError:
            print("imageio not installed. Install with: pip install imageio")
            print("Falling back to matplotlib animation...")
            use_imageio = False
    
    if not use_imageio:
        # Fallback to matplotlib animation
        print(f"Creating animation with {len(network_snapshots)} frames...")
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(network_snapshots),
            interval=500,
            blit=False,
            repeat=True,
        )
        
        with tqdm(total=len(network_snapshots), desc="Generating GIF frames", unit="frame") as pbar:
            anim.save(gif_path, writer="pillow", fps=10, dpi=80, 
                     progress_callback=lambda frame, total: pbar.update(1))
        print(f"GIF saved successfully to {gif_path}")

    # Also save a static plot of the final frame
    static_path = os.path.join(output_dir, f"load_distribution_final_{timestamp}.png")
    print("Saving static plot...")
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    print(f"Static plot saved to {static_path}")

    plt.close(fig)  # Close figure to free memory


def plot_load_distribution(
    env: RoutingEnvAsync,
    solver: BaseSolver,
    eval_seed: int,
    output_dir: str = "figs/load_distribution",
    save_snapshots_path: str = None,
):
    """
    Plot load distribution dynamics as an animated GIF showing satellite network load over time.

    Args:
        env: The routing environment
        solver: The trained solver to evaluate
        eval_seed: Random seed for evaluation
        output_dir: Directory to save the GIF
        save_snapshots_path: Optional path to save network snapshots

    Returns:
        List of network snapshots
    """
    # Initialize snapshot collection
    network_snapshots = []

    def debug_callback(env: RoutingEnvAsync):
        time = env.current_time / 1000.0

        # Capture complete network snapshot
        snapshot = {"time": time, "nodes": {}, "links": {}}

        # Capture node data
        for node in env.network.nodes.values():
            node_id = node.id
            name = node.name
            load = node.get_load_factor()
            longitude, latitude = node.get_projected_position()

            # Store node data in snapshot
            snapshot["nodes"][node_id] = {
                "id": node_id,
                "name": name,
                "is_satellite": node.is_satellite(),
                "longitude": longitude,
                "latitude": latitude,
                "load": load,
            }

        # Capture link data
        for link in env.network.links.values():
            if link.is_connected:
                link_id = f"{link.source.id}-{link.sink.id}"
                load = link.get_load_factor()
                
                # Get source and sink positions
                source_lon, source_lat = link.source.get_projected_position()
                sink_lon, sink_lat = link.sink.get_projected_position()
                
                # Store link data in snapshot
                snapshot["links"][link_id] = {
                    "id": link_id,
                    "source_id": link.source.id,
                    "sink_id": link.sink.id,
                    "source_name": link.source.name,
                    "sink_name": link.sink.name,
                    "source_longitude": source_lon,
                    "source_latitude": source_lat,
                    "sink_longitude": sink_lon,
                    "sink_latitude": sink_lat,
                    "load": load,
                    "is_connected": link.is_connected,
                }

        network_snapshots.append(snapshot)

    # Run simulation and collect snapshots
    print("Starting simulation...")
    env.reset(seed=eval_seed, start_time=0)
    solver.set_eval()
    
    # Add progress tracking for simulation
    simulation_start_time = datetime.now()
    print("Collecting network snapshots...")
    
    env.run(solver, debug_callback=debug_callback, callback_interval_ms=10)
    
    simulation_end_time = datetime.now()
    simulation_duration = (simulation_end_time - simulation_start_time).total_seconds()
    print(f"Simulation completed in {simulation_duration:.1f} seconds")
    print(f"Collected {len(network_snapshots)} snapshots")

    if not network_snapshots:
        print("No snapshots collected. Cannot create GIF.")
        return []

    # Save snapshots immediately after simulation (before GIF generation)
    if save_snapshots_path:
        print("\nSaving snapshots to disk...")
        save_network_snapshots(network_snapshots, save_snapshots_path)
        print("Snapshots saved successfully. You can now use them without re-running simulation.")

    # Create GIF showing load distribution dynamics
    print("\nGenerating GIF visualization...")
    create_load_distribution_gif_from_snapshots(network_snapshots, output_dir)

    return network_snapshots


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot satellite network load distribution as animated GIF"
    )
    parser.add_argument(
        "--config",
        default="configs/starlink_dvbs2_test.json",
        help="Environment config file",
    )
    parser.add_argument(
        "--solver_path",
        help="Path to trained solver (required if not loading snapshots)",
    )
    parser.add_argument("--eval_seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument(
        "--output_dir",
        default="figs/load_distribution",
        help="Output directory for GIF",
    )
    parser.add_argument("--save_snapshots", help="Path to save network snapshots")
    parser.add_argument(
        "--load_snapshots", help="Path to load existing network snapshots"
    )
    parser.add_argument(
        "--analyze_snapshots", help="Path to analyze existing network snapshots"
    )
    args = parser.parse_args()

    # Check if we're analyzing, loading existing snapshots, or generating new ones
    if args.analyze_snapshots:
        # Analyze existing snapshots
        print("Analyzing existing network snapshots...")
        analyze_network_snapshots(args.analyze_snapshots)
    elif args.load_snapshots:
        # Load existing snapshots and create GIF
        print("Loading existing network snapshots...")
        network_snapshots = load_network_snapshots(args.load_snapshots)
        create_load_distribution_gif_from_snapshots(network_snapshots, args.output_dir)
    else:
        # Generate new snapshots
        if not args.solver_path:
            parser.error(
                "--solver_path is required when not loading existing snapshots"
            )

        # Load configuration
        config = NamedDict.load("configs/starlink_dvbs2_test.json")

        # Create environment
        env = RoutingEnvAsync(config)

        # Load solver
        solver = load_solver_from(env, args.solver_path)

        # Generate load distribution GIF
        snapshots = plot_load_distribution(
            env, solver, args.eval_seed, args.output_dir, args.save_snapshots
        )
        
        # Auto-save snapshots if not specified
        if not args.save_snapshots and snapshots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_path = f"data/network_snapshots_{timestamp}.parquet"
            save_network_snapshots(snapshots, auto_save_path)


if __name__ == "__main__":
    main()
