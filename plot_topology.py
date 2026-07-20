"""
This script visualizes the topology of a satellite network using Cartopy.
It plots satellites and inter-satellite links (ISL_N) on a world map.
"""

import argparse
import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


from sat_net.network import SatelliteNetwork
from sat_net.util import NamedDict

# Set matplotlib to use Type 1 fonts (avoids Type 3)
plt.rcParams["ps.useafm"] = True  # Use Adobe Font Metrics (AFM) fonts
plt.rcParams["pdf.use14corefonts"] = True  # Use standard PDF fonts (for PDF export)
plt.rcParams["text.usetex"] = False  # Avoid using LaTeX which may override fonts


def load_config(config_path):
    return NamedDict.load(config_path).to_dict()


def create_network(config):
    network_config = config["network"]
    network = SatelliteNetwork(
        altitude=network_config["altitude"],
        inclination=network_config["inclination"],
        num_orbits=network_config["num_orbits"],
        num_sats_per_orbit=network_config["num_sats_per_orbit"],
        phasing=network_config["phasing"],
        min_elevation_angle_deg=network_config["min_elevation_angle_deg"],
        link_buffer_size=network_config["link_buffer_size"],
        isl_data_rate=network_config["isl_data_rate"],
    )
    return network


def plot_great_circle(ax, lon1, lat1, lon2, lat2, **kwargs):
    """Plot a great circle line that handles longitude wrapping correctly."""
    # Normalize longitudes to avoid crossing the date line
    lon1_norm = ((lon1 + 180) % 360) - 180
    lon2_norm = ((lon2 + 180) % 360) - 180

    # If the difference is greater than 180 degrees, we need to handle wrapping
    if abs(lon2_norm - lon1_norm) > 180:
        # Determine which way to wrap
        if lon1_norm < 0:
            lon1_wrap = lon1_norm + 360
        else:
            lon1_wrap = lon1_norm - 360

        # Create two line segments
        ax.plot([lon1_norm, lon1_wrap], [lat1, lat1], **kwargs)
        ax.plot([lon1_wrap, lon2_norm], [lat1, lat2], **kwargs)
    else:
        # Normal case - direct line
        ax.plot([lon1_norm, lon2_norm], [lat1, lat2], **kwargs)


def plot_topology(network, timestamp=0, save_path=None):
    _fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    network.update_topology(timestamp)

    # Plot satellites
    sat_lons, sat_lats = network.get_satellite_lon_lat()
    for lon, lat in zip(sat_lons, sat_lats):
        ax.plot(
            lon,
            lat,
            "ro",
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            transform=ccrs.PlateCarree(),
        )

    # Plot ISL_N only
    for sat_id, next_sat_id in zip(network.satellite_ids, network.isl_n):
        if next_sat_id < 0:
            continue
        source_lon, source_lat = sat_lons[int(sat_id)], sat_lats[int(sat_id)]
        sink_lon, sink_lat = sat_lons[int(next_sat_id)], sat_lats[int(next_sat_id)]
        plot_great_circle(
            ax,
            source_lon,
            source_lat,
            sink_lon,
            sink_lat,
            color="green",
            linewidth=0.75,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
        )

    # Legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Satellite",
        ),
        # plt.Line2D([0], [0], color="green", linewidth=2, label="ISL_N Connection"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    # title = f"Satellite Network Topology (ISL_N only)\n"
    # title += f"Altitude: {network.altitude}km, Orbits: {network.num_orbits}, "
    # title += f"Sats per Orbit: {network.num_sats_per_orbit}, "
    # title += f"Total Satellites: {network.num_satellites}"
    # if timestamp > 0:
    #     title += f"\nTime: {timestamp}ms"
    # ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.savefig("figs/topology.eps", dpi=600, bbox_inches="tight", format="eps")
    plt.savefig("figs/topology.svg", dpi=600, bbox_inches="tight", format="svg")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simple satellite network topology visualization (ISL_N only)")
    parser.add_argument(
        "--config",
        default="configs/main.json",
        help="Configuration file path",
    )
    parser.add_argument("--timestamp", type=int, default=0, help="Timestamp in milliseconds")
    parser.add_argument("--save", help="Path to save the plot")
    args = parser.parse_args()

    config = load_config(args.config)
    network = create_network(config)
    plot_topology(network, timestamp=args.timestamp, save_path=args.save)


if __name__ == "__main__":
    main()
