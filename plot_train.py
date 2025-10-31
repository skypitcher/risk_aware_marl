"""
This script provides functions to calculate and plot training metrics for RL-based routing algorithms.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up beautiful seaborn styling
sns.set_palette("husl")  # Beautiful, colorful palette

# Professional color palette - High contrast colors for better visibility
COLORS = [
    "#2ca02c",  # Green - 绿色
    "#1f77b4",  # Blue - 蓝色
    "#d62728",  # Red - 红色
    "#ff7f0e",  # Orange - 橙色
    "#9467bd",  # Purple - 紫色
    "#8c564b",  # Brown - 棕色
    "#e377c2",  # Pink - 粉色
    "#7f7f7f",  # Gray - 灰色
    "#bcbd22",  # Olive - 橄榄绿
    "#17becf",  # Cyan - 青色
]
LINE_STYLES = ["-", "-", "-", "-", "-"]  # All solid lines
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

GRID_ALPHA = 0.5

ARRIVAL_RATE = 10  # packets/ms
SIMULATION_TIME = 30 * 1000  # milliseconds
EXPECTED_TOTAL_PACKETS = ARRIVAL_RATE * SIMULATION_TIME

SAVE_FOLDER = "figs/train"

# Global figure settings
FIG_SIZE = (6.5, 4.5)
LABEL_FONT_SIZE = 16  # for x/y label
LEGEND_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LINE_WIDTH = 3

TRAIN_EVERY = 1  # train once every 1 ms
PLOT_INTERVAL_MILLISECONDS = 2000  # plot interval
MAX_EPOCH = 30  # max epoch to plot

DELAY_X_START = 0 * 1000
NUM_X_TICKS = 5
X_LABEL = "Training Iterations"

intervals_per_epoch = SIMULATION_TIME // PLOT_INTERVAL_MILLISECONDS
train_steps_per_interval = PLOT_INTERVAL_MILLISECONDS // TRAIN_EVERY

# Set global matplotlib parameters
plt.rcParams["font.size"] = LABEL_FONT_SIZE
plt.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
plt.rcParams["ytick.labelsize"] = TICK_FONT_SIZE
plt.rcParams["figure.figsize"] = FIG_SIZE
plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
plt.rcParams["lines.linewidth"] = LINE_WIDTH
plt.rcParams["figure.autolayout"] = True  # Global tight_layout
plt.rcParams["savefig.bbox"] = "tight"  # Remove redundant boxes when saving
plt.rcParams["savefig.pad_inches"] = 0.1  # Add small padding when saving

# Set matplotlib to use Type 1 fonts (avoids Type 3)
plt.rcParams["ps.useafm"] = True  # Use Adobe Font Metrics (AFM) fonts
plt.rcParams["pdf.use14corefonts"] = True  # Use standard PDF fonts (for PDF export)
plt.rcParams["text.usetex"] = False  # Avoid using LaTeX which may override fonts


def calculate_epoch_metrics(run_id, epoch, percentile_level, risk_level, cost_limit):
    """
    Calculate performance metrics for a specific epoch by loading packet data.

    Args:
        run_id: Training run identifier
        epoch: Epoch number
        percentile_level: Percentile level for quantile calculation
        risk_level: Risk level for CVaR calculation
        cost_limit: The cost limit for violation calculation.

    Returns:
        Dictionary with mean, std, percentile, and CVaR values
    """
    packet_file = f"runs/{run_id}/packets/packets_epoch_{epoch}.csv"

    try:
        # Load packet data for the epoch
        packet_df = pd.read_csv(packet_file)

        # Filter delivered packets and get costs and delays
        delivered_packets = packet_df[packet_df["delivered"] == True]
        dropped_packets = packet_df[packet_df["dropped"] == True]

        if len(delivered_packets) > 0:
            costs = delivered_packets["total_queue_cost"].values
            delays = delivered_packets["total_delay"].values

            # Calculate delivery and drop rates
            total_packets = len(packet_df)
            delivered_count = len(delivered_packets)
            dropped_count = len(dropped_packets)
            delivery_rate = delivered_count / total_packets if total_packets > 0 else 0
            drop_rate = dropped_count / total_packets if total_packets > 0 else 0

            # Calculate throughput based on actual delivered data size in Mbits
            total_delivered_mbits = delivered_packets["size"].sum()
            # Throughput = total delivered data / simulation time (Mbps)
            throughput_mbps = total_delivered_mbits / (SIMULATION_TIME / 1000)  # Convert ms to seconds

            # Calculate violation percentage (a packet is considered violated if its queuing cost > cost_limit OR it is dropped)
            violated_packets_count = np.sum(costs > cost_limit) + dropped_count
            violation_rate = violated_packets_count / total_packets if total_packets > 0 else 0

            # Calculate hard violation
            hard_violation_cost = np.maximum(0, costs - cost_limit).sum() / total_packets if total_packets > 0 else 0

            # Calculate CVaR (Conditional Value at Risk)
            # CVaR is the average of the worst (1-risk_level) portion of the data
            cvar_threshold = np.percentile(costs, (1 - risk_level) * 100)
            worst_costs = costs[costs >= cvar_threshold]
            cvar_value = np.mean(worst_costs) if len(worst_costs) > 0 else cvar_threshold

            # Calculate CVaR for packet delays
            delay_cvar_threshold = np.percentile(delays, (1 - risk_level) * 100)
            worst_delays = delays[delays >= delay_cvar_threshold]
            cvar_packet_delay = np.mean(worst_delays) if len(worst_delays) > 0 else delay_cvar_threshold

            return {
                "mean": np.mean(costs),
                "std": np.std(costs),
                "percentile": np.percentile(costs, percentile_level * 100),
                "cvar": cvar_value,
                "min": np.min(costs),
                "max": np.max(costs),
                "avg_packet_delay": np.mean(delays),
                "cvar_packet_delay": cvar_packet_delay,
                "num_packets": len(delivered_packets),
                "delivery_rate": delivery_rate,
                "drop_rate": drop_rate,
                "throughput_mbps": throughput_mbps,
                "violation_rate": violation_rate,
                "hard_violation_cost": hard_violation_cost,
            }
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not load packet data for epoch {epoch}: {e}")
        return None


def get_available_epochs(run_id):
    """
    Scan the packets directory to find all available epoch files.

    Args:
        run_id: Training run identifier

    Returns:
        List of epoch numbers sorted in ascending order
    """
    import os
    import re

    packets_dir = f"runs/{run_id}/packets"

    if not os.path.exists(packets_dir):
        print(f"Error: Packets directory {packets_dir} does not exist!")
        return []

    # Find all packet_epoch_*.csv files
    epoch_files = []
    for filename in os.listdir(packets_dir):
        match = re.match(r"packets_epoch_(\d+)\.csv", filename)
        if match:
            epoch_num = int(match.group(1))
            epoch_files.append(epoch_num)

    # Sort epochs in ascending order
    epoch_files.sort()
    print(f"Found {len(epoch_files)} epoch files: {epoch_files[:5]}{'...' if len(epoch_files) > 5 else ''}")

    return epoch_files


def calculate_interval_metrics(run_id, epoch, percentile_level, risk_level, cost_limit):
    """
    Calculate performance metrics for each interval within an epoch by loading packet data.
    Groups packets by PLOT_INTERVAL_MILLISECONDS intervals to show training dynamics.

    Args:
        run_id: Training run identifier
        epoch: Epoch number
        percentile_level: Percentile level for quantile calculation
        risk_level: Risk level for CVaR calculation
        cost_limit: The cost limit for violation calculation.

    Returns:
        List of dictionaries with metrics for each interval, or None if no data
    """
    packet_file = f"runs/{run_id}/packets/packets_epoch_{epoch}.csv"

    try:
        # Load packet data for the epoch
        packet_df = pd.read_csv(packet_file)

        if len(packet_df) == 0:
            print(f"Warning: No packet data found for epoch {epoch}")
            return None

        # Create intervals based on PLOT_INTERVAL_MILLISECONDS
        max_time = SIMULATION_TIME  # 5000ms
        intervals = []

        for interval_start in range(0, max_time, PLOT_INTERVAL_MILLISECONDS):
            interval_end = min(interval_start + PLOT_INTERVAL_MILLISECONDS, max_time)
            interval_center = interval_start + PLOT_INTERVAL_MILLISECONDS // 2

            # Filter packets that were generated in this interval
            interval_packets = packet_df[(packet_df["creation_time"] >= interval_start) & (packet_df["creation_time"] < interval_end)]

            if len(interval_packets) > 0:
                delivered_packets = interval_packets[interval_packets["delivered"] == True]
                dropped_packets = interval_packets[interval_packets["dropped"] == True]

                total_packets = len(interval_packets)
                delivered_count = len(delivered_packets)
                dropped_count = len(dropped_packets)

                # Calculate metrics for this interval
                if delivered_count > 0:
                    costs = delivered_packets["total_queue_cost"].values
                    delays = delivered_packets["total_delay"].values

                    # Basic metrics
                    delivery_rate = delivered_count / total_packets if total_packets > 0 else 0
                    drop_rate = dropped_count / total_packets if total_packets > 0 else 0

                    # Throughput calculation
                    total_delivered_mbits = delivered_packets["size"].sum()
                    throughput_mbps = total_delivered_mbits / (PLOT_INTERVAL_MILLISECONDS / 1000)

                    # Violation metrics
                    violated_packets_count = np.sum(costs > cost_limit) + dropped_count
                    violation_rate = violated_packets_count / total_packets if total_packets > 0 else 0
                    hard_violation_cost = np.maximum(0, costs - cost_limit).sum() / total_packets if total_packets > 0 else 0

                    # CVaR calculations
                    cvar_threshold = np.percentile(costs, (1 - risk_level) * 100)
                    worst_costs = costs[costs >= cvar_threshold]
                    cvar_value = np.mean(worst_costs) if len(worst_costs) > 0 else cvar_threshold

                    delay_cvar_threshold = np.percentile(delays, (1 - risk_level) * 100)
                    worst_delays = delays[delays >= delay_cvar_threshold]
                    cvar_packet_delay = np.mean(worst_delays) if len(worst_delays) > 0 else delay_cvar_threshold

                    interval_metrics = {
                        "interval_start": interval_start,
                        "interval_end": interval_end,
                        "interval_center": interval_center,
                        "mean": np.mean(costs),
                        "std": np.std(costs),
                        "percentile": np.percentile(costs, percentile_level * 100),
                        "cvar": cvar_value,
                        "min": np.min(costs),
                        "max": np.max(costs),
                        "avg_packet_delay": np.mean(delays),
                        "cvar_packet_delay": cvar_packet_delay,
                        "num_packets": delivered_count,
                        "total_packets": total_packets,
                        "delivery_rate": delivery_rate,
                        "drop_rate": drop_rate,
                        "throughput_mbps": throughput_mbps,
                        "violation_rate": violation_rate,
                        "hard_violation_cost": hard_violation_cost,
                    }
                else:
                    # No delivered packets in this interval
                    interval_metrics = {
                        "interval_start": interval_start,
                        "interval_end": interval_end,
                        "interval_center": interval_center,
                        "mean": 0,
                        "std": 0,
                        "percentile": 0,
                        "cvar": 0,
                        "min": 0,
                        "max": 0,
                        "avg_packet_delay": 0,
                        "cvar_packet_delay": 0,
                        "num_packets": 0,
                        "total_packets": total_packets,
                        "delivery_rate": 0,
                        "drop_rate": 1.0,
                        "throughput_mbps": 0,
                        "violation_rate": 1.0,
                        "hard_violation_cost": 0,
                    }

                intervals.append(interval_metrics)
            else:
                # No packets in this interval - create zero metrics
                interval_metrics = {
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                    "interval_center": interval_center,
                    "mean": 0,
                    "std": 0,
                    "percentile": 0,
                    "cvar": 0,
                    "min": 0,
                    "max": 0,
                    "avg_packet_delay": 0,
                    "cvar_packet_delay": 0,
                    "num_packets": 0,
                    "total_packets": 0,
                    "delivery_rate": 0,
                    "drop_rate": 0,
                    "throughput_mbps": 0,
                    "violation_rate": 0,
                    "hard_violation_cost": 0,
                }
                intervals.append(interval_metrics)

        return intervals

    except Exception as e:
        print(f"Warning: Could not load packet data for epoch {epoch}: {e}")
        return None


def calculate_algorithm_metrics(run_id, risk_level, percentile_level, cost_limit, max_epoch=None):
    """
    Calculate performance metrics for an algorithm across all epochs.

    Args:
        run_id: Training run identifier
        risk_level: Risk level for CVaR calculation
        percentile_level: Percentile level for quantile calculation
        cost_limit: Cost limit for violation calculation.
        max_epoch: Maximum epoch number to process (None for no limit)

    Returns:
        Dictionary with epochs and corresponding metrics
    """
    # Get all available epochs by scanning packet files
    epochs = get_available_epochs(run_id)

    # Filter epochs if max_epoch is specified
    if max_epoch is not None:
        epochs = [epoch for epoch in epochs if epoch <= max_epoch]
        print(f"Filtered to {len(epochs)} epochs (max_epoch={max_epoch})")

    if len(epochs) == 0:
        print(f"Error: No epoch files found for {run_id}. Check packet data directory.")
        return None

    print(f"Calculating metrics for {run_id} with {percentile_level * 100:.0f}th percentile...")

    results = {
        "epochs": [],
        "mean_values": [],
        "std_values": [],
        "percentile_values": [],
        "cvar_values": [],
        "min_values": [],
        "max_values": [],
        "avg_packet_delays": [],
        "cvar_packet_delays": [],
        "num_packets": [],
        "delivery_rates": [],
        "drop_rates": [],
        "throughput_mbps": [],
        "violation_rates": [],
        "hard_violation_costs": [],
    }

    for epoch in epochs:
        metrics = calculate_epoch_metrics(run_id, epoch, percentile_level, risk_level, cost_limit)

        if metrics is not None:
            results["epochs"].append(epoch)
            results["mean_values"].append(metrics["mean"])
            results["std_values"].append(metrics["std"])
            results["percentile_values"].append(metrics["percentile"])
            results["cvar_values"].append(metrics["cvar"])
            results["min_values"].append(metrics["min"])
            results["max_values"].append(metrics["max"])
            results["avg_packet_delays"].append(metrics["avg_packet_delay"])
            results["cvar_packet_delays"].append(metrics["cvar_packet_delay"])
            results["num_packets"].append(metrics["num_packets"])
            results["delivery_rates"].append(metrics["delivery_rate"])
            results["drop_rates"].append(metrics["drop_rate"])
            results["throughput_mbps"].append(metrics["throughput_mbps"])
            results["violation_rates"].append(metrics["violation_rate"])
            results["hard_violation_costs"].append(metrics["hard_violation_cost"])

            if epoch % 10 == 0 or epoch <= 5:  # Print progress for some epochs
                print(
                    f"  Epoch {epoch}: Mean={metrics['mean']:.2f}ms, {percentile_level * 100:.0f}th%={metrics['percentile']:.2f}ms, CVaR({risk_level:.2f})={metrics['cvar']:.2f}ms, AvgDelay={metrics['avg_packet_delay']:.2f}ms, CVaRDelay={metrics['cvar_packet_delay']:.2f}ms, DeliveryRate={metrics['delivery_rate']:.3f}, ViolationRate={metrics['violation_rate']:.3%}, HardViolation={metrics['hard_violation_cost']:.4f}, Throughput={metrics['throughput_mbps']:.2f}Mbps"
                )

    if len(results["epochs"]) == 0:
        print(f"Error: No data calculated for {run_id}. Check packet data files.")
        return None

    return results


def calculate_algorithm_metrics_by_intervals(run_id, risk_level, percentile_level, cost_limit, max_epoch=None):
    """
    Calculate performance metrics for an algorithm across all epochs and intervals.
    This provides finer granularity to show training dynamics within epochs.

    Args:
        run_id: Training run identifier
        risk_level: Risk level for CVaR calculation
        percentile_level: Percentile level for quantile calculation
        cost_limit: Cost limit for violation calculation.
        max_epoch: Maximum epoch number to process (None for no limit)

    Returns:
        Dictionary with training steps and corresponding metrics
    """
    # Get all available epochs by scanning packet files
    epochs = get_available_epochs(run_id)

    # Filter epochs if max_epoch is specified
    if max_epoch is not None:
        epochs = [epoch for epoch in epochs if epoch <= max_epoch]
        print(f"Filtered to {len(epochs)} epochs (max_epoch={max_epoch})")

    if len(epochs) == 0:
        print(f"Error: No epoch files found for {run_id}. Check packet data directory.")
        return None

    print(f"Calculating interval-based metrics for {run_id} with {percentile_level * 100:.0f}th percentile...")

    results = {
        "training_steps": [],  # Instead of epochs, use training steps
        "epochs": [],  # Keep track of which epoch each step belongs to
        "intervals": [],  # Keep track of which interval within epoch
        "mean_values": [],
        "std_values": [],
        "percentile_values": [],
        "cvar_values": [],
        "min_values": [],
        "max_values": [],
        "avg_packet_delays": [],
        "cvar_packet_delays": [],
        "num_packets": [],
        "delivery_rates": [],
        "drop_rates": [],
        "throughput_mbps": [],
        "violation_rates": [],
        "hard_violation_costs": [],
    }

    for epoch in epochs:
        interval_metrics_list = calculate_interval_metrics(run_id, epoch, percentile_level, risk_level, cost_limit)

        if interval_metrics_list is not None:
            for interval_idx, metrics in enumerate(interval_metrics_list):
                # Calculate training step as: epoch * intervals_per_epoch + interval_index
                training_step = ((epoch - 1) * intervals_per_epoch + (interval_idx + 1)) * train_steps_per_interval

                results["training_steps"].append(training_step)
                results["epochs"].append(epoch)
                results["intervals"].append(interval_idx)
                results["mean_values"].append(metrics["mean"])
                results["std_values"].append(metrics["std"])
                results["percentile_values"].append(metrics["percentile"])
                results["cvar_values"].append(metrics["cvar"])
                results["min_values"].append(metrics["min"])
                results["max_values"].append(metrics["max"])
                results["avg_packet_delays"].append(metrics["avg_packet_delay"])
                results["cvar_packet_delays"].append(metrics["cvar_packet_delay"])
                results["num_packets"].append(metrics["num_packets"])
                results["delivery_rates"].append(metrics["delivery_rate"])
                results["drop_rates"].append(metrics["drop_rate"])
                results["throughput_mbps"].append(metrics["throughput_mbps"])
                results["violation_rates"].append(metrics["violation_rate"])
                results["hard_violation_costs"].append(metrics["hard_violation_cost"])

        # Print progress for some epochs
        if epoch % 10 == 0 or epoch <= 5:
            if interval_metrics_list:
                avg_delivery_rate = np.mean([m["delivery_rate"] for m in interval_metrics_list])
                avg_mean_cost = np.mean([m["mean"] for m in interval_metrics_list])
                print(
                    f"  Epoch {epoch}: {len(interval_metrics_list)} intervals, Avg delivery rate={avg_delivery_rate:.3f}, Avg mean cost={avg_mean_cost:.2f}ms"
                )

    if len(results["training_steps"]) == 0:
        print(f"Error: No interval data calculated for {run_id}. Check packet data files.")
        return None

    print(f"Total training steps: {len(results['training_steps'])}")
    return results


def plot_cvar_queueing_cost(
    all_results,
    cost_limit,
    risk_level,
    max_delay,
    delay_step=5,
    max_training_step=None,
):
    """
    Create a comparison plot of multiple algorithms' CVaR cost performance.
    """
    if len(all_results) == 0:
        print("Error: No data provided for CVaR cost comparison.")
        return

    # Create single plot
    fig, ax = plt.subplots()

    # Add cost limit line
    ax.axhline(y=cost_limit, color="black", linestyle="--", label=f"Threshold", alpha=0.6)

    # Plot each algorithm
    for i, (name, results) in enumerate(all_results.items()):
        color = COLORS[i % len(COLORS)]
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        x_values = results["training_steps"]
        ax.plot(x_values, results["cvar_values"], label=name, color=color, linestyle=line_style)

    # Update layout
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(f"CVaR({risk_level:.2f}) Queuing Delay (ms)")
    ax.set_yticks(np.arange(0, max_delay + 1, delay_step))

    if max_training_step is not None:
        # Create fewer ticks for training steps (converted to thousands)
        step_interval_thousands = max(1, (max_training_step // 1000) // NUM_X_TICKS)  # About NUM_X_TICKS ticks in thousands
        tick_positions = np.arange(0, max_training_step + 1, step_interval_thousands * 1000)
        tick_labels = [f"{int(pos // 1000)}k" if pos > 0 else "0" for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, max_delay)
    if max_training_step is not None:
        ax.set_xlim(0, max_training_step)

    ax.set_ylim(0, max_delay)
    ax.legend(frameon=True)
    ax.grid(True, alpha=GRID_ALPHA)  # Disable grid lines but keep tick marks

    # Save the plot
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(SAVE_FOLDER, f"CVaR_queueing_cost.png"), dpi=300)
    plt.savefig(os.path.join(SAVE_FOLDER, f"CVaR_queueing_cost.eps"), format="eps")
    plt.close()


def plot_mean_queueing_cost_intervals(
    all_results,
    cost_limit=10.0,
    max_delay=100,
    delay_step=5,
    max_training_step=None,
):
    """
    Create a comparison plot of multiple algorithms' mean cost performance by training intervals.
    """
    if len(all_results) == 0:
        print("Error: No data provided for mean cost comparison.")
        return

    # Create single plot
    fig, ax = plt.subplots()

    # Add cost limit line
    ax.axhline(y=cost_limit, color="black", linestyle="--", label=f"Threshold", alpha=0.6)

    # Plot each algorithm
    for i, (name, results) in enumerate(all_results.items()):
        color = COLORS[i % len(COLORS)]
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        x_values = results["training_steps"]
        ax.plot(x_values, results["mean_values"], label=name, color=color, linestyle=line_style)

    # Update layout
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Average Queuing Delay (ms)")
    ax.set_yticks(np.arange(0, max_delay + 1, delay_step))
    if max_training_step is not None:
        # Create fewer ticks for training steps (converted to thousands)
        step_interval_thousands = max(1, (max_training_step // 1000) // NUM_X_TICKS)  # About NUM_X_TICKS ticks in thousands
        tick_positions = np.arange(0, max_training_step + 1, step_interval_thousands * 1000)
        tick_labels = [f"{int(pos // 1000)}k" if pos > 0 else "0" for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, max_delay)
    if max_training_step is not None:
        ax.set_xlim(0, max_training_step)
    ax.legend(frameon=True)
    ax.grid(True, alpha=GRID_ALPHA)

    # Save the plot
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(SAVE_FOLDER, "mean_queueing_cost.png"), dpi=300)
    plt.savefig(os.path.join(SAVE_FOLDER, "mean_queueing_cost.eps"), format="eps")
    plt.close()


def plot_avg_packet_delay_intervals(
    all_results,
    max_delay=100,
    delay_step=5,
    max_training_step=None,
):
    """
    Create a comparison plot of multiple algorithms' average packet delay performance by training intervals.
    """
    if len(all_results) == 0:
        print("Error: No data provided for average packet delay comparison.")
        return

    # Create single plot
    fig, ax = plt.subplots()

    # Plot each algorithm
    for i, (name, results) in enumerate(all_results.items()):
        color = COLORS[i % len(COLORS)]
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        x_values = results["training_steps"]
        ax.plot(x_values, results["avg_packet_delays"], label=name, color=color, linestyle=line_style)

    # Update layout
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Average E2E Delay (ms)")
    ax.set_yticks(np.arange(0, max_delay + 1, delay_step))
    if max_training_step is not None:
        # Create fewer ticks for training steps (converted to thousands)
        step_interval_thousands = max(1, (max_training_step // 1000) // NUM_X_TICKS)  # About NUM_X_TICKS ticks in thousands
        tick_positions = np.arange(0, max_training_step + 1, step_interval_thousands * 1000)
        tick_labels = [f"{int(pos // 1000)}k" if pos > 0 else "0" for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    ax.set_ylim(60, max_delay)
    if max_training_step is not None:
        ax.set_xlim(DELAY_X_START, max_training_step)
    ax.legend(frameon=True)
    ax.grid(True, alpha=GRID_ALPHA)

    # Save the plot
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(SAVE_FOLDER, "avg_packet_delay.png"), dpi=300)
    plt.savefig(os.path.join(SAVE_FOLDER, "avg_packet_delay.eps"), format="eps")
    plt.close()


def plot_delivery_rate_intervals(
    all_results,
    max_training_step=None,
):
    """
    Create a comparison plot of multiple algorithms' packet delivery rate performance by training intervals.
    """
    if len(all_results) == 0:
        print("Error: No data provided for delivery rate comparison.")
        return

    # Create a single plot for delivery rate
    fig, ax = plt.subplots()

    # Plot each algorithm
    for i, (name, results) in enumerate(all_results.items()):
        color = COLORS[i % len(COLORS)]
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        x_values = results["training_steps"]
        # Convert to percentage for display
        delivery_rates_percent = [rate * 100 for rate in results["delivery_rates"]]
        ax.plot(x_values, delivery_rates_percent, label=name, color=color, linestyle=line_style)

    # Update layout
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Packet delivery rate (%)")
    ax.set_ylim(0, 100)
    if max_training_step is not None:
        ax.set_xlim(0, max_training_step)
        step_interval_thousands = max(1, (max_training_step // 1000) // NUM_X_TICKS)  # About 10 ticks in thousands
        tick_positions = np.arange(0, max_training_step + 1, step_interval_thousands * 1000)
        tick_labels = [f"{int(pos // 1000)}k" if pos > 0 else "0" for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    ax.set_yticks(np.arange(0, 101, 10))  # 0%, 10%, 20%, ..., 100%
    ax.legend(frameon=True)
    ax.grid(True, alpha=GRID_ALPHA)

    # Save the plot
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(SAVE_FOLDER, "packet_delivery_rate.png"), dpi=300)
    plt.savefig(os.path.join(SAVE_FOLDER, "packet_delivery_rate.eps"), format="eps")
    plt.close()


def plot_drop_rate_intervals(
    all_results,
    max_training_step=None,
):
    """
    Create a comparison plot of multiple algorithms' packet drop rate performance by training intervals.
    """
    if len(all_results) == 0:
        print("Error: No data provided for drop rate comparison.")
        return

    # Calculate min and max values from all drop rate data
    all_drop_rates = []
    for results in all_results.values():
        all_drop_rates.extend(results["drop_rates"])

    if not all_drop_rates:
        print("Error: No drop rate data found.")
        return

    min_rate = min(all_drop_rates)
    max_rate = max(all_drop_rates)

    # Convert to percentage for range calculation
    min_rate_percent = min_rate * 100
    max_rate_percent = max_rate * 100

    # Focus on the actual range to better show differences
    if max_rate_percent <= 10:  # If all drop rates are <= 10%
        y_min = -5  # Start from -5% to make zero line clearly visible
        y_max = max(5, max_rate_percent + 1)  # At least 5% range, or add margin above maximum
        y_max = ((y_max + 9) // 10) * 10
    else:
        y_min = -5  # Start from -5%
        y_max = min(100, max_rate_percent + 5)  # Add 5% margin
        y_max = ((y_max + 9) // 10) * 10

    print(f"Drop rate range: {min_rate:.4f} - {max_rate:.4f}")
    print(f"Y-axis range (focused): {y_min:.1f}% - {y_max:.1f}% (step: 20%)")

    # Create a single plot for drop rate
    fig, ax = plt.subplots()

    # Plot each algorithm
    for i, (name, results) in enumerate(all_results.items()):
        color = COLORS[i % len(COLORS)]
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        x_values = results["training_steps"]
        # Convert to percentage for display
        drop_rates_percent = [rate * 100 for rate in results["drop_rates"]]
        ax.plot(x_values, drop_rates_percent, label=name, color=color, linestyle=line_style)

    # Update layout
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Packet Drop Rate (%)")
    ax.set_ylim(y_min, y_max)
    if max_training_step is not None:
        ax.set_xlim(0, max_training_step)
        step_interval_thousands = max(1, (max_training_step // 1000) // NUM_X_TICKS)  # About 10 ticks in thousands
        tick_positions = np.arange(0, max_training_step + 1, step_interval_thousands * 1000)
        tick_labels = [f"{int(pos // 1000)}k" if pos > 0 else "0" for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    ax.set_yticks(np.arange(0, y_max + 1, 20))
    ax.legend(frameon=True, fontsize=14)
    ax.grid(True, alpha=GRID_ALPHA)

    # Save the plot
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(SAVE_FOLDER, "packet_drop_rate.png"), dpi=300)
    plt.savefig(os.path.join(SAVE_FOLDER, "packet_drop_rate.eps"), format="eps")
    plt.close()


def plot_training_metrics_by_intervals(competitors: list[dict]):
    """
    Plot training metrics by intervals to show finer training dynamics.
    This shows performance every PLOT_INTERVAL_MILLISECONDS (1000ms) instead of per epoch.
    """
    # Define parameters
    cost_limit = 10.0
    risk_level = 0.25
    percentile_level = 0.95
    max_delay_for_packet = 120
    max_delay_for_queueing_cost = 80
    max_epoch = MAX_EPOCH  # Reduce for finer analysis

    # Calculate interval-based metrics for all algorithms
    print("Calculating interval-based metrics for all algorithms...")
    calculated_results = {}
    max_training_step = max_epoch * intervals_per_epoch * train_steps_per_interval

    for alg in competitors:
        run_id = alg["run_id"]
        name = alg.get("name", run_id)
        results = calculate_algorithm_metrics_by_intervals(
            run_id, risk_level=risk_level, percentile_level=percentile_level, cost_limit=cost_limit, max_epoch=max_epoch
        )
        if results is not None:
            calculated_results[name] = results

    # Print training step information for each algorithm
    print("\n" + "=" * 50)
    print("TRAINING STEP INFORMATION FOR EACH ALGORITHM")
    print("=" * 50)
    print(f"Maximum training step: {max_training_step}")

    print(f"Intervals per epoch: {intervals_per_epoch}")
    print(f"Plot interval: {PLOT_INTERVAL_MILLISECONDS}ms")

    for name, results in calculated_results.items():
        if len(results["training_steps"]) > 0:
            epoch_range = f"{min(results['epochs'])}-{max(results['epochs'])}"
            print(f"{name}: {len(results['training_steps'])} training steps, epochs {epoch_range}")
    print()

    # Generate interval-based plots
    plot_mean_queueing_cost_intervals(
        all_results=calculated_results,
        cost_limit=cost_limit,
        max_delay=max_delay_for_queueing_cost,
        delay_step=10,
        max_training_step=max_training_step,
    )

    plot_cvar_queueing_cost(
        all_results=calculated_results,
        cost_limit=cost_limit,
        risk_level=risk_level,
        max_delay=max_delay_for_queueing_cost,
        delay_step=10,
        max_training_step=max_training_step,
    )

    plot_avg_packet_delay_intervals(
        all_results=calculated_results,
        max_delay=max_delay_for_packet,
        delay_step=10,
        max_training_step=max_training_step,
    )

    plot_delivery_rate_intervals(
        all_results=calculated_results,
        max_training_step=max_training_step,
    )

    plot_drop_rate_intervals(
        all_results=calculated_results,
        max_training_step=max_training_step,
    )

    print("training dynamics plots saved to figs/train/")


if __name__ == "__main__":
    plot_training_metrics_by_intervals(
        competitors=[
            {"run_id": "MaDQN_2025-07-28_10-54-19", "name": "MADQN"},
            {"run_id": "PrimalAvg_2025-07-25_16-47-34", "name": "PRIMAL-Avg"},
            {"run_id": "PrimalCVaR_2025-07-26_02-14-04", "name": "PRIMAL-CVaR(0.25)"},
        ]
    )
