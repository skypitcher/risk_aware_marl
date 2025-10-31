"""
This script generates various plots for packet delay and queuing delay metrics.

It reads packet data from CSV files, calculates statistics, and creates visualizations 
such as CDFs, boxplots, violin plots, and bar charts.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CVAR_ALPHA = 0.25

# Global configuration for plot appearance
BOXPLOT_WIDTH = 0.25  # Adjust this value to change boxplot width (0.1-1.0)
COST_LIMIT = 10.0  # Cost limit line for queuing delay (ms)

# Global figure settings
FIG_SIZE = (6.5, 4.5)
LABEL_FONT_SIZE = 16  # for x/y label
LEGEND_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LINE_WIDTH = 3

# Simulation time from config
SIMULATION_TIME_SEC = 30.0
NUM_SEEDS = 5  # Number of seeds used in the simulation, i.e., number of runs


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


def plot_packet_delay_cdf(packets: dict, plot_dir: str):
    """
    Draw CDF of packet delays for all algorithms.
    """
    plt.figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (algorithm, df) in enumerate(packets.items()):
        # Filter delivered packets
        delivered = df[df["delivered"] == True]

        if len(delivered) == 0:
            print(f"Warning: No delivered packets for {algorithm}")
            continue

        delays = delivered["total_delay"].values

        # Calculate CDF
        sorted_delays = np.sort(delays)
        cdf_values = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)

        plt.plot(
            sorted_delays,
            cdf_values,
            label=algorithm,
            color=colors[i % len(colors)],
            linewidth=2,
        )

        # Print statistics
        print(f"{algorithm} - Delivered packets: {len(delivered)}, Avg delay: {np.mean(delays):.2f}ms")

    plt.xlabel("Packet Delay (ms)")
    plt.ylabel("Cumulative Probability")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plot_dir, "packet_delay_cdf.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "packet_delay_cdf.eps"), format="eps")
    plt.close()


def plot_packet_delay_boxplot(packets: dict, plot_dir: str):
    """
    Draw boxplot of packet delays for all algorithms.
    """
    # Prepare data for boxplot
    plot_data = []
    for algorithm, df in packets.items():
        delivered = df[df["delivered"] == True]
        if len(delivered) > 0:
            for delay in delivered["total_delay"]:
                plot_data.append({"Algorithm": algorithm, "Delay": delay})

    if not plot_data:
        print("No data available for boxplot")
        return

    plot_df = pd.DataFrame(plot_data)

    plt.figure()

    # Create boxplot
    sns.boxplot(data=plot_df, x="Algorithm", y="Delay", hue="Algorithm", palette="Set2", showfliers=True, legend=False, width=BOXPLOT_WIDTH)

    # Add scatter points for better visualization
    # sns.stripplot(
    #     data=plot_df, x="Algorithm", y="Delay", size=3, alpha=0.1, color="black"
    # )

    plt.ylabel("Packet Delay (ms)")
    plt.xlabel("")
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plot_dir, "packet_delay_boxplot.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "packet_delay_boxplot.eps"), format="eps")
    plt.close()


def plot_queueing_delay_cdf(packets: dict, plot_dir: str):
    """
    Draw CDF of queuing delays for all algorithms.
    Note: packet_delay = queueing_delay + propagation_delay + transmission_delay
    """
    plt.figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (algorithm, df) in enumerate(packets.items()):
        # Filter delivered packets
        delivered = df[df["delivered"] == True]

        if len(delivered) == 0:
            continue

        queueing_delays = delivered["queue_delay"].values

        # Calculate CDF
        sorted_delays = np.sort(queueing_delays)
        cdf_values = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)

        plt.plot(
            sorted_delays,
            cdf_values,
            label=algorithm,
            color=colors[i % len(colors)],
            linewidth=2,
        )

        # Print statistics
        print(f"{algorithm} - Avg queuing delay: {np.mean(queueing_delays):.2f}ms")

    plt.xlabel("Queuing Delay (ms)")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, alpha=0.3)

    # Add cost limit line
    plt.axvline(x=COST_LIMIT, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"Threshold ({COST_LIMIT}ms)")
    plt.legend()

    plt.savefig(os.path.join(plot_dir, "queueing_delay_cdf.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "queueing_delay_cdf.eps"), format="eps")
    plt.close()


def plot_queueing_delay_boxplot(packets: dict, plot_dir: str):
    """
    Draw boxplot of queuing delays for all algorithms.
    """
    # Prepare data for boxplot
    plot_data = []
    for algorithm, df in packets.items():
        delivered = df[df["delivered"] == True]
        if len(delivered) > 0:
            for delay in delivered["queue_delay"]:
                plot_data.append({"Algorithm": algorithm, "Queuing Delay": delay})

    if not plot_data:
        print("No data available for queuing delay boxplot")
        return

    plot_df = pd.DataFrame(plot_data)

    plt.figure()

    # Create boxplot
    sns.boxplot(
        data=plot_df,
        x="Algorithm",
        y="Queuing Delay",
        hue="Algorithm",
        palette="Set1",
        showfliers=True,
        legend=False,
        width=BOXPLOT_WIDTH,
    )

    # Add scatter points for better visualization
    # sns.stripplot(
    #     data=plot_df,
    #     x="Algorithm",
    #     y="Queuing Delay",
    #     size=3,
    #     alpha=0.1,
    #     color="black",
    # )

    plt.ylabel("Queuing Delay (ms)")
    plt.xlabel("")
    plt.grid(True, alpha=0.3)

    # Add cost limit line
    plt.axhline(y=COST_LIMIT, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"Threshold ({COST_LIMIT}ms)")

    plt.savefig(os.path.join(plot_dir, "queueing_delay_boxplot.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "queueing_delay_boxplot.eps"), format="eps")
    plt.close()


def plot_combined_delay_boxplot(packets: dict, plot_dir: str):
    """
    Draw combined boxplot comparing end-to-end delay and queuing delay for all algorithms.
    """
    # Prepare data for combined boxplot
    plot_data = []
    for algorithm, df in packets.items():
        delivered = df[df["delivered"] == True]
        if len(delivered) > 0:
            # Add end-to-end delay data
            for delay in delivered["total_delay"]:
                plot_data.append({"Algorithm": algorithm, "Delay Type": "End-to-End", "Delay": delay})
            # Add queuing delay data
            for delay in delivered["queue_delay"]:
                plot_data.append({"Algorithm": algorithm, "Delay Type": "Queuing", "Delay": delay})

    if not plot_data:
        print("No data available for combined delay boxplot")
        return

    plot_df = pd.DataFrame(plot_data)

    plt.figure()

    # Create combined boxplot
    sns.boxplot(
        data=plot_df,
        x="Algorithm",
        y="Delay",
        hue="Delay Type",
        palette=["#2ca02c", "#ff7f0e"],
        showfliers=True,
        width=BOXPLOT_WIDTH,
    )

    plt.ylabel("Delay (ms)")
    plt.xlabel("")
    plt.grid(True, alpha=0.3)

    # Add cost limit line
    plt.axhline(y=COST_LIMIT, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"Threshold ({COST_LIMIT}ms)")
    plt.legend()

    plt.savefig(os.path.join(plot_dir, "combined_delay_boxplot.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "combined_delay_boxplot.eps"), format="eps")
    plt.close()


def plot_packet_delay_violin(packets: dict, plot_dir: str):
    """
    Draw violin plot of packet delays for all algorithms.
    """
    # Prepare data for violin plot
    plot_data = []
    for algorithm, df in packets.items():
        delivered = df[df["delivered"] == True]
        if len(delivered) > 0:
            for delay in delivered["total_delay"]:
                plot_data.append({"Algorithm": algorithm, "Delay": delay})

    if not plot_data:
        print("No data available for violin plot")
        return

    plot_df = pd.DataFrame(plot_data)

    plt.figure()

    # Create violin plot
    sns.violinplot(data=plot_df, x="Algorithm", y="Delay", hue="Algorithm", palette="Set2", legend=False, inner="box")

    plt.ylabel("Packet Delay (ms)")
    plt.xlabel("")
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plot_dir, "packet_delay_violin.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "packet_delay_violin.eps"), format="eps")
    plt.close()


def plot_queueing_delay_violin(packets: dict, plot_dir: str):
    """
    Draw violin plot of queuing delays for all algorithms.
    """
    # Prepare data for violin plot
    plot_data = []
    for algorithm, df in packets.items():
        delivered = df[df["delivered"] == True]
        if len(delivered) > 0:
            for delay in delivered["queue_delay"]:
                plot_data.append({"Algorithm": algorithm, "Queuing Delay": delay})

    if not plot_data:
        print("No data available for queuing delay violin plot")
        return

    plot_df = pd.DataFrame(plot_data)

    plt.figure()

    # Create violin plot
    sns.violinplot(
        data=plot_df,
        x="Algorithm",
        y="Queuing Delay",
        hue="Algorithm",
        palette="Set1",
        legend=False,
        inner="box",
    )

    plt.ylabel("Queuing Delay (ms)")
    plt.xlabel("")
    plt.grid(True, alpha=0.3)

    # Add cost limit line
    plt.axhline(y=COST_LIMIT, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"Threshold ({COST_LIMIT}ms)")

    plt.savefig(os.path.join(plot_dir, "queueing_delay_violin.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "queueing_delay_violin.eps"), format="eps")
    plt.close()


def plot_combined_delay_violin(packets: dict, plot_dir: str):
    """
    Draw combined violin plot comparing end-to-end delay and queuing delay for all algorithms.
    """
    # Prepare data for combined violin plot
    plot_data = []
    for algorithm, df in packets.items():
        delivered = df[df["delivered"] == True]
        if len(delivered) > 0:
            # Add end-to-end delay data
            for delay in delivered["total_delay"]:
                plot_data.append({"Algorithm": algorithm, "Delay Type": "End-to-End", "Delay": delay})
            # Add queuing delay data
            for delay in delivered["queue_delay"]:
                plot_data.append({"Algorithm": algorithm, "Delay Type": "Queuing", "Delay": delay})

    if not plot_data:
        print("No data available for combined delay violin plot")
        return

    plot_df = pd.DataFrame(plot_data)

    plt.figure()

    # Create combined violin plot
    sns.violinplot(
        data=plot_df,
        x="Algorithm",
        y="Delay",
        hue="Delay Type",
        palette=["#2ca02c", "#ff7f0e"],
        split=False,
        inner="box",
    )

    plt.ylabel("Delay (ms)")
    plt.xlabel("")
    plt.grid(True, alpha=0.3)

    # Add cost limit line
    plt.axhline(y=COST_LIMIT, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"Threshold ({COST_LIMIT}ms)")
    plt.legend()

    plt.savefig(os.path.join(plot_dir, "combined_delay_violin.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "combined_delay_violin.eps"), format="eps")
    plt.close()


def plot_delay_components_bar(packets: dict, plot_dir: str):
    """
    Draw bar chart of average packet delay components to show the breakdown.
    Components: queueing_delay, transmission_delay, propagation_delay
    """
    # Calculate average delay components for each algorithm
    delay_components = {}

    for algorithm, df in packets.items():
        delivered = df[df["delivered"] == True]

        if len(delivered) == 0:
            continue

        avg_total_delay = delivered["total_delay"].mean()
        avg_queue = delivered["queue_delay"].mean()
        avg_transmission = delivered["transmission_delay"].mean()
        avg_propagation = delivered["propagation_delay"].mean()

        delay_components[algorithm] = {
            "Total": avg_total_delay,
            "Queuing": avg_queue,
            "Transmission": avg_transmission,
            "Propagation": avg_propagation,
        }

        total_calculated = avg_queue + avg_transmission + avg_propagation
        total_actual = delivered["total_delay"].mean()

        print(
            f"{algorithm}: Queue={avg_queue:.2f}ms, Trans={avg_transmission:.2f}ms, "
            f"Prop={avg_propagation:.2f}ms, Total={total_actual:.2f}ms "
            f"(Calculated={total_calculated:.2f}ms)"
        )

    if not delay_components:
        print("No data available for delay components chart")
        return

    # Prepare data for stacked bar chart
    algorithms = list(delay_components.keys())
    queueing_delays = [delay_components[alg]["Queuing"] for alg in algorithms]
    transmission_delays = [delay_components[alg]["Transmission"] for alg in algorithms]
    propagation_delays = [delay_components[alg]["Propagation"] for alg in algorithms]
    total_delays_actual = [delay_components[alg]["Total"] for alg in algorithms]
    total_delays_calculated = [q + t + p for q, t, p in zip(queueing_delays, transmission_delays, propagation_delays)]

    # Create stacked bar chart
    fig, ax = plt.subplots()

    bar_width = 0.6
    x_pos = np.arange(len(algorithms))

    # Create stacked bars
    ax.bar(
        x_pos,
        queueing_delays,
        bar_width,
        label="Queuing Delay",
        color="#ff7f0e",
        alpha=0.8,
    )
    ax.bar(
        x_pos,
        transmission_delays,
        bar_width,
        bottom=queueing_delays,
        label="Transmission Delay",
        color="#2ca02c",
        alpha=0.8,
    )
    ax.bar(
        x_pos,
        propagation_delays,
        bar_width,
        bottom=np.array(queueing_delays) + np.array(transmission_delays),
        label="Propagation Delay",
        color="#d62728",
        alpha=0.8,
    )

    # Add value labels on bars
    for i, alg in enumerate(algorithms):
        total_delay = total_delays_actual[i]
        ax.text(
            i,
            total_delay + 0.5,
            f"{total_delay:.1f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Average Delay (ms)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(os.path.join(plot_dir, "delay_components_bar.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "delay_components_bar.eps"), format="eps")
    plt.close()

    # Also create a side-by-side comparison
    fig, ax = plt.subplots()

    x_pos = np.arange(len(algorithms))
    width = 0.2

    bars1 = ax.bar(
        x_pos - 1.5 * width,
        total_delays_actual,
        width,
        label="End-to-End Delay",
        color="#1f77b4",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x_pos - 0.5 * width,
        queueing_delays,
        width,
        label="Queuing Delay",
        color="#ff7f0e",
        alpha=0.8,
    )
    bars3 = ax.bar(
        x_pos + 0.5 * width,
        transmission_delays,
        width,
        label="Transmission Delay",
        color="#2ca02c",
        alpha=0.8,
    )
    bars4 = ax.bar(
        x_pos + 1.5 * width,
        propagation_delays,
        width,
        label="Propagation Delay",
        color="#d62728",
        alpha=0.8,
    )

    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    # Add cost limit line
    plt.axhline(y=COST_LIMIT, color="black", linestyle="--", linewidth=2, alpha=0.7, label=f"Threshold ({COST_LIMIT}ms)")
    plt.legend()

    ax.set_ylim(0, 150)  # Set y-axis limit for better visibility
    ax.set_ylabel("Average Delay (ms)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(os.path.join(plot_dir, "delay_components_comparison.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "delay_components_comparison.eps"), format="eps")
    plt.close()


def plot_violation_metrics_bar(packets: dict, plot_dir: str):
    """
    Draw a combined bar chart comparing violation rate and average violation magnitude
    using a secondary y-axis. Violation magnitude is calculated only for packets that
    actually violate the constraint.
    """
    metrics = {}
    for algorithm, df in packets.items():
        total_packets = len(df)
        if total_packets == 0:
            metrics[algorithm] = {"violation_rate": 0, "hard_violation_cost": 0}
            continue

        delivered = df[df["delivered"] == True]
        dropped_count = len(df[df["dropped"] == True])

        costs = delivered["total_queue_cost"].values

        # A packet is considered violated if its queuing cost > cost_limit OR it is dropped
        violated_on_cost_count = np.sum(costs > COST_LIMIT)
        total_violated_count = violated_on_cost_count + dropped_count
        violation_rate = (total_violated_count / total_packets) * 100 if total_packets > 0 else 0

        # Calculate average hard violation (only for packets that actually violate the constraint)
        violating_costs = costs[costs > COST_LIMIT]
        if len(violating_costs) > 0:
            # Calculate magnitude of violations (how much over the limit)
            violation_magnitudes = violating_costs - COST_LIMIT
            avg_hard_violation = np.mean(violation_magnitudes)
        else:
            # No violations among delivered packets
            avg_hard_violation = 0.0

        metrics[algorithm] = {"violation_rate": violation_rate, "hard_violation_cost": avg_hard_violation}

        print(f"{algorithm}: Violation Rate={violation_rate:.2f}%, Avg Violation Magnitude={avg_hard_violation:.4f}ms")

    if not metrics:
        print("No data available for violation metrics bar chart")
        return

    algorithms = list(metrics.keys())
    violation_rates = [metrics[alg]["violation_rate"] for alg in algorithms]
    hard_violation_costs = [metrics[alg]["hard_violation_cost"] for alg in algorithms]
    x_pos = np.arange(len(algorithms))
    bar_width = 0.45

    # Create figure and primary y-axis
    fig, ax1 = plt.subplots()

    # Plot Violation Rate on the primary axis (ax1)
    bars1 = ax1.bar(x_pos - bar_width / 2, violation_rates, bar_width, label="Violation Rate (%)", color="skyblue", alpha=0.8)
    ax1.set_ylabel("Violation Rate (%)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms)
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Create secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + bar_width / 2, hard_violation_costs, bar_width, label="Average Violation Magnitude (ms)", color="salmon", alpha=0.8)
    ax2.set_ylabel("Average Violation Magnitude (ms)")

    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.25, f"{height:.2f}%", ha="center", va="bottom", fontweight="bold")

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}ms", ha="center", va="bottom", fontweight="bold")

    # Combined legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.savefig(os.path.join(plot_dir, "violation_metrics_comparison.png"), dpi=300)
    plt.savefig(os.path.join(plot_dir, "violation_metrics_comparison.eps"), format="eps")
    plt.close()


def print_throughput_metrics(packets: dict, result_path: str):
    """
    Calculate and print throughput (Mbps) for each algorithm.

    Args:
        packets: Dictionary of algorithm names to packet DataFrames
        result_path: Path to the result directory for reference
    """
    print("\n" + "=" * 60)
    print("THROUGHPUT ANALYSIS (Mbps)")
    print("=" * 60)

    throughput_results = {}

    for algorithm, df in packets.items():
        # Filter delivered packets only
        delivered = df[df["delivered"] == True]

        if len(delivered) == 0:
            print(f"{algorithm}: No delivered packets - Throughput = 0.00 Mbps")
            throughput_results[algorithm] = 0.0
            continue

        # Calculate total delivered data size in Mbits
        total_delivered_mbits = delivered["size"].sum()

        # Calculate throughput: total delivered data / simulation time (Mbps)
        throughput_mbps = total_delivered_mbits / SIMULATION_TIME_SEC / NUM_SEEDS

        # Calculate additional metrics
        total_packets = len(df)
        delivered_count = len(delivered)
        dropped_count = len(df[df["dropped"] == True])
        delivery_rate = delivered_count / total_packets if total_packets > 0 else 0
        drop_rate = dropped_count / total_packets if total_packets > 0 else 0

        # Calculate average packet size
        avg_packet_size = delivered["size"].mean() if len(delivered) > 0 else 0

        print(f"{algorithm}:")
        print(f"  Throughput: {throughput_mbps:.2f} Mbps")
        print(f"  Total delivered data: {total_delivered_mbits:.2f} Mbits")
        print(f"  Delivered packets: {delivered_count}/{total_packets} ({delivery_rate * 100:.1f}%)")
        print(f"  Dropped packets: {dropped_count} ({drop_rate * 100:.1f}%)")
        print(f"  Average packet size: {avg_packet_size:.3f} Mbits")
        print()

        throughput_results[algorithm] = throughput_mbps

    # Print summary comparison
    print("SUMMARY COMPARISON:")
    print("-" * 40)
    for algorithm, throughput in sorted(throughput_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{algorithm:15s}: {throughput:8.2f} Mbps")

    print("\n" + "=" * 60)

    return throughput_results


def print_comprehensive_metrics_table(packets: dict, result_path: str):
    """
    Print a comprehensive metrics table for all algorithms.

    Args:
        packets: Dictionary of algorithm names to packet DataFrames
        result_path: Path to the result directory for reference
    """
    print("\n" + "=" * 140)
    print("COMPREHENSIVE METRICS TABLE")
    print("=" * 140)

    # Total simulation time: 5 seeds × 30 seconds each = 150 seconds
    TOTAL_SIMULATION_TIME_SEC = SIMULATION_TIME_SEC * NUM_SEEDS

    # Prepare table data
    table_data = []

    for algorithm, df in packets.items():
        # Filter delivered packets only
        delivered = df[df["delivered"] == True]
        dropped = df[df["dropped"] == True]

        # Calculate throughput
        total_delivered_mbits = delivered["size"].sum() if len(delivered) > 0 else 0
        throughput_mbps = total_delivered_mbits / TOTAL_SIMULATION_TIME_SEC

        # Calculate drop rate
        total_packets = len(df)
        delivered_count = len(delivered)
        dropped_count = len(dropped)
        drop_rate = dropped_count / total_packets if total_packets > 0 else 0

        # Calculate E2E delay statistics
        if len(delivered) > 0:
            e2e_delays = delivered["total_delay"].values
            e2e_mean = np.mean(e2e_delays)
            e2e_std = np.std(e2e_delays)
            e2e_str = f"{e2e_mean:.1f}±{e2e_std:.1f}"
        else:
            e2e_str = "N/A"

        # Calculate queuing delay statistics
        if len(delivered) > 0:
            queue_delays = delivered["queue_delay"].values
            queue_mean = np.mean(queue_delays)
            queue_std = np.std(queue_delays)
            queue_str = f"{queue_mean:.1f}±{queue_std:.1f}"

            # Calculate CVaR(0.25) for queuing delay
            # CVaR(α) = E[X|X > VaR(α)] where VaR(α) is the α-quantile
            alpha = CVAR_ALPHA
            var_alpha = np.percentile(queue_delays, (1 - alpha) * 100)
            cvar_alpha = np.mean(queue_delays[queue_delays >= var_alpha])
            cvar_str = f"{cvar_alpha:.1f}"
        else:
            queue_str = "N/A"
            cvar_str = "N/A"

        # Calculate violation rate and violations
        # A packet is considered violated if its queuing cost > cost_limit OR it is dropped
        if len(delivered) > 0:
            costs = delivered["total_queue_cost"].values
            violated_on_cost_count = np.sum(costs > COST_LIMIT)
            total_violated_count = violated_on_cost_count + dropped_count
            violation_rate = (total_violated_count / total_packets) * 100 if total_packets > 0 else 0

            # Calculate violation magnitude (only for packets that actually violate the constraint)
            # For delivered packets, only consider those that exceed the cost limit
            violating_costs = costs[costs > COST_LIMIT]
            if len(violating_costs) > 0:
                # Calculate magnitude of violations (how much over the limit)
                violation_magnitudes = violating_costs - COST_LIMIT
                violation_mean = np.mean(violation_magnitudes)
                violation_std = np.std(violation_magnitudes)
                violation_str = f"{violation_mean:.2f}±{violation_std:.2f}"
            else:
                # No violations among delivered packets
                violation_str = "0.00±0.00"
        else:
            # If no delivered packets, all packets are considered violated (dropped)
            violation_rate = 100.0 if total_packets > 0 else 0
            violation_str = "N/A"

        table_data.append(
            {
                "Algorithm": algorithm,
                "Throughput (Mbps)": f"{throughput_mbps:.1f}",
                "Drop Rate (%)": f"{drop_rate * 100:.1f}",
                "E2E Delay (ms)": e2e_str,
                "Queuing Delay (ms)": queue_str,
                f"CVaR({CVAR_ALPHA}) (ms)": cvar_str,
                "Violation Rate (%)": f"{violation_rate:.1f}",
                "Violations (ms)": violation_str,
            }
        )

    # Print table header with better formatting
    print(
        f"{'Algorithm':<18} {'Throughput':<12} {'Drop Rate':<10} {'E2E Delay':<15} {'Queuing Delay':<15} {f'CVaR({CVAR_ALPHA})':<12} {'Violation Rate':<15} {'Violations':<15}"
    )
    print("-" * 140)

    # Print table rows
    for row in table_data:
        print(
            f"{row['Algorithm']:<18} {row['Throughput (Mbps)']:<12} {row['Drop Rate (%)']:<10} {row['E2E Delay (ms)']:<15} {row['Queuing Delay (ms)']:<15} {row[f'CVaR({CVAR_ALPHA}) (ms)']:<12} {row['Violation Rate (%)']:<15} {row['Violations (ms)']:<15}"
        )

    print("\n" + "=" * 140)

    return table_data


def plot_training_metrics(result_path: str):
    """
    Analyze and plot packet delay metrics from evaluation results.

    Args:
        result_path: Path to the directory containing algorithm_packets.csv files
    """

    display_name_table = {
        "MaDQN": "MADQN",
        "PrimalAvg": "Avg",
        "PrimalCVaR": "CVaR(0.25)",
        "SPF": "SPF",
    }
    display_order = ["SPF", "MADQN", "Avg", "CVaR(0.25)"]

    # Read packet data from CSV files
    packets = {}
    for file in os.listdir(result_path):
        if file.endswith(".csv"):
            algorithm_name = file.split("_")[0]
            df = pd.read_csv(os.path.join(result_path, file))
            display_name = display_name_table.get(algorithm_name, algorithm_name)
            packets[display_name] = df

    if not packets:
        print(f"No CSV files found in {result_path}")
        return

    packets = {k: packets[k] for k in display_order if k in packets}
    print(f"Found data for algorithms: {list(packets.keys())}")

    # Create output directory for plots
    plot_dir = "figs/test"
    os.makedirs(plot_dir, exist_ok=True)

    # Generate all plots
    plot_packet_delay_cdf(packets, plot_dir)
    plot_packet_delay_boxplot(packets, plot_dir)
    plot_packet_delay_violin(packets, plot_dir)
    plot_queueing_delay_cdf(packets, plot_dir)
    plot_queueing_delay_boxplot(packets, plot_dir)
    plot_queueing_delay_violin(packets, plot_dir)
    plot_combined_delay_boxplot(packets, plot_dir)
    plot_combined_delay_violin(packets, plot_dir)
    plot_delay_components_bar(packets, plot_dir)
    plot_violation_metrics_bar(packets, plot_dir)
    print_throughput_metrics(packets, result_path)
    print_comprehensive_metrics_table(packets, result_path)

    print(f"All plots saved to {plot_dir}")


def main():
    """
    Main function to run the plotting script with command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Plot training metrics from test results.")
    parser.add_argument("source", type=str, help="Path to the directory containing test result files (<algorithm_name>_packets.csv)")

    args = parser.parse_args()
    plot_training_metrics(result_path=args.source)


if __name__ == "__main__":
    main()
