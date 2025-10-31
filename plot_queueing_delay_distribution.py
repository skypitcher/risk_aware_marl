#!/usr/bin/env python3
"""
Plot queueing delay distribution comparison for different algorithms.
Reads packet data and generates publication-quality distribution plots.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


def load_packet_data(result_path: str):
    """
    Load packet data from test results.
    
    Args:
        result_path: Path to the directory containing CSV files
        
    Returns:
        Dictionary mapping algorithm names to DataFrames
    """
    packets = {}
    
    # Algorithm name mapping (file prefix -> display name)
    algorithm_mapping = {
        "PrimalAvg": "PrimalAvg",
        "PrimalCVaR": "PrimalCVaR",
        "MaDQN": "MaDQN",
    }
    
    for file in os.listdir(result_path):
        if file.endswith(".csv"):
            # Extract algorithm name from filename
            file_prefix = file.split("_")[0]
            
            if file_prefix in algorithm_mapping:
                df = pd.read_csv(os.path.join(result_path, file))
                display_name = algorithm_mapping[file_prefix]
                packets[display_name] = df
                print(f"Loaded {display_name}: {len(df)} packets")
    
    return packets


def calculate_statistics(queueing_delays, alpha=0.25):
    """
    Calculate statistics: mean, CVaR, VaR, etc.
    
    Args:
        queueing_delays: Array of queueing delay values
        alpha: Risk level for CVaR calculation (0.25 = top 25% worst delays)
        
    Returns:
        Dictionary containing statistics
    """
    mean_delay = np.mean(queueing_delays)
    
    # Compute CVaR(alpha) = E[X|X >= VaR(alpha)]
    var_alpha = np.percentile(queueing_delays, (1 - alpha) * 100)
    cvar_alpha = np.mean(queueing_delays[queueing_delays >= var_alpha])
    
    return {
        "mean": mean_delay,
        "cvar": cvar_alpha,
        "var": var_alpha,
        "min": np.min(queueing_delays),
        "max": np.max(queueing_delays),
        "std": np.std(queueing_delays),
        "median": np.median(queueing_delays),
        "p95": np.percentile(queueing_delays, 95),
        "p99": np.percentile(queueing_delays, 99),
    }


def plot_queueing_delay_distribution(
    packets: dict,
    output_dir: str = "figs/queueing_delay",
    alpha: float = 0.25,
):
    """
    Generate queueing delay distribution plots comparing different algorithms.
    
    Args:
        packets: Dictionary mapping algorithm names to DataFrames
        output_dir: Directory to save figures
        alpha: Risk level for CVaR calculation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("QUEUEING DELAY DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Collect queueing delay data for all algorithms
    algorithm_delays = {}
    algorithm_stats = {}
    
    for algorithm, df in packets.items():
        # Consider only successfully delivered packets
        delivered = df[df["delivered"] == True]
        
        if len(delivered) == 0:
            print(f"Warning: No delivered packets for {algorithm}")
            continue
        
        queueing_delays = delivered["queue_delay"].values
        algorithm_delays[algorithm] = queueing_delays
        
        # Calculate statistics
        stats_dict = calculate_statistics(queueing_delays, alpha=alpha)
        algorithm_stats[algorithm] = stats_dict
        
        print(f"\n{algorithm}:")
        print(f"  Packets delivered: {len(delivered):,}")
        print(f"  Mean delay: {stats_dict['mean']:.2f}ms")
        print(f"  Median delay: {stats_dict['median']:.2f}ms")
        print(f"  Std dev: {stats_dict['std']:.2f}ms")
        print(f"  Min/Max: {stats_dict['min']:.2f}ms / {stats_dict['max']:.2f}ms")
        print(f"  95th percentile: {stats_dict['p95']:.2f}ms")
        print(f"  99th percentile: {stats_dict['p99']:.2f}ms")
        print(f"  VaR(α={alpha}): {stats_dict['var']:.2f}ms")
        print(f"  CVaR(α={alpha}): {stats_dict['cvar']:.2f}ms")
    
    if not algorithm_delays:
        print("No data available for plotting")
        return
    
    # Determine global range for consistent axes
    all_delays = np.concatenate(list(algorithm_delays.values()))
    global_min = np.min(all_delays)
    global_max = np.max(all_delays)
    
    # Color mapping for algorithms
    colors = {
        "SPF": "#1f77b4",          # Blue
        "MaDQN": "#ff7f0e",        # Orange
        "PrimalAvg": "#2ca02c",    # Green
        "PrimalCVaR": "#d62728",   # Red
    }
    
    # ========== Figure 1: Probability Density Functions (PDF) ==========
    print("\nGenerating PDF comparison plot...")
    _, ax = plt.subplots(figsize=(10, 6))
    
    for algorithm, delays in algorithm_delays.items():
        # Calculate KDE
        kde = stats.gaussian_kde(delays)
        x_points = np.linspace(global_min, global_max, 500)
        y_points = kde(x_points)
        
        # Plot PDF
        color = colors.get(algorithm, "black")
        ax.plot(x_points, y_points, label=algorithm, color=color, linewidth=2)
        
        # Add mean line
        mean_val = algorithm_stats[algorithm]["mean"]
        ax.axvline(mean_val, color=color, linestyle="--", alpha=0.5, linewidth=1)
    
    ax.set_xlabel("Queuing Delay (ms)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Queuing Delay Distribution Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(output_dir, f"queueing_delay_pdf_{timestamp}.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PDF plot saved to {pdf_path}")
    
    # ========== Figure 2: Cumulative Distribution Functions (CDF) ==========
    print("Generating CDF comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algorithm, delays in algorithm_delays.items():
        # Sort delays for CDF
        sorted_delays = np.sort(delays)
        cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
        
        # Plot CDF
        color = colors.get(algorithm, "black")
        ax.plot(sorted_delays, cdf, label=algorithm, color=color, linewidth=2)
    
    # Add reference lines
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Median (50%)")
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="95th percentile")
    
    ax.set_xlabel("Queuing Delay (ms)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title("Queuing Delay CDF Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    cdf_path = os.path.join(output_dir, f"queueing_delay_cdf_{timestamp}.png")
    plt.savefig(cdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"CDF plot saved to {cdf_path}")
    
    # ========== Figure 3: Box Plot Comparison ==========
    print("Generating box plot comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    box_colors = []
    
    for algorithm in sorted(algorithm_delays.keys()):
        box_data.append(algorithm_delays[algorithm])
        box_labels.append(algorithm)
        box_colors.append(colors.get(algorithm, "black"))
    
    # Create box plot
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel("Queuing Delay (ms)", fontsize=12)
    ax.set_title("Queuing Delay Distribution (Box Plot)", fontsize=14, fontweight="bold")
    ax.grid(True, axis='y', alpha=0.3)
    
    box_path = os.path.join(output_dir, f"queueing_delay_boxplot_{timestamp}.png")
    plt.savefig(box_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Box plot saved to {box_path}")
    
    # ========== Figure 4: Statistical Comparison Bar Chart ==========
    print("Generating statistical comparison chart...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    algorithms = sorted(algorithm_delays.keys())
    x_pos = np.arange(len(algorithms))
    width = 0.6
    
    # Subplot 1: Mean and Median
    ax = axes[0, 0]
    means = [algorithm_stats[alg]["mean"] for alg in algorithms]
    medians = [algorithm_stats[alg]["median"] for alg in algorithms]
    
    x1 = x_pos - width/4
    x2 = x_pos + width/4
    
    for i, alg in enumerate(algorithms):
        color = colors.get(alg, "black")
        ax.bar(x1[i], means[i], width/2, label="Mean" if i == 0 else "", 
               color=color, alpha=0.7)
        ax.bar(x2[i], medians[i], width/2, label="Median" if i == 0 else "", 
               color=color, alpha=0.4)
    
    ax.set_ylabel("Delay (ms)", fontsize=10)
    ax.set_title("Mean and Median Delay", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Subplot 2: Standard Deviation
    ax = axes[0, 1]
    stds = [algorithm_stats[alg]["std"] for alg in algorithms]
    bar_colors = [colors.get(alg, "black") for alg in algorithms]
    
    ax.bar(x_pos, stds, width, color=bar_colors, alpha=0.7)
    ax.set_ylabel("Delay (ms)", fontsize=10)
    ax.set_title("Standard Deviation", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.grid(True, axis='y', alpha=0.3)
    
    # Subplot 3: 95th and 99th Percentiles
    ax = axes[1, 0]
    p95 = [algorithm_stats[alg]["p95"] for alg in algorithms]
    p99 = [algorithm_stats[alg]["p99"] for alg in algorithms]
    
    x1 = x_pos - width/4
    x2 = x_pos + width/4
    
    for i, alg in enumerate(algorithms):
        color = colors.get(alg, "black")
        ax.bar(x1[i], p95[i], width/2, label="95th %" if i == 0 else "", 
               color=color, alpha=0.7)
        ax.bar(x2[i], p99[i], width/2, label="99th %" if i == 0 else "", 
               color=color, alpha=0.4)
    
    ax.set_ylabel("Delay (ms)", fontsize=10)
    ax.set_title("95th and 99th Percentiles", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Subplot 4: VaR and CVaR
    ax = axes[1, 1]
    var_values = [algorithm_stats[alg]["var"] for alg in algorithms]
    cvar_values = [algorithm_stats[alg]["cvar"] for alg in algorithms]
    
    x1 = x_pos - width/4
    x2 = x_pos + width/4
    
    for i, alg in enumerate(algorithms):
        color = colors.get(alg, "black")
        ax.bar(x1[i], var_values[i], width/2, label=f"VaR(α={alpha})" if i == 0 else "", 
               color=color, alpha=0.7)
        ax.bar(x2[i], cvar_values[i], width/2, label=f"CVaR(α={alpha})" if i == 0 else "", 
               color=color, alpha=0.4)
    
    ax.set_ylabel("Delay (ms)", fontsize=10)
    ax.set_title(f"VaR and CVaR (α={alpha})", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    stats_path = os.path.join(output_dir, f"queueing_delay_stats_{timestamp}.png")
    plt.savefig(stats_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Statistical comparison saved to {stats_path}")
    
    # ========== Save Summary Statistics to CSV ==========
    print("\nSaving summary statistics to CSV...")
    summary_data = []
    
    for algorithm in algorithms:
        stats_dict = algorithm_stats[algorithm]
        summary_data.append({
            "Algorithm": algorithm,
            "Mean (ms)": f"{stats_dict['mean']:.2f}",
            "Median (ms)": f"{stats_dict['median']:.2f}",
            "Std Dev (ms)": f"{stats_dict['std']:.2f}",
            "Min (ms)": f"{stats_dict['min']:.2f}",
            "Max (ms)": f"{stats_dict['max']:.2f}",
            "P95 (ms)": f"{stats_dict['p95']:.2f}",
            "P99 (ms)": f"{stats_dict['p99']:.2f}",
            f"VaR(α={alpha}) (ms)": f"{stats_dict['var']:.2f}",
            f"CVaR(α={alpha}) (ms)": f"{stats_dict['cvar']:.2f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, f"queueing_delay_summary_{timestamp}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary statistics saved to {csv_path}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Plot queueing delay distribution for different routing algorithms"
    )
    parser.add_argument(
        "--result_path",
        default="runs_eval/2025-07-29_23-56-12",
        help="Path to evaluation results directory containing CSV files",
    )
    parser.add_argument(
        "--output_dir",
        default="figs/queueing_delay",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Risk level for CVaR calculation (default: 0.25 = top 25%% worst delays)",
    )
    args = parser.parse_args()
    
    # Load packet data
    print(f"Loading packet data from {args.result_path}...")
    packets = load_packet_data(args.result_path)
    
    if not packets:
        print("No valid packet data found!")
        return
    
    # Generate plots
    plot_queueing_delay_distribution(packets, args.output_dir, args.alpha)


if __name__ == "__main__":
    main()

