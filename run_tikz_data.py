import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

TIKZ_Y_MAX = 5.0
STEP_SIZE_MS = 0.2


def load_packet_data(result_path: str):
    """
    Load packet data from test results
    """
    packets = {}

    # Algorithm name mapping
    algorithm_mapping = {
        "PrimalAvg": "primal-avg",
        "PrimalCVaR": "primal-cvar",
        "MaDQN": "madqn",
    }

    for file in os.listdir(result_path):
        if file.endswith(".csv"):
            algorithm_name = file.split("_")[0]
            if algorithm_name in algorithm_mapping:
                df = pd.read_csv(os.path.join(result_path, file))
                display_name = algorithm_mapping[algorithm_name]
                packets[display_name] = df
                print(f"Loaded {display_name}: {len(df)} packets")

    return packets


def calculate_pdf_data(
    queueing_delays, min_delay, max_delay, step=1.0, scale_factor=1.0
):
    """
    Calculate PDF data points from min_delay to max_delay with given step size
    Using a unified scale factor
    """
    # Calculate PDF using kernel density estimation
    kde = stats.gaussian_kde(queueing_delays)

    # Generate x-axis points
    x_points = np.arange(min_delay, max_delay + step, step)

    # Calculate corresponding PDF values
    y_points = kde(x_points)

    # Apply unified scale factor
    y_points_scaled = y_points * scale_factor

    return x_points, y_points_scaled, y_points


def calculate_statistics(queueing_delays, alpha=0.25):
    """
    Calculate statistics: mean, CVaR, etc.
    """
    mean_delay = np.mean(queueing_delays)

    # Compute CVaR(alpha)
    # CVaR(α) = E[X|X > VaR(α)] where VaR(α) is the α-quantile
    var_alpha = np.percentile(queueing_delays, (1 - alpha) * 100)
    cvar_alpha = np.mean(queueing_delays[queueing_delays >= var_alpha])

    return {
        "mean": mean_delay,
        "cvar": cvar_alpha,
        "var": var_alpha,
        "min": np.min(queueing_delays),
        "max": np.max(queueing_delays),
        "std": np.std(queueing_delays),
    }


def generate_tikz_data(packets: dict):
    """
    Generate PDF data required for TikZ visualization
    """
    print("=" * 60)
    print("GENERATING TIKZ PDF DATA FOR QUEUING DELAY")
    print("=" * 60)

    # Collect queuing delay data for all algorithms
    all_delays = []
    algorithm_delays = {}

    for algorithm, df in packets.items():
        # Consider only successfully delivered packets
        delivered = df[df["delivered"] == True]

        if len(delivered) == 0:
            print(f"Warning: No delivered packets for {algorithm}")
            continue

        queueing_delays = delivered["queue_delay"].values
        algorithm_delays[algorithm] = queueing_delays
        all_delays.extend(queueing_delays)

        print(f"{algorithm}: {len(delivered)} delivered packets")

    if not all_delays:
        print("No data available for analysis")
        return

    # Determine global min and max
    global_min = np.min(all_delays)
    global_max = np.max(all_delays)

    print(f"\nGlobal delay range: {global_min:.2f}ms to {global_max:.2f}ms")

    # Step 1: Calculate maximum values of all PDFs to determine unified scale factor
    print(f"\n{'-' * 40}")
    print("CALCULATING UNIFIED SCALE FACTOR")
    print(f"{'-' * 40}")

    all_pdf_maxima = []
    algorithm_pdf_data = {}

    for algorithm, delays in algorithm_delays.items():
        # Calculate PDF data points
        x_points = np.arange(global_min, global_max + STEP_SIZE_MS, STEP_SIZE_MS)
        kde = stats.gaussian_kde(delays)
        y_points = kde(x_points)

        # Record maximum value
        max_pdf = np.max(y_points)
        all_pdf_maxima.append(max_pdf)
        algorithm_pdf_data[algorithm] = {
            "x_points": x_points,
            "y_points": y_points,
            "max_pdf": max_pdf,
        }

        print(f"{algorithm}: max PDF value = {max_pdf:.6f}")

    # Calculate unified scale factor
    global_max_pdf = max(all_pdf_maxima)
    unified_scale_factor = TIKZ_Y_MAX / global_max_pdf if global_max_pdf > 0 else 1.0

    print(f"\nGlobal max PDF value: {global_max_pdf:.6f}")
    print(f"Unified scale factor: {unified_scale_factor:.6f}")
    print(f"Target Y max: {TIKZ_Y_MAX}")

    # Create combined verification plot for all algorithms
    print(f"\n{'-' * 40}")
    print("GENERATING COMBINED VERIFICATION PLOT")
    print(f"{'-' * 40}")
    
    # Color mapping for algorithms
    colors = {
        "madqn": "#ff7f0e",        # Orange
        "primal-avg": "#2ca02c",   # Green
        "primal-cvar": "#d62728",  # Red
    }
    
    # Create single plot with original PDFs
    _, ax = plt.subplots(figsize=(10, 6))
    
    for algorithm, delays in algorithm_delays.items():
        delay_stats = calculate_statistics(delays, alpha=0.25)
        x_points, _, y_points_original = calculate_pdf_data(
            delays, global_min, global_max, step=STEP_SIZE_MS, scale_factor=1.0
        )
        
        color = colors.get(algorithm, "black")
        ax.plot(x_points, y_points_original, linewidth=2, label=f"{algorithm.upper()}", color=color)
        ax.axvline(delay_stats["mean"], color=color, linestyle="--", alpha=0.5, linewidth=1.5)
    
    # Add 10ms threshold line
    ax.axvline(10.0, color="black", linestyle="--", linewidth=2, alpha=0.7, label="Threshold (10ms)")
    
    ax.set_xlabel("Queueing Delay (ms)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Queueing Delay PDF Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_plot_path = "figs/tikz/pdf_verification_combined.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Combined verification plot saved to {combined_plot_path}")

    # 第二步：为每个算法生成PDF数据
    for algorithm, delays in algorithm_delays.items():
        print(f"\n{'-' * 40}")
        print(f"ANALYZING {algorithm.upper()}")
        print(f"{'-' * 40}")

        # Calculate statistics
        delay_stats = calculate_statistics(delays, alpha=0.25)

        print(f"Mean queuing delay: {delay_stats['mean']:.2f}ms")
        print(f"CVaR(0.25): {delay_stats['cvar']:.2f}ms")
        print(f"VaR(0.25): {delay_stats['var']:.2f}ms")
        print(f"Standard deviation: {delay_stats['std']:.2f}ms")
        print(f"Min: {delay_stats['min']:.2f}ms, Max: {delay_stats['max']:.2f}ms")

        # Generate PDF data points using unified scale factor
        x_points, y_points_scaled, y_points_original = calculate_pdf_data(
            delays,
            global_min,
            global_max,
            step=STEP_SIZE_MS,
            scale_factor=unified_scale_factor,
        )

        # Output TikZ format data
        print(f"\nTIKZ PDF DATA FOR {algorithm.upper()}:")
        print("=" * 50)

        # Output coordinates (using scaled values)
        tikz_points = []
        for x, y in zip(x_points, y_points_scaled):
            tikz_points.append(f"({x:.1f},{y:.6f})")

        print(" ".join(tikz_points))

        # Calculate PDF values for special points (original and scaled)
        kde = stats.gaussian_kde(delays)
        mean_pdf_original = float(kde(delay_stats["mean"]))
        cvar_pdf_original = float(kde(delay_stats["cvar"]))

        # Apply unified scale factor
        mean_pdf_scaled = mean_pdf_original * unified_scale_factor
        cvar_pdf_scaled = cvar_pdf_original * unified_scale_factor

        # Output special points (mean, CVaR)
        print(f"\nSPECIAL POINTS FOR {algorithm.upper()}:")
        print(f"Mean point (scaled): ({delay_stats['mean']:.1f},{mean_pdf_scaled:.6f})")
        print(f"CVaR point (scaled): ({delay_stats['cvar']:.1f},{cvar_pdf_scaled:.6f})")
        print(
            f"Mean point (original): ({delay_stats['mean']:.1f},{mean_pdf_original:.6f})"
        )
        print(
            f"CVaR point (original): ({delay_stats['cvar']:.1f},{cvar_pdf_original:.6f})"
        )
        print(f"Unified scale factor: {unified_scale_factor:.6f}")

        # Save to file
        output_file = f"figs/tikz/queuing_delay_distribution_{algorithm}.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"Saving data to {output_file}")

        # Write TikZ data file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# {algorithm.upper()} PDF Data for TikZ\n")
            f.write(f"# Mean: {delay_stats['mean']:.2f}ms\n")
            f.write(f"# CVaR(0.25): {delay_stats['cvar']:.2f}ms\n")
            f.write(f"# VaR(0.25): {delay_stats['var']:.2f}ms\n")
            f.write(f"# Standard deviation: {delay_stats['std']:.2f}ms\n")
            f.write(
                f"# Min: {delay_stats['min']:.2f}ms, Max: {delay_stats['max']:.2f}ms\n\n"
            )

            f.write("# PDF curve points (x,y):\n")
            f.write(" ".join(tikz_points))
            f.write("\n\n")

            f.write("# Special points (scaled):\n")
            f.write(f"# Mean: ({delay_stats['mean']:.1f},{mean_pdf_scaled:.6f})\n")
            f.write(f"# CVaR: ({delay_stats['cvar']:.1f},{cvar_pdf_scaled:.6f})\n")
            f.write(f"# Unified scale factor: {unified_scale_factor:.6f}\n")
            f.write("# Special points (original):\n")
            f.write(f"# Mean: ({delay_stats['mean']:.1f},{mean_pdf_original:.6f})\n")
            f.write(f"# CVaR: ({delay_stats['cvar']:.1f},{cvar_pdf_original:.6f})\n")

        print(f"Data saved to {output_file}")

        # Draw PDF plot for verification (including original and scaled curves)
        plt.figure(figsize=(12, 8))

        # Create subplots
        plt.subplot(2, 1, 1)
        plt.hist(delays, bins=50, density=True, alpha=0.7, label="Histogram")
        plt.plot(
            x_points, y_points_original, "r-", linewidth=2, label="Original KDE PDF"
        )
        plt.axvline(
            delay_stats["mean"],
            color="blue",
            linestyle="--",
            label=f"Mean: {delay_stats['mean']:.2f}ms",
        )
        plt.axvline(
            delay_stats["cvar"],
            color="green",
            linestyle="--",
            label=f"CVaR(0.25): {delay_stats['cvar']:.2f}ms",
        )
        plt.xlabel("Queuing Delay (ms)")
        plt.ylabel("Probability Density")
        plt.title(f"{algorithm.upper()} Queuing Delay PDF (Original)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(
            x_points,
            y_points_scaled,
            "b-",
            linewidth=2,
            label=f"Scaled KDE PDF (max={TIKZ_Y_MAX})",
        )
        plt.axvline(
            delay_stats["mean"],
            color="blue",
            linestyle="--",
            label=f"Mean: {delay_stats['mean']:.2f}ms",
        )
        plt.axvline(
            delay_stats["cvar"],
            color="green",
            linestyle="--",
            label=f"CVaR(0.25): {delay_stats['cvar']:.2f}ms",
        )
        plt.xlabel("Queuing Delay (ms)")
        plt.ylabel("Scaled Probability Density")
        plt.title(f"{algorithm.upper()} Queuing Delay PDF (Scaled for TikZ)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"figs/tikz/pdf_verification_{algorithm}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"PDF verification plot saved to pdf_verification_{algorithm}.png")


def main():
    """
    Main function
    """
    # Use test data path
    result_path = "runs_eval/2025-07-29_23-56-12"

    # Load data
    packets = load_packet_data(result_path)

    if not packets:
        print("No valid data found!")
        return

    # Generate TikZ data
    generate_tikz_data(packets)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
