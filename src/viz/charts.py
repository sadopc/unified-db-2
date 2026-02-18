"""All chart generation for the benchmark results.

7 chart types using matplotlib with consistent color scheme:
- DuckDB SQL: #2196F3 (blue)
- NumPy CPU: #9C27B0 (purple)
- MLX GPU: #FF9800 (orange)
- PCIe overhead: #F44336 (red)
- Unified: #4CAF50 (green)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.theoretical import TransferScenario
from src.bench.runner import BenchmarkResult
from src.bench.transfer import TransferBenchmark
from src.config import MEMORY_BW_GBS, PCIE4_BW_GBS, PCIE5_BW_GBS, QUERY_IDS, RESULTS_DIR

# Color scheme
C_DUCKDB = "#2196F3"
C_NUMPY = "#9C27B0"
C_MLX = "#FF9800"
C_PCIE = "#F44336"
C_UNIFIED = "#4CAF50"
C_PCIE5 = "#FF7043"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = RESULTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def chart_architecture_comparison() -> Path:
    """Chart 1: PCIe path vs unified memory path diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Memory Architecture Comparison", fontsize=14, fontweight="bold")

    # PCIe architecture
    ax1.set_title("Discrete GPU (PCIe)", fontsize=12)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis("off")

    # CPU + RAM
    ax1.add_patch(plt.Rectangle((0.5, 3.5), 3, 2, fill=True,
                                facecolor="#E3F2FD", edgecolor=C_DUCKDB, linewidth=2))
    ax1.text(2, 4.5, "CPU\n+ System RAM", ha="center", va="center", fontsize=10)

    # PCIe bus
    ax1.annotate("", xy=(5.5, 4.5), xytext=(3.5, 4.5),
                arrowprops=dict(arrowstyle="<->", color=C_PCIE, lw=3))
    ax1.text(4.5, 5.2, f"PCIe Bus\n25-50 GB/s", ha="center", va="center",
            fontsize=9, color=C_PCIE, fontweight="bold")

    # GPU + VRAM
    ax1.add_patch(plt.Rectangle((5.5, 3.5), 3, 2, fill=True,
                                facecolor="#FFF3E0", edgecolor=C_MLX, linewidth=2))
    ax1.text(7, 4.5, "GPU\n+ VRAM", ha="center", va="center", fontsize=10)

    # Bottleneck indicator
    ax1.text(4.5, 2.5, "BOTTLENECK", ha="center", va="center",
            fontsize=11, color=C_PCIE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=C_PCIE))

    # Unified architecture
    ax2.set_title("Apple Silicon (Unified Memory)", fontsize=12)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis("off")

    # Shared memory pool
    ax2.add_patch(plt.Rectangle((1, 1), 8, 2, fill=True,
                                facecolor="#E8F5E9", edgecolor=C_UNIFIED, linewidth=2))
    ax2.text(5, 2, f"Unified Memory Pool\n120 GB/s", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_UNIFIED)

    # CPU
    ax2.add_patch(plt.Rectangle((1.5, 3.5), 3, 2, fill=True,
                                facecolor="#E3F2FD", edgecolor=C_DUCKDB, linewidth=2))
    ax2.text(3, 4.5, "CPU Cores", ha="center", va="center", fontsize=10)

    # GPU
    ax2.add_patch(plt.Rectangle((5.5, 3.5), 3, 2, fill=True,
                                facecolor="#FFF3E0", edgecolor=C_MLX, linewidth=2))
    ax2.text(7, 4.5, "GPU Cores", ha="center", va="center", fontsize=10)

    # Arrows to shared memory
    ax2.annotate("", xy=(3, 3), xytext=(3, 3.5),
                arrowprops=dict(arrowstyle="<->", color=C_UNIFIED, lw=2))
    ax2.annotate("", xy=(7, 3), xytext=(7, 3.5),
                arrowprops=dict(arrowstyle="<->", color=C_UNIFIED, lw=2))

    ax2.text(5, 0.3, "No PCIe transfer needed", ha="center", va="center",
            fontsize=11, color=C_UNIFIED, fontweight="bold")

    return _save(fig, "01_architecture_comparison")


def chart_bandwidth_comparison() -> Path:
    """Chart 2: PCIe 4.0/5.0 vs M4 unified memory bandwidth."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["PCIe 3.0\nx16", "PCIe 4.0\nx16", "PCIe 5.0\nx16", "M4 Unified\nMemory"]
    values = [12, PCIE4_BW_GBS, PCIE5_BW_GBS, MEMORY_BW_GBS]
    colors = [C_PCIE, C_PCIE, C_PCIE5, C_UNIFIED]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val} GB/s", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("CPU↔GPU Data Transfer Bandwidth", fontweight="bold")
    ax.set_ylim(0, 145)

    ax.text(0.5, 0.95, "Scenario model: what bandwidth is available for initial data transfer",
           transform=ax.transAxes, ha="center", va="top", fontsize=8, style="italic",
           color="gray")

    return _save(fig, "02_bandwidth_comparison")


def chart_three_baseline(results: list[BenchmarkResult], sf: float) -> Path:
    """Chart 3: DuckDB SQL vs NumPy CPU vs MLX GPU execution time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    queries = QUERY_IDS
    x = np.arange(len(queries))
    width = 0.25

    duckdb_times = []
    numpy_times = []
    mlx_times = []

    for qid in queries:
        for r in results:
            if r.query_id == qid and r.sf == sf:
                if r.baseline == "duckdb":
                    duckdb_times.append(r.warm_mean_s * 1000)
                elif r.baseline == "numpy":
                    numpy_times.append(r.warm_mean_s * 1000)
                elif r.baseline == "mlx":
                    mlx_times.append(r.warm_mean_s * 1000)

    if not (duckdb_times and numpy_times and mlx_times):
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", va="center")
        return _save(fig, f"03_three_baseline_sf{sf}")

    ax.bar(x - width, duckdb_times, width, label="DuckDB SQL", color=C_DUCKDB)
    ax.bar(x, numpy_times, width, label="NumPy CPU", color=C_NUMPY)
    ax.bar(x + width, mlx_times, width, label="MLX GPU", color=C_MLX)

    ax.set_ylabel("Execution Time (ms)")
    ax.set_title(f"Three-Baseline Execution Time @ SF{sf}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.legend()
    ax.set_yscale("log")

    return _save(fig, f"03_three_baseline_sf{sf}")


def chart_conversion_overhead(results: list[BenchmarkResult], sf: float) -> Path:
    """Chart 4: Stacked bars showing DuckDB query + extraction + copy + GPU compute."""
    fig, ax = plt.subplots(figsize=(10, 6))

    queries = QUERY_IDS
    x = np.arange(len(queries))

    # We only have load_s (total) and compute for MLX.
    # For detailed breakdown, we'd need the LoadTiming from each query.
    # Approximate: load_s is ~80% extraction + 20% copy
    compute = []
    load = []

    for qid in queries:
        for r in results:
            if r.query_id == qid and r.sf == sf and r.baseline == "mlx":
                compute.append(r.warm_mean_s * 1000)
                load.append(r.load_s * 1000)

    if not compute:
        ax.text(0.5, 0.5, "No MLX data available", transform=ax.transAxes,
                ha="center", va="center")
        return _save(fig, f"04_conversion_overhead_sf{sf}")

    ax.bar(x, load, label="Data Loading (DuckDB→NumPy→MLX)", color=C_PCIE)
    ax.bar(x, compute, bottom=load, label="GPU Compute", color=C_MLX)

    ax.set_ylabel("Time (ms)")
    ax.set_title(f"End-to-End Breakdown: Loading vs Compute @ SF{sf}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.legend()

    # Add percentage labels
    for i in range(len(queries)):
        total = load[i] + compute[i]
        if total > 0:
            pct = load[i] / total * 100
            ax.text(i, total + total*0.02, f"{pct:.0f}% load",
                   ha="center", va="bottom", fontsize=8)

    return _save(fig, f"04_conversion_overhead_sf{sf}")


def chart_effective_bandwidth(
    bandwidth_data: list[dict],
) -> Path:
    """Chart 5: Achieved bytes/sec vs theoretical peak per query."""
    fig, ax = plt.subplots(figsize=(10, 6))

    queries = [d["query_id"] for d in bandwidth_data]
    achieved = [d["scan_bw_gbs"] for d in bandwidth_data]
    x = np.arange(len(queries))

    ax.bar(x, achieved, color=C_MLX, label="Achieved Scan BW")
    ax.axhline(y=MEMORY_BW_GBS, color=C_UNIFIED, linestyle="--", linewidth=2,
              label=f"M4 Peak ({MEMORY_BW_GBS} GB/s)")

    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Effective Scan Bandwidth vs Theoretical Peak", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.legend()

    # Add utilization labels
    for i, (q, bw) in enumerate(zip(queries, achieved)):
        pct = bw / MEMORY_BW_GBS * 100
        ax.text(i, bw + 2, f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

    return _save(fig, "05_effective_bandwidth")


def chart_scale_factor_scaling(all_results: list[BenchmarkResult]) -> Path:
    """Chart 6: SF 0.1/1/10 for all three baselines."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Scaling Across Scale Factors", fontsize=14, fontweight="bold")

    sfs = sorted(set(r.sf for r in all_results))

    for i, qid in enumerate(QUERY_IDS):
        ax = axes[i // 3][i % 3]

        for baseline, color, label in [
            ("duckdb", C_DUCKDB, "DuckDB"),
            ("numpy", C_NUMPY, "NumPy"),
            ("mlx", C_MLX, "MLX"),
        ]:
            times = []
            sf_vals = []
            for sf in sfs:
                for r in all_results:
                    if r.query_id == qid and r.sf == sf and r.baseline == baseline:
                        times.append(r.warm_mean_s * 1000)
                        sf_vals.append(sf)
            if times:
                ax.plot(sf_vals, times, "o-", color=color, label=label, linewidth=2)

        ax.set_title(qid)
        ax.set_xlabel("Scale Factor")
        ax.set_ylabel("Time (ms)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if i == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, "06_scale_factor_scaling")


def chart_transfer_scenario(scenarios: list[TransferScenario]) -> Path:
    """Chart 7: 'What if PCIe?' — modeled additional time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    queries = [s.query_id for s in scenarios]
    x = np.arange(len(queries))
    width = 0.25

    gpu_times = [s.gpu_compute_s * 1000 for s in scenarios]
    pcie4_times = [s.pcie4_transfer_s * 1000 for s in scenarios]
    pcie5_times = [s.pcie5_transfer_s * 1000 for s in scenarios]

    # Stacked: GPU compute + PCIe transfer
    ax.bar(x - width, gpu_times, width, label="GPU Compute", color=C_MLX)

    ax.bar(x, gpu_times, width, color=C_MLX, alpha=0.5)
    ax.bar(x, pcie4_times, width, bottom=gpu_times,
           label="+ PCIe 4.0 Transfer", color=C_PCIE)

    ax.bar(x + width, gpu_times, width, color=C_MLX, alpha=0.5)
    ax.bar(x + width, pcie5_times, width, bottom=gpu_times,
           label="+ PCIe 5.0 Transfer", color=C_PCIE5)

    ax.set_ylabel("Time (ms)")
    ax.set_title("Transfer Overhead Scenario: What If PCIe?", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.legend()

    ax.text(0.5, -0.12,
           "Left: unified (compute only) | Center: + PCIe 4.0 | Right: + PCIe 5.0",
           transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="gray")

    return _save(fig, "07_transfer_scenario")


def generate_all_charts(
    benchmark_results: list[BenchmarkResult],
    scenarios: list[TransferScenario],
    bandwidth_data: list[dict] | None = None,
) -> list[Path]:
    """Generate all charts and return paths."""
    paths = []
    sfs = sorted(set(r.sf for r in benchmark_results))

    paths.append(chart_architecture_comparison())
    paths.append(chart_bandwidth_comparison())

    for sf in sfs:
        paths.append(chart_three_baseline(benchmark_results, sf))
        paths.append(chart_conversion_overhead(benchmark_results, sf))

    if bandwidth_data:
        paths.append(chart_effective_bandwidth(bandwidth_data))

    if len(sfs) > 1:
        paths.append(chart_scale_factor_scaling(benchmark_results))

    if scenarios:
        paths.append(chart_transfer_scenario(scenarios))

    return paths
