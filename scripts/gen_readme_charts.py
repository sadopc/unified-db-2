#!/usr/bin/env python3
"""Generate charts for README.md — formal, clean, annotated."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import RESULTS_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load benchmark data
with open(RESULTS_DIR / "benchmark_results.json") as f:
    raw = json.load(f)

# Colors
C_DUCKDB = "#2196F3"
C_NUMPY = "#9C27B0"
C_MLX = "#FF9800"
C_PCIE = "#F44336"
C_UNIFIED = "#4CAF50"

QUERIES = ["Q1", "Q3", "Q5", "Q6", "Q12", "Q14"]


def _get(baseline: str, sf: float) -> dict[str, float]:
    """Get warm_mean_s per query for a baseline at a given SF."""
    out = {}
    for r in raw:
        if r["baseline"] == baseline and r["sf"] == sf:
            out[r["query_id"]] = r["warm_mean_s"]
    return out


def _annot_bar(ax, bars, fmt="{:.1f}", offset=3, fontsize=7):
    """Annotate bars with their value."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    fmt.format(h), ha="center", va="bottom", fontsize=fontsize)


# ── Chart 1: Three-baseline comparison at SF1 (ms, lower is better) ──
def chart_three_baseline_sf1():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ddb = _get("duckdb", 1.0)
    npy = _get("numpy", 1.0)
    mlx = _get("mlx", 1.0)

    x = np.arange(len(QUERIES))
    w = 0.25

    vals_d = [ddb.get(q, 0) * 1000 for q in QUERIES]
    vals_n = [npy.get(q, 0) * 1000 for q in QUERIES]
    vals_m = [mlx.get(q, 0) * 1000 for q in QUERIES]

    b1 = ax.bar(x - w, vals_d, w, label="DuckDB SQL", color=C_DUCKDB, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x, vals_n, w, label="NumPy CPU", color=C_NUMPY, edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + w, vals_m, w, label="MLX GPU", color=C_MLX, edgecolor="white", linewidth=0.5)

    _annot_bar(ax, b1, fmt="{:.1f}", offset=1, fontsize=7)
    _annot_bar(ax, b2, fmt="{:.1f}", offset=1, fontsize=7)
    _annot_bar(ax, b3, fmt="{:.1f}", offset=1, fontsize=7)

    ax.set_xlabel("TPC-H Query", fontsize=11)
    ax.set_ylabel("Warm Mean Execution Time (ms)", fontsize=11)
    ax.set_title("Three-Baseline Comparison at SF1 (~6M rows)\nLower is better", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(QUERIES, fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = RESULTS_DIR / "readme_three_baseline_sf1.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Chart 2: MLX GPU vs NumPy CPU speedup at SF1 and SF10 ──
def chart_gpu_speedup():
    fig, ax = plt.subplots(figsize=(10, 5))

    npy1 = _get("numpy", 1.0)
    mlx1 = _get("mlx", 1.0)
    npy10 = _get("numpy", 10.0)
    mlx10 = _get("mlx", 10.0)

    speedup_sf1 = [npy1[q] / mlx1[q] if mlx1.get(q, 0) > 0 else 0 for q in QUERIES]
    speedup_sf10 = [npy10[q] / mlx10[q] if mlx10.get(q, 0) > 0 else 0 for q in QUERIES]

    x = np.arange(len(QUERIES))
    w = 0.35

    b1 = ax.bar(x - w / 2, speedup_sf1, w, label="SF1 (~6M rows)", color=C_MLX, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w / 2, speedup_sf10, w, label="SF10 (~60M rows)", color="#E65100", edgecolor="white", linewidth=0.5)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                    f"{h:.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(QUERIES) - 0.5, 1.05, "breakeven (1.0x)", fontsize=7, color="gray", ha="right")

    ax.set_xlabel("TPC-H Query", fontsize=11)
    ax.set_ylabel("GPU Speedup (MLX / NumPy)", fontsize=11)
    ax.set_title("MLX GPU vs NumPy CPU Speedup\nHigher is better  |  Same algorithm, fair comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(QUERIES, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(speedup_sf1), max(speedup_sf10)) * 1.25)

    fig.tight_layout()
    path = RESULTS_DIR / "readme_gpu_speedup.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Chart 3: Scale factor scaling (SF 0.1 → 1 → 10, MLX GPU) ──
def chart_scaling():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    sfs = [0.1, 1.0, 10.0]
    sf_labels = ["SF 0.1", "SF 1", "SF 10"]
    colors = ["#FFC107", C_MLX, "#E65100"]

    x = np.arange(len(QUERIES))
    w = 0.25

    for i, sf in enumerate(sfs):
        mlx_data = _get("mlx", sf)
        vals = [mlx_data.get(q, 0) * 1000 for q in QUERIES]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=sf_labels[i],
                      color=colors[i], edgecolor="white", linewidth=0.5)
        _annot_bar(ax, bars, fmt="{:.0f}", offset=2, fontsize=6.5)

    ax.set_xlabel("TPC-H Query", fontsize=11)
    ax.set_ylabel("MLX GPU Warm Mean (ms)", fontsize=11)
    ax.set_title("MLX GPU Performance Across Scale Factors\nLower is better  |  Data scales ~10x per step",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(QUERIES, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_ylabel("MLX GPU Warm Mean (ms, log scale)", fontsize=11)

    fig.tight_layout()
    path = RESULTS_DIR / "readme_scaling.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Chart 4: PCIe transfer overhead scenario ──
def chart_pcie_overhead():
    from src.analysis.theoretical import QUERY_DATA_VOLUMES_SF1
    from src.config import PCIE4_BW_GBS, PCIE5_BW_GBS

    fig, ax = plt.subplots(figsize=(10, 5.5))

    mlx1 = _get("mlx", 1.0)
    x = np.arange(len(QUERIES))
    w = 0.2

    gpu_ms = []
    pcie4_ms = []
    pcie5_ms = []

    for q in QUERIES:
        compute = mlx1.get(q, 0) * 1000
        data_bytes = QUERY_DATA_VOLUMES_SF1.get(q, 0)
        p4 = data_bytes / (PCIE4_BW_GBS * 1e9) * 1000
        p5 = data_bytes / (PCIE5_BW_GBS * 1e9) * 1000
        gpu_ms.append(compute)
        pcie4_ms.append(p4)
        pcie5_ms.append(p5)

    b1 = ax.bar(x - w, gpu_ms, w, label="GPU Compute (unified)", color=C_MLX, edgecolor="white", linewidth=0.5)
    # Stacked: GPU + PCIe 4.0 transfer
    b2_base = gpu_ms
    b2 = ax.bar(x, b2_base, w, color=C_MLX, edgecolor="white", linewidth=0.5, alpha=0.5)
    b2t = ax.bar(x, pcie4_ms, w, bottom=b2_base, label="+ PCIe 4.0 transfer", color=C_PCIE, edgecolor="white", linewidth=0.5)
    # Stacked: GPU + PCIe 5.0 transfer
    b3 = ax.bar(x + w, b2_base, w, color=C_MLX, edgecolor="white", linewidth=0.5, alpha=0.5)
    b3t = ax.bar(x + w, pcie5_ms, w, bottom=b2_base, label="+ PCIe 5.0 transfer", color="#FF8A80", edgecolor="white", linewidth=0.5)

    # Annotate overhead %
    for i, q in enumerate(QUERIES):
        total_p4 = gpu_ms[i] + pcie4_ms[i]
        pct = pcie4_ms[i] / total_p4 * 100 if total_p4 > 0 else 0
        ax.text(x[i], total_p4 + 1, f"+{pct:.0f}%", ha="center", va="bottom", fontsize=7, color=C_PCIE, fontweight="bold")

    ax.set_xlabel("TPC-H Query", fontsize=11)
    ax.set_ylabel("Execution Time (ms)", fontsize=11)
    ax.set_title("Transfer Overhead Scenario at SF1\nWhat if the M4 GPU were behind a PCIe bus?  |  Lower is better",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(QUERIES, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = RESULTS_DIR / "readme_pcie_overhead.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Chart 5: Bus bandwidth comparison ──
def chart_bandwidth():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    labels = ["PCIe 3.0\nx16", "PCIe 4.0\nx16", "PCIe 5.0\nx16", "M4 Unified\nMemory"]
    values = [12, 25, 50, 120]
    colors = [C_PCIE, C_PCIE, C_PCIE, C_UNIFIED]
    alphas = [0.5, 0.7, 0.9, 1.0]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f"{v} GB/s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
    ax.set_title("Memory Bus Bandwidth Comparison\nHigher is better  |  Practical throughput",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 145)

    fig.tight_layout()
    path = RESULTS_DIR / "readme_bandwidth.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    chart_three_baseline_sf1()
    chart_gpu_speedup()
    chart_scaling()
    chart_pcie_overhead()
    chart_bandwidth()
    print("\nAll README charts generated.")
