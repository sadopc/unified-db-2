#!/usr/bin/env python3
"""Full benchmark suite: generate → validate → benchmark → analyze → chart.

Usage:
    uv run python scripts/run_all.py [--sf 0.1,1] [--warm 5] [--skip-gen] [--skip-validate]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.analysis.bandwidth_model import BandwidthMetrics, print_bandwidth_table
from src.analysis.comparison import build_comparison, context_table, print_comparison
from src.analysis.theoretical import (
    QUERY_DATA_VOLUMES_SF1,
    model_transfer_scenarios,
    print_scenarios,
)
from src.bench.memory import print_memory_budgets
from src.bench.runner import print_results_table, run_all_queries
from src.bench.transfer import measure_transfer, print_transfer_table
from src.bench.validators import validate_all
from src.config import QUERY_IDS, RESULTS_DIR, SCALE_FACTORS
from src.data.generate import generate
from src.viz.charts import generate_all_charts

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Unified-DB-2 Full Benchmark Suite")
    p.add_argument("--sf", type=str, default="0.1,1",
                   help="Comma-separated scale factors (default: 0.1,1)")
    p.add_argument("--warm", type=int, default=9,
                   help="Number of warm iterations (default: 9)")
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip data generation")
    p.add_argument("--skip-validate", action="store_true",
                   help="Skip validation step")
    p.add_argument("--transfer-reps", type=int, default=100,
                   help="Transfer measurement repetitions (default: 100)")
    return p.parse_args()


def main():
    args = parse_args()
    sfs = [float(s.strip()) for s in args.sf.split(",")]

    console.print("[bold green]=" * 60)
    console.print("[bold green]Unified-DB-2: Apple Silicon GPU-Accelerated Analytics[/bold green]")
    console.print("[bold green]=" * 60)

    t_start = time.perf_counter()

    # --- Step 1: Data generation ---
    if not args.skip_gen:
        console.print("\n[bold]Step 1: Data Generation[/bold]")
        for sf in sfs:
            generate(sf)
    else:
        console.print("\n[bold]Step 1: Data Generation (skipped)[/bold]")

    # --- Step 2: Memory budgets ---
    console.print("\n[bold]Step 2: Memory Budgets[/bold]")
    for sf in sfs:
        print_memory_budgets(sf)

    # --- Step 3: Validation ---
    if not args.skip_validate:
        console.print("\n[bold]Step 3: Validation (at smallest SF)[/bold]")
        val_sf = min(sfs)
        val_results = validate_all(val_sf)
        n_fail = sum(1 for r in val_results if not r.passed)
        if n_fail > 0:
            console.print(f"[bold red]{n_fail} validation(s) failed! Fix before benchmarking.[/bold red]")
            for r in val_results:
                if not r.passed:
                    console.print(f"  [red]{r.query_id} ({r.baseline}): {r.message}[/red]")
    else:
        console.print("\n[bold]Step 3: Validation (skipped)[/bold]")

    # --- Step 4: Benchmarks ---
    console.print("\n[bold]Step 4: Benchmarks[/bold]")
    all_results = []
    for sf in sfs:
        console.print(f"\n[bold cyan]Running benchmarks at SF{sf}...[/bold cyan]")
        results = run_all_queries(sf, n_warm=args.warm)
        all_results.extend(results)

    print_results_table(all_results)

    # --- Step 5: Transfer measurement ---
    console.print("\n[bold]Step 5: Transfer Overhead Measurement[/bold]")
    transfer_results = []
    for sf in sfs:
        tr = measure_transfer(sf, n_reps=args.transfer_reps)
        transfer_results.append(tr)
    print_transfer_table(transfer_results)

    # --- Step 6: Theoretical analysis ---
    console.print("\n[bold]Step 6: Theoretical Analysis[/bold]")
    context_table()

    # Build scenario model using benchmark data
    for sf in sfs:
        gpu_times = {}
        for r in all_results:
            if r.sf == sf and r.baseline == "mlx":
                gpu_times[r.query_id] = r.warm_mean_s

        if gpu_times:
            scenarios = model_transfer_scenarios(gpu_times, sf)
            print_scenarios(scenarios)

            summary = build_comparison(scenarios)
            print_comparison(summary)

    # --- Step 7: Bandwidth analysis ---
    console.print("\n[bold]Step 7: Bandwidth Analysis[/bold]")
    bw_data = []
    for sf in sfs:
        for r in all_results:
            if r.sf == sf and r.baseline == "mlx":
                data_bytes = int(QUERY_DATA_VOLUMES_SF1.get(r.query_id, 0) * sf)
                bw_data.append({
                    "query_id": r.query_id,
                    "sf": sf,
                    "bytes_scanned": data_bytes,
                    "compute_s": r.warm_mean_s,
                    "load_s": r.load_s,
                    "scan_bw_gbs": data_bytes / r.warm_mean_s / 1e9 if r.warm_mean_s > 0 else 0,
                })

    bw_metrics = [
        BandwidthMetrics(
            query_id=d["query_id"], sf=d["sf"],
            bytes_scanned=d["bytes_scanned"],
            compute_s=d["compute_s"], load_s=d["load_s"],
        ) for d in bw_data
    ]
    print_bandwidth_table(bw_metrics)

    # --- Step 8: Charts ---
    console.print("\n[bold]Step 8: Chart Generation[/bold]")
    # Get scenarios for the largest SF
    largest_sf = max(sfs)
    gpu_times_largest = {}
    for r in all_results:
        if r.sf == largest_sf and r.baseline == "mlx":
            gpu_times_largest[r.query_id] = r.warm_mean_s
    chart_scenarios = model_transfer_scenarios(gpu_times_largest, largest_sf) if gpu_times_largest else []

    chart_paths = generate_all_charts(all_results, chart_scenarios, bw_data)
    for p in chart_paths:
        console.print(f"  [green]Saved: {p}[/green]")

    # --- Step 9: Save raw results ---
    console.print("\n[bold]Step 9: Saving Results[/bold]")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = []
    for r in all_results:
        raw_data.append({
            "query_id": r.query_id,
            "baseline": r.baseline,
            "sf": r.sf,
            "cold_s": r.cold_s,
            "warm_mean_s": r.warm_mean_s,
            "warm_std_s": r.warm_std_s,
            "warm_median_s": r.warm_median_s,
            "load_s": r.load_s,
            "peak_memory_mb": r.peak_memory_mb,
            "warm_times_s": r.warm_times_s,
        })

    results_path = RESULTS_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    console.print(f"  [green]Saved: {results_path}[/green]")

    t_end = time.perf_counter()
    console.print(f"\n[bold green]Done in {t_end - t_start:.1f}s[/bold green]")


if __name__ == "__main__":
    main()
