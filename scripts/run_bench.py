#!/usr/bin/env python3
"""Run specific benchmarks.

Usage:
    uv run python scripts/run_bench.py --query Q6 --sf 0.1 --baseline mlx
    uv run python scripts/run_bench.py --query Q1,Q6 --sf 1 --baseline all
    uv run python scripts/run_bench.py --validate --sf 0.1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
from rich.console import Console

from src.bench.memory import check_memory_before_query, print_memory_budgets
from src.bench.runner import (
    print_results_table,
    run_duckdb,
    run_mlx,
    run_numpy,
)
from src.bench.validators import validate_all
from src.config import QUERY_IDS, db_path
from src.data.generate import generate

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Run specific benchmarks")
    p.add_argument("--query", type=str, default="all",
                   help="Comma-separated query IDs or 'all' (default: all)")
    p.add_argument("--sf", type=float, default=0.1,
                   help="Scale factor (default: 0.1)")
    p.add_argument("--baseline", type=str, default="all",
                   choices=["duckdb", "numpy", "mlx", "all"],
                   help="Which baseline(s) to run (default: all)")
    p.add_argument("--warm", type=int, default=9,
                   help="Number of warm iterations (default: 9)")
    p.add_argument("--validate", action="store_true",
                   help="Run validation only")
    p.add_argument("--generate", action="store_true",
                   help="Generate data before benchmarking")
    p.add_argument("--memory", action="store_true",
                   help="Show memory budgets only")
    return p.parse_args()


def main():
    args = parse_args()

    if args.query == "all":
        query_ids = QUERY_IDS
    else:
        query_ids = [q.strip().upper() for q in args.query.split(",")]
        for qid in query_ids:
            if qid not in QUERY_IDS:
                console.print(f"[red]Unknown query: {qid}. Valid: {QUERY_IDS}[/red]")
                sys.exit(1)

    sf = args.sf

    if args.memory:
        print_memory_budgets(sf)
        return

    if args.generate:
        generate(sf)

    if args.validate:
        validate_all(sf, query_ids)
        return

    # Ensure data exists
    if not db_path(sf).exists():
        console.print(f"[yellow]Generating TPC-H SF{sf} data...[/yellow]")
        generate(sf)

    conn = duckdb.connect(str(db_path(sf)), read_only=True)
    results = []

    for qid in query_ids:
        console.print(f"\n[bold cyan]--- {qid} @ SF{sf} ---[/bold cyan]")

        # Check memory budget
        budget = check_memory_before_query(qid, sf)

        baselines = ["duckdb", "numpy", "mlx"] if args.baseline == "all" else [args.baseline]

        for baseline in baselines:
            console.print(f"  {baseline}...", end=" ")

            if baseline == "duckdb":
                r = run_duckdb(conn, qid, args.warm)
            elif baseline == "numpy":
                r = run_numpy(conn, qid, args.warm)
            elif baseline == "mlx":
                r = run_mlx(conn, qid, args.warm)

            r.sf = sf
            results.append(r)
            console.print(f"cold={r.cold_s:.4f}s warm={r.warm_mean_s:.4f}s")

    conn.close()
    print_results_table(results)


if __name__ == "__main__":
    main()
