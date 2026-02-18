#!/usr/bin/env python3
"""Run a custom GPU-favorable analytical query across three baselines.

The query is intentionally scan + arithmetic + high-cardinality grouped
aggregation (no joins) to better expose GPU parallel throughput.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.table import Table

from src.config import db_path
from src.data.generate import generate
from src.data.loader import load_columns_mlx, load_columns_numpy
from src.gpu.primitives import group_by_count, group_by_sum

console = Console()
NUM_BUCKETS = 4096


def qx_lineitem_sql() -> str:
    """Columns for the custom grouped feature-synthesis query."""
    return """
    SELECT
        l_partkey::INTEGER AS partkey,
        l_extendedprice::FLOAT AS extendedprice,
        l_discount::FLOAT AS discount,
        l_tax::FLOAT AS tax,
        l_quantity::FLOAT AS quantity
    FROM lineitem
    """


DUCKDB_QX_SQL = f"""
WITH agg AS (
    SELECT
        (l_partkey % {NUM_BUCKETS})::INTEGER AS bucket,
        SUM(l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax)) AS sum_charge,
        SUM(l_quantity * (1.0 + 0.5 * l_discount)) AS sum_qty_adj,
        SUM(l_extendedprice * l_discount * l_discount + l_quantity * l_tax) AS sum_mix,
        COUNT(*) AS cnt
    FROM lineitem
    GROUP BY 1
)
SELECT
    SUM(
        (sum_charge / cnt) * (sum_qty_adj / cnt)
        + 0.25 * (sum_mix / cnt)
    ) AS score
FROM agg
"""


@dataclass
class BenchResult:
    """Benchmark summary for one baseline."""

    baseline: str
    cold_s: float
    warm_s: list[float] = field(default_factory=list)
    load_s: float = 0.0
    score: float = 0.0

    @property
    def warm_mean_s(self) -> float:
        return float(np.mean(self.warm_s)) if self.warm_s else 0.0


def _compute_numpy(d: dict[str, np.ndarray]) -> float:
    keys = (d["partkey"] % NUM_BUCKETS).astype(np.int32)
    price = d["extendedprice"]
    disc = d["discount"]
    tax = d["tax"]
    qty = d["quantity"]

    charge = price * (1.0 - disc) * (1.0 + tax)
    qty_adj = qty * (1.0 + 0.5 * disc)
    mix = price * disc * disc + qty * tax

    sum_charge = np.zeros(NUM_BUCKETS, dtype=np.float32)
    sum_qty_adj = np.zeros(NUM_BUCKETS, dtype=np.float32)
    sum_mix = np.zeros(NUM_BUCKETS, dtype=np.float32)
    counts = np.zeros(NUM_BUCKETS, dtype=np.float32)

    np.add.at(sum_charge, keys, charge)
    np.add.at(sum_qty_adj, keys, qty_adj)
    np.add.at(sum_mix, keys, mix)
    np.add.at(counts, keys, 1.0)

    safe_counts = np.where(counts > 0, counts, 1.0)
    score = np.sum(
        (sum_charge / safe_counts) * (sum_qty_adj / safe_counts)
        + 0.25 * (sum_mix / safe_counts),
        dtype=np.float64,
    )
    return float(score)


def _compute_mlx(d: dict[str, mx.array]) -> float:
    keys = d["partkey"] % NUM_BUCKETS

    price = d["extendedprice"]
    disc = d["discount"]
    tax = d["tax"]
    qty = d["quantity"]

    charge = price * (1.0 - disc) * (1.0 + tax)
    qty_adj = qty * (1.0 + 0.5 * disc)
    mix = price * disc * disc + qty * tax

    sum_charge = group_by_sum(keys, charge, NUM_BUCKETS)
    sum_qty_adj = group_by_sum(keys, qty_adj, NUM_BUCKETS)
    sum_mix = group_by_sum(keys, mix, NUM_BUCKETS)
    counts = group_by_count(keys, NUM_BUCKETS)

    safe_counts = mx.where(counts > 0, counts, mx.ones_like(counts))
    score = mx.sum(
        (sum_charge / safe_counts) * (sum_qty_adj / safe_counts)
        + 0.25 * (sum_mix / safe_counts)
    )
    mx.eval(score)
    return float(score.item())


def run_duckdb(conn: duckdb.DuckDBPyConnection, warm: int) -> BenchResult:
    t0 = time.perf_counter()
    score = float(conn.execute(DUCKDB_QX_SQL).fetchone()[0])
    t1 = time.perf_counter()
    cold = t1 - t0

    warm_times = []
    for _ in range(warm):
        t0 = time.perf_counter()
        score = float(conn.execute(DUCKDB_QX_SQL).fetchone()[0])
        t1 = time.perf_counter()
        warm_times.append(t1 - t0)

    return BenchResult("duckdb", cold_s=cold, warm_s=warm_times, score=score)


def run_numpy(conn: duckdb.DuckDBPyConnection, warm: int) -> BenchResult:
    d, _ = load_columns_numpy(conn, qx_lineitem_sql())

    t0 = time.perf_counter()
    score = _compute_numpy(d)
    t1 = time.perf_counter()
    cold = t1 - t0

    warm_times = []
    for _ in range(warm):
        t0 = time.perf_counter()
        score = _compute_numpy(d)
        t1 = time.perf_counter()
        warm_times.append(t1 - t0)

    return BenchResult("numpy", cold_s=cold, warm_s=warm_times, load_s=0.0, score=score)


def run_mlx(conn: duckdb.DuckDBPyConnection, warm: int) -> BenchResult:
    d, load_timing = load_columns_mlx(conn, qx_lineitem_sql())

    mx.clear_cache()
    t0 = time.perf_counter()
    score = _compute_mlx(d)
    t1 = time.perf_counter()
    cold = t1 - t0

    warm_times = []
    for _ in range(warm):
        t0 = time.perf_counter()
        score = _compute_mlx(d)
        t1 = time.perf_counter()
        warm_times.append(t1 - t0)

    return BenchResult(
        "mlx",
        cold_s=cold,
        warm_s=warm_times,
        load_s=load_timing.total_s,
        score=score,
    )


def rel_err(got: float, expected: float) -> float:
    if expected == 0:
        return abs(got)
    return abs(got - expected) / abs(expected)


def print_table(results: list[BenchResult]) -> None:
    table = Table(title="GPU-Showcase Query (QX) Benchmark")
    table.add_column("Baseline", style="cyan")
    table.add_column("Cold (ms)", justify="right")
    table.add_column("Warm Mean (ms)", justify="right")
    table.add_column("Pipeline Load (ms)", justify="right")
    table.add_column("Score", justify="right")

    for r in results:
        table.add_row(
            r.baseline,
            f"{r.cold_s * 1000:.2f}",
            f"{r.warm_mean_s * 1000:.2f}",
            f"{r.load_s * 1000:.2f}" if (r.baseline == "mlx" and r.load_s > 0) else "-",
            f"{r.score:.3e}",
        )

    console.print(table)


def parse_args():
    p = argparse.ArgumentParser(description="Run GPU-favored custom query benchmark")
    p.add_argument("--sf", type=float, default=10, help="Scale factor (default: 10)")
    p.add_argument("--warm", type=int, default=9, help="Warm iterations (default: 9)")
    p.add_argument("--generate", action="store_true", help="Generate DB if missing")
    return p.parse_args()


def main():
    args = parse_args()
    sf = args.sf

    if not db_path(sf).exists():
        if not args.generate:
            console.print(
                f"[red]Database not found: {db_path(sf)}. "
                "Run with --generate to create it.[/red]"
            )
            sys.exit(1)
        console.print(f"[yellow]Generating TPC-H SF{sf} data...[/yellow]")
        generate(sf)

    conn = duckdb.connect(str(db_path(sf)), read_only=True)
    try:
        duckdb_r = run_duckdb(conn, args.warm)
        numpy_r = run_numpy(conn, args.warm)
        mlx_r = run_mlx(conn, args.warm)
    finally:
        conn.close()

    results = [duckdb_r, numpy_r, mlx_r]
    print_table(results)

    np_vs_mlx = numpy_r.warm_mean_s / mlx_r.warm_mean_s if mlx_r.warm_mean_s > 0 else float("inf")
    dd_vs_mlx = duckdb_r.warm_mean_s / mlx_r.warm_mean_s if mlx_r.warm_mean_s > 0 else float("inf")
    err_np = rel_err(numpy_r.score, duckdb_r.score)
    err_mlx = rel_err(mlx_r.score, duckdb_r.score)

    console.print(f"\n[bold]MLX vs NumPy warm speedup:[/bold] {np_vs_mlx:.2f}x")
    console.print(f"[bold]DuckDB vs MLX warm ratio:[/bold] {dd_vs_mlx:.2f}x")
    console.print(f"[bold]Relative error vs DuckDB:[/bold] NumPy={err_np:.6f}, MLX={err_mlx:.6f}")


if __name__ == "__main__":
    main()
