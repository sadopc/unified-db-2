"""Measure DuckDB → NumPy → MLX conversion overhead.

100 repetitions of the full conversion pipeline with per-stage timing.
Honestly reports the copy overhead — the copy IS there, but occurs at
memory bandwidth (~120 GB/s) rather than PCIe bandwidth (25-50 GB/s).
"""

import time
from dataclasses import dataclass

import duckdb
import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.table import Table

from src.config import TRANSFER_REPS, db_path
from src.data.loader import LoadTiming, load_columns_mlx

console = Console()

# Use Q6 lineitem columns as the standard transfer benchmark
# (4 float/int columns, scales linearly with SF)
TRANSFER_SQL = """
SELECT
    l_extendedprice::FLOAT AS extendedprice,
    l_discount::FLOAT AS discount,
    l_quantity::FLOAT AS quantity,
    (l_shipdate - DATE '1970-01-01')::INTEGER AS shipdate
FROM lineitem
"""


@dataclass
class TransferBenchmark:
    """Aggregated transfer benchmark results."""
    sf: float
    n_reps: int
    total_bytes: int
    timings: list[LoadTiming]

    @property
    def mean_total_s(self) -> float:
        return float(np.mean([t.total_s for t in self.timings]))

    @property
    def mean_query_s(self) -> float:
        return float(np.mean([t.duckdb_query_s for t in self.timings]))

    @property
    def mean_extract_s(self) -> float:
        return float(np.mean([t.duckdb_extract_s for t in self.timings]))

    @property
    def mean_copy_s(self) -> float:
        return float(np.mean([t.numpy_to_mlx_s for t in self.timings]))

    @property
    def mean_eval_s(self) -> float:
        return float(np.mean([t.mlx_eval_s for t in self.timings]))

    @property
    def effective_bw_gbs(self) -> float:
        if self.mean_total_s == 0:
            return 0.0
        return self.total_bytes / self.mean_total_s / 1e9


def measure_transfer(sf: float, n_reps: int = TRANSFER_REPS) -> TransferBenchmark:
    """Measure conversion pipeline overhead.

    Args:
        sf: Scale factor.
        n_reps: Number of repetitions.

    Returns:
        TransferBenchmark with aggregated results.
    """
    conn = duckdb.connect(str(db_path(sf)), read_only=True)
    timings = []
    total_bytes = 0

    console.print(f"[bold]Measuring transfer overhead at SF{sf} ({n_reps} reps)...[/bold]")

    for i in range(n_reps):
        mx.clear_cache()
        data, timing = load_columns_mlx(conn, TRANSFER_SQL)
        timings.append(timing)
        if i == 0:
            total_bytes = timing.total_bytes
        # Free MLX arrays
        del data

    conn.close()

    result = TransferBenchmark(
        sf=sf, n_reps=n_reps, total_bytes=total_bytes, timings=timings,
    )

    console.print(f"  Total bytes: {total_bytes / 1e6:.1f} MB")
    console.print(f"  Mean total: {result.mean_total_s * 1000:.2f} ms")
    console.print(f"  Mean DuckDB query: {result.mean_query_s * 1000:.2f} ms")
    console.print(f"  Mean DuckDB extract: {result.mean_extract_s * 1000:.2f} ms")
    console.print(f"  Mean NumPy→MLX copy: {result.mean_copy_s * 1000:.2f} ms")
    console.print(f"  Mean MLX eval: {result.mean_eval_s * 1000:.2f} ms")
    console.print(f"  Effective bandwidth: {result.effective_bw_gbs:.2f} GB/s")

    return result


def print_transfer_table(results: list[TransferBenchmark]) -> None:
    """Print transfer benchmark results."""
    table = Table(title="Transfer Overhead (DuckDB → NumPy → MLX)")
    table.add_column("SF", justify="right")
    table.add_column("Data (MB)", justify="right")
    table.add_column("DuckDB Query (ms)", justify="right")
    table.add_column("Extract (ms)", justify="right")
    table.add_column("Copy (ms)", justify="right")
    table.add_column("MLX Eval (ms)", justify="right")
    table.add_column("Total (ms)", justify="right")
    table.add_column("BW (GB/s)", justify="right")

    for r in results:
        table.add_row(
            str(r.sf),
            f"{r.total_bytes / 1e6:.1f}",
            f"{r.mean_query_s * 1000:.2f}",
            f"{r.mean_extract_s * 1000:.2f}",
            f"{r.mean_copy_s * 1000:.2f}",
            f"{r.mean_eval_s * 1000:.2f}",
            f"{r.mean_total_s * 1000:.2f}",
            f"{r.effective_bw_gbs:.2f}",
        )

    console.print(table)
