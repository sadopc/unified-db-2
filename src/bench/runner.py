"""Benchmark orchestrator with cold/warm timing and statistics.

Protocol:
1. mx.clear_cache() — ensure clean state
2. Run 1 cold iteration (timed) — JIT compilation + cache population
3. Run N warm iterations (timed) — steady-state performance
4. Report cold and warm separately
"""

import time
from dataclasses import dataclass, field

import duckdb
import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.table import Table

from src.config import COLD_RUNS, QUERY_IDS, WARM_RUNS, db_path
from src.cpu.duckdb_queries import QUERIES as DUCKDB_QUERIES
from src.cpu.numpy_queries import QUERIES as NUMPY_QUERIES
from src.gpu.query_q1 import q1 as mlx_q1
from src.gpu.query_q3 import q3 as mlx_q3
from src.gpu.query_q5 import q5 as mlx_q5
from src.gpu.query_q6 import q6 as mlx_q6
from src.gpu.query_q12 import q12 as mlx_q12
from src.gpu.query_q14 import q14 as mlx_q14

console = Console()

MLX_QUERIES = {
    "Q1": mlx_q1,
    "Q3": mlx_q3,
    "Q5": mlx_q5,
    "Q6": mlx_q6,
    "Q12": mlx_q12,
    "Q14": mlx_q14,
}


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single query."""
    query_id: str
    baseline: str        # "duckdb", "numpy", "mlx"
    sf: float
    cold_s: float
    warm_times_s: list[float] = field(default_factory=list)
    load_s: float = 0.0          # Data loading time (for numpy/mlx)
    peak_memory_mb: float = 0.0  # Peak memory for MLX

    @property
    def warm_mean_s(self) -> float:
        return float(np.mean(self.warm_times_s)) if self.warm_times_s else 0.0

    @property
    def warm_std_s(self) -> float:
        return float(np.std(self.warm_times_s)) if self.warm_times_s else 0.0

    @property
    def warm_median_s(self) -> float:
        return float(np.median(self.warm_times_s)) if self.warm_times_s else 0.0


def run_duckdb(conn: duckdb.DuckDBPyConnection, query_id: str,
               n_warm: int = WARM_RUNS) -> BenchmarkResult:
    """Benchmark a DuckDB SQL query."""
    qfn = DUCKDB_QUERIES[query_id]

    # Cold run
    result = qfn(conn)
    cold_s = result.elapsed_s

    # Warm runs
    warm = []
    for _ in range(n_warm):
        result = qfn(conn)
        warm.append(result.elapsed_s)

    return BenchmarkResult(
        query_id=query_id, baseline="duckdb", sf=0,
        cold_s=cold_s, warm_times_s=warm,
    )


def run_numpy(conn: duckdb.DuckDBPyConnection, query_id: str,
              n_warm: int = WARM_RUNS) -> BenchmarkResult:
    """Benchmark a NumPy CPU kernel."""
    qfn = NUMPY_QUERIES[query_id]

    # Cold run
    result = qfn(conn)
    cold_s = result.compute_s
    load_s = result.load_s

    # Warm runs (re-execute full pipeline each time for honest timing)
    warm = []
    for _ in range(n_warm):
        result = qfn(conn)
        warm.append(result.compute_s)

    return BenchmarkResult(
        query_id=query_id, baseline="numpy", sf=0,
        cold_s=cold_s, warm_times_s=warm, load_s=load_s,
    )


def run_mlx(conn: duckdb.DuckDBPyConnection, query_id: str,
            n_warm: int = WARM_RUNS) -> BenchmarkResult:
    """Benchmark an MLX GPU kernel."""
    qfn = MLX_QUERIES[query_id]

    # Cold run: clear cache first
    mx.clear_cache()
    mx.reset_peak_memory()

    data, compute_s, load_timing = qfn(conn)
    cold_s = compute_s

    peak_mem = mx.get_peak_memory() / 1024**2  # MB

    # Warm runs
    warm = []
    for _ in range(n_warm):
        mx.clear_cache()
        _, compute_s, _ = qfn(conn)
        warm.append(compute_s)

    return BenchmarkResult(
        query_id=query_id, baseline="mlx", sf=0,
        cold_s=cold_s, warm_times_s=warm,
        load_s=load_timing.total_s,
        peak_memory_mb=peak_mem,
    )


def run_all_queries(sf: float, n_warm: int = WARM_RUNS,
                    query_ids: list[str] | None = None) -> list[BenchmarkResult]:
    """Run all three baselines for specified queries at a given scale factor.

    Args:
        sf: Scale factor.
        n_warm: Number of warm iterations.
        query_ids: Which queries to run (default: all).

    Returns:
        List of BenchmarkResult for each query/baseline combination.
    """
    if query_ids is None:
        query_ids = QUERY_IDS

    conn = duckdb.connect(str(db_path(sf)), read_only=True)
    results = []

    for qid in query_ids:
        console.print(f"\n[bold cyan]--- {qid} @ SF{sf} ---[/bold cyan]")

        # DuckDB
        console.print(f"  DuckDB SQL...", end=" ")
        r = run_duckdb(conn, qid, n_warm)
        r.sf = sf
        results.append(r)
        console.print(f"cold={r.cold_s:.4f}s warm={r.warm_mean_s:.4f}s")

        # NumPy
        console.print(f"  NumPy CPU...", end=" ")
        r = run_numpy(conn, qid, n_warm)
        r.sf = sf
        results.append(r)
        console.print(f"cold={r.cold_s:.4f}s warm={r.warm_mean_s:.4f}s")

        # MLX
        console.print(f"  MLX GPU...", end=" ")
        r = run_mlx(conn, qid, n_warm)
        r.sf = sf
        results.append(r)
        console.print(f"cold={r.cold_s:.4f}s warm={r.warm_mean_s:.4f}s peak={r.peak_memory_mb:.0f}MB")

    conn.close()
    return results


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a rich table."""
    table = Table(title="Benchmark Results")
    table.add_column("Query", style="cyan")
    table.add_column("Baseline", style="magenta")
    table.add_column("SF", justify="right")
    table.add_column("Cold (s)", justify="right")
    table.add_column("Warm Mean (s)", justify="right")
    table.add_column("Warm Std (s)", justify="right")
    table.add_column("Peak Mem (MB)", justify="right")

    for r in results:
        table.add_row(
            r.query_id, r.baseline, str(r.sf),
            f"{r.cold_s:.4f}", f"{r.warm_mean_s:.4f}",
            f"{r.warm_std_s:.4f}",
            f"{r.peak_memory_mb:.0f}" if r.peak_memory_mb > 0 else "-",
        )

    console.print(table)
