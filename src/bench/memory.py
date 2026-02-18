"""Memory budgeting and runtime tracking.

Pre-computes memory budgets per query and validates before execution.
Tracks peak usage at runtime using MLX memory APIs.
"""

from dataclasses import dataclass

import mlx.core as mx
from rich.console import Console
from rich.table import Table

from src.config import (
    CHUNK_THRESHOLD_BYTES,
    MAX_USABLE_BYTES,
    MEMORY_BUDGETS_SF1,
    MEMORY_BUDGETS_SF10,
    QUERY_IDS,
)

console = Console()


@dataclass
class MemoryBudget:
    """Memory budget for a query at a given scale factor."""
    query_id: str
    sf: float
    input_bytes: int            # Steady-state input arrays
    transient_peak_bytes: int   # Peak during loading (3x input)
    fits_in_memory: bool
    needs_chunking: bool


@dataclass
class MemoryUsage:
    """Runtime memory usage for a query execution."""
    query_id: str
    sf: float
    peak_mb: float
    active_mb: float
    budget_mb: float
    within_budget: bool


def compute_budget(query_id: str, sf: float) -> MemoryBudget:
    """Pre-compute memory budget for a query at a given scale factor.

    At SF1, budgets are pre-defined. At other SFs, scale linearly.
    Transient peak is ~3x input (DuckDB buffer + NumPy + MLX coexist
    briefly during per-column loading).
    """
    if sf <= 1:
        base = MEMORY_BUDGETS_SF1.get(query_id, 256 * 1024**2)
        input_bytes = int(base * sf)
    else:
        base = MEMORY_BUDGETS_SF10.get(query_id, 2560 * 1024**2)
        input_bytes = int(base * sf / 10)

    # Transient peak: up to 3x during loading (with per-column loading
    # this is reduced to ~2x of the largest single column set)
    transient_peak = int(input_bytes * 2.5)

    fits = transient_peak < MAX_USABLE_BYTES
    needs_chunking = transient_peak > CHUNK_THRESHOLD_BYTES

    return MemoryBudget(
        query_id=query_id,
        sf=sf,
        input_bytes=input_bytes,
        transient_peak_bytes=transient_peak,
        fits_in_memory=fits,
        needs_chunking=needs_chunking,
    )


def check_memory_before_query(query_id: str, sf: float) -> MemoryBudget:
    """Validate memory budget before executing a query.

    Returns the budget. Prints a warning if memory is tight.
    """
    budget = compute_budget(query_id, sf)

    if not budget.fits_in_memory:
        console.print(
            f"[bold red]WARNING: {query_id} at SF{sf} may exceed memory "
            f"({budget.transient_peak_bytes / 1e9:.1f} GB transient peak)[/bold red]"
        )

    if budget.needs_chunking:
        console.print(
            f"[yellow]{query_id} at SF{sf}: chunked processing recommended "
            f"({budget.input_bytes / 1e9:.1f} GB input)[/yellow]"
        )

    return budget


def track_memory_usage(query_id: str, sf: float, budget: MemoryBudget) -> MemoryUsage:
    """Capture current MLX memory usage after query execution."""
    peak = mx.get_peak_memory() / 1024**2
    active = mx.get_active_memory() / 1024**2
    budget_mb = budget.input_bytes / 1024**2

    usage = MemoryUsage(
        query_id=query_id,
        sf=sf,
        peak_mb=peak,
        active_mb=active,
        budget_mb=budget_mb,
        within_budget=peak * 1024**2 < budget.transient_peak_bytes * 1.5,
    )

    return usage


def print_memory_budgets(sf: float) -> None:
    """Print memory budgets for all queries at a given scale factor."""
    table = Table(title=f"Memory Budgets @ SF{sf}")
    table.add_column("Query", style="cyan")
    table.add_column("Input (MB)", justify="right")
    table.add_column("Peak (MB)", justify="right")
    table.add_column("Fits?", justify="center")
    table.add_column("Chunking?", justify="center")

    for qid in QUERY_IDS:
        b = compute_budget(qid, sf)
        table.add_row(
            qid,
            f"{b.input_bytes / 1e6:.0f}",
            f"{b.transient_peak_bytes / 1e6:.0f}",
            "[green]yes[/green]" if b.fits_in_memory else "[red]NO[/red]",
            "[yellow]yes[/yellow]" if b.needs_chunking else "no",
        )

    console.print(table)
