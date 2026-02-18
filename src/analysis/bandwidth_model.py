"""Operator-level bandwidth metrics.

Replaces FLOPS-heavy roofline with more defensible metrics for
integer-heavy, branch-heavy SQL analytics:
- Effective scan bandwidth
- Gather/scatter throughput
- Selectivity sensitivity
- End-to-end transfer fraction
"""

from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table

from src.config import MEMORY_BW_GBS, QUERY_IDS

console = Console()


@dataclass
class BandwidthMetrics:
    """Bandwidth metrics for a single query execution."""
    query_id: str
    sf: float
    bytes_scanned: int
    compute_s: float          # GPU compute time
    load_s: float             # Data loading pipeline time

    @property
    def scan_bw_gbs(self) -> float:
        """Effective scan bandwidth: bytes_scanned / compute_time."""
        if self.compute_s == 0:
            return 0.0
        return self.bytes_scanned / self.compute_s / 1e9

    @property
    def bw_utilization(self) -> float:
        """Fraction of theoretical peak memory bandwidth utilized."""
        return self.scan_bw_gbs / MEMORY_BW_GBS

    @property
    def transfer_fraction(self) -> float:
        """Fraction of total time spent on data conversion (load vs compute)."""
        total = self.load_s + self.compute_s
        if total == 0:
            return 0.0
        return self.load_s / total

    @property
    def total_s(self) -> float:
        return self.load_s + self.compute_s


def compute_bandwidth_metrics(
    query_id: str,
    sf: float,
    bytes_scanned: int,
    compute_s: float,
    load_s: float,
) -> BandwidthMetrics:
    """Create bandwidth metrics for a query execution."""
    return BandwidthMetrics(
        query_id=query_id,
        sf=sf,
        bytes_scanned=bytes_scanned,
        compute_s=compute_s,
        load_s=load_s,
    )


@dataclass
class SelectivityAnalysis:
    """How filter selectivity affects achieved bandwidth."""
    query_id: str
    total_rows: int
    filtered_rows: int     # Rows passing filters

    @property
    def selectivity(self) -> float:
        """Fraction of rows passing filters (0-1)."""
        if self.total_rows == 0:
            return 0.0
        return self.filtered_rows / self.total_rows

    @property
    def access_pattern(self) -> str:
        """Characterize the memory access pattern."""
        s = self.selectivity
        if s > 0.8:
            return "sequential (high selectivity)"
        elif s > 0.3:
            return "semi-random (moderate selectivity)"
        else:
            return "sparse/random (low selectivity)"


# Approximate selectivities for TPC-H queries
QUERY_SELECTIVITIES = {
    "Q6":  {"desc": "date+discount+qty filter", "approx": 0.015},
    "Q1":  {"desc": "date filter only", "approx": 0.98},
    "Q14": {"desc": "date filter (1 month)", "approx": 0.08},
    "Q12": {"desc": "mode+date+commit filter", "approx": 0.003},
    "Q3":  {"desc": "segment+date+shipdate", "approx": 0.01},
    "Q5":  {"desc": "region+date+nation match", "approx": 0.005},
}


def print_bandwidth_table(metrics: list[BandwidthMetrics]) -> None:
    """Print bandwidth metrics as a table."""
    table = Table(title="Effective Bandwidth Metrics")
    table.add_column("Query", style="cyan")
    table.add_column("SF", justify="right")
    table.add_column("Scanned (MB)", justify="right")
    table.add_column("Compute (ms)", justify="right")
    table.add_column("Load (ms)", justify="right")
    table.add_column("Scan BW (GB/s)", justify="right")
    table.add_column("BW Util%", justify="right")
    table.add_column("Transfer%", justify="right")

    for m in metrics:
        table.add_row(
            m.query_id,
            str(m.sf),
            f"{m.bytes_scanned / 1e6:.1f}",
            f"{m.compute_s * 1000:.2f}",
            f"{m.load_s * 1000:.2f}",
            f"{m.scan_bw_gbs:.2f}",
            f"{m.bw_utilization * 100:.1f}%",
            f"{m.transfer_fraction * 100:.1f}%",
        )

    console.print(table)
