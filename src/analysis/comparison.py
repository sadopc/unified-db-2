"""Normalized comparison vs PCIe scenario model.

Does NOT directly compare M4 custom kernels to published GPU-DB numbers.
Instead builds a transfer overhead scenario model:
- Given PCIe practical bandwidth, how much time would data transfer add?
- What speedup would we lose if we had a PCIe bus?
- Models "what if our M4 GPU had a PCIe bus"
"""

from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table

from src.analysis.theoretical import TransferScenario
from src.config import MEMORY_BW_GBS, PCIE4_BW_GBS, PCIE5_BW_GBS

console = Console()


@dataclass
class ComparisonSummary:
    """Summary of unified memory advantage across all queries."""
    scenarios: list[TransferScenario]

    @property
    def mean_pcie4_overhead_pct(self) -> float:
        vals = [s.pcie4_overhead_pct for s in self.scenarios if s.gpu_compute_s > 0]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_pcie5_overhead_pct(self) -> float:
        vals = [s.pcie5_overhead_pct for s in self.scenarios if s.gpu_compute_s > 0]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_unified_speedup_vs_pcie4(self) -> float:
        vals = [s.unified_speedup_vs_pcie4 for s in self.scenarios if s.gpu_compute_s > 0]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def max_pcie4_overhead_pct(self) -> float:
        vals = [s.pcie4_overhead_pct for s in self.scenarios if s.gpu_compute_s > 0]
        return float(max(vals)) if vals else 0.0

    @property
    def max_query_pcie4(self) -> str:
        if not self.scenarios:
            return ""
        return max(
            (s for s in self.scenarios if s.gpu_compute_s > 0),
            key=lambda s: s.pcie4_overhead_pct,
        ).query_id

    @property
    def total_data_gb(self) -> float:
        return sum(s.data_bytes for s in self.scenarios) / 1e9


def build_comparison(scenarios: list[TransferScenario]) -> ComparisonSummary:
    """Build a comparison summary from transfer scenarios."""
    return ComparisonSummary(scenarios=scenarios)


def print_comparison(summary: ComparisonSummary) -> None:
    """Print comparison summary."""
    console.print("\n[bold]Unified Memory Advantage Summary[/bold]")
    console.print(f"  Total data scanned: {summary.total_data_gb:.2f} GB")
    console.print(f"  Mean PCIe 4.0 overhead: {summary.mean_pcie4_overhead_pct:.1f}%")
    console.print(f"  Mean PCIe 5.0 overhead: {summary.mean_pcie5_overhead_pct:.1f}%")
    console.print(f"  Mean speedup vs PCIe 4.0: {summary.mean_unified_speedup_vs_pcie4:.2f}x")
    console.print(f"  Worst case: {summary.max_query_pcie4} "
                  f"({summary.max_pcie4_overhead_pct:.1f}% PCIe 4.0 overhead)")

    console.print("\n[dim]Note: PCIe numbers are a scenario model, not direct hardware comparison.[/dim]")
    console.print("[dim]Newer NVIDIA systems (NVLink-C2C, PCIe 6.0) reduce this gap.[/dim]")


def context_table() -> None:
    """Print context: bandwidth hierarchy for reference."""
    table = Table(title="Memory/Bus Bandwidth Reference")
    table.add_column("Architecture", style="cyan")
    table.add_column("Bandwidth (GB/s)", justify="right")
    table.add_column("Notes")

    table.add_row("PCIe 3.0 x16", "~12", "Older NVIDIA GPUs")
    table.add_row("PCIe 4.0 x16", "~25", "RTX 3090, A100 (host↔device)")
    table.add_row("PCIe 5.0 x16", "~50", "RTX 4090, H100 (host↔device)")
    table.add_row("Apple M4 Unified", "~120", "No bus transfer needed")
    table.add_row("NVIDIA HBM3 (on-chip)", "~3,350", "H100 GPU VRAM bandwidth")
    table.add_row("NVLink-C2C", "~900", "Grace Hopper unified (reduces gap)")

    console.print(table)
    console.print("[dim]Key insight: unified memory eliminates the PCIe transfer,[/dim]")
    console.print("[dim]but NVIDIA HBM3 on-chip bandwidth far exceeds M4's 120 GB/s.[/dim]")
    console.print("[dim]The advantage is eliminating the bottleneck, not raw bandwidth.[/dim]")
