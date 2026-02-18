"""Theoretical bandwidth analysis: PCIe vs unified memory scenario model.

For each query: data volume scanned, transfer time under PCIe 4.0/5.0,
access time on M4 unified memory.

Key argument: effective_bandwidth = min(memory_bw, transfer_bw).
On discrete GPU: PCIe bottleneck. On M4: no transfer, full 120 GB/s usable.
"""

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from src.config import MEMORY_BW_GBS, PCIE4_BW_GBS, PCIE5_BW_GBS, QUERY_IDS

console = Console()


@dataclass
class TransferScenario:
    """Modeled transfer overhead for a query under different bus architectures."""
    query_id: str
    data_bytes: int
    # Times in seconds
    pcie4_transfer_s: float   # PCIe 4.0 x16 transfer time
    pcie5_transfer_s: float   # PCIe 5.0 x16 transfer time
    unified_transfer_s: float  # Unified memory (no explicit transfer)
    gpu_compute_s: float       # Measured MLX GPU compute time

    @property
    def pcie4_total_s(self) -> float:
        """Total time if data had to cross PCIe 4.0 bus + GPU compute."""
        return self.pcie4_transfer_s + self.gpu_compute_s

    @property
    def pcie5_total_s(self) -> float:
        return self.pcie5_transfer_s + self.gpu_compute_s

    @property
    def unified_total_s(self) -> float:
        """On unified memory: just GPU compute, no bus transfer."""
        return self.gpu_compute_s

    @property
    def pcie4_overhead_pct(self) -> float:
        """Percentage of total time spent on PCIe 4.0 transfer."""
        if self.pcie4_total_s == 0:
            return 0.0
        return 100.0 * self.pcie4_transfer_s / self.pcie4_total_s

    @property
    def pcie5_overhead_pct(self) -> float:
        if self.pcie5_total_s == 0:
            return 0.0
        return 100.0 * self.pcie5_transfer_s / self.pcie5_total_s

    @property
    def unified_speedup_vs_pcie4(self) -> float:
        """Speedup from eliminating PCIe 4.0 transfer."""
        if self.unified_total_s == 0:
            return 0.0
        return self.pcie4_total_s / self.unified_total_s

    @property
    def unified_speedup_vs_pcie5(self) -> float:
        if self.unified_total_s == 0:
            return 0.0
        return self.pcie5_total_s / self.unified_total_s


# Per-query data volumes at SF1 (bytes, 4 bytes per column per row)
# lineitem ~6M rows at SF1
_LI_ROWS_SF1 = 6_001_215
_ORDERS_ROWS_SF1 = 1_500_000
_CUSTOMER_ROWS_SF1 = 150_000
_SUPPLIER_ROWS_SF1 = 10_000
_PART_ROWS_SF1 = 200_000

QUERY_DATA_VOLUMES_SF1 = {
    "Q6":  4 * _LI_ROWS_SF1 * 4,                          # 4 cols
    "Q1":  7 * _LI_ROWS_SF1 * 4,                          # 7 cols
    "Q14": 4 * _LI_ROWS_SF1 * 4 + 2 * _PART_ROWS_SF1 * 4,
    "Q12": 5 * _LI_ROWS_SF1 * 4 + 2 * _ORDERS_ROWS_SF1 * 4,
    "Q3":  4 * _LI_ROWS_SF1 * 4 + 4 * _ORDERS_ROWS_SF1 * 4 + 2 * _CUSTOMER_ROWS_SF1 * 4,
    "Q5":  (4 * _LI_ROWS_SF1 + 3 * _ORDERS_ROWS_SF1 + 2 * _CUSTOMER_ROWS_SF1
            + 2 * _SUPPLIER_ROWS_SF1 + 3 * 25 + 2 * 5) * 4,
}


def model_transfer_scenarios(
    gpu_compute_times: dict[str, float],
    sf: float = 1.0,
) -> list[TransferScenario]:
    """Build transfer overhead scenario model.

    Models 'what if our M4 GPU had a PCIe bus' to isolate the
    unified memory advantage.

    Args:
        gpu_compute_times: dict of query_id → measured MLX GPU compute time (seconds).
        sf: Scale factor (data volumes scale linearly).

    Returns:
        List of TransferScenario for each query.
    """
    scenarios = []
    for qid in QUERY_IDS:
        data_bytes = int(QUERY_DATA_VOLUMES_SF1.get(qid, 0) * sf)
        gpu_time = gpu_compute_times.get(qid, 0.0)

        pcie4_s = data_bytes / (PCIE4_BW_GBS * 1e9)
        pcie5_s = data_bytes / (PCIE5_BW_GBS * 1e9)

        scenarios.append(TransferScenario(
            query_id=qid,
            data_bytes=data_bytes,
            pcie4_transfer_s=pcie4_s,
            pcie5_transfer_s=pcie5_s,
            unified_transfer_s=0.0,
            gpu_compute_s=gpu_time,
        ))

    return scenarios


def print_scenarios(scenarios: list[TransferScenario]) -> None:
    """Print transfer scenario model as a table."""
    table = Table(title="Transfer Overhead Scenario Model")
    table.add_column("Query", style="cyan")
    table.add_column("Data (MB)", justify="right")
    table.add_column("GPU (ms)", justify="right")
    table.add_column("PCIe4 xfer (ms)", justify="right")
    table.add_column("PCIe4 OH%", justify="right")
    table.add_column("PCIe5 xfer (ms)", justify="right")
    table.add_column("PCIe5 OH%", justify="right")
    table.add_column("Unified Speedup\nvs PCIe4", justify="right")

    for s in scenarios:
        table.add_row(
            s.query_id,
            f"{s.data_bytes / 1e6:.1f}",
            f"{s.gpu_compute_s * 1000:.2f}",
            f"{s.pcie4_transfer_s * 1000:.2f}",
            f"{s.pcie4_overhead_pct:.1f}%",
            f"{s.pcie5_transfer_s * 1000:.2f}",
            f"{s.pcie5_overhead_pct:.1f}%",
            f"{s.unified_speedup_vs_pcie4:.2f}x",
        )

    console.print(table)
