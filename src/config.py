"""Constants, TPC-H dates, hardware specs, memory budgets."""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SCALE_FACTORS = [0.1, 1, 10]

# --- TPC-H date constants (days since Unix epoch 1970-01-01) ---
# DuckDB encodes DATE as int32 days-since-epoch
EPOCH_1993_01_01 = 8401   # 1993-01-01
EPOCH_1993_07_01 = 8582   # 1993-07-01
EPOCH_1993_10_01 = 8674   # 1993-10-01
EPOCH_1994_01_01 = 8766   # 1994-01-01
EPOCH_1995_01_01 = 9131   # 1995-01-01
EPOCH_1995_03_15 = 9204   # 1995-03-15
EPOCH_1995_09_01 = 9374   # 1995-09-01
EPOCH_1995_10_01 = 9404   # 1995-10-01
EPOCH_1996_01_01 = 9497   # 1996-01-01
EPOCH_1996_12_31 = 9862   # 1996-12-31
EPOCH_1997_01_01 = 9862   # 1997-01-01
EPOCH_1998_09_02 = 10471  # 1998-09-02
EPOCH_1998_12_01 = 10561  # 1998-12-01

# --- Hardware specs (Apple M4 base) ---
MEMORY_BW_GBS = 120       # GB/s unified memory bandwidth
PCIE4_BW_GBS = 25         # GB/s practical PCIe 4.0 x16
PCIE5_BW_GBS = 50         # GB/s practical PCIe 5.0 x16
GPU_CORES = 10            # M4 base GPU cores
TOTAL_RAM_GB = 16         # System RAM

# --- Benchmark config ---
COLD_RUNS = 1
WARM_RUNS = 9
TRANSFER_REPS = 100

# --- Validation ---
FLOAT_RTOL = 1e-3         # 0.1% relative tolerance for float32 aggregates
                          # float32 scatter-add over millions of rows accumulates
                          # rounding errors beyond 0.01%; 0.1% is realistic

# --- TPC-H queries we benchmark ---
QUERY_IDS = ["Q1", "Q3", "Q5", "Q6", "Q12", "Q14"]

# --- Memory budgets (bytes) per query at SF1 ---
# lineitem ~6M rows, each column 4 bytes
MEMORY_BUDGETS_SF1 = {
    "Q6":  144 * 1024**2,   # ~144 MB
    "Q1":  336 * 1024**2,   # ~336 MB
    "Q14": 100 * 1024**2,   # ~100 MB
    "Q12": 144 * 1024**2,   # ~144 MB
    "Q3":  162 * 1024**2,   # ~162 MB
    "Q5":  256 * 1024**2,   # ~256 MB
}

# At SF10, multiply by ~10 (lineitem ~60M rows)
MEMORY_BUDGETS_SF10 = {k: v * 10 for k, v in MEMORY_BUDGETS_SF1.items()}

# Max usable memory (leave ~4-5 GB for OS + DuckDB)
MAX_USABLE_BYTES = 11 * 1024**3  # ~11 GB

# Threshold for chunked processing
CHUNK_THRESHOLD_BYTES = 10 * 1024**3  # 10 GB


def db_path(sf: float) -> Path:
    """Path to TPC-H database file for a given scale factor."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Format: integer SFs as "1", fractional as "0.1"
    sf_str = str(int(sf)) if sf == int(sf) else str(sf)
    return DATA_DIR / f"tpch_sf{sf_str}.duckdb"
