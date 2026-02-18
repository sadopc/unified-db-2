"""Q1 Pricing Summary Report — group-by aggregation.

7 columns from lineitem, composite key = returnflag*2 + linestatus (6 groups),
scatter-add for SUM/COUNT, then compute AVG.

MLX doesn't support boolean fancy indexing, so we use the overflow bin
pattern: masked rows route to an overflow group (index NUM_Q1_GROUPS)
and their values are zeroed out.
"""

import time

import duckdb
import mlx.core as mx

from src.config import EPOCH_1998_09_02
from src.data.encodings import NUM_Q1_GROUPS
from src.data.loader import LoadTiming, load_columns_mlx, q1_lineitem_sql


def q1(conn: duckdb.DuckDBPyConnection) -> tuple[dict, float, LoadTiming]:
    """Execute Q1 on GPU via MLX.

    Returns:
        Tuple of (result dict, compute_seconds, load_timing).
    """
    data, load_timing = load_columns_mlx(conn, q1_lineitem_sql())

    t0 = time.perf_counter()
    mask = data["shipdate"] <= EPOCH_1998_09_02

    # Compute composite key: returnflag * 2 + linestatus
    raw_keys = data["returnflag"] * 2 + data["linestatus"]

    # Overflow bin pattern: invalid rows → group n (discarded), values zeroed
    n = NUM_Q1_GROUPS
    overflow = n  # Extra bin for masked-out rows
    keys = mx.where(mask, raw_keys, mx.full(raw_keys.shape, overflow, dtype=raw_keys.dtype))

    qty = mx.where(mask, data["quantity"], mx.zeros_like(data["quantity"]))
    price = mx.where(mask, data["extendedprice"], mx.zeros_like(data["extendedprice"]))
    disc = mx.where(mask, data["discount"], mx.zeros_like(data["discount"]))
    tax = mx.where(mask, data["tax"], mx.zeros_like(data["tax"]))

    # Scatter-add with n+1 groups (last is overflow bin)
    total_groups = n + 1

    sum_qty = mx.zeros((total_groups,), dtype=mx.float32)
    sum_qty = sum_qty.at[keys].add(qty)

    sum_base = mx.zeros((total_groups,), dtype=mx.float32)
    sum_base = sum_base.at[keys].add(price)

    disc_price = price * (1 - disc)
    sum_disc_price = mx.zeros((total_groups,), dtype=mx.float32)
    sum_disc_price = sum_disc_price.at[keys].add(disc_price)

    charge = disc_price * (1 + tax)
    sum_charge = mx.zeros((total_groups,), dtype=mx.float32)
    sum_charge = sum_charge.at[keys].add(charge)

    sum_disc = mx.zeros((total_groups,), dtype=mx.float32)
    sum_disc = sum_disc.at[keys].add(disc)

    ones = mx.where(mask, mx.ones(mask.shape, dtype=mx.float32), mx.zeros(mask.shape, dtype=mx.float32))
    counts = mx.zeros((total_groups,), dtype=mx.float32)
    counts = counts.at[keys].add(ones)

    # Slice off overflow bin, keep only real groups [0..n-1]
    sum_qty = sum_qty[:n]
    sum_base = sum_base[:n]
    sum_disc_price = sum_disc_price[:n]
    sum_charge = sum_charge[:n]
    sum_disc = sum_disc[:n]
    counts = counts[:n]

    avg_qty = mx.where(counts > 0, sum_qty / counts, mx.zeros_like(sum_qty))
    avg_price = mx.where(counts > 0, sum_base / counts, mx.zeros_like(sum_base))
    avg_disc = mx.where(counts > 0, sum_disc / counts, mx.zeros_like(sum_disc))

    # Force evaluation
    mx.eval(sum_qty, sum_base, sum_disc_price, sum_charge,
            avg_qty, avg_price, avg_disc, counts)
    t1 = time.perf_counter()

    return {
        "sum_qty": sum_qty,
        "sum_base_price": sum_base,
        "sum_disc_price": sum_disc_price,
        "sum_charge": sum_charge,
        "avg_qty": avg_qty,
        "avg_price": avg_price,
        "avg_disc": avg_disc,
        "count_order": counts,
    }, t1 - t0, load_timing
