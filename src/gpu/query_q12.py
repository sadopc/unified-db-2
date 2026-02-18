"""Q12 Shipping Modes and Order Priority — 2-way join + conditional agg.

Build order_priority_lookup[orderkey] (sparse!), gather for lineitem,
conditional COUNT by shipmode.

Uses overflow bin pattern since MLX doesn't support boolean fancy indexing.
"""

import time

import duckdb
import mlx.core as mx

from src.config import EPOCH_1994_01_01, EPOCH_1995_01_01
from src.data.encodings import NUM_SHIPMODES, SHIPMODE_ENC
from src.data.loader import (
    LoadTiming,
    load_columns_mlx,
    q12_lineitem_sql,
    q12_orders_sql,
)
from src.gpu.primitives import safe_index_join_int


def q12(conn: duckdb.DuckDBPyConnection) -> tuple[dict, float, LoadTiming]:
    """Execute Q12 on GPU via MLX.

    Returns:
        Tuple of (result dict, compute_seconds, load_timing).
    """
    li, li_timing = load_columns_mlx(conn, q12_lineitem_sql())
    od, od_timing = load_columns_mlx(conn, q12_orders_sql())

    t0 = time.perf_counter()
    # Join: get order priority for each lineitem (sparse orderkey)
    prio, valid = safe_index_join_int(li["orderkey"], od["orderpriority"], od["orderkey"])

    # Filter: MAIL or SHIP modes
    mail_enc = SHIPMODE_ENC["MAIL"]
    ship_enc = SHIPMODE_ENC["SHIP"]
    mode_ok = (li["shipmode"] == mail_enc) | (li["shipmode"] == ship_enc)

    # Date conditions
    date_ok = (
        (li["commitdate"] < li["receiptdate"]) &
        (li["shipdate"] < li["commitdate"]) &
        (li["receiptdate"] >= EPOCH_1994_01_01) &
        (li["receiptdate"] < EPOCH_1995_01_01)
    )

    mask = mode_ok & date_ok & valid

    # Overflow bin pattern: masked rows → shipmode overflow bin
    overflow_sm = NUM_SHIPMODES  # Extra bin
    sm = mx.where(mask, li["shipmode"],
                  mx.full(li["shipmode"].shape, overflow_sm, dtype=li["shipmode"].dtype))

    is_high = (prio == 0) | (prio == 1)  # 1-URGENT or 2-HIGH

    # Scatter-add counts by shipmode (with overflow bin)
    total_bins = NUM_SHIPMODES + 1

    high_ones = mx.where(mask & is_high,
                         mx.ones(mask.shape, dtype=mx.float32),
                         mx.zeros(mask.shape, dtype=mx.float32))
    low_ones = mx.where(mask & ~is_high,
                        mx.ones(mask.shape, dtype=mx.float32),
                        mx.zeros(mask.shape, dtype=mx.float32))

    high_count = mx.zeros((total_bins,), dtype=mx.float32)
    high_count = high_count.at[sm].add(high_ones)

    low_count = mx.zeros((total_bins,), dtype=mx.float32)
    low_count = low_count.at[sm].add(low_ones)

    # Slice off overflow bin
    high_count = high_count[:NUM_SHIPMODES]
    low_count = low_count[:NUM_SHIPMODES]

    mx.eval(high_count, low_count)
    t1 = time.perf_counter()

    combined_timing = LoadTiming(
        duckdb_query_s=li_timing.duckdb_query_s + od_timing.duckdb_query_s,
        duckdb_extract_s=li_timing.duckdb_extract_s + od_timing.duckdb_extract_s,
        numpy_to_mlx_s=li_timing.numpy_to_mlx_s + od_timing.numpy_to_mlx_s,
        mlx_eval_s=li_timing.mlx_eval_s + od_timing.mlx_eval_s,
        total_s=li_timing.total_s + od_timing.total_s,
        num_columns=li_timing.num_columns + od_timing.num_columns,
        total_bytes=li_timing.total_bytes + od_timing.total_bytes,
    )

    return {
        "high_line_count": high_count,
        "low_line_count": low_count,
    }, t1 - t0, combined_timing
