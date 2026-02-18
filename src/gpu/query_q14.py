"""Q14 Promotion Effect — 2-way join + conditional aggregation.

Build promo_lookup[partkey] from part table, gather for lineitem,
conditional SUM to compute promo revenue percentage.
"""

import time

import duckdb
import mlx.core as mx

from src.config import EPOCH_1995_09_01, EPOCH_1995_10_01
from src.data.loader import (
    LoadTiming,
    load_columns_mlx,
    q14_lineitem_sql,
    q14_part_sql,
)


def q14(conn: duckdb.DuckDBPyConnection) -> tuple[dict, float, LoadTiming]:
    """Execute Q14 on GPU via MLX.

    Returns:
        Tuple of (result dict, compute_seconds, load_timing).
    """
    li, li_timing = load_columns_mlx(conn, q14_lineitem_sql())
    pt, pt_timing = load_columns_mlx(conn, q14_part_sql())

    t0 = time.perf_counter()
    # Build promo lookup: partkey → is_promo (contiguous keys, use direct index)
    max_pk = mx.max(pt["partkey"])
    mx.eval(max_pk)
    max_pk_int = int(max_pk.item())

    promo_lookup = mx.zeros((max_pk_int + 1,), dtype=mx.int32)
    promo_lookup = promo_lookup.at[pt["partkey"]].add(pt["is_promo"])

    # Filter lineitem by date — use mx.where instead of boolean indexing
    date_mask = (li["shipdate"] >= EPOCH_1995_09_01) & (li["shipdate"] < EPOCH_1995_10_01)

    revenue = li["extendedprice"] * (1 - li["discount"])

    # Gather promo flag for each lineitem (all rows)
    safe_pk = mx.clip(li["partkey"], 0, max_pk_int)
    is_promo = promo_lookup[safe_pk]

    # Combined mask: date_mask AND is_promo
    promo_mask = date_mask & (is_promo == 1)

    promo_rev = mx.sum(mx.where(promo_mask, revenue, mx.zeros_like(revenue)))
    total_rev = mx.sum(mx.where(date_mask, revenue, mx.zeros_like(revenue)))

    result = 100.0 * promo_rev / total_rev

    mx.eval(result)
    t1 = time.perf_counter()

    # Combine load timings
    combined_timing = LoadTiming(
        duckdb_query_s=li_timing.duckdb_query_s + pt_timing.duckdb_query_s,
        duckdb_extract_s=li_timing.duckdb_extract_s + pt_timing.duckdb_extract_s,
        numpy_to_mlx_s=li_timing.numpy_to_mlx_s + pt_timing.numpy_to_mlx_s,
        mlx_eval_s=li_timing.mlx_eval_s + pt_timing.mlx_eval_s,
        total_s=li_timing.total_s + pt_timing.total_s,
        num_columns=li_timing.num_columns + pt_timing.num_columns,
        total_bytes=li_timing.total_bytes + pt_timing.total_bytes,
    )

    return {"promo_revenue": float(result.item())}, t1 - t0, combined_timing
