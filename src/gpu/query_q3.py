"""Q3 Shipping Priority — 3-way join + sort + LIMIT 10.

Chain: customer(BUILDING) → orders(date < 1995-03-15) → lineitem(shipdate > 1995-03-15)
Group by orderkey, sort by revenue DESC / orderdate ASC, take top 10.

Uses mx.where masking instead of boolean fancy indexing (unsupported in MLX).
For the scatter-add grouping, masked rows route to overflow bins.
"""

import time

import duckdb
import mlx.core as mx
import numpy as np

from src.config import EPOCH_1995_03_15
from src.data.loader import (
    LoadTiming,
    load_columns_mlx,
    q3_customer_sql,
    q3_lineitem_sql,
    q3_orders_sql,
)


def q3(conn: duckdb.DuckDBPyConnection) -> tuple[dict, float, LoadTiming]:
    """Execute Q3 on GPU via MLX.

    Returns:
        Tuple of (result dict, compute_seconds, load_timing).
    """
    li, li_t = load_columns_mlx(conn, q3_lineitem_sql())
    od, od_t = load_columns_mlx(conn, q3_orders_sql())
    cu, cu_t = load_columns_mlx(conn, q3_customer_sql())

    t0 = time.perf_counter()

    # --- Customer filter: is_building == 1 ---
    max_ck = mx.max(cu["custkey"])
    mx.eval(max_ck)
    max_ck_int = int(max_ck.item())

    cust_building = mx.zeros((max_ck_int + 1,), dtype=mx.int32)
    cust_building = cust_building.at[cu["custkey"]].add(cu["is_building"])

    # --- Filter orders: date < 1995-03-15 AND customer is BUILDING ---
    o_ck_safe = mx.clip(od["custkey"], 0, max_ck_int)
    o_cust_ok = cust_building[o_ck_safe] == 1
    o_date_ok = od["orderdate"] < EPOCH_1995_03_15
    o_mask = o_cust_ok & o_date_ok

    # --- Build sparse order lookups using mx.where ---
    # Route masked orders to overflow bin (index 0); shift valid keys up
    max_ok = mx.max(od["orderkey"])
    mx.eval(max_ok)
    max_ok_int = int(max_ok.item())

    # Use direct scatter: only valid orders get written to lookup
    order_date = mx.zeros((max_ok_int + 1,), dtype=mx.int32)
    order_prio = mx.zeros((max_ok_int + 1,), dtype=mx.int32)
    order_exists = mx.zeros((max_ok_int + 1,), dtype=mx.int32)

    # Scatter only valid orders
    valid_okeys = mx.where(o_mask, od["orderkey"], mx.zeros_like(od["orderkey"]))
    valid_dates = mx.where(o_mask, od["orderdate"], mx.zeros_like(od["orderdate"]))
    valid_prio = mx.where(o_mask, od["shippriority"], mx.zeros_like(od["shippriority"]))
    valid_ones = mx.where(o_mask,
                          mx.ones(o_mask.shape, dtype=mx.int32),
                          mx.zeros(o_mask.shape, dtype=mx.int32))

    order_date = order_date.at[valid_okeys].add(valid_dates)
    order_prio = order_prio.at[valid_okeys].add(valid_prio)
    order_exists = order_exists.at[valid_okeys].add(valid_ones)

    # --- Filter lineitem: shipdate > 1995-03-15 AND matching order exists ---
    l_ship_ok = li["shipdate"] > EPOCH_1995_03_15
    safe_ok = mx.clip(li["orderkey"], 0, max_ok_int)
    l_exists = (order_exists[safe_ok] > 0) & (li["orderkey"] <= max_ok_int) & (li["orderkey"] >= 0)
    l_mask = l_ship_ok & l_exists

    revenue = li["extendedprice"] * (1 - li["discount"])

    # --- Group by orderkey: scatter-add revenue (masked) ---
    masked_revenue = mx.where(l_mask, revenue, mx.zeros_like(revenue))
    masked_okey = mx.where(l_mask, li["orderkey"], mx.zeros_like(li["orderkey"]))

    rev_by_order = mx.zeros((max_ok_int + 1,), dtype=mx.float32)
    rev_by_order = rev_by_order.at[masked_okey].add(masked_revenue)

    mx.eval(rev_by_order)

    # --- Extract results to NumPy for sorting (MLX lacks lexsort) ---
    rev_np = np.array(rev_by_order)
    # Get orderkeys with revenue > 0
    unique_keys = np.where(rev_np > 0)[0].astype(np.int32)

    if len(unique_keys) == 0:
        mx.eval()
        t1 = time.perf_counter()
        return {
            "l_orderkey": mx.array([], dtype=mx.int32),
            "revenue": mx.array([], dtype=mx.float32),
            "o_orderdate": mx.array([], dtype=mx.int32),
            "o_shippriority": mx.array([], dtype=mx.int32),
        }, t1 - t0, _combine_timings(li_t, od_t, cu_t)

    revs = rev_np[unique_keys]
    order_date_np = np.array(order_date)
    order_prio_np = np.array(order_prio)
    dates = order_date_np[np.clip(unique_keys, 0, max_ok_int)]
    prios = order_prio_np[np.clip(unique_keys, 0, max_ok_int)]

    # Sort by revenue DESC, then orderdate ASC — take top 10
    sort_idx = np.lexsort((dates, -revs))[:10]

    result_keys = mx.array(unique_keys[sort_idx])
    result_revs = mx.array(revs[sort_idx].astype(np.float32))
    result_dates = mx.array(dates[sort_idx])
    result_prios = mx.array(prios[sort_idx])

    mx.eval(result_keys, result_revs, result_dates, result_prios)
    t1 = time.perf_counter()

    return {
        "l_orderkey": result_keys,
        "revenue": result_revs,
        "o_orderdate": result_dates,
        "o_shippriority": result_prios,
    }, t1 - t0, _combine_timings(li_t, od_t, cu_t)


def _combine_timings(*timings: LoadTiming) -> LoadTiming:
    return LoadTiming(
        duckdb_query_s=sum(t.duckdb_query_s for t in timings),
        duckdb_extract_s=sum(t.duckdb_extract_s for t in timings),
        numpy_to_mlx_s=sum(t.numpy_to_mlx_s for t in timings),
        mlx_eval_s=sum(t.mlx_eval_s for t in timings),
        total_s=sum(t.total_s for t in timings),
        num_columns=sum(t.num_columns for t in timings),
        total_bytes=sum(t.total_bytes for t in timings),
    )
