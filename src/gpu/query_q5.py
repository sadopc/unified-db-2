"""Q5 Local Supplier Volume — 6-way join (hardest query).

Chain: region→nation→customer→orders→lineitem←supplier
Condition: c_nationkey == s_nationkey (local supplier)
Region: ASIA, Year: 1994

Uses mx.where masking instead of boolean fancy indexing.
"""

import time

import duckdb
import mlx.core as mx
import numpy as np

from src.config import EPOCH_1994_01_01, EPOCH_1995_01_01
from src.data.loader import (
    LoadTiming,
    load_columns_mlx,
    q5_customer_sql,
    q5_lineitem_sql,
    q5_nation_sql,
    q5_orders_sql,
    q5_region_sql,
    q5_supplier_sql,
)


def q5(conn: duckdb.DuckDBPyConnection) -> tuple[dict, float, LoadTiming]:
    """Execute Q5 on GPU via MLX.

    Returns:
        Tuple of (result dict, compute_seconds, load_timing).
    """
    li, li_t = load_columns_mlx(conn, q5_lineitem_sql())
    od, od_t = load_columns_mlx(conn, q5_orders_sql())
    cu, cu_t = load_columns_mlx(conn, q5_customer_sql())
    su, su_t = load_columns_mlx(conn, q5_supplier_sql())
    na, na_t = load_columns_mlx(conn, q5_nation_sql())
    re, re_t = load_columns_mlx(conn, q5_region_sql())

    t0 = time.perf_counter()

    # --- region → nation: filter nations in ASIA (region_enc == 2) ---
    # Build region lookup
    max_rk = mx.max(re["regionkey"])
    mx.eval(max_rk)
    max_rk_int = int(max_rk.item())

    asia_mask_r = re["region_enc"] == 2
    region_is_asia = mx.zeros((max_rk_int + 1,), dtype=mx.int32)
    asia_flag = mx.where(asia_mask_r,
                         mx.ones(asia_mask_r.shape, dtype=mx.int32),
                         mx.zeros(asia_mask_r.shape, dtype=mx.int32))
    region_is_asia = region_is_asia.at[re["regionkey"]].add(asia_flag)

    # Nation: check if in ASIA region
    max_nk = mx.max(na["nationkey"])
    mx.eval(max_nk)
    max_nk_int = int(max_nk.item())

    nation_rk_safe = mx.clip(na["regionkey"], 0, max_rk_int)
    nation_in_asia_mask = region_is_asia[nation_rk_safe] > 0

    # Build nation_is_asia lookup
    nation_is_asia = mx.zeros((max_nk_int + 1,), dtype=mx.int32)
    n_asia_flag = mx.where(nation_in_asia_mask,
                           mx.ones(nation_in_asia_mask.shape, dtype=mx.int32),
                           mx.zeros(nation_in_asia_mask.shape, dtype=mx.int32))
    nation_is_asia = nation_is_asia.at[na["nationkey"]].add(n_asia_flag)

    mx.eval(nation_is_asia)

    # Get ASIA nation keys and names via numpy (for string handling)
    nation_is_asia_np = np.array(nation_is_asia)
    na_nk_np = np.array(na["nationkey"])
    asia_nation_indices = np.where(np.array(nation_in_asia_mask))[0]
    asia_nkeys_np = na_nk_np[asia_nation_indices]
    n_asia = len(asia_nkeys_np)

    # Map nation key → group index (via numpy, then back to MLX)
    nation_to_group_np = np.zeros(max_nk_int + 1, dtype=np.int32)
    for i, nk in enumerate(asia_nkeys_np):
        nation_to_group_np[nk] = i
    nation_to_group = mx.array(nation_to_group_np)

    # --- Filter customers in ASIA nations ---
    max_ck = mx.max(cu["custkey"])
    mx.eval(max_ck)
    max_ck_int = int(max_ck.item())

    cu_nk_safe = mx.clip(cu["nationkey"], 0, max_nk_int)
    cu_in_asia = nation_is_asia[cu_nk_safe] > 0

    # Build customer lookups
    cust_exists = mx.zeros((max_ck_int + 1,), dtype=mx.int32)
    cust_nationkey = mx.zeros((max_ck_int + 1,), dtype=mx.int32)

    cu_valid = mx.where(cu_in_asia,
                        mx.ones(cu_in_asia.shape, dtype=mx.int32),
                        mx.zeros(cu_in_asia.shape, dtype=mx.int32))
    cu_nk_masked = mx.where(cu_in_asia, cu["nationkey"], mx.zeros_like(cu["nationkey"]))
    cu_ck_masked = mx.where(cu_in_asia, cu["custkey"], mx.zeros_like(cu["custkey"]))

    cust_exists = cust_exists.at[cu_ck_masked].add(cu_valid)
    cust_nationkey = cust_nationkey.at[cu_ck_masked].add(cu_nk_masked)

    # --- Filter orders: date in [1994-01-01, 1995-01-01) AND customer in ASIA ---
    o_date_ok = (od["orderdate"] >= EPOCH_1994_01_01) & (od["orderdate"] < EPOCH_1995_01_01)
    o_ck_safe = mx.clip(od["custkey"], 0, max_ck_int)
    o_cust_ok = cust_exists[o_ck_safe] > 0
    o_mask = o_date_ok & o_cust_ok

    # Build sparse order lookup
    max_ok = mx.max(od["orderkey"])
    mx.eval(max_ok)
    max_ok_int = int(max_ok.item())

    order_exists = mx.zeros((max_ok_int + 1,), dtype=mx.int32)
    order_custkey = mx.zeros((max_ok_int + 1,), dtype=mx.int32)

    o_valid = mx.where(o_mask,
                       mx.ones(o_mask.shape, dtype=mx.int32),
                       mx.zeros(o_mask.shape, dtype=mx.int32))
    o_ok_masked = mx.where(o_mask, od["orderkey"], mx.zeros_like(od["orderkey"]))
    o_ck_val = mx.where(o_mask, od["custkey"], mx.zeros_like(od["custkey"]))

    order_exists = order_exists.at[o_ok_masked].add(o_valid)
    order_custkey = order_custkey.at[o_ok_masked].add(o_ck_val)

    # --- Build supplier nation lookup ---
    max_sk = mx.max(su["suppkey"])
    mx.eval(max_sk)
    max_sk_int = int(max_sk.item())

    supp_nationkey = mx.zeros((max_sk_int + 1,), dtype=mx.int32)
    supp_nationkey = supp_nationkey.at[su["suppkey"]].add(su["nationkey"])

    # --- Filter lineitem ---
    safe_ok = mx.clip(li["orderkey"], 0, max_ok_int)
    li_order_ok = (order_exists[safe_ok] > 0) & (li["orderkey"] <= max_ok_int) & (li["orderkey"] >= 0)

    safe_sk = mx.clip(li["suppkey"], 0, max_sk_int)
    li_supp_nk = supp_nationkey[safe_sk]

    # Customer nation from order → custkey → nationkey
    li_custkey = order_custkey[safe_ok]
    li_cust_nk_safe = mx.clip(li_custkey, 0, max_ck_int)
    li_cust_nk = cust_nationkey[li_cust_nk_safe]

    # Match: supplier nation == customer nation
    li_nation_match = li_supp_nk == li_cust_nk
    # Supplier must be in ASIA
    li_supp_nk_safe = mx.clip(li_supp_nk, 0, max_nk_int)
    li_supp_asia = nation_is_asia[li_supp_nk_safe] > 0

    li_mask = li_order_ok & li_nation_match & li_supp_asia

    # Compute revenue with masking
    revenue = li["extendedprice"] * (1 - li["discount"])
    masked_revenue = mx.where(li_mask, revenue, mx.zeros_like(revenue))

    # Get group index for each lineitem's supplier nation
    li_group = nation_to_group[li_supp_nk_safe]

    # Use overflow bin for non-matching rows
    overflow = n_asia
    masked_group = mx.where(li_mask, li_group,
                           mx.full(li_group.shape, overflow, dtype=li_group.dtype))

    # Group by nation
    rev_by_nation = mx.zeros((n_asia + 1,), dtype=mx.float32)
    rev_by_nation = rev_by_nation.at[masked_group].add(masked_revenue)
    rev_by_nation = rev_by_nation[:n_asia]  # Drop overflow

    mx.eval(rev_by_nation)

    # Sort by revenue DESC (use numpy for string nation names)
    rev_np = np.array(rev_by_nation)
    sort_idx = np.argsort(-rev_np)

    # Get nation names from DuckDB
    name_result = conn.execute(
        "SELECT n_nationkey, n_name FROM nation ORDER BY n_nationkey"
    ).fetchall()
    nk_to_name = {row[0]: row[1] for row in name_result}
    sorted_names = [nk_to_name[int(asia_nkeys_np[i])] for i in sort_idx]
    sorted_rev = mx.array(rev_np[sort_idx].astype(np.float32))

    mx.eval(sorted_rev)
    t1 = time.perf_counter()

    return {
        "n_name": sorted_names,
        "revenue": sorted_rev,
    }, t1 - t0, _combine_timings(li_t, od_t, cu_t, su_t, na_t, re_t)


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
