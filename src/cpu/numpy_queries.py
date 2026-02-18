"""TPC-H queries as NumPy CPU kernels.

These use identical algorithms to the MLX GPU implementations — same
scatter-add for group-by, same index lookups for joins — but using
NumPy operations on CPU. This is the fair CPU-vs-GPU comparison.
"""

import time
from dataclasses import dataclass

import duckdb
import numpy as np

from src.config import (
    EPOCH_1994_01_01,
    EPOCH_1995_01_01,
    EPOCH_1995_03_15,
    EPOCH_1995_09_01,
    EPOCH_1995_10_01,
    EPOCH_1998_09_02,
)
from src.data.encodings import (
    NUM_Q1_GROUPS,
    NUM_SHIPMODES,
    SHIPMODE_ENC,
)
from src.data.loader import (
    load_columns_numpy,
    q1_lineitem_sql,
    q3_customer_sql,
    q3_lineitem_sql,
    q3_orders_sql,
    q5_customer_sql,
    q5_lineitem_sql,
    q5_nation_sql,
    q5_orders_sql,
    q5_region_sql,
    q5_supplier_sql,
    q6_lineitem_sql,
    q12_lineitem_sql,
    q12_orders_sql,
    q14_lineitem_sql,
    q14_part_sql,
)


@dataclass
class NumpyResult:
    """Result from a NumPy CPU kernel execution."""
    data: dict               # Output arrays/scalars
    compute_s: float         # Pure compute time (excludes data loading)
    load_s: float            # Data loading time


def q1(conn: duckdb.DuckDBPyConnection) -> NumpyResult:
    """Q1 Pricing Summary — scatter-add group-by."""
    d, load_t = load_columns_numpy(conn, q1_lineitem_sql())

    t0 = time.perf_counter()
    mask = d["shipdate"] <= EPOCH_1998_09_02
    rf = d["returnflag"][mask]
    ls = d["linestatus"][mask]
    qty = d["quantity"][mask]
    price = d["extendedprice"][mask]
    disc = d["discount"][mask]
    tax = d["tax"][mask]

    keys = rf * 2 + ls
    n_groups = NUM_Q1_GROUPS

    sum_qty = np.zeros(n_groups, dtype=np.float32)
    sum_base = np.zeros(n_groups, dtype=np.float32)
    sum_disc_price = np.zeros(n_groups, dtype=np.float32)
    sum_charge = np.zeros(n_groups, dtype=np.float32)
    sum_disc = np.zeros(n_groups, dtype=np.float32)
    counts = np.zeros(n_groups, dtype=np.float32)

    np.add.at(sum_qty, keys, qty)
    np.add.at(sum_base, keys, price)
    disc_price = price * (1 - disc)
    np.add.at(sum_disc_price, keys, disc_price)
    charge = disc_price * (1 + tax)
    np.add.at(sum_charge, keys, charge)
    np.add.at(sum_disc, keys, disc)
    np.add.at(counts, keys, np.ones_like(qty))

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_qty = np.where(counts > 0, sum_qty / counts, 0)
        avg_price = np.where(counts > 0, sum_base / counts, 0)
        avg_disc = np.where(counts > 0, sum_disc / counts, 0)
    t1 = time.perf_counter()

    return NumpyResult(
        data={
            "sum_qty": sum_qty, "sum_base_price": sum_base,
            "sum_disc_price": sum_disc_price, "sum_charge": sum_charge,
            "avg_qty": avg_qty, "avg_price": avg_price, "avg_disc": avg_disc,
            "count_order": counts,
        },
        compute_s=t1 - t0, load_s=load_t,
    )


def q3(conn: duckdb.DuckDBPyConnection) -> NumpyResult:
    """Q3 Shipping Priority — 3-way join + sort + LIMIT 10."""
    li, t1 = load_columns_numpy(conn, q3_lineitem_sql())
    od, t2 = load_columns_numpy(conn, q3_orders_sql())
    cu, t3 = load_columns_numpy(conn, q3_customer_sql())
    load_t = t1 + t2 + t3

    t0 = time.perf_counter()
    # Build customer lookup: custkey → is_building
    max_ck = int(cu["custkey"].max())
    cust_building = np.zeros(max_ck + 1, dtype=np.int32)
    cust_building[cu["custkey"]] = cu["is_building"]

    # Filter orders: o_orderdate < 1995-03-15 AND customer is BUILDING
    o_cust_ok = cust_building[np.clip(od["custkey"], 0, max_ck)] == 1
    o_date_ok = od["orderdate"] < EPOCH_1995_03_15
    o_mask = o_cust_ok & o_date_ok
    o_keys = od["orderkey"][o_mask]
    o_dates = od["orderdate"][o_mask]
    o_prio = od["shippriority"][o_mask]

    # Build sparse order lookup
    max_ok = int(o_keys.max()) if len(o_keys) > 0 else 0
    order_date_lookup = np.zeros(max_ok + 1, dtype=np.int32)
    order_prio_lookup = np.zeros(max_ok + 1, dtype=np.int32)
    order_exists = np.zeros(max_ok + 1, dtype=np.int32)
    order_date_lookup[o_keys] = o_dates
    order_prio_lookup[o_keys] = o_prio
    order_exists[o_keys] = 1

    # Filter lineitem: l_shipdate > 1995-03-15 AND matching order exists
    l_ship_ok = li["shipdate"] > EPOCH_1995_03_15
    l_ok = li["orderkey"] <= max_ok
    safe_ok = np.clip(li["orderkey"], 0, max_ok)
    l_exists = (order_exists[safe_ok] > 0) & l_ok
    l_mask = l_ship_ok & l_exists

    l_okey = li["orderkey"][l_mask]
    revenue = li["extendedprice"][l_mask] * (1 - li["discount"][l_mask])

    # Group by orderkey: scatter-add revenue
    max_grouped = int(l_okey.max()) if len(l_okey) > 0 else 0
    rev_by_order = np.zeros(max_grouped + 1, dtype=np.float32)
    np.add.at(rev_by_order, l_okey, revenue)

    # Get unique orderkeys with revenue > 0
    unique_keys = np.where(rev_by_order > 0)[0]
    revs = rev_by_order[unique_keys]
    dates = order_date_lookup[np.clip(unique_keys, 0, max_ok)]
    prios = order_prio_lookup[np.clip(unique_keys, 0, max_ok)]

    # Sort by revenue DESC, orderdate ASC — take top 10
    sort_idx = np.lexsort((dates, -revs))[:10]
    t1_end = time.perf_counter()

    return NumpyResult(
        data={
            "l_orderkey": unique_keys[sort_idx],
            "revenue": revs[sort_idx],
            "o_orderdate": dates[sort_idx],
            "o_shippriority": prios[sort_idx],
        },
        compute_s=t1_end - t0, load_s=load_t,
    )


def q5(conn: duckdb.DuckDBPyConnection) -> NumpyResult:
    """Q5 Local Supplier Volume — 6-way join."""
    li, t1 = load_columns_numpy(conn, q5_lineitem_sql())
    od, t2 = load_columns_numpy(conn, q5_orders_sql())
    cu, t3 = load_columns_numpy(conn, q5_customer_sql())
    su, t4 = load_columns_numpy(conn, q5_supplier_sql())
    na, t5 = load_columns_numpy(conn, q5_nation_sql())
    re, t6 = load_columns_numpy(conn, q5_region_sql())
    load_t = t1 + t2 + t3 + t4 + t5 + t6

    # Load nation names separately (strings can't go to MLX)
    name_result = conn.execute(
        "SELECT n_nationkey, n_name FROM nation ORDER BY n_nationkey"
    ).fetchall()
    nk_to_name = {row[0]: row[1] for row in name_result}

    t0 = time.perf_counter()
    # region → nation: filter nations in ASIA (region_enc == 2)
    asia_rkeys = re["regionkey"][re["region_enc"] == 2]
    nation_in_asia = np.isin(na["regionkey"], asia_rkeys)
    asia_nkeys = na["nationkey"][nation_in_asia]
    asia_nnames = np.array([nk_to_name[int(nk)] for nk in asia_nkeys])

    # Nation name lookup
    max_nk = int(na["nationkey"].max())
    nation_name_idx = np.zeros(max_nk + 1, dtype=np.int32)
    nation_in_asia_flag = np.zeros(max_nk + 1, dtype=np.int32)
    for i, nk in enumerate(asia_nkeys):
        nation_name_idx[nk] = i
        nation_in_asia_flag[nk] = 1

    # Filter customers by nation in ASIA
    cu_nk = np.clip(cu["nationkey"], 0, max_nk)
    cu_in_asia = nation_in_asia_flag[cu_nk] == 1
    asia_custkeys = cu["custkey"][cu_in_asia]
    asia_cust_nkeys = cu["nationkey"][cu_in_asia]

    # Customer lookup
    max_ck = int(cu["custkey"].max())
    cust_exists = np.zeros(max_ck + 1, dtype=np.int32)
    cust_nationkey = np.zeros(max_ck + 1, dtype=np.int32)
    cust_exists[asia_custkeys] = 1
    cust_nationkey[asia_custkeys] = asia_cust_nkeys

    # Filter orders: date in [1994-01-01, 1995-01-01) AND customer in ASIA
    o_date_ok = (od["orderdate"] >= EPOCH_1994_01_01) & (od["orderdate"] < EPOCH_1995_01_01)
    o_ck = np.clip(od["custkey"], 0, max_ck)
    o_cust_ok = cust_exists[o_ck] == 1
    o_mask = o_date_ok & o_cust_ok
    valid_okeys = od["orderkey"][o_mask]
    valid_o_custkeys = od["custkey"][o_mask]

    # Order lookup (sparse)
    max_ok = int(od["orderkey"].max()) if len(od["orderkey"]) > 0 else 0
    order_exists = np.zeros(max_ok + 1, dtype=np.int32)
    order_custkey = np.zeros(max_ok + 1, dtype=np.int32)
    order_exists[valid_okeys] = 1
    order_custkey[valid_okeys] = valid_o_custkeys

    # Supplier nation lookup
    max_sk = int(su["suppkey"].max())
    supp_nationkey = np.zeros(max_sk + 1, dtype=np.int32)
    supp_nationkey[su["suppkey"]] = su["nationkey"]

    # Filter lineitem: matching order + supplier nation == customer nation
    safe_ok = np.clip(li["orderkey"], 0, max_ok)
    li_order_ok = (order_exists[safe_ok] == 1) & (li["orderkey"] <= max_ok)

    safe_sk = np.clip(li["suppkey"], 0, max_sk)
    li_supp_nk = supp_nationkey[safe_sk]
    li_cust_nk = cust_nationkey[np.clip(order_custkey[safe_ok], 0, max_ck)]
    li_nation_match = li_supp_nk == li_cust_nk
    # Also check supplier nation is in ASIA
    li_supp_asia = nation_in_asia_flag[np.clip(li_supp_nk, 0, max_nk)] == 1

    li_mask = li_order_ok & li_nation_match & li_supp_asia
    revenue = li["extendedprice"][li_mask] * (1 - li["discount"][li_mask])
    li_nk = li_supp_nk[li_mask]

    # Group by nation
    n_asia_nations = len(asia_nkeys)
    rev_by_nation = np.zeros(n_asia_nations, dtype=np.float64)
    for i, nk in enumerate(asia_nkeys):
        rev_by_nation[i] = revenue[li_nk == nk].sum()

    # Sort by revenue DESC
    sort_idx = np.argsort(-rev_by_nation)
    t1_end = time.perf_counter()

    return NumpyResult(
        data={
            "n_name": asia_nnames[sort_idx],
            "revenue": rev_by_nation[sort_idx].astype(np.float32),
        },
        compute_s=t1_end - t0, load_s=load_t,
    )


def q6(conn: duckdb.DuckDBPyConnection) -> NumpyResult:
    """Q6 Forecasting Revenue — scan + filter + reduce."""
    d, load_t = load_columns_numpy(conn, q6_lineitem_sql())

    t0 = time.perf_counter()
    mask = (
        (d["shipdate"] >= EPOCH_1994_01_01) &
        (d["shipdate"] < EPOCH_1995_01_01) &
        (d["discount"] >= 0.05) &
        (d["discount"] <= 0.07) &
        (d["quantity"] < 24)
    )
    revenue = np.sum(d["extendedprice"][mask] * d["discount"][mask])
    t1 = time.perf_counter()

    return NumpyResult(
        data={"revenue": float(revenue)},
        compute_s=t1 - t0, load_s=load_t,
    )


def q12(conn: duckdb.DuckDBPyConnection) -> NumpyResult:
    """Q12 Shipping Modes — 2-way join + conditional aggregation."""
    li, t1 = load_columns_numpy(conn, q12_lineitem_sql())
    od, t2 = load_columns_numpy(conn, q12_orders_sql())
    load_t = t1 + t2

    t0 = time.perf_counter()
    # Build order priority lookup (sparse)
    max_ok = int(od["orderkey"].max())
    order_prio = np.zeros(max_ok + 1, dtype=np.int32)
    order_exists = np.zeros(max_ok + 1, dtype=np.int32)
    order_prio[od["orderkey"]] = od["orderpriority"]
    order_exists[od["orderkey"]] = 1

    # Filter lineitem: MAIL/SHIP + date conditions
    mail_enc = SHIPMODE_ENC["MAIL"]
    ship_enc = SHIPMODE_ENC["SHIP"]
    mode_ok = (li["shipmode"] == mail_enc) | (li["shipmode"] == ship_enc)
    date_ok = (
        (li["commitdate"] < li["receiptdate"]) &
        (li["shipdate"] < li["commitdate"]) &
        (li["receiptdate"] >= EPOCH_1994_01_01) &
        (li["receiptdate"] < EPOCH_1995_01_01)
    )
    safe_ok = np.clip(li["orderkey"], 0, max_ok)
    exists_ok = (order_exists[safe_ok] > 0) & (li["orderkey"] <= max_ok)
    mask = mode_ok & date_ok & exists_ok

    sm = li["shipmode"][mask]
    ok = li["orderkey"][mask]
    prio = order_prio[np.clip(ok, 0, max_ok)]

    is_high = (prio == 0) | (prio == 1)  # 1-URGENT or 2-HIGH

    high_count = np.zeros(NUM_SHIPMODES, dtype=np.int64)
    low_count = np.zeros(NUM_SHIPMODES, dtype=np.int64)
    np.add.at(high_count, sm, is_high.astype(np.int64))
    np.add.at(low_count, sm, (~is_high).astype(np.int64))
    t1_end = time.perf_counter()

    return NumpyResult(
        data={
            "high_line_count": high_count,
            "low_line_count": low_count,
        },
        compute_s=t1_end - t0, load_s=load_t,
    )


def q14(conn: duckdb.DuckDBPyConnection) -> NumpyResult:
    """Q14 Promotion Effect — 2-way join + conditional aggregation."""
    li, t1 = load_columns_numpy(conn, q14_lineitem_sql())
    pt, t2 = load_columns_numpy(conn, q14_part_sql())
    load_t = t1 + t2

    t0 = time.perf_counter()
    # Build promo lookup: partkey → is_promo
    max_pk = int(pt["partkey"].max())
    promo_lookup = np.zeros(max_pk + 1, dtype=np.int32)
    promo_lookup[pt["partkey"]] = pt["is_promo"]

    # Filter lineitem by date
    date_mask = (li["shipdate"] >= EPOCH_1995_09_01) & (li["shipdate"] < EPOCH_1995_10_01)
    pk = li["partkey"][date_mask]
    price = li["extendedprice"][date_mask]
    disc = li["discount"][date_mask]

    revenue = price * (1 - disc)
    safe_pk = np.clip(pk, 0, max_pk)
    is_promo = promo_lookup[safe_pk] == 1

    promo_rev = np.sum(revenue[is_promo])
    total_rev = np.sum(revenue)

    result = 100.0 * promo_rev / total_rev if total_rev > 0 else 0.0
    t1_end = time.perf_counter()

    return NumpyResult(
        data={"promo_revenue": float(result)},
        compute_s=t1_end - t0, load_s=load_t,
    )


QUERIES = {
    "Q1": q1,
    "Q3": q3,
    "Q5": q5,
    "Q6": q6,
    "Q12": q12,
    "Q14": q14,
}
