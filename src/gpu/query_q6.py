"""Q6 Forecasting Revenue Change — pure scan + filter + reduce.

The simplest query: 4 columns from lineitem, apply date/discount/quantity
filters, sum(extendedprice * discount). Pure memory bandwidth test.
"""

import time

import duckdb
import mlx.core as mx

from src.config import EPOCH_1994_01_01, EPOCH_1995_01_01
from src.data.loader import LoadTiming, load_columns_mlx, q6_lineitem_sql


def q6(conn: duckdb.DuckDBPyConnection) -> tuple[dict, float, LoadTiming]:
    """Execute Q6 on GPU via MLX.

    Returns:
        Tuple of (result dict, compute_seconds, load_timing).
    """
    data, load_timing = load_columns_mlx(conn, q6_lineitem_sql())

    t0 = time.perf_counter()
    mask = (
        (data["shipdate"] >= EPOCH_1994_01_01) &
        (data["shipdate"] < EPOCH_1995_01_01) &
        (data["discount"] >= 0.05) &
        (data["discount"] <= 0.07) &
        (data["quantity"] < 24)
    )
    # Use mx.where to zero out non-matching rows, then sum
    revenue = mx.sum(mx.where(mask, data["extendedprice"] * data["discount"],
                               mx.zeros_like(data["extendedprice"])))
    mx.eval(revenue)
    t1 = time.perf_counter()

    return {"revenue": float(revenue.item())}, t1 - t0, load_timing
