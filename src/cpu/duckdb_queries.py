"""TPC-H queries as DuckDB SQL with timed execution.

These serve as the optimized SQL engine baseline. DuckDB uses vectorized
execution with a full query optimizer — not an apples-to-apples comparison
with hand-written kernels, but shows what a production engine achieves.
"""

import time
from dataclasses import dataclass

import duckdb

from src.config import db_path


@dataclass
class SQLResult:
    """Result from a DuckDB SQL query execution."""
    rows: list[tuple]
    columns: list[str]
    elapsed_s: float


def _run(conn: duckdb.DuckDBPyConnection, sql: str) -> SQLResult:
    t0 = time.perf_counter()
    result = conn.execute(sql)
    rows = result.fetchall()
    cols = [desc[0] for desc in result.description] if result.description else []
    t1 = time.perf_counter()
    return SQLResult(rows=rows, columns=cols, elapsed_s=t1 - t0)


def q1(conn: duckdb.DuckDBPyConnection) -> SQLResult:
    """Q1 Pricing Summary Report."""
    return _run(conn, """
        SELECT
            l_returnflag,
            l_linestatus,
            SUM(l_quantity) AS sum_qty,
            SUM(l_extendedprice) AS sum_base_price,
            SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            AVG(l_quantity) AS avg_qty,
            AVG(l_extendedprice) AS avg_price,
            AVG(l_discount) AS avg_disc,
            COUNT(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
    """)


def q3(conn: duckdb.DuckDBPyConnection) -> SQLResult:
    """Q3 Shipping Priority."""
    return _run(conn, """
        SELECT
            l_orderkey,
            SUM(l_extendedprice * (1 - l_discount)) AS revenue,
            o_orderdate,
            o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
            AND c_custkey = o_custkey
            AND l_orderkey = o_orderkey
            AND o_orderdate < DATE '1995-03-15'
            AND l_shipdate > DATE '1995-03-15'
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10
    """)


def q5(conn: duckdb.DuckDBPyConnection) -> SQLResult:
    """Q5 Local Supplier Volume."""
    return _run(conn, """
        SELECT
            n_name,
            SUM(l_extendedprice * (1 - l_discount)) AS revenue
        FROM customer, orders, lineitem, supplier, nation, region
        WHERE c_custkey = o_custkey
            AND l_orderkey = o_orderkey
            AND l_suppkey = s_suppkey
            AND c_nationkey = s_nationkey
            AND s_nationkey = n_nationkey
            AND n_regionkey = r_regionkey
            AND r_name = 'ASIA'
            AND o_orderdate >= DATE '1994-01-01'
            AND o_orderdate < DATE '1995-01-01'
        GROUP BY n_name
        ORDER BY revenue DESC
    """)


def q6(conn: duckdb.DuckDBPyConnection) -> SQLResult:
    """Q6 Forecasting Revenue Change."""
    return _run(conn, """
        SELECT
            SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01'
            AND l_shipdate < DATE '1995-01-01'
            AND l_discount BETWEEN 0.05 AND 0.07
            AND l_quantity < 24
    """)


def q12(conn: duckdb.DuckDBPyConnection) -> SQLResult:
    """Q12 Shipping Modes and Order Priority."""
    return _run(conn, """
        SELECT
            l_shipmode,
            SUM(CASE
                WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH'
                THEN 1 ELSE 0
            END) AS high_line_count,
            SUM(CASE
                WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH'
                THEN 1 ELSE 0
            END) AS low_line_count
        FROM orders, lineitem
        WHERE o_orderkey = l_orderkey
            AND l_shipmode IN ('MAIL', 'SHIP')
            AND l_commitdate < l_receiptdate
            AND l_shipdate < l_commitdate
            AND l_receiptdate >= DATE '1994-01-01'
            AND l_receiptdate < DATE '1995-01-01'
        GROUP BY l_shipmode
        ORDER BY l_shipmode
    """)


def q14(conn: duckdb.DuckDBPyConnection) -> SQLResult:
    """Q14 Promotion Effect."""
    return _run(conn, """
        SELECT
            100.00 * SUM(CASE
                WHEN p_type LIKE 'PROMO%'
                THEN l_extendedprice * (1 - l_discount)
                ELSE 0
            END) / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
        FROM lineitem, part
        WHERE l_partkey = p_partkey
            AND l_shipdate >= DATE '1995-09-01'
            AND l_shipdate < DATE '1995-10-01'
    """)


# Map query ID to function
QUERIES = {
    "Q1": q1,
    "Q3": q3,
    "Q5": q5,
    "Q6": q6,
    "Q12": q12,
    "Q14": q14,
}
