"""Validation: compare GPU/CPU kernel results against DuckDB ground truth.

Precision policy:
- Exact match for: dimension keys, counts, sort order
- Relative tolerance (0.01%) only for: float aggregates (SUM, AVG of monetary values)
"""

from dataclasses import dataclass

import duckdb
import mlx.core as mx
import numpy as np
from rich.console import Console

from src.config import FLOAT_RTOL, QUERY_IDS, db_path
from src.cpu.duckdb_queries import QUERIES as DUCKDB_QUERIES
from src.cpu.numpy_queries import QUERIES as NUMPY_QUERIES
from src.data.encodings import (
    RETURNFLAG_DEC,
    SHIPMODE_DEC,
    SHIPMODE_ENC,
)
from src.gpu.query_q1 import q1 as mlx_q1
from src.gpu.query_q3 import q3 as mlx_q3
from src.gpu.query_q5 import q5 as mlx_q5
from src.gpu.query_q6 import q6 as mlx_q6
from src.gpu.query_q12 import q12 as mlx_q12
from src.gpu.query_q14 import q14 as mlx_q14

console = Console()

MLX_QUERIES = {
    "Q1": mlx_q1, "Q3": mlx_q3, "Q5": mlx_q5,
    "Q6": mlx_q6, "Q12": mlx_q12, "Q14": mlx_q14,
}


@dataclass
class ValidationResult:
    """Result of validating a query."""
    query_id: str
    baseline: str
    passed: bool
    message: str


def _to_numpy(val):
    """Convert MLX array or scalar to numpy."""
    if isinstance(val, mx.array):
        return np.array(val)
    return val


def _check_float(name: str, got: float, expected: float, rtol: float = FLOAT_RTOL) -> str | None:
    """Check float with relative tolerance. Returns error message or None."""
    if expected == 0:
        if abs(got) < 1e-6:
            return None
        return f"{name}: got {got}, expected 0"
    rel_err = abs(got - expected) / abs(expected)
    if rel_err > rtol:
        return f"{name}: got {got:.6f}, expected {expected:.6f}, rel_err={rel_err:.6f}"
    return None


def validate_q1(conn: duckdb.DuckDBPyConnection, result: dict, baseline: str) -> ValidationResult:
    """Validate Q1 against DuckDB."""
    duckdb_result = DUCKDB_QUERIES["Q1"](conn)
    errors = []

    for row in duckdb_result.rows:
        rf_str, ls_str = row[0], row[1]
        rf_enc = {"A": 0, "N": 1, "R": 2}[rf_str]
        ls_enc = {"F": 0, "O": 1}[ls_str]
        idx = rf_enc * 2 + ls_enc

        sum_qty = _to_numpy(result["sum_qty"])[idx]
        count = _to_numpy(result["count_order"])[idx]

        # Exact match for count
        expected_count = row[9]
        if int(count) != expected_count:
            errors.append(f"{rf_str}-{ls_str} count: got {int(count)}, expected {expected_count}")

        # Float tolerance for aggregates
        for name, got_val, exp_val in [
            ("sum_qty", float(sum_qty), float(row[2])),
            ("sum_base_price", float(_to_numpy(result["sum_base_price"])[idx]), float(row[3])),
            ("sum_disc_price", float(_to_numpy(result["sum_disc_price"])[idx]), float(row[4])),
            ("sum_charge", float(_to_numpy(result["sum_charge"])[idx]), float(row[5])),
        ]:
            err = _check_float(f"{rf_str}-{ls_str} {name}", got_val, exp_val)
            if err:
                errors.append(err)

    passed = len(errors) == 0
    msg = "OK" if passed else "; ".join(errors[:3])
    return ValidationResult("Q1", baseline, passed, msg)


def validate_q3(conn: duckdb.DuckDBPyConnection, result: dict, baseline: str) -> ValidationResult:
    """Validate Q3 against DuckDB."""
    duckdb_result = DUCKDB_QUERIES["Q3"](conn)
    errors = []

    got_keys = _to_numpy(result["l_orderkey"])
    got_revs = _to_numpy(result["revenue"])

    n = min(len(duckdb_result.rows), len(got_keys))
    for i in range(n):
        exp_key = duckdb_result.rows[i][0]
        exp_rev = float(duckdb_result.rows[i][1])

        # Exact match for orderkey
        if int(got_keys[i]) != exp_key:
            errors.append(f"row {i}: key got {int(got_keys[i])}, expected {exp_key}")

        # Float tolerance for revenue
        err = _check_float(f"row {i} revenue", float(got_revs[i]), exp_rev)
        if err:
            errors.append(err)

    if len(got_keys) != len(duckdb_result.rows):
        errors.append(f"row count: got {len(got_keys)}, expected {len(duckdb_result.rows)}")

    passed = len(errors) == 0
    msg = "OK" if passed else "; ".join(errors[:3])
    return ValidationResult("Q3", baseline, passed, msg)


def validate_q5(conn: duckdb.DuckDBPyConnection, result: dict, baseline: str) -> ValidationResult:
    """Validate Q5 against DuckDB."""
    duckdb_result = DUCKDB_QUERIES["Q5"](conn)
    errors = []

    got_names = result["n_name"]
    got_revs = _to_numpy(result["revenue"])

    for i, row in enumerate(duckdb_result.rows):
        exp_name = row[0]
        exp_rev = float(row[1])

        if i < len(got_names):
            got_name = got_names[i] if isinstance(got_names[i], str) else str(got_names[i])
            if got_name != exp_name:
                errors.append(f"row {i}: name got '{got_name}', expected '{exp_name}'")

            err = _check_float(f"row {i} revenue", float(got_revs[i]), exp_rev)
            if err:
                errors.append(err)

    passed = len(errors) == 0
    msg = "OK" if passed else "; ".join(errors[:3])
    return ValidationResult("Q5", baseline, passed, msg)


def validate_q6(conn: duckdb.DuckDBPyConnection, result: dict, baseline: str) -> ValidationResult:
    """Validate Q6 against DuckDB."""
    duckdb_result = DUCKDB_QUERIES["Q6"](conn)
    exp_rev = float(duckdb_result.rows[0][0])
    got_rev = float(result["revenue"])

    err = _check_float("revenue", got_rev, exp_rev)
    passed = err is None
    msg = "OK" if passed else err
    return ValidationResult("Q6", baseline, passed, msg)


def validate_q12(conn: duckdb.DuckDBPyConnection, result: dict, baseline: str) -> ValidationResult:
    """Validate Q12 against DuckDB."""
    duckdb_result = DUCKDB_QUERIES["Q12"](conn)
    errors = []

    got_high = _to_numpy(result["high_line_count"])
    got_low = _to_numpy(result["low_line_count"])

    for row in duckdb_result.rows:
        mode_str = row[0]
        exp_high = int(row[1])
        exp_low = int(row[2])
        mode_enc = SHIPMODE_ENC[mode_str]

        if int(got_high[mode_enc]) != exp_high:
            errors.append(f"{mode_str} high: got {int(got_high[mode_enc])}, expected {exp_high}")
        if int(got_low[mode_enc]) != exp_low:
            errors.append(f"{mode_str} low: got {int(got_low[mode_enc])}, expected {exp_low}")

    passed = len(errors) == 0
    msg = "OK" if passed else "; ".join(errors[:3])
    return ValidationResult("Q12", baseline, passed, msg)


def validate_q14(conn: duckdb.DuckDBPyConnection, result: dict, baseline: str) -> ValidationResult:
    """Validate Q14 against DuckDB."""
    duckdb_result = DUCKDB_QUERIES["Q14"](conn)
    exp_rev = float(duckdb_result.rows[0][0])
    got_rev = float(result["promo_revenue"])

    err = _check_float("promo_revenue", got_rev, exp_rev)
    passed = err is None
    msg = "OK" if passed else err
    return ValidationResult("Q14", baseline, passed, msg)


VALIDATORS = {
    "Q1": validate_q1,
    "Q3": validate_q3,
    "Q5": validate_q5,
    "Q6": validate_q6,
    "Q12": validate_q12,
    "Q14": validate_q14,
}


def validate_all(sf: float, query_ids: list[str] | None = None) -> list[ValidationResult]:
    """Validate all queries for both NumPy and MLX against DuckDB.

    Args:
        sf: Scale factor to validate at.
        query_ids: Which queries to validate (default: all).

    Returns:
        List of ValidationResult.
    """
    if query_ids is None:
        query_ids = QUERY_IDS

    conn = duckdb.connect(str(db_path(sf)), read_only=True)
    results = []

    for qid in query_ids:
        console.print(f"\n[bold]Validating {qid} @ SF{sf}[/bold]")
        validator = VALIDATORS[qid]

        # Validate NumPy
        np_result = NUMPY_QUERIES[qid](conn)
        vr = validator(conn, np_result.data, "numpy")
        results.append(vr)
        status = "[green]PASS[/green]" if vr.passed else f"[red]FAIL: {vr.message}[/red]"
        console.print(f"  NumPy: {status}")

        # Validate MLX
        mlx_data, _, _ = MLX_QUERIES[qid](conn)
        vr = validator(conn, mlx_data, "mlx")
        results.append(vr)
        status = "[green]PASS[/green]" if vr.passed else f"[red]FAIL: {vr.message}[/red]"
        console.print(f"  MLX:   {status}")

    conn.close()

    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    console.print(f"\n[bold]Validation: {n_pass}/{n_total} passed[/bold]")

    return results
