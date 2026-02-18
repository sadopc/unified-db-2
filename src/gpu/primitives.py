"""Reusable MLX GPU primitives for TPC-H query implementations.

Building blocks: scatter-add group-by, safe index join for sparse keys,
composite key encoding, top-k selection.
"""

import mlx.core as mx


def group_by_sum(keys: mx.array, values: mx.array, num_groups: int) -> mx.array:
    """Scatter-add: sum values grouped by integer keys.

    Args:
        keys: int32 array of group indices.
        values: float32 array of values to sum.
        num_groups: number of output groups.

    Returns:
        float32 array of shape (num_groups,) with summed values.
    """
    out = mx.zeros((num_groups,), dtype=values.dtype)
    return out.at[keys].add(values)


def group_by_count(keys: mx.array, num_groups: int) -> mx.array:
    """Scatter-add: count elements per group.

    Args:
        keys: int32 array of group indices.
        num_groups: number of output groups.

    Returns:
        float32 array of shape (num_groups,) with counts.
    """
    out = mx.zeros((num_groups,), dtype=mx.float32)
    ones = mx.ones((keys.shape[0],), dtype=mx.float32)
    return out.at[keys].add(ones)


def safe_index_join(
    fact_fk: mx.array,
    dim_values: mx.array,
    dim_keys: mx.array,
) -> tuple[mx.array, mx.array]:
    """Join with bounds checking for sparse key domains.

    Uses zero-initialized lookup + separate existence tracking.
    PK uniqueness guarantees .at[pk].add(value) = 0 + value = value.

    Args:
        fact_fk: Foreign keys from fact table (int32).
        dim_values: Values from dimension table to look up.
        dim_keys: Primary keys from dimension table (int32).

    Returns:
        Tuple of (gathered values, validity mask).
        Invalid rows have value 0 and valid=False.
    """
    max_key = mx.max(dim_keys)
    mx.eval(max_key)
    max_key_int = int(max_key.item())

    # Value lookup: 0-initialized, scatter values at PK positions
    lookup = mx.zeros((max_key_int + 1,), dtype=dim_values.dtype)
    lookup = lookup.at[dim_keys].add(dim_values)

    # Existence lookup: track which keys actually exist in dim table
    exists = mx.zeros((max_key_int + 1,), dtype=mx.int32)
    exists = exists.at[dim_keys].add(mx.ones((dim_keys.shape[0],), dtype=mx.int32))

    # Bounds-safe gather: clamp fact_fk to [0, max_key] to prevent OOB access
    safe_fk = mx.clip(fact_fk, 0, max_key_int)
    raw_result = lookup[safe_fk]
    valid = (exists[safe_fk] > 0) & (fact_fk >= 0) & (fact_fk <= max_key_int)

    # Apply validity mask — clamped invalid keys must not leak into aggregates
    result = mx.where(valid, raw_result, mx.zeros_like(raw_result))

    return result, valid


def safe_index_join_int(
    fact_fk: mx.array,
    dim_values: mx.array,
    dim_keys: mx.array,
) -> tuple[mx.array, mx.array]:
    """Same as safe_index_join but for int32 dimension values."""
    max_key = mx.max(dim_keys)
    mx.eval(max_key)
    max_key_int = int(max_key.item())

    lookup = mx.zeros((max_key_int + 1,), dtype=mx.int32)
    lookup = lookup.at[dim_keys].add(dim_values)

    exists = mx.zeros((max_key_int + 1,), dtype=mx.int32)
    exists = exists.at[dim_keys].add(mx.ones((dim_keys.shape[0],), dtype=mx.int32))

    safe_fk = mx.clip(fact_fk, 0, max_key_int)
    raw_result = lookup[safe_fk]
    valid = (exists[safe_fk] > 0) & (fact_fk >= 0) & (fact_fk <= max_key_int)

    result = mx.where(valid, raw_result, mx.zeros_like(raw_result))
    return result, valid


def encode_composite_key(*columns: mx.array, multipliers: list[int]) -> mx.array:
    """Combine multiple columns into a single composite integer key.

    Example: returnflag * 2 + linestatus gives 6 unique groups.

    Args:
        columns: Integer arrays to combine.
        multipliers: Multiplier for each column (last one is implicitly 1).

    Returns:
        int32 array of composite keys.
    """
    assert len(columns) == len(multipliers) + 1
    result = columns[0] * multipliers[0]
    for i in range(1, len(columns) - 1):
        result = result + columns[i] * multipliers[i]
    result = result + columns[-1]
    return result


def topk_desc(values: mx.array, k: int) -> mx.array:
    """Get indices of top-k values in descending order.

    Args:
        values: 1D array to sort.
        k: Number of top elements.

    Returns:
        int32 array of indices into values, sorted by descending value.
    """
    idx = mx.argsort(-values)
    return idx[:k]
