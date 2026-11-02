from __future__ import annotations

from typing import Any


DEFAULT_MIN_CELL_SIZE = 11


def suppress_small_cell(value: Any, n: int, min_cell_size: int = DEFAULT_MIN_CELL_SIZE) -> Any:
    """Return None when a group is too small to report safely."""
    if n < min_cell_size:
        return None
    return value


def assert_aggregate_only(rows: list[dict[str, Any]], count_key: str = "n", min_cell_size: int = DEFAULT_MIN_CELL_SIZE) -> None:
    """Fail if a result set contains small cells that should not leave the private layer."""
    unsafe = [row for row in rows if int(row.get(count_key) or 0) < min_cell_size]
    if unsafe:
        raise ValueError(
            f"Refusing to return {len(unsafe)} rows with {count_key} < {min_cell_size}. "
            "Broaden the filters or increase aggregation."
        )

