#!/usr/bin/env python3
"""Differential privacy for custom RSA-911 SQL queries.

Every custom query the agent writes gets random noise (OpenDP's discrete Laplace
mechanism) before anyone sees it, so subtracting two overlapping answers no longer
singles out one person. Each new query spends part of a lifetime privacy budget kept
in a ledger next to the database; repeating the same query returns the saved answer
without spending more. Prebuilt summaries and benchmark ground truth stay exact.

Assumes each row is a different person (one exit per person in the data).
"""
from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import opendp.prelude as dp
import sqlglot
from sqlglot import exp

from rsa911.config import work_dir
from rsa911.safe_sql import SafeSqlError, _is_indicator, _unalias

dp.enable_features("contrib")

# AVG over a plain column is only allowed for these columns. Values are clamped to the
# range so one person can move a sum by at most the range's largest absolute value.
MEAN_BOUNDS: dict[str, tuple[float, float]] = {
    "exithourlywage": (0.0, 100.0),
    "ipehourlywage": (0.0, 100.0),
    "exitweeklyhoursworked": (0.0, 80.0),
    "ipeweeklyhoursworked": (0.0, 80.0),
    "age_app": (0.0, 100.0),
    "age_ipe": (0.0, 100.0),
    "age_exit": (0.0, 100.0),
}
# Sums for means are taken in hundredths (cents for wages) so all noise is on whole numbers.
MEAN_SCALE = 100
# Share of each query's epsilon spent on the group size n; the rest is split evenly.
N_SHARE = 0.5
MAX_GROUPS = 10_000


@dataclass(frozen=True)
class DPConfig:
    epsilon_per_query: float = 1.0
    delta_per_query: float = 1e-6
    total_epsilon: float = 50.0
    ledger_path: Path = field(default_factory=lambda: work_dir() / "dp_ledger.json")

    def __post_init__(self) -> None:
        if not (self.epsilon_per_query > 0 and 0 < self.delta_per_query < 1 and self.total_epsilon > 0):
            raise ValueError("Differential privacy settings must be positive (and delta below 1).")

    @classmethod
    def from_env(cls) -> "DPConfig | None":
        """Settings for the agent. RSA911_DP=off turns noise off (exact answers)."""
        if os.getenv("RSA911_DP", "on").strip().lower() in {"off", "0", "false", "no"}:
            return None
        return cls(
            epsilon_per_query=float(os.getenv("RSA911_DP_EPSILON", "1.0")),
            delta_per_query=float(os.getenv("RSA911_DP_DELTA", "1e-6")),
            total_epsilon=float(os.getenv("RSA911_DP_BUDGET", "50")),
        )


def group_threshold(config: DPConfig, min_cell_size: int, n_epsilon: float | None = None) -> int:
    """Smallest noisy n a group needs to be shown.

    The margin above min_cell_size makes the chance that a group just at the cutoff
    appears at most delta, which covers both group discovery and the min-cell rule.
    """
    n_epsilon = n_epsilon or config.epsilon_per_query * N_SHARE
    return min_cell_size + math.ceil(math.log(1 / config.delta_per_query) / n_epsilon)


def prompt_rules(config: DPConfig, min_cell_size: int) -> list[str]:
    return [
        "Results include random noise (differential privacy), so numbers are close estimates, not exact counts.",
        f"Groups with fewer than about {group_threshold(config, min_cell_size)} people are hidden.",
        "HAVING is not allowed. ORDER BY and LIMIT are applied after the noise is added.",
        f"AVG over a plain column is only allowed for: {', '.join(sorted(MEAN_BOUNDS))}.",
        "Each new query uses part of a limited privacy budget; repeating an identical query is free.",
    ]


@functools.lru_cache(maxsize=64)
def _mechanism(scale: float):
    return dp.m.make_laplace(dp.atom_domain(T=int, nan=False), dp.absolute_distance(T=int), scale=scale)


def _whole(value: Any) -> int:
    return int(round(float(value))) if value is not None else 0


def _aggregate(node: exp.Expression) -> exp.Expression:
    node = _unalias(node)
    while isinstance(node, (exp.Round, exp.Cast, exp.TryCast)):
        node = node.this
    return node


def _never_null(indicator: exp.Expression) -> bool:
    """True when a 0/1 CASE/IIF has an explicit ELSE, so it is never NULL."""
    if isinstance(indicator, exp.Case):
        return indicator.args.get("default") is not None
    return indicator.args.get("false") is not None


def _plan(query: exp.Select, kinds: dict[str, str]) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Rewrite the validated query so every released number is a count or a bounded sum.

    Returns the rewritten SQL, how to build each output column, and each noised
    column's sensitivity (how much one person can change it).
    """
    select_sql: list[str] = []
    outputs: list[dict[str, Any]] = []
    sensitivity: dict[str, int] = {}
    for i, projection in enumerate(query.expressions):
        kind = kinds[projection.alias_or_name.lower()]
        name = projection.alias_or_name or _unalias(projection).sql(dialect="sqlite")
        if name.lower() == "n":
            select_sql.append("COUNT(*) AS n")
            sensitivity["n"] = 1
            outputs.append({"name": name, "kind": "n"})
            continue
        if kind in {"group", "label"}:
            outputs.append({"name": name, "kind": "key", "position": len(select_sql)})
            select_sql.append(projection.sql(dialect="sqlite"))
            continue
        agg = _aggregate(projection)
        if kind == "count":
            key = f"__dp_{i}"
            select_sql.append(f"{agg.sql(dialect='sqlite')} AS {key}")
            sensitivity[key] = 1
            outputs.append({"name": name, "kind": "count", "sum": key})
            continue
        value = agg.this
        value_sql = value.sql(dialect="sqlite")
        sum_key, den_key = f"__dp_{i}_sum", f"__dp_{i}_den"
        if _is_indicator(value):
            select_sql.append(f"SUM({value_sql}) AS {sum_key}")
            sensitivity[sum_key] = 1
            if _never_null(value):
                # Every row counts in the denominator, so it is n; reuse the noisy n.
                outputs.append({"name": name, "kind": "rate", "sum": sum_key, "den": "n"})
                continue
            outputs.append({"name": name, "kind": "rate", "sum": sum_key, "den": den_key})
        else:
            column = value.find(exp.Column)
            bounds = MEAN_BOUNDS.get(column.name.lower()) if column else None
            if bounds is None:
                raise SafeSqlError(
                    f"AVG of {value_sql} is not allowed with differential privacy: it has no safe value range. "
                    f"Allowed: {', '.join(sorted(MEAN_BOUNDS))}, or a 0/1 CASE for rates."
                )
            low, high = bounds
            select_sql.append(
                f"SUM(CAST(ROUND(MIN(MAX(CAST({value_sql} AS REAL), {low}), {high}) * {MEAN_SCALE}) AS INTEGER)) AS {sum_key}"
            )
            sensitivity[sum_key] = math.ceil(max(abs(low), abs(high)) * MEAN_SCALE)
            outputs.append({"name": name, "kind": "mean", "sum": sum_key, "den": den_key, "bounds": bounds})
        select_sql.append(f"COUNT({value_sql}) AS {den_key}")
        sensitivity[den_key] = 1

    rewritten = query.copy()
    rewritten.set("order", None)
    rewritten.set("limit", None)
    rewritten.set("expressions", [sqlglot.parse_one(sql, read="sqlite") for sql in select_sql])
    return rewritten.sql(dialect="sqlite"), outputs, sensitivity


def _scales(config: DPConfig, sensitivity: dict[str, int]) -> tuple[dict[str, float], float]:
    """Noise scale per column: the query's epsilon is split, half to n and the rest evenly."""
    others = [key for key in sensitivity if key != "n"]
    n_epsilon = config.epsilon_per_query * (N_SHARE if others else 1.0)
    other_epsilon = (config.epsilon_per_query - n_epsilon) / len(others) if others else 0.0
    scales = {"n": sensitivity["n"] / n_epsilon}
    scales.update({key: sensitivity[key] / other_epsilon for key in others})
    return scales, n_epsilon


def _noisy_row(
    raw: tuple[Any, ...],
    names: list[str],
    outputs: list[dict[str, Any]],
    scales: dict[str, float],
    threshold: int,
    min_cell_size: int,
) -> tuple[dict[str, Any] | None, int]:
    by_name = dict(zip(names, raw))
    noisy = {key: _mechanism(scale)(_whole(by_name.get(key))) for key, scale in scales.items()}
    n = noisy["n"]
    if n < threshold:
        return None, 0
    row: dict[str, Any] = {}
    suppressed = 0
    for out in outputs:
        name, kind = out["name"], out["kind"]
        if kind == "key":
            row[name] = raw[out["position"]]
        elif kind == "n":
            row[name] = n
        elif kind == "count":
            count = min(max(noisy[out["sum"]], 0), n)
            row[name] = None if 0 < count < min_cell_size or 0 < n - count < min_cell_size else count
        else:
            den = noisy[out["den"]]
            total = noisy[out["sum"]]
            if den < min_cell_size:
                row[name] = None
            elif kind == "rate":
                total = min(max(total, 0), den)
                small = 0 < total < min_cell_size or 0 < den - total < min_cell_size
                row[name] = None if small else round(total / den, 3)
            else:
                low, high = out["bounds"]
                row[name] = round(min(max(total / MEAN_SCALE / den, low), high), 1)
        if kind not in {"key", "n"} and row[name] is None:
            suppressed += 1
    return row, suppressed


def _sort_and_limit(
    rows: list[dict[str, Any]], query: exp.Select, columns: list[str], max_rows: int
) -> list[dict[str, Any]]:
    """Apply the query's ORDER BY and LIMIT to the noisy rows (post-processing only)."""
    order = query.args.get("order")
    for ordered in reversed(order.expressions if order else []):
        key = _order_column(ordered.this, query, columns)
        if key is None:
            continue
        present = [row for row in rows if row.get(key) is not None]
        missing = [row for row in rows if row.get(key) is None]
        present.sort(key=lambda row: (type(row[key]).__name__, row[key]), reverse=bool(ordered.args.get("desc")))
        rows = present + missing
    limit = query.args.get("limit")
    limit_value = limit.expression if limit is not None else None
    if isinstance(limit_value, exp.Literal) and not limit_value.is_string:
        max_rows = min(max_rows, int(limit_value.this))
    return rows[:max_rows]


def _order_column(target: exp.Expression, query: exp.Select, columns: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    if isinstance(target, exp.Literal) and not target.is_string:
        index = int(target.this) - 1
        return columns[index] if 0 <= index < len(columns) else None
    if isinstance(target, exp.Column) and target.name.lower() in lowered:
        return lowered[target.name.lower()]
    target_sql = target.sql(dialect="sqlite").lower()
    for projection, column in zip(query.expressions, columns):
        if _unalias(projection).sql(dialect="sqlite").lower() == target_sql:
            return column
    return None


@contextmanager
def _ledger(path: Path) -> Iterator[dict[str, Any]]:
    """Read the budget ledger under an exclusive lock; save it only if the block succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "a", encoding="utf-8") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"epsilon_spent": 0.0, "delta_spent": 0.0, "queries": 0, "answers": {}}
        )
        yield ledger
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(path)


def run_dp_query(
    cleaned: str,
    query: exp.Select,
    kinds: dict[str, str],
    conn: sqlite3.Connection,
    min_cell_size: int,
    max_rows: int,
    config: DPConfig,
) -> dict[str, Any]:
    if query.args.get("having"):
        raise SafeSqlError("HAVING is not allowed with differential privacy: it filters on exact totals. Use WHERE.")
    rewritten, outputs, sensitivity = _plan(query, kinds)
    scales, n_epsilon = _scales(config, sensitivity)
    threshold = group_threshold(config, min_cell_size, n_epsilon)
    # Saved answers belong to one build of the database; a rebuild gets fresh answers (the budget still adds up).
    db_file = conn.execute("PRAGMA database_list").fetchone()[2]
    db_stat = os.stat(db_file) if db_file else None
    db_version = [db_stat.st_size, db_stat.st_mtime_ns] if db_stat else None
    cache_key = hashlib.sha256(
        json.dumps(
            [
                query.sql(dialect="sqlite").lower(),
                config.epsilon_per_query,
                config.delta_per_query,
                min_cell_size,
                max_rows,
                db_version,
            ]
        ).encode("utf-8")
    ).hexdigest()

    with _ledger(config.ledger_path) as ledger:
        cached = ledger["answers"].get(cache_key)
        if cached is None:
            if ledger["epsilon_spent"] + config.epsilon_per_query > config.total_epsilon + 1e-9:
                raise SafeSqlError(
                    f"The RSA-911 privacy budget is used up ({ledger['epsilon_spent']:g} of {config.total_epsilon:g}). "
                    "No new custom queries can run; repeated queries and the prebuilt summaries still work."
                )
            cur = conn.execute(
                f"SELECT * FROM ({rewritten}) AS safe_result WHERE CAST(n AS INTEGER) >= ? LIMIT ?",
                (min_cell_size, MAX_GROUPS),
            )
            names = [desc[0] for desc in cur.description]
            columns = [out["name"] for out in outputs]
            rows: list[dict[str, Any]] = []
            suppressed_cells = 0
            for raw in cur.fetchall():
                row, suppressed = _noisy_row(raw, names, outputs, scales, threshold, min_cell_size)
                if row is not None:
                    rows.append(row)
                    suppressed_cells += suppressed
            rows = _sort_and_limit(rows, query, columns, max_rows)
            cached = {
                "sql": cleaned,
                "min_cell_size": min_cell_size,
                "round_base": 0,
                "row_count": len(rows),
                "columns": columns,
                "rows": rows,
                "suppressed_cells": suppressed_cells,
                "differential_privacy": {
                    "mechanism": "OpenDP discrete Laplace",
                    "epsilon": config.epsilon_per_query,
                    "delta": config.delta_per_query,
                    "group_threshold": threshold,
                },
                "privacy_note": (
                    f"Values include random noise (differential privacy, epsilon={config.epsilon_per_query:g} per query), "
                    f"so they are close estimates, not exact numbers. Groups with fewer than about {threshold} people are hidden."
                ),
            }
            ledger["answers"][cache_key] = cached
            ledger["epsilon_spent"] += config.epsilon_per_query
            ledger["delta_spent"] += config.delta_per_query
            ledger["queries"] += 1
            reused = False
        else:
            reused = True
        result = json.loads(json.dumps(cached))
        result["differential_privacy"].update(
            {
                "reused_saved_answer": reused,
                "budget_spent": round(ledger["epsilon_spent"], 6),
                "budget_total": config.total_epsilon,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Show or reset the RSA-911 differential privacy budget.")
    parser.add_argument("--reset", action="store_true", help="Clear the budget and saved answers.")
    parser.add_argument("--yes", action="store_true", help="Confirm --reset.")
    args = parser.parse_args()
    # Read the same .env settings the agent uses, so the limit shown is the one runs enforce.
    from dotenv import load_dotenv

    load_dotenv()
    config = DPConfig.from_env() or DPConfig()
    if args.reset:
        if not args.yes:
            raise SystemExit(
                "Resetting gives every query a fresh budget, which weakens the privacy guarantee. "
                "Only do this when the data changes. Re-run with --reset --yes to confirm."
            )
        with _ledger(config.ledger_path) as ledger:
            ledger.update({"epsilon_spent": 0.0, "delta_spent": 0.0, "queries": 0, "answers": {}})
        print("Privacy budget reset.")
        return 0
    with _ledger(config.ledger_path) as ledger:
        print(f"Epsilon spent: {ledger['epsilon_spent']:g} of {config.total_epsilon:g}")
        print(f"Delta spent: {ledger['delta_spent']:g}")
        print(f"New queries answered: {ledger['queries']}")
        print(f"Per query: epsilon={config.epsilon_per_query:g}, delta={config.delta_per_query:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
