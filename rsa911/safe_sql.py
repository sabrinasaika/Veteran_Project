#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlglot
from sqlglot import exp

from rsa911.config import connect_existing, sqlite_path, work_dir
from rsa911.privacy import DEFAULT_MIN_CELL_SIZE

if TYPE_CHECKING:
    from rsa911.dp import DPConfig


TABLE = "rsa911_exiters"
DEFAULT_MAX_ROWS = 50
DEFAULT_ROUND_BASE = 0
MAX_GROUP_BY = 3

# Only these functions may appear anywhere in a query. MIN/MAX are left out on
# purpose: they return one person's exact value.
ALLOWED_FUNCTIONS = (
    exp.Count,
    exp.Sum,
    exp.Avg,
    exp.Case,
    exp.If,
    exp.Trim,
    exp.Cast,
    exp.TryCast,
    exp.Round,
    exp.Coalesce,
    exp.Lower,
    exp.Upper,
)

# Constructs that let a query reach row-level data or combine result sets.
FORBIDDEN_NODES = (
    exp.Subquery,
    exp.With,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.Join,
    exp.Window,
    exp.Lateral,
    exp.Values,
    exp.Offset,
)

BLOCKED_KEYWORDS = {
    "alter",
    "attach",
    "create",
    "delete",
    "detach",
    "drop",
    "insert",
    "pragma",
    "replace",
    "update",
    "vacuum",
}

BLOCKED_COLUMNS = {
    "individualid",
    "applicationdate",
    "eligibilitydate",
    "twestartdate",
    "tweenddate",
    "oosplacementdate",
    "oosexitdate",
    "vrservicestartdate",
    "careerservicedate",
    "exitdate",
    "ipeinitialdate",
    "ipeextensiondate",
}

CORE_SCHEMA_COLUMNS = [
    "source_file",
    "programyear",
    "agencycode",
    "sex",
    "amerindian",
    "asian",
    "black",
    "hawaiian",
    "white",
    "hispanic",
    "veteran",
    "referral",
    "student",
    "disability",
    "primdisability",
    "primdisabilitycause",
    "seconddisability",
    "seconddisabilitycause",
    "disabilitysigcode",
    "ipesupportedempgoal",
    "ipeempstatus",
    "ipeprimaryocc",
    "ipehourlywage",
    "ipeweeklyhoursworked",
    "longtermunemp",
    "homelessorrunaway",
    "exoffenderstatus",
    "lowincomestatus",
    "englishlearner",
    "basicskillsdeficient",
    "singleparent",
    "dishomemaker",
    "vocrehab",
    "wpempservice",
    "exitType".lower(),
    "exitreason",
    "exitempoutcome",
    "exitprimoccupation",
    "exithourlywage",
    "exitweeklyhoursworked",
    "age_app",
    "age_ipe",
    "age_exit",
]

SERVICE_COLUMNS = [
    "jpaVRAgencyStaff",
    "jpaVRAgencyPurchase",
    "jsaVRAgencyStaff",
    "jsaVRAgencyPurchase",
    "sesVRAgencyStaff",
    "sesVRAgencyPurchase",
    "bcVRAgencyStaff",
    "bcVRAgencyPurchase",
    "cesVRAgencyStaff",
    "cesVRAgencyPurchase",
    "wblevVRAgencyStaff",
    "wblevVRAgencyPurchase",
    "tranVRAgencyStaff",
    "tranVRAgencyPurchase",
    "pasVRAgencyStaff",
    "pasVRAgencyPurchase",
    "homemodifyVRAgencyStaff",
    "homemodifyVRAgencyPurchase",
]

OUTCOME_HELP = {
    "employed_blank_as_not_employed": (
        "Use AVG(CASE WHEN trim(exitempoutcome) IN ('1','2','3','4','5') THEN 1.0 ELSE 0.0 END). "
        "Denominator includes blanks as not employed."
    ),
    "employment_among_known_exit_outcomes": (
        "Filter WHERE trim(exitempoutcome) <> ''. Use AVG(CASE WHEN trim(exitempoutcome) IN ('1','2','3','4','5') THEN 1.0 ELSE 0.0 END)."
    ),
    "competitive_integrated_among_known": (
        "Filter WHERE trim(exitempoutcome) <> ''. Use AVG(CASE WHEN trim(exitempoutcome) = '1' THEN 1.0 ELSE 0.0 END)."
    ),
}


class SafeSqlError(ValueError):
    pass


def _columns(conn: sqlite3.Connection) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{TABLE}")').fetchall()]


def _service_columns_clean() -> list[str]:
    return [re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower() for col in SERVICE_COLUMNS]


# Allowlist: a query may only reference these columns. Anything else (dates,
# IDs, columns nobody has reviewed yet) is rejected by default.
ALLOWED_COLUMNS = {col.lower() for col in CORE_SCHEMA_COLUMNS} | set(_service_columns_clean())


def schema_for_prompt(sqlite_file: Path = sqlite_path(), topic: str = "") -> dict[str, Any]:
    conn = connect_existing(sqlite_file)
    try:
        available = set(_columns(conn))
    finally:
        conn.close()

    service_clean = _service_columns_clean()
    cols = [col for col in CORE_SCHEMA_COLUMNS if col in available]
    cols.extend(col for col in service_clean if col in available)
    cols = list(dict.fromkeys(cols))
    return {
        "table": TABLE,
        "minimum_cell_size": DEFAULT_MIN_CELL_SIZE,
        "required_result_count_column": "n",
        "allowed_columns_core": cols,
        "blocked_columns": sorted(BLOCKED_COLUMNS),
        "outcome_definitions": OUTCOME_HELP,
        "rules": [
            "Only one SELECT aggregate query over rsa911_exiters is allowed: no subqueries, joins, WITH, UNION, window functions, or OFFSET.",
            "The select list must include COUNT(*) AS n at the top level.",
            f"Every other selected column must be an aggregate or a GROUP BY expression (at most {MAX_GROUP_BY} GROUP BY expressions).",
            "Allowed functions: COUNT, SUM, AVG, CASE/IIF, TRIM, CAST, ROUND, COALESCE, LOWER, UPPER. MIN and MAX are not allowed.",
            "SUM is only allowed as a count, e.g. SUM(CASE WHEN ... THEN 1 ELSE 0 END). Use AVG for means and rates.",
            f"Rows with n < {DEFAULT_MIN_CELL_SIZE} are removed. Counts and rates implying fewer than {DEFAULT_MIN_CELL_SIZE} people are returned as null.",
            (
                f"Counts are rounded to the nearest {DEFAULT_ROUND_BASE}; rates to 3 decimals; means to 1 decimal."
                if DEFAULT_ROUND_BASE
                else "Rates are rounded to 3 decimals and means to 1 decimal; counts are exact."
            ),
            "Only columns in allowed_columns_core may be used. Do not SELECT *.",
            "Use trim(column) when comparing categorical codes stored as text.",
            (
                "Disability columns hold numeric codes, never words, so never use LIKE '%PTSD%' or similar. "
                "primdisability/seconddisability = impairment type: 10 mobility orthopedic/neurological, "
                "12 mobility and dexterity, 17 cognitive, 18 psychosocial, 19 other mental. "
                "primdisabilitycause/seconddisabilitycause = cause: 03 amputations, 04 anxiety disorders "
                "(RSA-911 has no PTSD code; PTSD is recorded here), 15 depressive and mood disorders, "
                "35 spinal cord injury, 37 traumatic brain injury. Codes may be stored without the leading zero, "
                "so match both forms, e.g. trim(primdisabilitycause) IN ('4','04')."
            ),
            "Always filter trim(veteran) = '1' when the question is about veterans.",
            "programyear holds the program year as 'PY2021', 'PY2022', 'PY2023'; use it for comparisons across years.",
        ],
    }


def _clean_sql(sql: str) -> str:
    sql = str(sql or "").strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def _unalias(node: exp.Expression) -> exp.Expression:
    return node.this if isinstance(node, exp.Alias) else node


def _is_zero_one_literal(node: exp.Expression | None) -> bool:
    if node is None:
        return True
    if isinstance(node, exp.Paren):
        node = node.this
    if not isinstance(node, exp.Literal) or node.is_string:
        return False
    try:
        return float(node.this) in (0.0, 1.0)
    except ValueError:
        return False


def _is_indicator(node: exp.Expression) -> bool:
    """True for CASE/IIF expressions whose every result is the literal 0 or 1."""
    if isinstance(node, exp.Case):
        results = [branch.args.get("true") for branch in node.args.get("ifs") or []]
        results.append(node.args.get("default"))
    elif isinstance(node, exp.If):
        results = [node.args.get("true"), node.args.get("false")]
    else:
        return False
    return all(_is_zero_one_literal(result) for result in results)


def _is_plain_column(node: exp.Expression) -> bool:
    """True for a column, optionally wrapped in TRIM/CAST/COALESCE(column, literal)."""
    while isinstance(node, (exp.Trim, exp.Cast, exp.TryCast, exp.Coalesce, exp.Paren)):
        if isinstance(node, exp.Coalesce) and not all(
            isinstance(arg, exp.Literal) for arg in node.args.get("expressions") or []
        ):
            return False
        node = node.this
    return isinstance(node, exp.Column)


def _aggregate_kind(node: exp.Expression) -> str:
    """Classify a select expression as 'count', 'mean', 'group' or 'label'."""
    inner = node
    if isinstance(inner, (exp.Round, exp.Cast, exp.TryCast)):
        inner = inner.this
    if isinstance(inner, exp.Count):
        return "count"
    if isinstance(inner, exp.Sum):
        if not _is_indicator(inner.this):
            raise SafeSqlError(
                "SUM is only allowed as a count, e.g. SUM(CASE WHEN ... THEN 1 ELSE 0 END). Use AVG for means."
            )
        return "count"
    if isinstance(inner, exp.Avg):
        # AVG(CASE WHEN <one person> THEN wage ELSE 0 END) * n would recover that
        # person's value, so AVG takes a plain column or a 0/1 indicator only.
        if not (_is_plain_column(inner.this) or _is_indicator(inner.this)):
            raise SafeSqlError(
                "AVG is only allowed over a column (e.g. AVG(exithourlywage)) or a 0/1 CASE for rates. "
                "Use WHERE to choose the group instead of CASE inside AVG."
            )
        return "mean"
    if node.find(exp.AggFunc):
        raise SafeSqlError("Aggregates may only be wrapped in ROUND or CAST, e.g. ROUND(AVG(x), 3) AS avg_x.")
    return "group" if node.find(exp.Column) else "label"


def _check_query(sql: str) -> tuple[str, exp.Select, dict[str, str]]:
    """Parse and validate one query. Returns the cleaned SQL, its tree, and each output column's kind."""
    cleaned = _clean_sql(sql)
    lowered = cleaned.lower()
    if not cleaned:
        raise SafeSqlError("SQL is empty.")
    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        raise SafeSqlError("Comments are not allowed.")
    tokens = set(re.findall(r"[a-z_][a-z0-9_]*", lowered))
    blocked = sorted(tokens & BLOCKED_KEYWORDS)
    if blocked:
        raise SafeSqlError(f"Blocked SQL keyword(s): {', '.join(blocked)}")
    blocked_cols = sorted(tokens & BLOCKED_COLUMNS)
    if blocked_cols:
        raise SafeSqlError(f"Blocked sensitive column(s): {', '.join(blocked_cols)}")

    try:
        statements = [stmt for stmt in sqlglot.parse(cleaned, read="sqlite") if stmt is not None]
    except sqlglot.errors.ParseError as exc:
        raise SafeSqlError(f"Could not parse SQL: {exc}") from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise SafeSqlError("Only one SELECT query is allowed.")
    query = statements[0]

    for node_type in FORBIDDEN_NODES:
        if query.find(node_type):
            raise SafeSqlError(
                f"{node_type.__name__} is not allowed. Use one flat aggregate SELECT over {TABLE}."
            )
    if any(select is not query for select in query.find_all(exp.Select)):
        raise SafeSqlError(f"Subqueries are not allowed. Use one flat aggregate SELECT over {TABLE}.")

    tables = list(query.find_all(exp.Table))
    if len(tables) != 1 or tables[0].name.lower() != TABLE or tables[0].db:
        raise SafeSqlError(f"Query must read from {TABLE} only.")

    for func in query.find_all(exp.Func):
        # sqlglot models AND/OR as functions; they are plain logic, not data access.
        if not isinstance(func, ALLOWED_FUNCTIONS + (exp.Connector,)):
            raise SafeSqlError(
                f"Function not allowed: {func.sql(dialect='sqlite')[:60]}. "
                "Allowed: COUNT, SUM, AVG, CASE/IIF, TRIM, CAST, ROUND, COALESCE, LOWER, UPPER."
            )
    for star in query.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            raise SafeSqlError("SELECT * is not allowed.")

    aliases = {projection.alias.lower() for projection in query.expressions if projection.alias}
    for column in query.find_all(exp.Column):
        name = column.name.lower()
        if name not in ALLOWED_COLUMNS and name not in aliases:
            raise SafeSqlError(
                f"Column not allowed: {column.name}. Use describe_rsa911_sql_schema for the allowed columns."
            )

    group = query.args.get("group")
    group_exprs = list(group.expressions) if group else []
    if len(group_exprs) > MAX_GROUP_BY:
        raise SafeSqlError(f"At most {MAX_GROUP_BY} GROUP BY expressions are allowed.")
    group_sqls = {expr.sql(dialect="sqlite").lower() for expr in group_exprs}
    group_names = {expr.name.lower() for expr in group_exprs if isinstance(expr, exp.Column)}

    kinds: dict[str, str] = {}
    for projection in query.expressions:
        name = projection.alias_or_name.lower()
        inner = _unalias(projection)
        kind = _aggregate_kind(inner)
        if kind == "group" and inner.sql(dialect="sqlite").lower() not in group_sqls and name not in group_names:
            raise SafeSqlError(f"Selected column {name} must be an aggregate or appear in GROUP BY.")
        if name in kinds:
            raise SafeSqlError(f"Duplicate output column name: {name}.")
        kinds[name] = kind

    n_projection = next(
        (_unalias(projection) for projection in query.expressions if projection.alias_or_name.lower() == "n"),
        None,
    )
    if not (isinstance(n_projection, exp.Count) and isinstance(n_projection.this, exp.Star)):
        raise SafeSqlError("Query must include COUNT(*) AS n in the top-level select list.")
    return cleaned, query, kinds


def validate_sql(sql: str) -> str:
    cleaned, _, _ = _check_query(sql)
    return cleaned


def _round_to_base(value: float, base: int) -> int:
    if not base:
        return int(value)
    return int(base * round(float(value) / base))


def _protect_row(row: dict[str, Any], kinds: dict[str, str], min_cell_size: int, round_base: int) -> tuple[dict[str, Any], int]:
    """Null out counts/rates that imply a small group, then round what is left."""
    n_key = next(key for key in row if key.lower() == "n")
    n = int(float(row[n_key]))
    protected = dict(row)
    suppressed = 0
    for col, value in row.items():
        kind = kinds.get(col.lower())
        if col == n_key or value is None or kind not in {"count", "mean"}:
            continue
        value = float(value)
        if kind == "mean" and not 0.0 <= value <= 1.0:
            protected[col] = round(value, 1)
            continue
        # A count, or a rate whose implied count is value * n. Both the count and
        # its complement must be 0 or at least min_cell_size.
        count = round(value if kind == "count" else value * n)
        if 0 < count < min_cell_size or 0 < n - count < min_cell_size:
            protected[col] = None
            suppressed += 1
            continue
        protected[col] = _round_to_base(value, round_base) if kind == "count" else round(value, 3)
    protected[n_key] = _round_to_base(n, round_base)
    return protected, suppressed


def _check_aliases(conn: sqlite3.Connection, query: exp.Select) -> None:
    # SQLite resolves a name to the real column before an alias, so an alias
    # must not reuse the name of a column outside the allowlist.
    hidden = {col.lower() for col in _columns(conn)} - ALLOWED_COLUMNS
    clashes = sorted(projection.alias for projection in query.expressions if projection.alias.lower() in hidden)
    if clashes:
        raise SafeSqlError(f"Alias names must not match restricted columns: {', '.join(clashes)}")


def run_safe_sql(
    sql: str,
    sqlite_file: Path = sqlite_path(),
    min_cell_size: int = DEFAULT_MIN_CELL_SIZE,
    max_rows: int = DEFAULT_MAX_ROWS,
    round_base: int = DEFAULT_ROUND_BASE,
    dp_config: "DPConfig | None" = None,
) -> dict[str, Any]:
    """Run one validated aggregate query. With dp_config, answers get differential privacy noise."""
    cleaned, query, kinds = _check_query(sql)
    if dp_config is not None:
        from rsa911.dp import run_dp_query

        conn = connect_existing(sqlite_file)
        try:
            _check_aliases(conn, query)
            return run_dp_query(cleaned, query, kinds, conn, min_cell_size, max_rows, dp_config)
        finally:
            conn.close()

    wrapped = f'SELECT * FROM ({cleaned}) AS safe_result WHERE CAST(n AS INTEGER) >= ? LIMIT ?'
    conn = connect_existing(sqlite_file)
    try:
        _check_aliases(conn, query)
        cur = conn.execute(wrapped, (min_cell_size, max_rows))
        columns = [desc[0] for desc in cur.description]
        raw_rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    if not any(col.lower() == "n" for col in columns):
        raise SafeSqlError("Result must include an n column.")
    rows = []
    suppressed_cells = 0
    for row in raw_rows:
        n_val = next(value for key, value in row.items() if key.lower() == "n")
        if int(float(n_val)) < min_cell_size:
            raise SafeSqlError("Unsafe small cell escaped validation.")
        protected, suppressed = _protect_row(row, kinds, min_cell_size, round_base)
        rows.append(protected)
        suppressed_cells += suppressed

    return {
        "sql": cleaned,
        "min_cell_size": min_cell_size,
        "round_base": round_base,
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
        "suppressed_cells": suppressed_cells,
        "privacy_note": (
            "Only aggregate result rows with n >= min_cell_size are returned. Counts and rates implying fewer "
            "than min_cell_size people are null."
            + (f" Counts are rounded to the nearest {round_base}." if round_base else "")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validated aggregate-only SQL against private RSA-911 SQLite.")
    parser.add_argument("--sql", default="", help="Aggregate SELECT SQL to run.")
    parser.add_argument("--schema", action="store_true", help="Print JSON schema/rules for text-to-SQL prompting.")
    parser.add_argument("--sqlite", default=str(sqlite_path()))
    parser.add_argument("--out-dir", default=str(work_dir()))
    parser.add_argument("--min-cell-size", type=int, default=DEFAULT_MIN_CELL_SIZE)
    args = parser.parse_args()

    sqlite_file = Path(args.sqlite).expanduser()
    if args.schema:
        print(json.dumps(schema_for_prompt(sqlite_file), indent=2))
        return 0
    if not args.sql:
        raise SystemExit("--sql is required unless --schema is used")

    try:
        result = run_safe_sql(args.sql, sqlite_file=sqlite_file, min_cell_size=args.min_cell_size)
    except SafeSqlError as exc:
        print(f"Safe SQL rejected: {exc}")
        return 2
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "safe_sql_result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote safe SQL aggregate result to {out_path}")
    print(f"Returned rows: {result['row_count']}")
    print("No rows with n < min_cell_size were returned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
