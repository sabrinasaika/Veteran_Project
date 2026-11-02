#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

from rsa911.config import connect_existing, sqlite_path, work_dir
from rsa911.privacy import DEFAULT_MIN_CELL_SIZE


ALIASES = {
    "veteran": ["veteran", "element_016", "e016", "c_16"],
    "primary_disability": ["primdisability", "primary_disability", "element_043", "e043", "c_43"],
    "primary_disability_cause": ["primdisabilitycause", "primary_disability_cause"],
    "secondary_disability": ["seconddisability", "secondary_disability", "element_044", "e044", "c_44"],
    "secondary_disability_cause": ["seconddisabilitycause", "secondary_disability_cause"],
    "employment_outcome_exit": ["exitempoutcome", "employment_outcome_at_exit", "employment_outcome_exit", "element_356", "e356", "c_356"],
    "occupation_exit": ["exitprimoccupation", "primary_occupation_at_exit", "occupation_at_exit", "element_357", "e357", "c_357"],
    "hourly_wage_exit": ["exithourlywage", "hourly_wage_at_exit", "element_359", "e359", "c_359"],
    "hours_exit": ["exitweeklyhoursworked", "hours_worked_in_a_week_at_exit", "hours_worked_week_exit", "element_360", "e360", "c_360"],
}


DISABILITY_SOURCE = {
    "00": "cause unknown",
    "01": "accident/injury other than TBI or SCI",
    "02": "alcohol abuse or dependence",
    "03": "amputation",
    "04": "anxiety disorders",
    "05": "arthritis and rheumatism",
    "06": "asthma and other allergies",
    "07": "ADHD",
    "08": "autism",
    "09": "blood disorders",
    "10": "cancer",
    "11": "cardiac/circulatory conditions",
    "12": "cerebral palsy",
    "13": "congenital condition or birth injury",
    "14": "cystic fibrosis",
    "15": "depressive and other mood disorders",
    "16": "diabetes mellitus",
    "17": "digestive disorders",
    "18": "drug abuse or dependence",
    "19": "eating disorders",
    "20": "end-stage renal disease/genitourinary disorders",
    "21": "epilepsy",
    "22": "HIV or AIDS",
    "23": "immune deficiencies excluding HIV/AIDS",
    "24": "mental illness not listed elsewhere",
    "25": "intellectual disability",
    "26": "multiple sclerosis",
    "27": "muscular dystrophy",
    "28": "Parkinson's disease and other neurological disorders",
    "29": "personality disorders",
    "30": "physical disorders/conditions not listed elsewhere",
    "31": "polio",
    "32": "respiratory disorders other than cystic fibrosis/asthma",
    "33": "schizophrenia and other psychotic disorders",
    "34": "specific learning disabilities",
    "35": "spinal cord injury",
    "36": "stroke",
    "37": "traumatic brain injury",
}


EMPLOYMENT_OUTCOME_EXIT = {
    "1": "competitive integrated employment",
    "2": "self-employment",
    "3": "Randolph-Sheppard BEP",
    "4": "state agency managed BEP",
    "5": "supported employment in competitive integrated employment",
    "7": "homemaker",
}


SOC_LABELS = {
    "119013": "Farmers, Ranchers, and Other Agricultural Managers",
    "119199": "Managers, All Other",
    "211093": "Social and Human Service Assistants",
    "291141": "Registered Nurses",
    "339032": "Security Guards",
    "352021": "Food Preparation Workers",
    "353023": "Fast Food and Counter Workers",
    "359021": "Dishwashers",
    "372011": "Janitors and Cleaners, Except Maids and Housekeeping Cleaners",
    "372012": "Maids and Housekeeping Cleaners",
    "373011": "Landscaping and Groundskeeping Workers",
    "412011": "Cashiers",
    "412031": "Retail Salespersons",
    "434051": "Customer Service Representatives",
    "439061": "Office Clerks, General",
    "439199": "Office and Administrative Support Workers, All Other",
    "493023": "Automotive Service Technicians and Mechanics",
    "499071": "Maintenance and Repair Workers, General",
    "519198": "Helpers--Production Workers",
    "519199": "Production Workers, All Other",
    "533031": "Driver/Sales Workers",
    "533032": "Heavy and Tractor-Trailer Truck Drivers",
    "533033": "Light Truck Drivers",
    "537062": "Laborers and Freight, Stock, and Material Movers, Hand",
    "537065": "Stockers and Order Fillers",
}


KEY_DISABILITY_GROUPS = {
    "tbi": "37",
    "spinal_cord_injury": "35",
    "anxiety": "04",
    "depression_mood": "15",
    "multiple_sclerosis": "26",
    "amputation": "03",
}

# Groups by impairment type (PrimDisability / SecondDisability) rather than cause.
# Mobility covers far more people than spinal cord injury (cause 35) alone.
KEY_IMPAIRMENT_GROUPS = {
    "mobility": (["10", "12"], "Mobility orthopedic/neurological impairment (impairment types 10 and 12)"),
    "psychosocial": (["18"], "Psychosocial impairment (impairment type 18)"),
}


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [str(row[1]) for row in rows]


def _find_col(columns: list[str], key: str) -> Optional[str]:
    normalized = {re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower(): col for col in columns}
    for alias in ALIASES[key]:
        if alias in normalized:
            return normalized[alias]
    return None


def _source_code_expr(col: str) -> str:
    # RSA-911 disability values are commonly "type;source" such as "17;37".
    return f"""
    CASE
      WHEN instr({col}, ';') > 0 THEN substr({col}, instr({col}, ';') + 1, 2)
      ELSE NULL
    END
    """


def _code_expr(type_col: str, cause_col: Optional[str]) -> str:
    if cause_col:
        return f'"{cause_col}"'
    return _source_code_expr(f'"{type_col}"')


def _text_expr(expr: str) -> str:
    return f"trim(CAST(({expr}) AS TEXT))"


def _label_code(mapping: dict[str, str], value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "blank/unknown"
    candidates = [text]
    if text.isdigit():
        candidates.append(text.zfill(2))
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return "other/unknown"


def _normalize_code_expr(expr: str, width: int = 2) -> str:
    text = _text_expr(expr)
    return f"""
    CASE
      WHEN {text} = '' THEN ''
      WHEN length({text}) < {width} THEN substr('000000' || {text}, -{width}, {width})
      ELSE {text}
    END
    """


def _count_where(
    conn: sqlite3.Connection,
    table: str,
    where_sql: str,
    params: list[object],
) -> int:
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {where_sql}', params).fetchone()[0])


EMPLOYED_CODES = ("1", "2", "3", "4", "5")


def _employment_rate(
    conn: sqlite3.Connection,
    table: str,
    where_sql: str,
    params: list[object],
    outcome_col: Optional[str],
    n: int,
    min_cell_size: int,
) -> Optional[float]:
    """Share employed at exit, counting a blank outcome as not employed; None when either side is a small cell."""
    if not outcome_col or n < min_cell_size:
        return None
    placeholders = ", ".join("?" for _ in EMPLOYED_CODES)
    employed = _count_where(
        conn, table, f'{where_sql} AND trim("{outcome_col}") IN ({placeholders})', [*params, *EMPLOYED_CODES]
    )
    if 0 < employed < min_cell_size or 0 < n - employed < min_cell_size:
        return None
    return round(employed / n, 3)


def _safe_group_query(
    conn: sqlite3.Connection,
    table: str,
    group_col: str,
    where_sql: str,
    params: list[object],
    min_cell_size: int,
    limit: int,
    normalize_width: Optional[int] = None,
) -> list[dict]:
    normalized_group = (
        _normalize_code_expr(group_col, normalize_width)
        if normalize_width is not None
        else _text_expr(group_col)
    )
    denominator = _count_where(
        conn,
        table,
        f"{where_sql} AND {normalized_group} <> ''",
        params,
    )
    sql = f"""
        SELECT {normalized_group} AS value, COUNT(*) AS n
        FROM "{table}"
        WHERE {where_sql} AND {normalized_group} <> ''
        GROUP BY {normalized_group}
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
        LIMIT ?
    """
    rows = conn.execute(sql, [*params, min_cell_size, limit]).fetchall()
    return [
        {
            "value": row[0],
            "n": row[1],
            "percent": round(row[1] / denominator, 4) if denominator else None,
        }
        for row in rows
    ]


def _numeric_stats(
    conn: sqlite3.Connection,
    table: str,
    col: Optional[str],
    where_sql: str,
    params: list[object],
    min_cell_size: int,
    positive_only: bool = False,
) -> Optional[dict]:
    if not col:
        return None
    text_col = _text_expr(f'"{col}"')
    value_expr = f"CAST({text_col} AS REAL)"
    positive_clause = f"AND {value_expr} > 0" if positive_only else ""
    filter_sql = f"""
        FROM "{table}"
        WHERE {where_sql}
          AND {text_col} <> ''
          AND {value_expr} >= 0
          {positive_clause}
    """
    n, mean = conn.execute(f"SELECT COUNT(*), AVG({value_expr}) {filter_sql}", params).fetchone()
    n = int(n or 0)
    if n < min_cell_size:
        return None

    # No min/max: each is one person's exact value. The 10th/90th percentiles are shown only when
    # at least min_cell_size people fall below the 10th and above the 90th, so neither points to
    # one person; with fewer, they are None.
    def percentile(q: float) -> Optional[float]:
        if n < 10 * min_cell_size:
            return None
        position = q * (n - 1)
        lower = int(position)
        values = [
            float(row[0])
            for row in conn.execute(
                f"SELECT {value_expr} AS v {filter_sql} ORDER BY v LIMIT 2 OFFSET ?", [*params, lower]
            ).fetchall()
        ]
        upper_value = values[1] if len(values) > 1 else values[0]
        return round(values[0] + (upper_value - values[0]) * (position - lower), 2)

    return {
        "n": n,
        "mean": round(float(mean), 2) if mean is not None else None,
        "p10": percentile(0.1),
        "p90": percentile(0.9),
    }


def _disability_where(
    primary_col: str,
    primary_cause_col: Optional[str],
    secondary_col: Optional[str],
    secondary_cause_col: Optional[str],
    code: str,
) -> str:
    # Zero-pad before comparing: Excel exports can store cause 04 as "4".
    clauses = [_normalize_code_expr(_code_expr(primary_col, primary_cause_col)) + " IN (?, ?)"]
    if secondary_col:
        clauses.append(_normalize_code_expr(_code_expr(secondary_col, secondary_cause_col)) + " IN (?, ?)")
    return "(" + " OR ".join(clauses) + ")"


def _type_code_expr(col: str) -> str:
    # Impairment type is the part before ';' when a field holds "type;cause", else the whole value.
    return f"""
    CASE
      WHEN instr("{col}", ';') > 0 THEN substr("{col}", 1, instr("{col}", ';') - 1)
      ELSE "{col}"
    END
    """


def _impairment_where(primary_col: str, secondary_col: Optional[str], codes: list[str]) -> tuple[str, list[object]]:
    placeholders = ", ".join("?" for _ in codes)
    columns = [primary_col] + ([secondary_col] if secondary_col else [])
    clauses = [f"{_normalize_code_expr(_type_code_expr(col))} IN ({placeholders})" for col in columns]
    return "(" + " OR ".join(clauses) + ")", [code for _ in columns for code in codes]


def summarize_veterans(conn: sqlite3.Connection, table: str, min_cell_size: int) -> dict:
    cols = _columns(conn, table)
    veteran_col = _find_col(cols, "veteran")
    primary_col = _find_col(cols, "primary_disability")
    primary_cause_col = _find_col(cols, "primary_disability_cause")
    secondary_col = _find_col(cols, "secondary_disability")
    secondary_cause_col = _find_col(cols, "secondary_disability_cause")
    outcome_col = _find_col(cols, "employment_outcome_exit")
    occupation_col = _find_col(cols, "occupation_exit")
    wage_col = _find_col(cols, "hourly_wage_exit")
    hours_col = _find_col(cols, "hours_exit")

    if not veteran_col:
        raise ValueError("Could not find the Veteran column. Run inspect_schema and map aliases if needed.")

    where = f'"{veteran_col}" = ?'
    params: list[object] = ["1"]
    total_veterans = conn.execute(
        f'SELECT COUNT(*) FROM "{table}" WHERE {where}', params
    ).fetchone()[0]

    columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
    source_files = (
        [row[0] for row in conn.execute(f'SELECT DISTINCT source_file FROM "{table}" ORDER BY source_file')]
        if "source_file" in columns
        else []
    )

    payload: dict = {
        "min_cell_size": min_cell_size,
        # File names only (e.g. PY23exiters.csv), so answers can say which program years are loaded.
        "source_files": source_files,
        "total_veteran_records": total_veterans if total_veterans >= min_cell_size else None,
        "employment_rate_blank_as_not_employed": _employment_rate(
            conn, table, where, params, outcome_col, total_veterans, min_cell_size
        ),
        "available_columns": {
            "veteran": veteran_col,
            "primary_disability": primary_col,
            "primary_disability_cause": primary_cause_col,
            "secondary_disability": secondary_col,
            "secondary_disability_cause": secondary_cause_col,
            "employment_outcome_exit": outcome_col,
            "occupation_exit": occupation_col,
            "hourly_wage_exit": wage_col,
            "hours_exit": hours_col,
        },
        "note": "All tables suppress cells below min_cell_size. Results are aggregate records, not individual rows.",
    }

    if primary_col:
        expr = _code_expr(primary_col, primary_cause_col)
        rows = _safe_group_query(conn, table, expr, where, params, min_cell_size, 25, normalize_width=2)
        for row in rows:
            row["label"] = _label_code(DISABILITY_SOURCE, row["value"])
        payload["primary_disability_source_counts"] = rows

    if outcome_col:
        rows = _safe_group_query(
            conn, table, f'"{outcome_col}"', where, params, min_cell_size, 25
        )
        for row in rows:
            row["label"] = _label_code(EMPLOYMENT_OUTCOME_EXIT, row["value"])
        payload["employment_outcome_exit_counts"] = rows

    if occupation_col:
        rows = _safe_group_query(
            conn, table, f'"{occupation_col}"', where, params, min_cell_size, 25
        )
        for row in rows:
            row["label"] = SOC_LABELS.get(str(row["value"] or "").strip(), "SOC label not mapped")
        payload["occupation_exit_counts"] = rows

    payload["exit_hourly_wage_stats"] = _numeric_stats(conn, table, wage_col, where, params, min_cell_size)
    payload["exit_hourly_wage_positive_stats"] = _numeric_stats(
        conn, table, wage_col, where, params, min_cell_size, positive_only=True
    )
    payload["exit_weekly_hours_stats"] = _numeric_stats(conn, table, hours_col, where, params, min_cell_size)
    payload["exit_weekly_hours_positive_stats"] = _numeric_stats(
        conn, table, hours_col, where, params, min_cell_size, positive_only=True
    )

    if primary_col:
        group_filters: list[tuple[str, str, list[object], str]] = []
        for group_name, code in KEY_DISABILITY_GROUPS.items():
            disability_clause = _disability_where(
                primary_col,
                primary_cause_col,
                secondary_col,
                secondary_cause_col,
                code,
            )
            if secondary_col:
                disability_params = [code, code.zfill(2), code, code.zfill(2)]
            else:
                disability_params = [code, code.zfill(2)]
            group_filters.append((group_name, disability_clause, disability_params, DISABILITY_SOURCE[code]))
        for group_name, (codes, label) in KEY_IMPAIRMENT_GROUPS.items():
            impairment_clause, impairment_params = _impairment_where(primary_col, secondary_col, codes)
            group_filters.append((group_name, impairment_clause, impairment_params, label))

        group_summaries = {}
        for group_name, disability_clause, disability_params, label in group_filters:
            group_where = f"{where} AND {disability_clause}"
            group_params = [*params, *disability_params]
            group_n = _count_where(conn, table, group_where, group_params)
            if group_n < min_cell_size:
                continue

            item: dict = {
                "n": group_n,
                "label": label,
                "employment_rate_blank_as_not_employed": _employment_rate(
                    conn, table, group_where, group_params, outcome_col, group_n, min_cell_size
                ),
            }
            if outcome_col:
                outcome_rows = _safe_group_query(
                    conn, table, f'"{outcome_col}"', group_where, group_params, min_cell_size, 10
                )
                for row in outcome_rows:
                    row["label"] = _label_code(EMPLOYMENT_OUTCOME_EXIT, row["value"])
                item["employment_outcome_exit_counts"] = outcome_rows
            if occupation_col:
                occupation_rows = _safe_group_query(
                    conn, table, f'"{occupation_col}"', group_where, group_params, min_cell_size, 10
                )
                for row in occupation_rows:
                    row["label"] = SOC_LABELS.get(str(row["value"] or "").strip(), "SOC label not mapped")
                item["occupation_exit_counts"] = occupation_rows
            item["exit_hourly_wage_stats"] = _numeric_stats(
                conn, table, wage_col, group_where, group_params, min_cell_size
            )
            item["exit_hourly_wage_positive_stats"] = _numeric_stats(
                conn, table, wage_col, group_where, group_params, min_cell_size, positive_only=True
            )
            item["exit_weekly_hours_stats"] = _numeric_stats(
                conn, table, hours_col, group_where, group_params, min_cell_size
            )
            item["exit_weekly_hours_positive_stats"] = _numeric_stats(
                conn, table, hours_col, group_where, group_params, min_cell_size, positive_only=True
            )
            group_summaries[group_name] = item
        payload["key_disability_group_summaries"] = group_summaries

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Write privacy-safe aggregate summaries from local RSA-911 SQLite.")
    parser.add_argument("--sqlite", default=str(sqlite_path()))
    parser.add_argument("--table", default="rsa911_exiters")
    parser.add_argument("--out-dir", default=str(work_dir()))
    parser.add_argument("--min-cell-size", type=int, default=DEFAULT_MIN_CELL_SIZE)
    args = parser.parse_args()

    db_path = Path(args.sqlite).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_existing(db_path)
    try:
        payload = summarize_veterans(conn, args.table, args.min_cell_size)
    finally:
        conn.close()

    out_path = out_dir / "aggregate_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote privacy-safe aggregate summary to {out_path}")
    print(f"Minimum cell size: {args.min_cell_size}")
    print("No individual rows were printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
