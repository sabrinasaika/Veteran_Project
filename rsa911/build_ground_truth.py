from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from .code_labels import codebook_note, label_code
from .config import sqlite_path, work_dir
from .safe_sql import SafeSqlError, run_safe_sql


# Fills in RSA-911 ground truth for the expanded benchmark by running each question's
# aggregate query through safe_sql (n >= 11 enforcement and small-cell suppression still
# apply). The merged benchmark is written under data/rsa911_private/ because it contains
# numbers derived from the private data; the committed benchmark is left unchanged.

VETERAN = "trim(veteran) = '1'"
EMPLOYED = "trim(exitempoutcome) IN ('1','2','3','4','5')"
EMP_RATE = f"AVG(CASE WHEN {EMPLOYED} THEN 1.0 ELSE 0.0 END) AS employed_rate"
RATE_NOTE = "employment rate at exit, counting blank exitEmpOutcome as not employed"


def _grouped(column: str, extra_where: str = "", order: str = "") -> str:
    where = f"{VETERAN} AND {extra_where}" if extra_where else VETERAN
    sql = (
        f"SELECT trim({column}) AS {column}, COUNT(*) AS n, {EMP_RATE} "
        f"FROM rsa911_exiters WHERE {where} GROUP BY trim({column})"
    )
    return f"{sql} ORDER BY {order}" if order else sql


def _indicator(column: str) -> str:
    return f"SELECT COUNT(*) AS n, {EMP_RATE} FROM rsa911_exiters WHERE {VETERAN} AND trim({column}) = '1'"


MENTAL = "trim(primdisability) LIKE '17%' OR trim(primdisability) LIKE '18%' OR trim(primdisability) LIKE '19%'"
PHYSICAL = " OR ".join(f"trim(primdisability) LIKE '{code}%'" for code in range(10, 17))

# benchmark id -> list of (label, sql, grouping column or None)
QUERIES: dict[str, list[tuple[str, str, str | None]]] = {
    "rsa911_overall_veteran_rate": [
        ("Veterans vs non-veterans", f"SELECT trim(veteran) AS veteran, COUNT(*) AS n, {EMP_RATE} FROM rsa911_exiters GROUP BY trim(veteran)", "veteran"),
    ],
    "black_veteran_employment_disparity": [("Among veterans, by Black indicator", _grouped("black"), "black")],
    "homelessness_employment_rates": [("Among veterans, by homeless/runaway status", _grouped("homelessorrunaway"), "homelessorrunaway")],
    "race_ethnicity_rates": [
        (f"Among {name} veterans", _indicator(col), None)
        for name, col in [
            ("Black", "black"),
            ("White", "white"),
            ("Asian", "asian"),
            ("American Indian/Alaska Native", "amerindian"),
            ("Native Hawaiian/Pacific Islander", "hawaiian"),
            ("Hispanic", "hispanic"),
        ]
    ],
    "primary_disability_rates": [("Among veterans, by primary disability", _grouped("primdisability", order="employed_rate DESC"), "primdisability")],
    "rsa911_sex_comparison": [("Among veterans, by sex", _grouped("sex"), "sex")],
    # The PY23 file has no program-year column; each source file is one program year.
    "rsa911_program_year_trend": [
        ("Among veterans, by program year (one source file per year; only loaded years appear)", _grouped("source_file", order="source_file"), "source_file"),
    ],
    "rsa911_significance_of_disability": [("Among veterans, by significance of disability", _grouped("disabilitysigcode"), "disabilitysigcode")],
    "rsa911_long_term_unemployed": [("Among veterans, by long-term unemployment at entry", _grouped("longtermunemp"), "longtermunemp")],
    "rsa911_exoffender": [("Among veterans, by ex-offender status", _grouped("exoffenderstatus"), "exoffenderstatus")],
    "rsa911_low_income": [("Among veterans, by low-income status", _grouped("lowincomestatus"), "lowincomestatus")],
    "rsa911_supported_employment_goal": [("Among veterans, by supported employment goal on the IPE", _grouped("ipesupportedempgoal"), "ipesupportedempgoal")],
    "rsa911_agency_comparison": [("Among veterans, by state agency", _grouped("agencycode", order="employed_rate DESC"), "agencycode")],
    "rsa911_mental_health_vs_physical": [
        (
            "Among veterans, mental (primary disability types 17-19) vs physical (types 10-16)",
            f"SELECT CASE WHEN {MENTAL} THEN 'mental' WHEN {PHYSICAL} THEN 'physical' ELSE 'other' END AS group_name, "
            f"COUNT(*) AS n, {EMP_RATE} FROM rsa911_exiters WHERE {VETERAN} "
            f"GROUP BY CASE WHEN {MENTAL} THEN 'mental' WHEN {PHYSICAL} THEN 'physical' ELSE 'other' END",
            "group_name",
        ),
    ],
    "rsa911_exit_reasons": [
        ("Most common exit reasons among veterans", f"SELECT trim(exitreason) AS exitreason, COUNT(*) AS n FROM rsa911_exiters WHERE {VETERAN} GROUP BY trim(exitreason) ORDER BY n DESC", "exitreason"),
    ],
    "rsa911_hourly_wage_at_exit": [
        (
            "Average hourly wage at exit among employed veterans, by primary disability",
            f"SELECT trim(primdisability) AS primdisability, COUNT(*) AS n, AVG(CAST(exithourlywage AS REAL)) AS avg_hourly_wage "
            f"FROM rsa911_exiters WHERE {VETERAN} AND {EMPLOYED} AND CAST(exithourlywage AS REAL) > 0 "
            f"GROUP BY trim(primdisability) ORDER BY avg_hourly_wage DESC",
            "primdisability",
        ),
    ],
    "rsa911_weekly_hours": [
        (
            "Weekly hours among employed veterans",
            f"SELECT COUNT(*) AS n, AVG(CAST(exitweeklyhoursworked AS REAL)) AS avg_weekly_hours, "
            f"SUM(CASE WHEN CAST(exitweeklyhoursworked AS REAL) >= 35 THEN 1 ELSE 0 END) AS worked_35_plus "
            f"FROM rsa911_exiters WHERE {VETERAN} AND {EMPLOYED} AND CAST(exitweeklyhoursworked AS REAL) > 0",
            None,
        ),
    ],
}

# Questions the tools cannot answer as asked; recorded so reviewers see why no numbers were added.
NOT_COMPUTED = {
    "rsa911_stats_wage_predictors": (
        "Not computed: the statistical analysis layer models employment at exit and excludes exithourlywage "
        "as an outcome-adjacent column, so it cannot identify wage predictors. A good answer should say so."
    ),
}


def _value(value: Any) -> str:
    return "suppressed (fewer than 11 people)" if value is None else str(value)


def _describe_row(row: dict[str, Any], column: str | None) -> str:
    parts = []
    if column:
        code = row.get(column)
        parts.append(f"{label_code(column, code)} [code {code if code not in (None, '') else 'blank'}]")
    parts.append(f"n={_value(row.get('n'))}")
    for key, value in row.items():
        if key not in {"n", column}:
            parts.append(f"{key}={_value(value)}")
    return ", ".join(parts)


def _fact_text(label: str, result: dict[str, Any], column: str | None) -> str:
    rows = result["rows"]
    if not rows:
        return f"{label}: no reportable groups (all groups had fewer than 11 people)."
    if len(rows) > 10:
        rows = rows[:5] + rows[-5:]
        label += " (top five and bottom five reportable groups)"
    return f"{label} ({RATE_NOTE} where a rate is shown): " + "; ".join(_describe_row(r, column) for r in rows) + "."


def _statistical_fact(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    importance = summary.get("feature_importance", {})
    if importance.get("status") != "completed":
        return None
    top = importance.get("top_absolute_coefficients", [])[:5]
    listed = "; ".join(f"{f['feature']} ({f['direction']}, coefficient {f['coefficient']})" for f in top)
    rate = summary.get("descriptive_statistics", {}).get("employment_rate_blank_as_not_employed")
    return (
        f"The RSA-911 statistical analysis layer (L2 logistic regression, holdout AUC {importance.get('auc_holdout')}) "
        f"reports a veteran employment rate of {rate} (blank exitEmpOutcome as not employed); the five features with "
        f"the largest absolute coefficients are: {listed}. These are associations, not causal effects."
    )


def build(benchmark: Path, sqlite_file: Path, out_path: Path, summary_path: Path) -> dict[str, Any]:
    cases = json.loads(benchmark.read_text())
    stamp = f"{date.today().isoformat()}: computed from private RSA-911 data with rsa911/safe_sql.py (aggregate only, n >= 11)"
    report: dict[str, Any] = {"computed": [], "failed": {}, "not_computed": NOT_COMPUTED}
    for case in cases:
        cid = case["id"]
        gt = case.get("ground_truth") or {}
        facts: list[dict[str, Any]] = []
        if cid in QUERIES:
            for i, (label, sql, column) in enumerate(QUERIES[cid], 1):
                try:
                    result = run_safe_sql(sql, sqlite_file=sqlite_file)
                except (SafeSqlError, sqlite3.Error) as exc:
                    report["failed"][f"{cid}#{i}"] = f"{type(exc).__name__}: {exc}"
                    continue
                facts.append({
                    "id": f"RSA911-{cid.upper()}-{i}",
                    "fact": _fact_text(label, result, column),
                    "source": "rsa911/safe_sql.py",
                    "basis": "computed",
                    "query": sql,
                    "verified": stamp,
                })
        elif cid == "statistical_feature_importance":
            text = _statistical_fact(summary_path)
            if text:
                facts.append({
                    "id": "RSA911-STATISTICAL_FEATURE_IMPORTANCE-1",
                    "fact": text,
                    "source": "rsa911/statistical_analysis.py",
                    "basis": "computed",
                    "verified": f"{date.today().isoformat()}: read from {summary_path.name} (aggregate model summary)",
                })
            else:
                report["failed"][cid] = f"{summary_path} missing or model suppressed; run rsa911.statistical_analysis first"
        if cid in NOT_COMPUTED:
            gt["notes"] = NOT_COMPUTED[cid]
        if facts:
            gt["key_facts"] = facts
            gt["notes"] = f"RSA-911 values computed {date.today().isoformat()} from private data. {codebook_note()}"
            report["computed"].append(cid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill RSA-911 ground truth for the expanded benchmark from private data.")
    parser.add_argument("--benchmark", default="eval_benchmark_expanded.json")
    parser.add_argument("--sqlite", default=str(sqlite_path()))
    parser.add_argument("--summary", default=str(work_dir() / "statistical_analysis_summary.json"))
    parser.add_argument("--out", default=str(work_dir() / "eval_benchmark_expanded_rsa911.json"))
    args = parser.parse_args()

    sqlite_file = Path(args.sqlite).expanduser()
    if not sqlite_file.exists():
        raise SystemExit(f"{sqlite_file} not found. Build it first with: python -m rsa911.build_sqlite --replace")
    out_path = Path(args.out).expanduser()
    report = build(Path(args.benchmark), sqlite_file, out_path, Path(args.summary).expanduser())
    print(f"Computed RSA-911 ground truth for {len(report['computed'])} questions.")
    for key, reason in report["failed"].items():
        print(f"  FAILED {key}: {reason}")
    for key, reason in report["not_computed"].items():
        print(f"  SKIPPED {key}: {reason}")
    print(f"Wrote merged benchmark to {out_path} (private; do not commit).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
