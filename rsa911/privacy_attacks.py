#!/usr/bin/env python3
"""Empirical privacy attacks against the RSA-911 SQL tool.

Plays an attacker who can call the same validated SQL interface the agent uses
(run_safe_sql) and knows a target veteran's quasi-identifiers (state agency, sex,
age at exit, race/ethnicity, primary disability). Three measurements:

1. Re-identification risk: how many veteran records are unique on those
   quasi-identifiers (k-anonymity class sizes).
2. Membership inference: a differencing attack decides whether the target's
   record is in the data. Count everyone, count everyone except people matching
   the target, and subtract. Each trial removes the target from a working copy
   with probability 1/2, as in the standard membership-inference game.
3. Attribute inference: the same differencing on employed veterans reveals the
   target's employment outcome at exit.

Each attack is run against the exact interface (small-cell suppression only) and
the differential privacy interface, averaging k repeated queries. Repeats get a
no-op "AND i = i" so the saved-answer cache does not return the same noise.

Works on a temporary veterans-only copy of the database and a temporary DP
ledger, so the real privacy budget is untouched. Only aggregate attack metrics
are printed and saved; no individual values leave this script.

Usage:
  python -m rsa911.privacy_attacks --targets 200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sqlite3
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from rsa911.config import connect_existing, sqlite_path, work_dir
from rsa911.dp import DPConfig
from rsa911.privacy import DEFAULT_MIN_CELL_SIZE
from rsa911.safe_sql import run_safe_sql

TABLE = "rsa911_exiters"
VETERAN = "TRIM(veteran) = '1'"
EMPLOYED = "TRIM(exitempoutcome) IN ('1','2','3','4','5')"
RACE_COLUMNS = ["amerindian", "asian", "black", "hawaiian", "white", "hispanic"]

# Nested quasi-identifier sets an attacker might plausibly know about a target.
QI_LEVELS = {
    "state + sex + age": ["agencycode", "sex", "age_exit"],
    "+ race/ethnicity": ["agencycode", "sex", "age_exit", *RACE_COLUMNS],
    "+ primary disability": ["agencycode", "sex", "age_exit", *RACE_COLUMNS, "primdisability"],
}
REPEATS = [1, 5, 25]


def _value_expr(col: str) -> str:
    return f"COALESCE(TRIM({col}), '')"


def _literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _match(qi: list[str], values: dict[str, Any]) -> str:
    return " AND ".join(f"{_value_expr(col)} = {_literal(values[col])}" for col in qi)


def _make_working_copy(source: Path, dest: Path) -> list[str]:
    """Copy veteran rows only (the attack's population) into a private temporary database."""
    src = connect_existing(source)
    try:
        columns = [row[1] for row in src.execute(f'PRAGMA table_info("{TABLE}")')]
        dst = sqlite3.connect(dest)
        try:
            dst.execute(src.execute("SELECT sql FROM sqlite_master WHERE name = ?", (TABLE,)).fetchone()[0])
            placeholders = ", ".join("?" for _ in columns)
            rows = src.execute(f'SELECT * FROM "{TABLE}" WHERE {VETERAN}')
            dst.executemany(f'INSERT INTO "{TABLE}" VALUES ({placeholders})', rows)
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()
    os.chmod(dest, 0o600)
    return columns


def _uniqueness(conn: sqlite3.Connection, qi: list[str]) -> dict[str, Any]:
    keys = ", ".join(_value_expr(col) for col in qi)
    sizes = [row[0] for row in conn.execute(f'SELECT COUNT(*) FROM "{TABLE}" GROUP BY {keys}')]
    total = sum(sizes)
    return {
        "quasi_identifiers": qi,
        "records": total,
        "equivalence_classes": len(sizes),
        "share_unique": round(sum(s for s in sizes if s == 1) / total, 4),
        "share_in_class_below_5": round(sum(s for s in sizes if s < 5) / total, 4),
        "share_in_class_below_min_cell": round(sum(s for s in sizes if s < DEFAULT_MIN_CELL_SIZE) / total, 4),
    }


def _query_n(sql: str, db: Path, dp_config: DPConfig | None) -> int:
    result = run_safe_sql(sql, sqlite_file=db, dp_config=dp_config)
    return int(result["rows"][0]["n"]) if result["rows"] else 0


def _difference(base_where: str, match: str, db: Path, dp_config: DPConfig | None, k: int) -> list[float]:
    """n(base) - n(base AND NOT target), repeated k times; returns the running mean after each repeat."""
    diffs = []
    for i in range(1, k + 1):
        noop = f" AND {i} = {i}" if dp_config is not None else ""
        everyone = _query_n(f"SELECT COUNT(*) AS n FROM {TABLE} WHERE {base_where}{noop}", db, dp_config)
        others = _query_n(f"SELECT COUNT(*) AS n FROM {TABLE} WHERE {base_where} AND NOT ({match}){noop}", db, dp_config)
        diffs.append(everyone - others)
    return [sum(diffs[:j]) / j for j in range(1, k + 1)]


def _summary(guesses: list[int], truths: list[int]) -> dict[str, Any]:
    n = len(truths)
    correct = sum(g == t for g, t in zip(guesses, truths))
    pos = [g for g, t in zip(guesses, truths) if t == 1]
    neg = [g for g, t in zip(guesses, truths) if t == 0]
    tpr = sum(pos) / len(pos) if pos else None
    fpr = sum(neg) / len(neg) if neg else None
    acc = correct / n if n else None
    # 95% Wilson interval for accuracy.
    z = 1.96
    centre = (acc + z * z / (2 * n)) / (1 + z * z / n) if n else None
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n) if n else None
    return {
        "trials": n,
        "accuracy": round(acc, 4) if acc is not None else None,
        "accuracy_95ci": [round(centre - half, 4), round(centre + half, 4)] if n else None,
        "tpr": round(tpr, 4) if tpr is not None else None,
        "fpr": round(fpr, 4) if fpr is not None else None,
        "advantage": round(tpr - fpr, 4) if tpr is not None and fpr is not None else None,
    }


def _dp_bound(epsilon: float, delta: float) -> float:
    """Best possible accuracy of any attacker against an (epsilon, delta)-DP release with a 50/50 prior."""
    return round(min(1.0, math.exp(epsilon) / (1 + math.exp(epsilon)) + delta), 4)


def run_attacks(source: Path, n_targets: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    base_config = DPConfig.from_env() or DPConfig()
    tmp = Path(tempfile.mkdtemp(prefix="privacy_attack_", dir=work_dir()))
    os.chmod(tmp, 0o700)
    try:
        db = tmp / "veterans.sqlite3"
        columns = _make_working_copy(source, db)
        conn = sqlite3.connect(db)
        try:
            qi_levels = {name: [c for c in qi if c in columns] for name, qi in QI_LEVELS.items()}
            uniqueness = {name: _uniqueness(conn, qi) for name, qi in qi_levels.items()}

            # Targets: veterans unique on the strongest quasi-identifier set (the worst case for privacy).
            qi = qi_levels["+ primary disability"]
            keys = ", ".join(_value_expr(col) for col in qi)
            unique_rowids = [
                row[0]
                for row in conn.execute(
                    f'SELECT MIN(rowid) FROM "{TABLE}" GROUP BY {keys} HAVING COUNT(*) = 1'
                )
            ]
            targets = rng.sample(unique_rowids, min(n_targets, len(unique_rowids)))

            membership = {"exact": ([], []), **{k: ([], []) for k in REPEATS}}
            attribute = {"exact": ([], []), **{k: ([], []) for k in REPEATS}}
            employed_truth: list[int] = []
            for index, rowid in enumerate(targets, 1):
                values = dict(zip(qi, conn.execute(f'SELECT {keys} FROM "{TABLE}" WHERE rowid = ?', (rowid,)).fetchone()))
                match = _match(qi, values)
                employed = conn.execute(
                    f'SELECT CASE WHEN {EMPLOYED} THEN 1 ELSE 0 END FROM "{TABLE}" WHERE rowid = ?', (rowid,)
                ).fetchone()[0]
                employed_truth.append(employed)
                ledger = tmp / f"ledger_{index}.json"
                dp_config = replace(base_config, total_epsilon=1e9, ledger_path=ledger)

                # Attribute inference: the target is known to be in the data.
                exact = _difference(f"{VETERAN} AND {EMPLOYED}", match, db, None, 1)[-1]
                attribute["exact"][0].append(int(exact > 0.5))
                attribute["exact"][1].append(employed)
                means = _difference(f"{VETERAN} AND {EMPLOYED}", match, db, dp_config, max(REPEATS))
                for k in REPEATS:
                    attribute[k][0].append(int(means[k - 1] > 0.5))
                    attribute[k][1].append(employed)

                # Membership inference: remove the target with probability 1/2.
                member = rng.random() < 0.5
                saved = None
                if not member:
                    saved = conn.execute(f'SELECT rowid, * FROM "{TABLE}" WHERE rowid = ?', (rowid,)).fetchone()
                    conn.execute(f'DELETE FROM "{TABLE}" WHERE rowid = ?', (rowid,))
                    conn.commit()
                try:
                    exact = _difference(VETERAN, match, db, None, 1)[-1]
                    membership["exact"][0].append(int(exact > 0.5))
                    membership["exact"][1].append(int(member))
                    ledger.unlink(missing_ok=True)
                    means = _difference(VETERAN, match, db, dp_config, max(REPEATS))
                    for k in REPEATS:
                        membership[k][0].append(int(means[k - 1] > 0.5))
                        membership[k][1].append(int(member))
                finally:
                    if saved is not None:
                        placeholders = ", ".join("?" for _ in saved)
                        conn.execute(f'INSERT INTO "{TABLE}" (rowid, {", ".join(columns)}) VALUES ({placeholders})', saved)
                        conn.commit()
                ledger.unlink(missing_ok=True)
                if index % 20 == 0:
                    print(f"  {index}/{len(targets)} targets done")
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    base_rate = sum(employed_truth) / len(employed_truth) if employed_truth else 0.0
    eps, delta = base_config.epsilon_per_query, base_config.delta_per_query

    def rows(results: dict) -> list[dict[str, Any]]:
        out = [{"interface": "exact (suppression only)", "repeats": 1, "epsilon_spent": 0, "dp_accuracy_bound": None,
                **_summary(*results["exact"])}]
        for k in REPEATS:
            # Only the "everyone" query depends on the target, so the bound uses k queries; the attack spends 2k.
            out.append({"interface": "differential privacy", "repeats": k, "epsilon_spent": 2 * k * eps,
                        "dp_accuracy_bound": _dp_bound(k * eps, k * delta), **_summary(*results[k])})
        return out

    return {
        "privacy_note": "Aggregate attack metrics only. No individual values are included.",
        "population": "veteran exiters",
        "targets": len(targets),
        "target_selection": "random veterans unique on: " + ", ".join(qi),
        "dp_settings": {"epsilon_per_query": eps, "delta_per_query": delta, "lifetime_budget": base_config.total_epsilon},
        "reidentification_risk": uniqueness,
        "membership_inference": rows(membership),
        "attribute_inference": {
            "attribute": "employed at exit (codes 1-5)",
            "majority_class_baseline_accuracy": round(max(base_rate, 1 - base_rate), 4),
            "results": rows(attribute),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run membership and attribute inference attacks on the RSA-911 SQL tool.")
    parser.add_argument("--sqlite", default=str(sqlite_path()))
    parser.add_argument("--targets", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(work_dir()))
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    summary = run_attacks(Path(args.sqlite).expanduser(), args.targets, args.seed)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "privacy_attack_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nRe-identification risk ({summary['population']}):")
    for name, u in summary["reidentification_risk"].items():
        print(f"  {name:<24} unique {u['share_unique']:.1%}   class < 5: {u['share_in_class_below_5']:.1%}")
    for title, results in [("Membership inference", summary["membership_inference"]),
                           ("Attribute inference", summary["attribute_inference"]["results"])]:
        print(f"\n{title} ({summary['targets']} targets):")
        print(f"  {'interface':<26}{'k':>4}{'eps':>6}{'acc':>8}{'adv':>8}{'bound':>8}")
        for r in results:
            bound = "--" if r["dp_accuracy_bound"] is None else f"{r['dp_accuracy_bound']:.3f}"
            print(f"  {r['interface']:<26}{r['repeats']:>4}{r['epsilon_spent']:>6g}{r['accuracy']:>8.3f}{r['advantage']:>8.3f}{bound:>8}")
    print(f"  majority-class baseline: {summary['attribute_inference']['majority_class_baseline_accuracy']:.3f}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
