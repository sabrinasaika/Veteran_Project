#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rsa911.code_labels import codebook_note, label_code
from rsa911.config import connect_existing, sqlite_path, work_dir
from rsa911.privacy import DEFAULT_MIN_CELL_SIZE


TABLE = "rsa911_exiters"
EMPLOYED_EXIT_CODES = {"1", "2", "3", "4", "5"}

RACE_ETHNICITY_INDICATOR_COLUMNS = {
    "amerindian",
    "asian",
    "black",
    "hawaiian",
    "white",
    "hispanic",
}

CORE_CATEGORICAL_COLUMNS = [
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
    "exittype",
    "exitreason",
    "exitprimoccupation",
]

NUMERIC_COLUMNS = [
    "age_app",
    "age_ipe",
    "age_exit",
    "ipehourlywage",
    "ipeweeklyhoursworked",
    "exithourlywage",
    "exitweeklyhoursworked",
]

OUTCOME_ADJACENT_COLUMNS = {
    "exitempoutcome",
    "exittype",
    "exitreason",
    "exitprimoccupation",
    "exithourlywage",
    "exitweeklyhoursworked",
}

SERVICE_PREFIXES = [
    "jec",
    "wble",
    "ceo",
    "wrt",
    "isa",
    "gcut",
    "fycut",
    "jcct",
    "ovt",
    "ojt",
    "rat",
    "barlt",
    "jrt",
    "drst",
    "mt",
    "rset",
    "ct",
    "ass",
    "dti",
    "vrcg",
    "jsa",
    "jpa",
    "stjs",
    "ses",
    "irs",
    "bc",
    "ces",
    "es",
    "tran",
    "mnt",
    "rt",
    "pas",
    "tas",
    "rs",
    "is",
    "os",
    "wblev",
    "homemodify",
]


def _table_columns(conn: sqlite3.Connection, table: str = TABLE) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _analysis_columns(conn: sqlite3.Connection) -> list[str]:
    available = set(_table_columns(conn))
    selected = ["veteran", "exitempoutcome"]
    selected.extend(col for col in CORE_CATEGORICAL_COLUMNS if col in available)
    selected.extend(col for col in NUMERIC_COLUMNS if col in available)
    for col in sorted(available):
        if any(col.startswith(prefix) for prefix in SERVICE_PREFIXES) and (
            col.endswith("vragencystaff") or col.endswith("vragencypurchase")
        ):
            selected.append(col)
    return list(dict.fromkeys(selected))


def _read_veteran_frame(conn: sqlite3.Connection, columns: list[str]) -> pd.DataFrame:
    quoted = ", ".join(f'"{col}"' for col in columns)
    return pd.read_sql_query(f'SELECT {quoted} FROM "{TABLE}" WHERE trim(veteran) = "1"', conn)


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    exit_outcome = _clean_text(out.get("exitempoutcome", pd.Series("", index=out.index)))
    out["_employed_blank_as_not_employed"] = exit_outcome.isin(EMPLOYED_EXIT_CODES).astype(int)
    out["_known_exit_outcome"] = exit_outcome.ne("").astype(int)
    return out


def _rate(successes: int, n: int) -> float | None:
    if n <= 0:
        return None
    return round(float(successes) / float(n), 4)


def _safe_count(value: int, min_cell_size: int) -> int | None:
    return int(value) if value >= min_cell_size else None


def _safe_successes(value: int, min_cell_size: int) -> int | None:
    if value == 0 or value >= min_cell_size:
        return int(value)
    return None


def _cramers_v_from_counts(counts: np.ndarray) -> float | None:
    if counts.size == 0:
        return None
    total = float(counts.sum())
    if total <= 0:
        return None
    row_sum = counts.sum(axis=1, keepdims=True)
    col_sum = counts.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    valid = expected > 0
    if not valid.any():
        return None
    chi2 = (((counts - expected) ** 2) / np.where(valid, expected, 1.0))[valid].sum()
    r, k = counts.shape
    denom = total * max(1, min(k - 1, r - 1))
    if denom <= 0:
        return None
    return round(float(math.sqrt(chi2 / denom)), 4)


def _categorical_associations(
    df: pd.DataFrame,
    categorical_cols: list[str],
    min_cell_size: int,
    top_n: int,
) -> list[dict[str, Any]]:
    y = df["_employed_blank_as_not_employed"].astype(int)
    results: list[dict[str, Any]] = []
    for col in categorical_cols:
        values = _clean_text(df[col]).replace("", "(blank)")
        table = pd.crosstab(values, y)
        if table.shape[0] < 2:
            continue
        if 0 not in table.columns:
            table[0] = 0
        if 1 not in table.columns:
            table[1] = 0
        table = table[[0, 1]]
        v = _cramers_v_from_counts(table.to_numpy(dtype=float))
        if v is None:
            continue

        groups = []
        for code, row in table.iterrows():
            n = int(row.sum())
            if n < min_cell_size:
                continue
            successes = int(row[1])
            groups.append(
                {
                    "code": str(code),
                    "label": label_code(col, "" if str(code) == "(blank)" else str(code)),
                    "n": n,
                    "employed_count": _safe_successes(successes, min_cell_size),
                    "employment_rate": _rate(successes, n),
                }
            )
        groups = sorted(groups, key=lambda item: item["employment_rate"] or 0, reverse=True)[:8]
        results.append(
            {
                "feature": col,
                "method": "Cramer's V between feature categories and employed_at_exit",
                "score": v,
                "reportable_groups": groups,
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_n]


def _numeric_associations(df: pd.DataFrame, numeric_cols: list[str], min_cell_size: int) -> list[dict[str, Any]]:
    y = df["_employed_blank_as_not_employed"].astype(float)
    results: list[dict[str, Any]] = []
    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.notna()
        n = int(valid.sum())
        if n < min_cell_size:
            continue
        employed_values = x.loc[valid & y.eq(1)]
        not_employed_values = x.loc[valid & y.eq(0)]
        if len(employed_values) < min_cell_size or len(not_employed_values) < min_cell_size:
            continue
        corr = float(np.corrcoef(x.loc[valid].astype(float), y.loc[valid])[0, 1])
        if math.isnan(corr):
            continue
        pooled_sd = float(x.loc[valid].std(ddof=1) or 0)
        mean_diff = float(employed_values.mean() - not_employed_values.mean())
        smd = mean_diff / pooled_sd if pooled_sd > 0 else None
        results.append(
            {
                "feature": col,
                "method": "point-biserial correlation / standardized mean difference",
                "n": n,
                "correlation_with_employed": round(corr, 4),
                "employed_mean": round(float(employed_values.mean()), 4),
                "not_employed_mean": round(float(not_employed_values.mean()), 4),
                "standardized_mean_difference": round(float(smd), 4) if smd is not None else None,
            }
        )
    return sorted(results, key=lambda item: abs(item["correlation_with_employed"]), reverse=True)


def _feature_label(name: str) -> str:
    if "__" not in name:
        return name
    _, rest = name.split("__", 1)
    if rest.endswith("_missing"):
        return f"{rest[:-8]}=(blank / missing)"
    if "_" not in rest:
        return rest
    column, code = rest.rsplit("_", 1)
    label = label_code(column, code)
    if "label not verified" in label:
        return f"{rest} (label not verified)"
    return f"{column}={code} ({label})"


def _logistic_feature_importance(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    min_cell_size: int,
    top_n: int,
) -> dict[str, Any]:
    y = df["_employed_blank_as_not_employed"].astype(int)
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if len(df) < min_cell_size or positives < min_cell_size or negatives < min_cell_size:
        return {
            "status": "suppressed",
            "reason": "Not enough positive and negative cases.",
            "n": _safe_count(len(df), min_cell_size),
            "positive_cases": _safe_successes(positives, min_cell_size),
            "negative_cases": _safe_successes(negatives, min_cell_size),
        }

    X = df[categorical_cols + numeric_cols].copy()
    for col in categorical_cols:
        X[col] = _clean_text(X[col]).replace("", "missing")
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=min_cell_size)),
                    ]
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = round(float(roc_auc_score(y_test, proba)), 4) if len(set(y_test)) > 1 else None

    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    ranked = sorted(zip(feature_names, coefs), key=lambda item: abs(float(item[1])), reverse=True)
    features = [
        {
            "feature": _feature_label(str(name)),
            "coefficient": round(float(coef), 4),
            "direction": "higher predicted employment" if coef > 0 else "lower predicted employment",
        }
        for name, coef in ranked[:top_n]
    ]
    return {
        "status": "completed",
        "method": "L2 logistic regression with one-hot categorical features and balanced class weights",
        "n": int(len(df)),
        "positive_cases": positives,
        "negative_cases": negatives,
        "positive_rate": _rate(positives, len(df)),
        "auc_holdout": auc,
        "top_absolute_coefficients": features,
        "interpretation_warning": (
            "Feature importance is associational and model-dependent. It does not prove causality "
            "or identify interventions that would change employment outcomes."
        ),
    }


def _feature_relationships(
    df: pd.DataFrame,
    categorical_cols: list[str],
    min_cell_size: int,
    top_n: int,
) -> list[dict[str, Any]]:
    candidates = [col for col in categorical_cols if _clean_text(df[col]).nunique(dropna=False) >= 2]
    relationships: list[dict[str, Any]] = []
    for left, right in itertools.combinations(candidates, 2):
        if left in RACE_ETHNICITY_INDICATOR_COLUMNS and right in RACE_ETHNICITY_INDICATOR_COLUMNS:
            continue
        left_values = _clean_text(df[left]).replace("", "(blank)")
        right_values = _clean_text(df[right]).replace("", "(blank)")
        table = pd.crosstab(left_values, right_values)
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        if int(table.to_numpy().sum()) < min_cell_size:
            continue
        v = _cramers_v_from_counts(table.to_numpy(dtype=float))
        if v is None:
            continue
        relationships.append(
            {
                "feature_a": left,
                "feature_b": right,
                "method": "Cramer's V",
                "score": v,
                "n": int(table.to_numpy().sum()),
            }
        )
    return sorted(relationships, key=lambda item: item["score"], reverse=True)[:top_n]


def build_statistical_summary(
    sqlite_file: Path,
    min_cell_size: int = DEFAULT_MIN_CELL_SIZE,
    top_n: int = 20,
) -> dict[str, Any]:
    conn = connect_existing(sqlite_file)
    try:
        columns = _analysis_columns(conn)
        df = _read_veteran_frame(conn, columns)
    finally:
        conn.close()

    df = _add_outcome(df)
    available_categorical = [
        col
        for col in CORE_CATEGORICAL_COLUMNS
        if col in df.columns and col not in {"source_file"} and col not in OUTCOME_ADJACENT_COLUMNS
    ]
    service_categorical = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in SERVICE_PREFIXES)
        and (col.endswith("vragencystaff") or col.endswith("vragencypurchase"))
    ]
    categorical_cols = list(dict.fromkeys(available_categorical + service_categorical))
    numeric_cols = [col for col in NUMERIC_COLUMNS if col in df.columns and col not in OUTCOME_ADJACENT_COLUMNS]

    y = df["_employed_blank_as_not_employed"].astype(int)
    positives = int(y.sum())
    total = int(len(df))
    descriptive = {
        "veteran_records": total,
        "employed_count": _safe_successes(positives, min_cell_size),
        "not_employed_or_blank_count": _safe_successes(total - positives, min_cell_size),
        "employment_rate_blank_as_not_employed": _rate(positives, total),
        "outcome_definition": "exitEmpOutcome in 1-5 is employed; blank and other codes are treated as not employed.",
    }

    return {
        "source": str(sqlite_file),
        "table": TABLE,
        "privacy_note": "Outputs are aggregate/model summaries only. No individual rows are returned.",
        "min_cell_size": min_cell_size,
        "codebook_note": codebook_note(),
        "excluded_outcome_adjacent_columns": sorted(OUTCOME_ADJACENT_COLUMNS),
        "descriptive_statistics": descriptive,
        "feature_discriminant_analysis": {
            "categorical_feature_scores": _categorical_associations(
                df, categorical_cols, min_cell_size=min_cell_size, top_n=top_n
            ),
            "numeric_feature_scores": _numeric_associations(df, numeric_cols, min_cell_size=min_cell_size)[:top_n],
        },
        "feature_importance": _logistic_feature_importance(
            df,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            min_cell_size=min_cell_size,
            top_n=top_n,
        ),
        "feature_relationships": _feature_relationships(
            df,
            categorical_cols=categorical_cols,
            min_cell_size=min_cell_size,
            top_n=top_n,
        ),
        "interpretation_guidance": [
            "Use 'associated with employment outcome', not 'causes employment'.",
            "Treat blank exitEmpOutcome as not employed only as a provisional definition.",
            "Mapped labels are based on the local RSA-911 codebook mapping; raw codes marked label not verified need official codebook review before publication.",
            "Use these results to guide follow-up modeling and domain review, not as causal recommendations.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run privacy-safe RSA-911 statistical analyses.")
    parser.add_argument("--sqlite", default=str(sqlite_path()))
    parser.add_argument("--out-dir", default=str(work_dir()))
    parser.add_argument("--min-cell-size", type=int, default=DEFAULT_MIN_CELL_SIZE)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    sqlite_file = Path(args.sqlite).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_statistical_summary(
        sqlite_file,
        min_cell_size=args.min_cell_size,
        top_n=args.top_n,
    )
    out_path = out_dir / "statistical_analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote privacy-safe statistical analysis summary to {out_path}")
    print(f"Veteran records analyzed: {summary['descriptive_statistics']['veteran_records']}")
    print("No individual rows were printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
