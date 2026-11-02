#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from rsa911.config import connect_existing, sqlite_path, work_dir
from rsa911.privacy import DEFAULT_MIN_CELL_SIZE


TABLE = "rsa911_exiters"

EMPLOYED_EXIT_CODES = {"1", "2", "3", "4", "5"}
COMPETITIVE_INTEGRATED_CODE = "1"

OUTCOME_DEFINITIONS = {
    "known_exit_employment": (
        "Denominator is records with populated exitEmpOutcome; numerator is exitEmpOutcome in 1-5."
    ),
    "blank_as_not_employed": (
        "Denominator is all veteran exiter records; numerator is exitEmpOutcome in 1-5. "
        "Blank exitEmpOutcome is treated as not employed."
    ),
    "competitive_integrated_known": (
        "Denominator is records with populated exitEmpOutcome; numerator is exitEmpOutcome = 1."
    ),
}

CORE_COLUMNS = [
    "source_file",
    "veteran",
    "black",
    "sex",
    "hispanic",
    "amerindian",
    "asian",
    "hawaiian",
    "white",
    "student",
    "disability",
    "primdisability",
    "primdisabilitycause",
    "seconddisability",
    "seconddisabilitycause",
    "disabilitysigcode",
    "ipesupportedempgoal",
    "ipeempstatus",
    "adult",
    "adulted",
    "dislocatedworker",
    "vocrehab",
    "wpempservice",
    "youth",
    "longtermunemp",
    "homelessorrunaway",
    "exoffenderstatus",
    "lowincomestatus",
    "englishlearner",
    "basicskillsdeficient",
    "culturalbarriers",
    "singleparent",
    "dishomemaker",
    "migrantfarmworker",
    "schoolgradecompleted",
    "age_app",
    "age_ipe",
    "age_startvrservice",
    "age_vrservicestartdate",
    "age_exit",
    "exitempoutcome",
]

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


def _available_columns(conn: sqlite3.Connection) -> list[str]:
    cols = set(_table_columns(conn))
    selected = [col for col in CORE_COLUMNS if col in cols]
    for col in sorted(cols):
        if any(col.startswith(prefix) for prefix in SERVICE_PREFIXES) and (
            col.endswith("vragencystaff") or col.endswith("vragencypurchase")
        ):
            selected.append(col)
    return list(dict.fromkeys(selected))


def _read_private_frame(conn: sqlite3.Connection, columns: list[str]) -> pd.DataFrame:
    quoted = ", ".join(f'"{col}"' for col in columns)
    return pd.read_sql_query(f'SELECT {quoted} FROM "{TABLE}" WHERE trim(veteran) = "1"', conn)


def _clean_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    exit_outcome = _clean_text_series(out.get("exitempoutcome", pd.Series("", index=out.index)))
    known = exit_outcome.ne("")
    employed = exit_outcome.isin(EMPLOYED_EXIT_CODES)
    competitive = exit_outcome.eq(COMPETITIVE_INTEGRATED_CODE)
    out["_known_exit_employment"] = np.where(known, employed.astype(int), np.nan)
    out["_blank_as_not_employed"] = employed.astype(int)
    out["_competitive_integrated_known"] = np.where(known, competitive.astype(int), np.nan)
    return out


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _comparison_row(df: pd.DataFrame, group_mask: pd.Series, outcome_col: str, min_cell_size: int) -> dict[str, Any]:
    values = df.loc[group_mask, outcome_col].dropna()
    denominator = int(values.shape[0])
    numerator = int(values.sum()) if denominator else 0
    return {
        "n": denominator if denominator >= min_cell_size else None,
        "successes": numerator if denominator >= min_cell_size and numerator >= min_cell_size else None,
        "rate": _rate(numerator, denominator) if denominator >= min_cell_size else None,
        "suppressed": denominator < min_cell_size or (0 < numerator < min_cell_size),
    }


def analyze_black_disparity(df: pd.DataFrame, min_cell_size: int = DEFAULT_MIN_CELL_SIZE) -> dict[str, Any]:
    black = _clean_text_series(df.get("black", pd.Series("", index=df.index)))
    black_mask = black.eq("1")
    nonblack_mask = black.eq("0")

    results = {}
    for key, description in OUTCOME_DEFINITIONS.items():
        outcome_col = f"_{key}"
        black_row = _comparison_row(df, black_mask, outcome_col, min_cell_size)
        nonblack_row = _comparison_row(df, nonblack_mask, outcome_col, min_cell_size)
        diff = None
        if black_row.get("rate") is not None and nonblack_row.get("rate") is not None:
            diff = round(float(black_row["rate"]) - float(nonblack_row["rate"]), 4)
        results[key] = {
            "description": description,
            "black_veterans": black_row,
            "nonblack_veterans": nonblack_row,
            "rate_difference_black_minus_nonblack": diff,
        }

    return {
        "analysis": "black_veteran_employment_disparity",
        "min_cell_size": min_cell_size,
        "interpretation_warning": (
            "These are unadjusted descriptive comparisons. They do not prove discrimination, causality, "
            "or service effectiveness. The final outcome definition should be confirmed with the domain expert."
        ),
        "results": results,
    }


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    leakage = {
        "source_file",
        "veteran",
        "black",
        "exitempoutcome",
        "_known_exit_employment",
        "_blank_as_not_employed",
        "_competitive_integrated_known",
    }
    numeric_candidates = [
        col
        for col in ["age_app", "age_ipe", "age_startvrservice", "age_vrservicestartdate", "age_exit"]
        if col in df.columns
    ]
    categorical = [col for col in df.columns if col not in leakage and col not in numeric_candidates]
    return categorical, numeric_candidates


def _feature_label(name: str) -> str:
    if "__" not in name:
        return name
    _, rest = name.split("__", 1)
    return rest


def model_black_veteran_predictors(
    df: pd.DataFrame,
    outcome_definition: str = "blank_as_not_employed",
    min_cell_size: int = DEFAULT_MIN_CELL_SIZE,
) -> dict[str, Any]:
    if outcome_definition not in OUTCOME_DEFINITIONS:
        raise ValueError(f"Unsupported outcome definition: {outcome_definition}")

    black = _clean_text_series(df.get("black", pd.Series("", index=df.index)))
    subset = df.loc[black.eq("1")].copy()
    outcome_col = f"_{outcome_definition}"
    subset = subset.loc[subset[outcome_col].notna()].copy()
    y = subset[outcome_col].astype(int)

    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if len(subset) < min_cell_size or positives < min_cell_size or negatives < min_cell_size:
        return {
            "analysis": "black_veteran_employment_predictors",
            "outcome_definition": outcome_definition,
            "status": "suppressed",
            "reason": "Not enough positive and negative cases under the selected outcome definition.",
            "n": len(subset),
            "positive_cases": positives if positives >= min_cell_size else None,
            "negative_cases": negatives if negatives >= min_cell_size else None,
        }

    categorical_cols, numeric_cols = _feature_columns(subset)
    X = subset[categorical_cols + numeric_cols].copy()
    for col in categorical_cols:
        X[col] = _clean_text_series(X[col]).replace("", "missing")
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
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

    stratify = y if positives >= 2 and negatives >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = round(float(roc_auc_score(y_test, proba)), 4) if len(set(y_test)) > 1 else None

    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    ranked = sorted(zip(feature_names, coefs), key=lambda item: abs(float(item[1])), reverse=True)

    positive_features = []
    negative_features = []
    for name, coef in ranked:
        item = {"feature": _feature_label(str(name)), "coefficient": round(float(coef), 4)}
        if coef > 0 and len(positive_features) < 15:
            positive_features.append(item)
        elif coef < 0 and len(negative_features) < 15:
            negative_features.append(item)
        if len(positive_features) >= 15 and len(negative_features) >= 15:
            break

    return {
        "analysis": "black_veteran_employment_predictors",
        "outcome_definition": outcome_definition,
        "outcome_description": OUTCOME_DEFINITIONS[outcome_definition],
        "status": "completed",
        "n": int(len(subset)),
        "positive_cases": positives,
        "negative_cases": negatives,
        "positive_rate": _rate(positives, len(subset)),
        "model": "L2 logistic regression with one-hot categorical features and balanced class weights",
        "auc_holdout": auc,
        "interpretation_warning": (
            "Predictors are associations in administrative data, not causal recommendations. "
            "Do not translate these directly into advice like 'apply for X to get employed fastest' "
            "without domain review and causal validation."
        ),
        "features_associated_with_higher_predicted_employment": positive_features,
        "features_associated_with_lower_predicted_employment": negative_features,
    }


def build_analysis_summary(
    sqlite_file: Path,
    min_cell_size: int = DEFAULT_MIN_CELL_SIZE,
) -> dict[str, Any]:
    conn = connect_existing(sqlite_file)
    try:
        columns = _available_columns(conn)
        df = _read_private_frame(conn, columns)
    finally:
        conn.close()

    df = _add_outcomes(df)
    return {
        "source": str(sqlite_file),
        "table": TABLE,
        "privacy_note": "Outputs are aggregate/model summaries only. No individual rows are returned.",
        "min_cell_size": min_cell_size,
        "veteran_records_loaded": int(len(df)),
        "outcome_definitions": OUTCOME_DEFINITIONS,
        "black_disparity": analyze_black_disparity(df, min_cell_size=min_cell_size),
        "black_predictors_blank_as_not_employed": model_black_veteran_predictors(
            df,
            outcome_definition="blank_as_not_employed",
            min_cell_size=min_cell_size,
        ),
        "black_predictors_known_exit_employment": model_black_veteran_predictors(
            df,
            outcome_definition="known_exit_employment",
            min_cell_size=min_cell_size,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run privacy-safe RSA-911 disparity and predictor analyses.")
    parser.add_argument("--sqlite", default=str(sqlite_path()))
    parser.add_argument("--out-dir", default=str(work_dir()))
    parser.add_argument("--min-cell-size", type=int, default=DEFAULT_MIN_CELL_SIZE)
    args = parser.parse_args()

    sqlite_file = Path(args.sqlite).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_analysis_summary(sqlite_file, min_cell_size=args.min_cell_size)
    out_path = out_dir / "analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote privacy-safe analysis summary to {out_path}")
    print(f"Veteran records analyzed: {summary['veteran_records_loaded']}")
    print("No individual rows were printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
