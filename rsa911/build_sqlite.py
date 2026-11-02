#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd

from rsa911.config import EXPECTED_FILES, discover_csv_files, raw_dir, sqlite_path


# Disability type and cause codes are two digits in the codebook ("04"), but Excel exports drop
# the leading zero on some rows ("4"), which splits one category in two and breaks "= '04'" filters.
TWO_DIGIT_CODE_COLUMNS = ("primdisability", "seconddisability", "primdisabilitycause", "seconddisabilitycause")


def _pad_code(value: object) -> object:
    if not isinstance(value, str):
        return value
    parts = value.strip().split(";")
    return ";".join(part.zfill(2) if part.isdigit() and len(part) == 1 else part for part in parts)


# PY21 and PY22 exports carry a ProgramYear column ("PY2021"); the PY23 export does not, so its rows
# would have an empty program year. Fill only empty values, from the year in the file name.
def _program_year_from_name(name: str) -> str:
    match = re.search(r"PY(20)?(\d{2})", name, flags=re.IGNORECASE)
    return f"PY20{match.group(2)}" if match else ""


def _clean_column(name: object, index: int) -> str:
    text = str(name or "").strip()
    if not text or text.lower().startswith("unnamed:"):
        text = f"element_{index + 1:03d}"
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    if not text:
        text = f"element_{index + 1:03d}"
    if text[0].isdigit():
        text = f"c_{text}"
    return text


def _dedupe_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for col in columns:
        count = seen.get(col, 0)
        seen[col] = count + 1
        out.append(col if count == 0 else f"{col}_{count + 1}")
    return out


def _normalized_columns_for_file(path: Path) -> list[str]:
    header = pd.read_csv(path, dtype=str, nrows=0, encoding="utf-8-sig").columns
    return _dedupe_columns([_clean_column(col, i) for i, col in enumerate(header)])


def _union_columns(paths: list[Path]) -> list[str]:
    cols = ["source_file"]
    seen = set(cols)
    for path in paths:
        if not path.exists():
            continue
        for col in _normalized_columns_for_file(path):
            if col not in seen:
                cols.append(col)
                seen.add(col)
    return cols


def import_file(
    conn: sqlite3.Connection,
    path: Path,
    table: str,
    chunksize: int,
    all_columns: list[str],
) -> int:
    total = 0
    padded = 0
    filled_year = 0
    for chunk_idx, df in enumerate(
        pd.read_csv(path, dtype=str, chunksize=chunksize, encoding="utf-8-sig", low_memory=False)
    ):
        df.columns = _dedupe_columns([_clean_column(col, i) for i, col in enumerate(df.columns)])
        for col in TWO_DIGIT_CODE_COLUMNS:
            if col in df.columns:
                fixed = df[col].map(_pad_code)
                padded += int((fixed.fillna("") != df[col].fillna("")).sum())
                df[col] = fixed
        df.insert(0, "source_file", path.name)
        for col in all_columns:
            if col not in df.columns:
                df[col] = None
        year = _program_year_from_name(path.name)
        if year and "programyear" in all_columns:
            empty = df["programyear"].fillna("").str.strip() == ""
            filled_year += int(empty.sum())
            df.loc[empty, "programyear"] = year
        df = df[all_columns]
        df.to_sql(table, conn, if_exists="append", index=False)
        total += len(df)
        print(f"{path.name}: imported {total} rows", flush=True)
    print(f"{path.name}: restored the leading zero on {padded} disability code values", flush=True)
    if filled_year:
        print(f"{path.name}: filled empty program year with {_program_year_from_name(path.name)} on {filled_year} rows", flush=True)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a local private SQLite database from RSA-911 CSV files. Prints counts only."
    )
    parser.add_argument("--raw-dir", default=str(raw_dir()))
    parser.add_argument("--sqlite", default=str(sqlite_path()))
    parser.add_argument("--table", default="rsa911_exiters")
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit filenames. By default, all CSV files in raw-dir are imported.",
    )
    parser.add_argument("--replace", action="store_true", help="Drop and rebuild the destination table.")
    args = parser.parse_args()

    src = Path(args.raw_dir).expanduser()
    db_path = Path(args.sqlite).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        if args.replace:
            conn.execute(f'DROP TABLE IF EXISTS "{args.table}"')
            conn.commit()

        totals = {}
        paths = discover_csv_files(src, args.files)
        if not paths:
            print(f"No CSV files found in {src}")
        existing_paths = [path for path in paths if path.exists()]
        all_columns = _union_columns(existing_paths)
        print(f"Unified schema columns: {len(all_columns)}")
        for path in paths:
            if not path.exists():
                print(f"Skipping missing file: {path}")
                continue
            totals[path.name] = import_file(conn, path, args.table, args.chunksize, all_columns)

        found_names = set(totals)
        expected_missing = [name for name in EXPECTED_FILES if name not in found_names]
        if expected_missing:
            print("Expected names not found:", ", ".join(expected_missing))

        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{args.table}_source_file ON "{args.table}" (source_file)')
        conn.commit()
    finally:
        conn.close()

    print(f"Wrote private SQLite database to {db_path}")
    print("Imported row counts:")
    for name, count in totals.items():
        print(f"- {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
