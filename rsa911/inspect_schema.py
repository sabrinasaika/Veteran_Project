#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rsa911.config import EXPECTED_FILES, discover_csv_files, raw_dir, work_dir


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def inspect_file(path: Path, count_rows: bool) -> dict:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])

    return {
        "file": path.name,
        "exists": path.exists(),
        "columns": len(header),
        "headers": header,
        "row_count": _count_csv_rows(path) if count_rows else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect RSA-911 CSV schemas without printing any data rows."
    )
    parser.add_argument("--raw-dir", default=str(raw_dir()))
    parser.add_argument("--out-dir", default=str(work_dir()))
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit filenames. By default, all CSV files in raw-dir are inspected.",
    )
    parser.add_argument("--count-rows", action="store_true", help="Count rows; reads files but prints no values.")
    args = parser.parse_args()

    src = Path(args.raw_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_csv_files(src, args.files)
    results = []
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(path.name)
            results.append({"file": path.name, "exists": False})
            continue
        results.append(inspect_file(path, count_rows=args.count_rows))

    discovered_names = {path.name for path in paths if path.exists()}
    expected_missing = [name for name in EXPECTED_FILES if name not in discovered_names]

    payload = {
        "raw_dir": str(src),
        "files": results,
        "missing_files": missing,
        "expected_files_not_found": expected_missing,
        "privacy_note": "This report includes file names, headers, column counts, and optional row counts only. It contains no data rows.",
    }

    out_path = out_dir / "schema_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote private schema summary to {out_path}")
    print("Files found:", sum(1 for row in results if row.get("exists")))
    if not results:
        print(f"No CSV files found in {src}")
    for row in results:
        if row.get("exists"):
            count = row.get("row_count")
            suffix = f", rows={count}" if count is not None else ""
            print(f"- {row['file']}: columns={row['columns']}{suffix}")
        else:
            print(f"- {row['file']}: missing")
    if expected_missing:
        print("Expected names not found:", ", ".join(expected_missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
