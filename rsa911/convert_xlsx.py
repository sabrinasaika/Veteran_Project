from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import openpyxl

from .config import raw_dir


# Converts an RSA-911 Excel export to CSV for build_sqlite, streaming rows so large files
# do not need to fit in memory. Prints only counts, never cell values. The source .xlsx is
# shredded only after the CSV is complete and passes basic checks.

EXCEL_MAX_ROWS = 1_048_576


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def convert(src: Path) -> tuple[int, int]:
    out = src.with_suffix(".csv")
    tmp = out.with_suffix(".csv.partial")
    workbook = openpyxl.load_workbook(src, read_only=True, data_only=True)
    sheets = len(workbook.sheetnames)
    rows = 0
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for row in workbook.worksheets[0].iter_rows(values_only=True):
                writer.writerow([_cell(value) for value in row])
                rows += 1
                if rows % 100_000 == 0:
                    print(f"  {rows:,} rows converted...", flush=True)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        workbook.close()
    tmp.chmod(0o600)
    tmp.rename(out)
    return sheets, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a private RSA-911 .xlsx export to .csv (counts only are printed).")
    parser.add_argument("name", help="File name inside the raw folder, e.g. PY23exiters.xlsx")
    parser.add_argument("--raw-dir", default=str(raw_dir()))
    parser.add_argument("--keep-xlsx", action="store_true", help="Do not shred the .xlsx after a successful conversion.")
    args = parser.parse_args()

    src = Path(args.raw_dir).expanduser() / args.name
    if not src.exists():
        raise SystemExit(f"{src} not found. Copy it to the server first.")
    print(f"Converting {src.name} (this can take a while for large files)...", flush=True)
    sheets, rows = convert(src)
    print(f"Sheets in file: {sheets}")
    print(f"Rows written (including header): {rows:,}")

    problems = []
    if sheets > 1:
        problems.append("the file has more than one sheet; only the first was converted")
    if rows >= EXCEL_MAX_ROWS:
        problems.append("the sheet hit Excel's row limit, so the data was probably cut off when saved as Excel")
    if rows < 2:
        problems.append("no data rows were found")
    if problems:
        for problem in problems:
            print(f"WARNING: {problem}.")
        print(f"Kept {src.name} so nothing is lost. Ask before deleting it.")
        return 1
    if args.keep_xlsx:
        print(f"Kept {src.name} (--keep-xlsx).")
    else:
        subprocess.run(["shred", "-u", str(src)], check=True)
        print(f"Converted successfully; {src.name} securely deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
