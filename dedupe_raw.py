#!/usr/bin/env python3
from __future__ import annotations
import hashlib, csv
from pathlib import Path

ROOT = Path(".").resolve()
RAW = ROOT / "_raw"
DUPES_DIR = RAW / "_dupes"
DUPES_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    files = [p for p in RAW.rglob("*") if p.is_file() and "_dupes" not in p.parts]
    seen = {}
    moved = 0

    out_csv = ROOT / "dupes.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sha256","kept_path","dupe_path"])

        for p in files:
            try:
                h = sha256_file(p)
            except Exception:
                continue

            if h not in seen:
                seen[h] = p
            else:
                kept = seen[h]
                target = DUPES_DIR / p.name
                # avoid collisions
                if target.exists():
                    target = DUPES_DIR / f"{p.stem}-{p.stat().st_size}{p.suffix}"
                p.rename(target)
                w.writerow([h, str(kept.relative_to(ROOT)), str(target.relative_to(ROOT))])
                moved += 1

    print("Total files scanned:", len(files))
    print("Duplicates moved to:", DUPES_DIR)
    print("Duplicates moved count:", moved)
    print("CSV:", out_csv)

if __name__ == "__main__":
    main()
