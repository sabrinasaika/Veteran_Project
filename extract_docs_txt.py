#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader
from docx import Document
import openpyxl

ROOT = Path(".").resolve()
RAW = ROOT / "_raw"
OUT = ROOT / "docs_text"
OUT.mkdir(exist_ok=True)

SKIP_DIR_NAMES = {"_dupes"}  # ignore moved duplicates

def sha12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def pdf_to_text(p: Path) -> str:
    reader = PdfReader(str(p))
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t.strip())
    return "\n\n".join(parts).strip()

def docx_to_text(p: Path) -> str:
    doc = Document(str(p))
    parts = [para.text.strip() for para in doc.paragraphs if para.text and para.text.strip()]
    return "\n".join(parts).strip()

def xlsx_to_text(p: Path, max_rows: int = 400) -> str:
    wb = openpyxl.load_workbook(str(p), data_only=True, read_only=True)
    out_lines = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        out_lines.append(f"=== SHEET: {sheet} ===")
        count = 0
        for row in ws.iter_rows(values_only=True):
            count += 1
            if count > max_rows:
                out_lines.append("...(truncated)")
                break
            vals = ["" if v is None else str(v) for v in row]
            if any(v.strip() for v in vals):
                out_lines.append("\t".join(vals))
        out_lines.append("")
    return "\n".join(out_lines).strip()

def main():
    files = []
    for p in RAW.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        files.append(p)

    made = 0
    skipped = 0
    too_short = 0

    for p in files:
        ext = p.suffix.lower()
        doc_id = sha12(str(p.relative_to(ROOT)))

        try:
            if ext == ".pdf":
                text = pdf_to_text(p)
            elif ext == ".docx":
                text = docx_to_text(p)
            elif ext == ".xlsx":
                text = xlsx_to_text(p)
            elif ext in [".csv", ".txt"]:
                text = p.read_text(encoding="utf-8", errors="ignore")
            else:
                skipped += 1
                continue

            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) < 250:
                too_short += 1
                continue

            header = f"FILENAME: {p.name}\nPATH: {p.relative_to(ROOT)}\n\n"
            (OUT / f"{doc_id}.txt").write_text(header + text, encoding="utf-8", errors="ignore")
            made += 1

        except Exception:
            skipped += 1

    print("Extracted txt files:", made)
    print("Too short (ignored):", too_short)
    print("Skipped (unsupported/errors):", skipped)
    print("Output folder:", OUT)

if __name__ == "__main__":
    main()
