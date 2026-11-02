#!/usr/bin/env python3
"""
Crawl + Process (no categorization)

- Reads seed URLs from sources.txt
- Crawls within each seed domain (polite, depth-limited)
- Downloads documents: pdf, doc/docx, xls/xlsx, csv, ppt/pptx, txt, zip (kept raw)
- Saves raw files into: _raw/<domain>/
- Saves extracted text into: _processed/<doc_id>.txt
- Updates manifest.csv automatically (no manual work)

Run:
  python crawl_process.py --max-pages 250 --max-depth 2 --delay 1.0

Notes:
- This is for research. It rate-limits and avoids infinite crawling.
- Focus is building a RAG corpus: raw + normalized text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from slugify import slugify
from tqdm import tqdm

# Processors
from pypdf import PdfReader
from docx import Document
import openpyxl

try:
    import trafilatura
except Exception:
    trafilatura = None

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


DOC_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx", ".txt", ".rtf", ".zip"
}

# If a page has less than this many characters after cleaning, we skip saving it.
MIN_HTML_TEXT_CHARS = 900

# Skip obvious junk pages
JUNK_URL_PATTERNS = re.compile(
    r"(login|sign[-_]?in|signin|account|password|reset|certificate|quiz|form|contact|search)",
    re.IGNORECASE,
)

# Only keep HTML pages that likely contain research/policy content
GOOD_TEXT_KEYWORDS = re.compile(
    r"(vocational rehabilitation|vr\b|rsa\b|rsa-?911|wioa|pirl|directive|guidance|policy|"
    r"evidence|research|study|evaluation|report|outcome|employment)",
    re.IGNORECASE,
)


@dataclass
class ManifestRow:
    doc_id: str
    source_url: str
    final_url: str
    domain: str
    title: str
    fetched_at: str
    content_type: str
    raw_path: str
    processed_text_path: str
    status: str
    notes: str


def normalize_url(url: str) -> str:
    url, _frag = urldefrag(url)
    return url.strip()


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def get_ext(url: str) -> str:
    p = urlparse(url).path
    return (Path(p).suffix or "").lower()


def looks_like_doc(url: str, content_type: str = "") -> bool:
    ext = get_ext(url)
    if ext in DOC_EXTENSIONS:
        return True
    ct = (content_type or "").lower()
    if any(x in ct for x in [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument",
        "application/vnd.ms-excel",
        "text/csv",
        "application/vnd.ms-powerpoint",
        "text/plain",
        "application/zip"
    ]):
        return True
    return False


def sha12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def safe_base_name(title: str, url: str) -> str:
    base = slugify(title)[:80] if title else ""
    if not base:
        last = Path(urlparse(url).path).name or "document"
        base = slugify(last)[:80] or "document"
    return base


def read_sources(path: Path) -> List[str]:
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def ensure_manifest(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "doc_id", "source_url", "final_url", "domain", "title", "fetched_at",
            "content_type", "raw_path", "processed_text_path", "status", "notes"
        ])


def load_manifest_urls(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fu = (row.get("final_url") or "").strip()
            if fu:
                seen.add(fu)
    return seen


def append_manifest(path: Path, row: ManifestRow) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            row.doc_id, row.source_url, row.final_url, row.domain, row.title, row.fetched_at,
            row.content_type, row.raw_path, row.processed_text_path, row.status, row.notes
        ])


def fetch(session: requests.Session, url: str, timeout: int) -> Optional[requests.Response]:
    try:
        resp = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        print(f"[fetch] status={resp.status_code} ct={ct} url={url} -> {resp.url}")
        if resp.status_code >= 400:
            snippet = (resp.text or "")[:200].replace("\n", " ")
            print(f"[fetch] body-snippet: {snippet}")
            return None
        return resp
    except requests.RequestException as e:
        print(f"[fetch] EXCEPTION url={url} err={repr(e)}")
        return None


def parse_html_title_and_links(base_url: str, html: str) -> Tuple[str, List[str]]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "").strip()

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "javascript:")):
            continue
        abs_url = normalize_url(urljoin(base_url, href))
        links.append(abs_url)

    # de-dup
    out, seen = [], set()
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return title, out


def clean_html_to_text(url: str, html: str) -> str:
    # 1) Try trafilatura first
    if trafilatura is not None:
        extracted = trafilatura.extract(html, url=url, include_tables=True) or ""
        extracted = extracted.strip()
        # If it's too short, likely a listing page / JS-heavy page → fall back
        if len(extracted) >= 1200:
            return extracted

    # 2) Fallback: soup text
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text



def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t.strip())
    return "\n\n".join(parts).strip()


def extract_docx_text(docx_path: Path) -> str:
    doc = Document(str(docx_path))
    parts = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    return "\n".join(parts).strip()


def extract_xlsx_text(xlsx_path: Path, max_rows: int = 200) -> str:
    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True, read_only=True)
    out_lines = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        out_lines.append(f"=== SHEET: {sheet_name} ===")
        row_count = 0
        for row in ws.iter_rows(values_only=True):
            row_count += 1
            if row_count > max_rows:
                out_lines.append("...(truncated)")
                break
            # Convert row to TSV-ish line
            vals = ["" if v is None else str(v) for v in row]
            # Drop fully empty rows
            if any(v.strip() for v in vals):
                out_lines.append("\t".join(vals))
        out_lines.append("")
    return "\n".join(out_lines).strip()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", errors="ignore")


def allowed_same_domain(target_url: str, seed_domain: str, include_subdomains: bool) -> bool:
    d = get_domain(target_url)
    if d == seed_domain:
        return True
    if include_subdomains and d.endswith("." + seed_domain):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Employment folder (default=.)")
    ap.add_argument("--max-pages", type=int, default=250)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--include-subdomains", action="store_true", default=False)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sources_path = root / "sources.txt"
    raw_root = root / "_raw"
    processed_root = root / "_processed"
    manifest_path = root / "manifest.csv"

    raw_root.mkdir(exist_ok=True)
    processed_root.mkdir(exist_ok=True)
    ensure_manifest(manifest_path)

    seeds = read_sources(sources_path)
    if not seeds:
        raise SystemExit("sources.txt is empty.")

    seen_final_urls = load_manifest_urls(manifest_path)

    session = requests.Session()

    total_html_pages = 0
    total_docs_downloaded = 0
    total_text_outputs = 0

    for seed in seeds:
        seed = normalize_url(seed)
        seed_domain = get_domain(seed)
        q: List[Tuple[str, int, str]] = [(seed, 0, seed)]  # (url, depth, source_url)

        visited: Set[str] = set()

        pbar = tqdm(total=args.max_pages, desc=f"Crawling {seed_domain}", unit="page")

        while q and total_html_pages < args.max_pages:
            url, depth, source_url = q.pop(0)
            url = normalize_url(url)

            if url in visited:
                continue
            visited.add(url)

            if not allowed_same_domain(url, seed_domain, args.include_subdomains):
                continue

            resp = fetch(session, url, timeout=args.timeout)
            time.sleep(args.delay)
            if not resp:
                continue

            final_url = normalize_url(resp.url)
            if final_url in seen_final_urls:
                continue

            content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()

            # If it's a document, download raw and process if possible
            if looks_like_doc(final_url, content_type=content_type):
                # decide extension
                ext = get_ext(final_url)
                if not ext:
                    # guess from content-type
                    if "pdf" in content_type:
                        ext = ".pdf"
                    elif "word" in content_type:
                        ext = ".docx"
                    elif "excel" in content_type or "spreadsheet" in content_type:
                        ext = ".xlsx"
                    elif "csv" in content_type:
                        ext = ".csv"
                    else:
                        ext = ".bin"

                domain_dir = raw_root / seed_domain
                domain_dir.mkdir(parents=True, exist_ok=True)

                # best effort title from filename
                title = Path(urlparse(final_url).path).name or final_url
                base = safe_base_name(title, final_url)
                doc_id = sha12(final_url)
                raw_path = domain_dir / f"{base}-{doc_id}{ext}"

                try:
                    raw_path.write_bytes(resp.content)
                except Exception as e:
                    # record failure
                    append_manifest(manifest_path, ManifestRow(
                        doc_id=doc_id, source_url=source_url, final_url=final_url, domain=seed_domain,
                        title=title, fetched_at=datetime.utcnow().isoformat(),
                        content_type=content_type or f"ext:{ext}", raw_path=str(raw_path.relative_to(root)),
                        processed_text_path="", status="raw_write_failed", notes=str(e)[:200]
                    ))
                    seen_final_urls.add(final_url)
                    continue

                # Process to text (only for supported formats)
                processed_path = processed_root / f"{doc_id}.txt"
                status = "downloaded"
                notes = ""
                extracted_text = ""

                try:
                    if ext == ".pdf":
                        extracted_text = extract_pdf_text(raw_path)
                    elif ext == ".docx":
                        extracted_text = extract_docx_text(raw_path)
                    elif ext == ".xlsx":
                        extracted_text = extract_xlsx_text(raw_path)
                    elif ext in [".txt", ".csv"]:
                        extracted_text = raw_path.read_text(encoding="utf-8", errors="ignore")
                    else:
                        # leave unprocessed (pptx/doc/xls/zip etc.)
                        extracted_text = ""
                except Exception as e:
                    status = "process_failed"
                    notes = f"process_error: {str(e)[:200]}"

                if extracted_text and len(extracted_text) >= 300:
                    write_text(processed_path, extracted_text)
                    total_text_outputs += 1
                    processed_rel = str(processed_path.relative_to(root))
                else:
                    processed_rel = ""

                append_manifest(manifest_path, ManifestRow(
                    doc_id=doc_id, source_url=source_url, final_url=final_url, domain=seed_domain,
                    title=title, fetched_at=datetime.utcnow().isoformat(),
                    content_type=content_type or f"ext:{ext}", raw_path=str(raw_path.relative_to(root)),
                    processed_text_path=processed_rel, status=status, notes=notes
                ))
                seen_final_urls.add(final_url)
                total_docs_downloaded += 1
                continue

            # Otherwise HTML page: save processed text if contentful + keep crawling links
            if "text/html" in content_type or content_type == "" or "text/" in content_type:
                html = resp.text or ""
                title, links = parse_html_title_and_links(final_url, html)

                # Filter out junk URLs early
                if JUNK_URL_PATTERNS.search(final_url):
                    continue

                text = clean_html_to_text(final_url, html)
                doc_id = sha12(final_url)

                processed_path = processed_root / f"{doc_id}.txt"
                processed_rel = ""

                if text and len(text) >= MIN_HTML_TEXT_CHARS and GOOD_TEXT_KEYWORDS.search(text):
                    write_text(processed_path, f"TITLE: {title}\nURL: {final_url}\n\n{text}")
                    processed_rel = str(processed_path.relative_to(root))
                    total_text_outputs += 1

                # record in manifest (even if we didn't save text, useful for trace)
                append_manifest(manifest_path, ManifestRow(
                    doc_id=doc_id, source_url=source_url, final_url=final_url, domain=seed_domain,
                    title=title or final_url, fetched_at=datetime.utcnow().isoformat(),
                    content_type=content_type or "text/html",
                    raw_path="", processed_text_path=processed_rel,
                    status="html_crawled", notes=""
                ))
                seen_final_urls.add(final_url)

                total_html_pages += 1
                pbar.update(1)

            # First: immediately download any document links on this page
            for u in links:
                if not allowed_same_domain(u, seed_domain, args.include_subdomains):
                    continue
                # If it looks like a doc link, fetch it right away by adding to the front of the queue
                if get_ext(u) in DOC_EXTENSIONS:
                    q.insert(0, (u, depth + 1, source_url))

            # Then: enqueue normal HTML links
            if depth < args.max_depth:
                for u in links:
                    if allowed_same_domain(u, seed_domain, args.include_subdomains) and u not in visited:
                        q.append((u, depth + 1, source_url))

            else:
                # unknown content type - skip but record
                doc_id = sha12(final_url)
                append_manifest(manifest_path, ManifestRow(
                    doc_id=doc_id, source_url=source_url, final_url=final_url, domain=seed_domain,
                    title=final_url, fetched_at=datetime.utcnow().isoformat(),
                    content_type=content_type or "unknown",
                    raw_path="", processed_text_path="",
                    status="skipped_unknown_type", notes=""
                ))
                seen_final_urls.add(final_url)

        pbar.close()

    print("\n✅ Done.")
    print(f"Raw docs downloaded: {total_docs_downloaded}")
    print(f"Text files created in _processed/: {total_text_outputs}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
