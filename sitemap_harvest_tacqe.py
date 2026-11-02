#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, re, time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}
DOC_RE = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|csv|zip)(\?|#|$)", re.I)
UPLOAD_HINT = re.compile(r"wp-content/uploads", re.I)

ROOT = Path(".").resolve()
RAW_DIR = ROOT / "_raw" / "tacqe.com"
PROCESSED_DIR = ROOT / "_processed"
MANIFEST = ROOT / "manifest.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def sha12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def ensure_manifest():
    if MANIFEST.exists():
        return
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id","source","url","content_type","raw_path","processed_text_path","status","notes"])

def append_manifest(row):
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def parse_sitemap_urls(sitemap_url: str) -> list[str]:
    r = requests.get(sitemap_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text.strip() for loc in root.findall(".//sm:url/sm:loc", ns)]

def parse_sitemap_index(index_url: str) -> list[str]:
    r = requests.get(index_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text.strip() for loc in root.findall(".//sm:sitemap/sm:loc", ns)]

def page_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script","style","nav","footer","header","noscript","svg"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def extract_doc_links(page_url: str, html: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        full = urljoin(page_url, href)
        if DOC_RE.search(full) or UPLOAD_HINT.search(full):
            links.add(full)
    return links

def download_file(file_url: str) -> tuple[str, str]:
    # returns (content_type, relative_path)
    r = requests.get(file_url, headers=HEADERS, timeout=45, allow_redirects=True)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    ext = Path(urlparse(r.url).path).suffix or ""
    doc_id = sha12(r.url)
    name = Path(urlparse(r.url).path).name or f"file-{doc_id}{ext}"
    out = RAW_DIR / f"{doc_id}-{name}"
    out.write_bytes(r.content)
    return ct, str(out.relative_to(ROOT))

def main():
    ensure_manifest()

    sitemap_index = "https://tacqe.com/sitemap_index.xml"
    sitemaps = parse_sitemap_index(sitemap_index)

    # choose only the useful ones first
    wanted = []
    for s in sitemaps:
        if any(k in s for k in ["trainings-sitemap.xml","eri-presenters-sitemap.xml","post-sitemap.xml","page-sitemap.xml"]):
            wanted.append(s)

    print("Using sitemaps:")
    for s in wanted:
        print(" -", s)

    # gather page urls
    page_urls: list[str] = []
    for sm in wanted:
        page_urls.extend(parse_sitemap_urls(sm))

    # de-dup preserve order
    seen = set()
    page_urls = [u for u in page_urls if not (u in seen or seen.add(u))]
    print("Total pages from sitemap:", len(page_urls))

    delay = 0.8
    max_pages = 800  # adjust as needed

    for i, page_url in enumerate(page_urls[:max_pages], 1):
        try:
            r = requests.get(page_url, headers=HEADERS, timeout=35, allow_redirects=True)
            ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            print(f"[{i}/{min(len(page_urls), max_pages)}] page {r.status_code} {page_url}")
            if r.status_code >= 400 or "text/html" not in ct:
                append_manifest([sha12(page_url), "tacqe_sitemap", page_url, ct, "", "", "page_skip", "non-html or error"])
                time.sleep(delay)
                continue

            text = page_to_text(r.text)
            page_doc_id = sha12(page_url)
            processed_path = PROCESSED_DIR / f"{page_doc_id}.txt"
            processed_rel = ""
            if len(text) > 300:
                processed_path.write_text(f"URL: {page_url}\n\n{text}", encoding="utf-8", errors="ignore")
                processed_rel = str(processed_path.relative_to(ROOT))
            append_manifest([page_doc_id, "tacqe_sitemap", page_url, "text/html", "", processed_rel, "page_saved", ""])

            # download linked docs
            doc_links = extract_doc_links(page_url, r.text)
            for dl in sorted(doc_links):
                try:
                    fct, raw_rel = download_file(dl)
                    append_manifest([sha12(dl), "tacqe_page_link", dl, fct, raw_rel, "", "doc_downloaded", f"from:{page_url}"])
                except Exception as e:
                    append_manifest([sha12(dl), "tacqe_page_link", dl, "", "", "", "doc_failed", f"{type(e).__name__}: {e}"])

            time.sleep(delay)

        except Exception as e:
            append_manifest([sha12(page_url), "tacqe_sitemap", page_url, "", "", "", "page_failed", f"{type(e).__name__}: {e}"])
            time.sleep(delay)

    print("Done. Raw docs:", sum(1 for _ in RAW_DIR.glob("*")))
    print("Processed txt:", sum(1 for _ in PROCESSED_DIR.glob("*.txt")))
    print("Manifest:", MANIFEST)

if __name__ == "__main__":
    main()
