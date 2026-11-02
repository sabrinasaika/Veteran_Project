#!/usr/bin/env python3
"""
Download JVR PDFs relevant to employment pathways for veterans with disabilities.

The script:
1. Scans Journal of Vocational Rehabilitation records from OpenAlex.
2. Filters title/abstract text using veteran + employment-pathway keyword groups.
3. Looks up each DOI in Crossref to find PDF URLs.
4. Attempts to download PDFs into `_raw/jvr/`.
5. Writes a metadata sidecar JSON for provenance.
6. Appends manifest rows in the same style as the rest of the corpus.

Notes:
- Many publisher PDF URLs may still return 403 without institutional access.
- Those candidates are still recorded in `manifest.csv` with enough provenance
  to fetch them manually later.

Run:
  python3 crawl_jvr_pdfs.py --max-matches 25
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, urldefrag, urlparse

import requests
from tqdm import tqdm

OPENALEX_JVR_URL = (
    "https://api.openalex.org/works"
    "?filter=primary_location.source.issn:1878-6316"
    "&sort=publication_date:desc"
    "&per-page=200"
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

PDF_HEADERS = {
    **HEADERS,
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
}

VETERAN_TERMS = [
    "veteran",
    "veterans",
    "military",
    "service member",
    "servicemember",
    "service-connected",
    "deployment",
    "va ",
    "vha",
    "vre",
    "vr&e",
    "chapter 31",
]

PATHWAY_TERMS = [
    "employment",
    "employed",
    "unemployment",
    "job",
    "jobs",
    "career",
    "careers",
    "work",
    "workforce",
    "return to work",
    "supported employment",
    "job placement",
    "job retention",
    "employment outcome",
    "employment outcomes",
    "employment pathway",
    "employment pathways",
    "vocational rehabilitation",
]

DISABILITY_TERMS = [
    "disability",
    "disabilities",
    "disabled",
    "traumatic brain injury",
    "tbi",
    "ptsd",
    "spinal cord",
    "vision loss",
    "hearing loss",
    "amputation",
    "multiple sclerosis",
    "chronic pain",
]


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


def sha12(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()[:12]


def ensure_manifest(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "doc_id",
                "source_url",
                "final_url",
                "domain",
                "title",
                "fetched_at",
                "content_type",
                "raw_path",
                "processed_text_path",
                "status",
                "notes",
            ]
        )


def append_manifest(path: Path, row: ManifestRow) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                row.doc_id,
                row.source_url,
                row.final_url,
                row.domain,
                row.title,
                row.fetched_at,
                row.content_type,
                row.raw_path,
                row.processed_text_path,
                row.status,
                row.notes,
            ]
        )


def load_downloaded_doc_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    doc_ids: Set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            if status != "jvr_pdf_downloaded":
                continue
            doc_id = (row.get("doc_id") or "").strip()
            if doc_id:
                doc_ids.add(doc_id)
    return doc_ids


def _fetch_json(session: requests.Session, url: str, timeout: int) -> Optional[dict]:
    try:
        resp = session.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[warn] json fetch failed url={url} err={type(exc).__name__}: {exc}")
        return None


def _openalex_abstract(item: dict) -> str:
    inverted = item.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""
    max_pos = -1
    for positions in inverted.values():
        if isinstance(positions, list) and positions:
            max_pos = max(max_pos, max(int(pos) for pos in positions))
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            pos = int(pos)
            if 0 <= pos < len(words):
                words[pos] = word
    return " ".join(word for word in words if word).strip()


def _clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _matching_terms(text: str, terms: Sequence[str]) -> List[str]:
    lowered = f" {_clean_text(text).lower()} "
    hits: List[str] = []
    for term in terms:
        candidate = term.lower().strip()
        if not candidate:
            continue
        if candidate.isalpha() and len(candidate) >= 4:
            pattern = rf"\b{re.escape(candidate)}\b"
            if re.search(pattern, lowered):
                hits.append(term)
        elif candidate in lowered:
            hits.append(term)
    return hits


def _match_candidate(item: dict) -> Optional[Dict[str, List[str]]]:
    title = str(item.get("title") or "")
    abstract = _openalex_abstract(item)
    text = f"{title}\n{abstract}"
    veteran_hits = _matching_terms(text, VETERAN_TERMS)
    pathway_hits = _matching_terms(text, PATHWAY_TERMS)
    disability_hits = _matching_terms(text, DISABILITY_TERMS)

    if not veteran_hits:
        return None
    if not pathway_hits:
        return None

    return {
        "veteran_hits": veteran_hits,
        "pathway_hits": pathway_hits,
        "disability_hits": disability_hits,
    }


def _fetch_openalex_matches(
    session: requests.Session,
    *,
    timeout: int,
    delay: float,
    max_matches: int,
    max_scan: int,
) -> List[Tuple[dict, Dict[str, List[str]]]]:
    cursor = "*"
    scanned = 0
    matched: List[Tuple[dict, Dict[str, List[str]]]] = []

    while True:
        url = f"{OPENALEX_JVR_URL}&cursor={quote(cursor, safe='')}"
        payload = _fetch_json(session, url, timeout=timeout)
        time.sleep(delay)
        if not payload:
            break

        results = payload.get("results") or []
        if not isinstance(results, list) or not results:
            break

        for item in results:
            if not isinstance(item, dict):
                continue
            scanned += 1
            match = _match_candidate(item)
            if match:
                matched.append((item, match))
                if max_matches > 0 and len(matched) >= max_matches:
                    return matched
            if max_scan > 0 and scanned >= max_scan:
                return matched

        cursor = ((payload.get("meta") or {}).get("next_cursor") or "").strip()
        if not cursor:
            break

    return matched


def _crossref_lookup(session: requests.Session, doi: str, timeout: int, delay: float) -> dict:
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    payload = _fetch_json(session, url, timeout=timeout)
    time.sleep(delay)
    if not payload:
        return {}
    message = payload.get("message")
    return message if isinstance(message, dict) else {}


def _candidate_pdf_urls(doi: str, crossref: dict) -> List[str]:
    urls: List[str] = []

    for link in crossref.get("link") or []:
        if not isinstance(link, dict):
            continue
        url = str(link.get("URL") or "").strip()
        content_type = str(link.get("content-type") or "").lower()
        if not url:
            continue
        if "pdf" in content_type or "/doi/pdf/" in url.lower() or "download?id=" in url.lower():
            urls.append(url)

    primary_url = str((((crossref.get("resource") or {}).get("primary") or {}).get("URL")) or "").strip()
    if primary_url:
        urls.append(primary_url)
        if "/doi/" in primary_url:
            urls.append(primary_url.replace("/doi/", "/doi/pdf/"))

    doi_url = f"https://doi.org/{doi}"
    urls.append(doi_url)
    urls.append(f"https://journals.sagepub.com/doi/pdf/{doi}")
    urls.append(f"https://content.iospress.com/download?id={doi}")

    deduped: List[str] = []
    seen: Set[str] = set()
    for url in urls:
        normalized = normalize_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _looks_like_pdf(content_type: str, content: bytes) -> bool:
    content_type = (content_type or "").lower()
    if "application/pdf" in content_type:
        return True
    return content.startswith(b"%PDF")


def _download_pdf(session: requests.Session, url: str, timeout: int) -> Tuple[Optional[bytes], str, Optional[str], int]:
    try:
        resp = session.get(url, headers=PDF_HEADERS, timeout=timeout, allow_redirects=True)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", None, 0

    content_type = str(resp.headers.get("Content-Type") or "")
    final_url = normalize_url(resp.url)
    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}", final_url, resp.status_code
    if not _looks_like_pdf(content_type, resp.content[:8]):
        return None, f"not_pdf:{content_type or 'unknown'}", final_url, resp.status_code
    return resp.content, "", final_url, resp.status_code


def _slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = slug[:80].strip("-")
    return slug or fallback


def _write_metadata_sidecar(
    *,
    path: Path,
    openalex_item: dict,
    crossref_item: dict,
    match: Dict[str, List[str]],
    doi: str,
    doi_url: str,
    source_url: str,
    attempted_pdf_urls: List[str],
    selected_pdf_url: str,
    selected_final_url: str,
    download_status: str,
) -> None:
    payload = {
        "source_type": "jvr_pdf_candidate",
        "doi": doi,
        "doi_url": doi_url,
        "source_url": source_url,
        "attempted_pdf_urls": attempted_pdf_urls,
        "selected_pdf_url": selected_pdf_url,
        "selected_final_url": selected_final_url,
        "download_status": download_status,
        "match": match,
        "openalex": openalex_item,
        "crossref": crossref_item,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _process_candidate(
    session: requests.Session,
    *,
    item: dict,
    match: Dict[str, List[str]],
    raw_dir: Path,
    manifest_path: Path,
    downloaded_doc_ids: Set[str],
    timeout: int,
    delay: float,
) -> bool:
    doi_url = str(item.get("doi") or "").strip()
    if not doi_url:
        return False
    doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    doc_id = sha12(doi.lower())
    if doc_id in downloaded_doc_ids:
        return False

    crossref = _crossref_lookup(session, doi, timeout=timeout, delay=delay)
    title = str(item.get("title") or (crossref.get("title") or ["Untitled article"])[0]).strip()
    source_url = str((((crossref.get("resource") or {}).get("primary") or {}).get("URL")) or doi_url).strip() or doi_url
    pdf_urls = _candidate_pdf_urls(doi, crossref)

    raw_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title, fallback=f"jvr-{doc_id}")
    pdf_path = raw_dir / f"{slug}-{doc_id}.pdf"
    meta_path = raw_dir / f"{slug}-{doc_id}.json"
    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    selected_pdf_url = ""
    selected_final_url = ""
    failure_reasons: List[str] = []

    for pdf_url in pdf_urls:
        content, error, final_url, status_code = _download_pdf(session, pdf_url, timeout=timeout)
        time.sleep(delay)
        if content is None:
            reason = f"{pdf_url} -> {error}"
            failure_reasons.append(reason)
            continue

        selected_pdf_url = pdf_url
        selected_final_url = final_url or pdf_url
        pdf_path.write_bytes(content)
        _write_metadata_sidecar(
            path=meta_path,
            openalex_item=item,
            crossref_item=crossref,
            match=match,
            doi=doi,
            doi_url=doi_url,
            source_url=source_url,
            attempted_pdf_urls=pdf_urls,
            selected_pdf_url=selected_pdf_url,
            selected_final_url=selected_final_url,
            download_status="downloaded",
        )
        append_manifest(
            manifest_path,
            ManifestRow(
                doc_id=doc_id,
                source_url=source_url,
                final_url=selected_final_url,
                domain=_domain(selected_final_url),
                title=title,
                fetched_at=fetched_at,
                content_type="application/pdf",
                raw_path=str(pdf_path.relative_to(manifest_path.parent)),
                processed_text_path="",
                status="jvr_pdf_downloaded",
                notes=(
                    f"doi:{doi} | pdf_source:{selected_pdf_url} | "
                    f"match:veteran={','.join(match['veteran_hits'])};"
                    f"pathway={','.join(match['pathway_hits'])};"
                    f"disability={','.join(match['disability_hits'])} | "
                    f"metadata:{meta_path.name}"
                ),
            ),
        )
        downloaded_doc_ids.add(doc_id)
        return True

    _write_metadata_sidecar(
        path=meta_path,
        openalex_item=item,
        crossref_item=crossref,
        match=match,
        doi=doi,
        doi_url=doi_url,
        source_url=source_url,
        attempted_pdf_urls=pdf_urls,
        selected_pdf_url="",
        selected_final_url="",
        download_status="unavailable",
    )
    append_manifest(
        manifest_path,
        ManifestRow(
            doc_id=doc_id,
            source_url=source_url,
            final_url=doi_url,
            domain=_domain(doi_url),
            title=title,
            fetched_at=fetched_at,
            content_type="application/pdf",
            raw_path="",
            processed_text_path="",
            status="jvr_pdf_unavailable",
            notes=(
                f"doi:{doi} | attempted:{len(pdf_urls)} pdf urls | "
                f"match:veteran={','.join(match['veteran_hits'])};"
                f"pathway={','.join(match['pathway_hits'])};"
                f"disability={','.join(match['disability_hits'])} | "
                f"metadata:{meta_path.name} | "
                f"failures:{' || '.join(failure_reasons[:5])}"
            ),
        ),
    )
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Employment folder (default=.)")
    ap.add_argument("--max-matches", type=int, default=25, help="Maximum matching JVR articles to attempt")
    ap.add_argument("--max-scan", type=int, default=0, help="Maximum JVR records to scan from OpenAlex (0 = no limit)")
    ap.add_argument("--delay", type=float, default=0.8, help="Delay between HTTP requests in seconds")
    ap.add_argument("--timeout", type=int, default=25, help="HTTP timeout in seconds")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    raw_dir = root / "_raw" / "jvr"
    manifest_path = root / "manifest.csv"

    raw_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(manifest_path)
    downloaded_doc_ids = load_downloaded_doc_ids(manifest_path)

    session = requests.Session()
    matches = _fetch_openalex_matches(
        session,
        timeout=args.timeout,
        delay=args.delay,
        max_matches=args.max_matches,
        max_scan=args.max_scan,
    )

    print(f"Matching JVR candidate articles found: {len(matches)}")

    downloaded = 0
    unavailable = 0
    pbar = tqdm(matches, desc="Downloading JVR PDFs", unit="paper")
    for item, match in pbar:
        ok = _process_candidate(
            session,
            item=item,
            match=match,
            raw_dir=raw_dir,
            manifest_path=manifest_path,
            downloaded_doc_ids=downloaded_doc_ids,
            timeout=args.timeout,
            delay=args.delay,
        )
        if ok:
            downloaded += 1
        else:
            unavailable += 1
    pbar.close()

    print("\n✅ Done.")
    print(f"Matching JVR candidate articles found: {len(matches)}")
    print(f"PDFs downloaded: {downloaded}")
    print(f"Candidates without downloadable PDF: {unavailable}")
    print(f"_raw/jvr/: {raw_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
