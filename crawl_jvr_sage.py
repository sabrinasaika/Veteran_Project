#!/usr/bin/env python3
"""
Harvest Journal of Vocational Rehabilitation (JVR) articles into docs_text/.

Discovery comes from scholarly APIs (OpenAlex + Crossref) because the live Sage
site blocks direct HTTP scraping from `requests` with 403 responses. The script
still attempts a best-effort landing-page fetch for article body text, but the
core ingest path is article metadata + abstract, which is enough to add indexed
JVR documents to the local corpus.

Run:
  python crawl_jvr_sage.py --max-issues 20
  python crawl_jvr_sage.py --max-articles 200
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import quote, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

JVR_HOME_URL = "https://journals.sagepub.com/home/jvr"
OPENALEX_JVR_URL = "https://api.openalex.org/works?filter=primary_location.source.issn:1878-6316&sort=publication_date:desc&per-page=200"
SAGE_DOMAIN = "journals.sagepub.com"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

ARTICLE_RE = re.compile(r"/doi/(?:abs/|full/|pdf/|epub/|xml/)?(10\.3233/JVR-[A-Za-z0-9]+)", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTIBLANK_RE = re.compile(r"\n{3,}")


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


def load_manifest_urls(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    urls: Set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            final_url = (row.get("final_url") or "").strip()
            if final_url:
                urls.add(final_url)
    return urls


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTIBLANK_RE.sub("\n\n", text)
    return text.strip()


def _fetch(session: requests.Session, url: str, timeout: int) -> Optional[requests.Response]:
    try:
        resp = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code >= 400:
            print(f"[warn] status={resp.status_code} url={url}")
            return None
        return resp
    except requests.RequestException as exc:
        print(f"[warn] fetch failed url={url} err={type(exc).__name__}: {exc}")
        return None


def _is_sage_url(url: str) -> bool:
    return urlparse(url).netloc.lower() == SAGE_DOMAIN


def _canonicalize_article_url(url: str) -> Optional[str]:
    match = ARTICLE_RE.search(url)
    if not match:
        return None
    doi = match.group(1)
    return f"https://{SAGE_DOMAIN}/doi/{doi}"


def _openalex_abstract(item: dict) -> str:
    inverted = item.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""
    max_pos = -1
    for positions in inverted.values():
        if isinstance(positions, list) and positions:
            max_pos = max(max_pos, max(int(p) for p in positions))
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
    return _clean_text(" ".join(word for word in words if word))


def _strip_jats(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return _clean_text(text)


def _crossref_lookup(session: requests.Session, doi: str, timeout: int, delay: float) -> dict:
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    resp = _fetch(session, url, timeout=timeout)
    time.sleep(delay)
    if not resp:
        return {}
    try:
        payload = resp.json()
    except Exception:
        return {}
    message = payload.get("message")
    return message if isinstance(message, dict) else {}


def _issue_key(item: dict) -> Tuple[str, str]:
    biblio = item.get("biblio") or {}
    volume = str(biblio.get("volume") or "")
    issue = str(biblio.get("issue") or "")
    return volume, issue


def _fetch_openalex_articles(
    session: requests.Session,
    *,
    timeout: int,
    delay: float,
    max_issues: int,
    max_articles: int,
) -> List[dict]:
    cursor = "*"
    selected: List[dict] = []
    issue_order: List[Tuple[str, str]] = []

    while True:
        url = f"{OPENALEX_JVR_URL}&cursor={quote(cursor, safe='')}"
        resp = _fetch(session, url, timeout=timeout)
        time.sleep(delay)
        if not resp:
            break
        try:
            payload = resp.json()
        except Exception:
            break

        results = payload.get("results") or []
        if not isinstance(results, list) or not results:
            break

        for item in results:
            if not isinstance(item, dict):
                continue
            key = _issue_key(item)
            if max_issues > 0 and key != ("", "") and key not in issue_order and len(issue_order) >= max_issues:
                continue
            if key != ("", "") and key not in issue_order:
                issue_order.append(key)
            selected.append(item)
            if max_articles > 0 and len(selected) >= max_articles:
                return selected

        cursor = ((payload.get("meta") or {}).get("next_cursor") or "").strip()
        if not cursor:
            break
        if max_issues > 0 and issue_order and len(issue_order) >= max_issues and max_articles <= 0:
            break

    return selected


def _pick_meta(soup: BeautifulSoup, *names: str) -> str:
    lowered = {name.lower() for name in names}
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").strip().lower()
        if name in lowered:
            value = (meta.get("content") or "").strip()
            if value:
                return value
    return ""


def _pick_meta_list(soup: BeautifulSoup, name: str) -> List[str]:
    out: List[str] = []
    wanted = name.lower()
    for meta in soup.find_all("meta"):
        meta_name = (meta.get("name") or meta.get("property") or "").strip().lower()
        if meta_name != wanted:
            continue
        value = (meta.get("content") or "").strip()
        if value:
            out.append(value)
    return out


def _remove_page_chrome(node: Tag) -> None:
    for selector in [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "button",
        "noscript",
        "svg",
    ]:
        for tag in node.select(selector):
            tag.decompose()


def _extract_main_text(soup: BeautifulSoup) -> str:
    main = soup.find("main")
    if not isinstance(main, Tag):
        body = soup.find("body")
        if not isinstance(body, Tag):
            return ""
        main = body
    main = BeautifulSoup(str(main), "lxml")
    body = main.find("body")
    container = body if isinstance(body, Tag) else main
    _remove_page_chrome(container)
    text = container.get_text("\n", strip=True)
    return _clean_text(text)


def _extract_abstract(soup: BeautifulSoup, main_text: str) -> str:
    for meta_name in ("citation_abstract", "dc.description", "description"):
        value = _pick_meta(soup, meta_name)
        if value and len(value) >= 60:
            return _clean_text(value)

    patterns = [
        re.compile(
            r"\bAbstract\b\s*(.+?)(?=\b(?:Keywords?|References|Acknowledg(?:e)?ments?|Funding|Conflict of interest|1\s+Introduction|Introduction)\b)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\bAbstract\b\s*(.+?)(?=\b(?:View Options|Access options|PDF/EPUB|Metrics and citations)\b)",
            re.IGNORECASE | re.DOTALL,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(main_text)
        if match:
            candidate = _clean_text(match.group(1))
            if candidate:
                return candidate
    return ""


def _detect_access(main_text: str) -> str:
    lowered = main_text.lower()
    if "open access" in lowered or "free access" in lowered:
        return "open"
    if "restricted access" in lowered:
        return "restricted"
    return "unknown"


def _slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = slug[:80].strip("-")
    return slug or fallback


def _write_raw_snapshot(
    *,
    raw_dir: Path,
    doc_id: str,
    title: str,
    final_url: str,
    doi: str,
    openalex_item: dict,
    crossref_item: dict,
    access: str,
    source_note: str,
    landing_status_code: Optional[int],
    landing_html: str,
) -> Tuple[Path, Optional[Path]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_name = _slugify(title, fallback=f"jvr-{doc_id}")
    json_path = raw_dir / f"{base_name}-{doc_id}.json"
    html_path: Optional[Path] = None

    payload = {
        "source_type": "jvr_article_snapshot",
        "title": title,
        "final_url": final_url,
        "doi": doi,
        "access": access,
        "source_note": source_note,
        "landing_status_code": landing_status_code,
        "openalex": openalex_item,
        "crossref": crossref_item,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if landing_html:
        html_path = raw_dir / f"{base_name}-{doc_id}.html"
        html_path.write_text(landing_html, encoding="utf-8", errors="ignore")

    return json_path, html_path


def _format_article_doc(
    *,
    title: str,
    url: str,
    doi: str,
    journal: str,
    published: str,
    volume: str,
    issue: str,
    pages: str,
    access: str,
    authors: List[str],
    keywords: List[str],
    abstract: str,
    full_text: str,
    source_note: str,
) -> str:
    lines = [
        f"TITLE: {title or 'Untitled article'}",
        f"URL: {url}",
        f"DOI: {doi or 'Unknown DOI'}",
        f"JOURNAL: {journal or 'Journal of Vocational Rehabilitation'}",
        f"PUBLISHED: {published or 'Unknown'}",
        f"VOLUME: {volume or 'Unknown'}",
        f"ISSUE: {issue or 'Unknown'}",
        f"PAGES: {pages or 'Unknown'}",
        f"ACCESS: {access}",
        f"SOURCE_NOTE: {source_note}",
    ]
    if authors:
        lines.append("AUTHORS: " + "; ".join(authors))
    if keywords:
        lines.append("KEYWORDS: " + "; ".join(keywords))
    lines.append("")
    if abstract:
        lines.append("ABSTRACT:")
        lines.append(abstract)
        lines.append("")
    if full_text:
        lines.append("FULL TEXT:")
        lines.append(full_text)
    return "\n".join(lines).strip() + "\n"


def _harvest_article(
    session: requests.Session,
    *,
    item: dict,
    timeout: int,
    delay: float,
    docs_dir: Path,
    raw_dir: Path,
    manifest_path: Path,
    seen_final_urls: Set[str],
) -> bool:
    doi_url = str(item.get("doi") or "").strip()
    if not doi_url:
        return False
    doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    final_url = doi_url if doi_url.startswith("http") else f"https://doi.org/{doi}"
    if final_url in seen_final_urls:
        return False
    biblio = item.get("biblio") or {}
    title = str(item.get("title") or final_url).strip()
    journal = str((((item.get("primary_location") or {}).get("source") or {}).get("display_name")) or "Journal of Vocational Rehabilitation")
    published = str(item.get("publication_date") or item.get("publication_year") or "")
    volume = str(biblio.get("volume") or "")
    issue = str(biblio.get("issue") or "")
    first_page = str(biblio.get("first_page") or "")
    last_page = str(biblio.get("last_page") or "")
    pages = f"{first_page}-{last_page}" if first_page and last_page else first_page
    authors = [
        str(((auth.get("author") or {}).get("display_name")) or "").strip()
        for auth in (item.get("authorships") or [])
        if isinstance(auth, dict)
    ]
    authors = [a for a in authors if a]
    keywords: List[str] = []

    abstract = _openalex_abstract(item)
    crossref = {}
    if not abstract or len(abstract) < 120:
        crossref = _crossref_lookup(session, doi, timeout=timeout, delay=delay)
        abstract = abstract or _strip_jats(str(crossref.get("abstract") or ""))
    else:
        crossref = _crossref_lookup(session, doi, timeout=timeout, delay=delay)

    landing_page = str((((item.get("primary_location") or {}).get("landing_page_url")) or final_url)).strip() or final_url
    full_text = ""
    landing_status_code: Optional[int] = None
    landing_html = ""
    landing_resp = _fetch(session, landing_page, timeout=timeout)
    time.sleep(delay)
    access = "unknown"
    source_note = "openalex"
    if landing_resp:
        source_note = "openalex+landing-page"
        landing_status_code = landing_resp.status_code
        landing_html = landing_resp.text or ""
        final_url = normalize_url(landing_resp.url)
        soup = BeautifulSoup(landing_resp.text or "", "lxml")
        main_text = _extract_main_text(soup)
        access = _detect_access(main_text)
        full_text = main_text
        if abstract and full_text:
            full_text = full_text.replace(abstract, "", 1).strip()
        if len(full_text) < 400:
            full_text = ""
    else:
        primary = item.get("primary_location") or {}
        access = "open" if bool(primary.get("is_oa")) else "blocked"
        if crossref.get("abstract"):
            source_note = "openalex+crossref"

    article_body = _format_article_doc(
        title=title,
        url=final_url,
        doi=doi,
        journal=journal,
        published=published,
        volume=volume,
        issue=issue,
        pages=pages,
        access=access,
        authors=authors,
        keywords=keywords,
        abstract=abstract,
        full_text=full_text,
        source_note=source_note,
    )

    if len(article_body) < 300:
        return False

    doc_id = sha12(final_url)
    filename_slug = _slugify(title, fallback=f"jvr-{doc_id}")
    display_name = f"{filename_slug}.txt"
    docs_path = docs_dir / f"{doc_id}.txt"
    header = f"FILENAME: {display_name}\nPATH: {final_url}\n\n"
    docs_path.write_text(header + article_body, encoding="utf-8", errors="ignore")
    raw_json_path, raw_html_path = _write_raw_snapshot(
        raw_dir=raw_dir,
        doc_id=doc_id,
        title=title,
        final_url=final_url,
        doi=doi,
        openalex_item=item,
        crossref_item=crossref,
        access=access,
        source_note=source_note,
        landing_status_code=landing_status_code,
        landing_html=landing_html,
    )

    notes = [f"access:{access}", f"raw_json:{raw_json_path.name}"]
    if raw_html_path is not None:
        notes.append(f"raw_html:{raw_html_path.name}")

    append_manifest(
        manifest_path,
        ManifestRow(
            doc_id=doc_id,
            source_url=JVR_HOME_URL,
            final_url=final_url,
            domain=SAGE_DOMAIN,
            title=title,
            fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            content_type="application/json",
            raw_path=str(raw_json_path.relative_to(manifest_path.parent)),
            processed_text_path=str(docs_path.relative_to(manifest_path.parent)),
            status="jvr_article_saved",
            notes=" | ".join(notes),
        ),
    )
    seen_final_urls.add(final_url)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Employment folder (default=.)")
    ap.add_argument("--max-issues", type=int, default=24, help="Maximum distinct journal issues to ingest, newest first (0 = no limit)")
    ap.add_argument("--max-articles", type=int, default=0, help="Maximum articles to save, newest first (0 = no limit)")
    ap.add_argument("--delay", type=float, default=0.8, help="Delay between requests in seconds")
    ap.add_argument("--timeout", type=int, default=25, help="HTTP timeout in seconds")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    docs_dir = root / "docs_text"
    raw_dir = root / "_raw" / "jvr"
    manifest_path = root / "manifest.csv"

    docs_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(manifest_path)
    seen_final_urls = load_manifest_urls(manifest_path)

    session = requests.Session()

    article_items = _fetch_openalex_articles(
        session,
        timeout=args.timeout,
        delay=args.delay,
        max_issues=args.max_issues,
        max_articles=args.max_articles,
    )
    distinct_issues = sorted({key for key in (_issue_key(item) for item in article_items) if key != ("", "")})
    print(f"OpenAlex JVR articles selected: {len(article_items)}")
    print(f"Distinct issues represented: {len(distinct_issues)}")

    saved = 0
    pbar = tqdm(article_items, desc="Harvesting JVR articles", unit="article")
    for item in pbar:
        if _harvest_article(
            session,
            item=item,
            timeout=args.timeout,
            delay=args.delay,
            docs_dir=docs_dir,
            raw_dir=raw_dir,
            manifest_path=manifest_path,
            seen_final_urls=seen_final_urls,
        ):
            saved += 1
    pbar.close()

    print("\n✅ Done.")
    print(f"OpenAlex JVR articles selected: {len(article_items)}")
    print(f"Distinct issues represented: {len(distinct_issues)}")
    print(f"New JVR article docs saved: {saved}")
    print(f"_raw/jvr/: {raw_dir}")
    print(f"docs_text/: {docs_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
