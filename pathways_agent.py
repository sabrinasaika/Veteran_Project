#!/usr/bin/env python3
"""
Veterans Employment Pathways Agent
------------------------------------
Agno agent with retrieval tools:
  1. Local RAG      – ChromaDB + BM25 + cross-encoder (existing corpus)
  2. PubMed         – live academic paper search
  3. Web Search     – DuckDuckGo (current VA programs, resources)
  4. RSA-911        – privacy-safe aggregate VR outcome patterns
  5. O*NET          – occupation matching (online search)

Requires env vars (set in .env or shell):
  GROQ_API_KEY      – from console.groq.com (free)
  (or AGNO_MODEL_PROVIDER=ollama + OLLAMA_MODEL to run a local model instead)
  AGENTOPS_API_KEY  – from agentops.ai (free tier)
"""

from __future__ import annotations

import json
import io
import os
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# ── Agno (must be imported before AgentOps init) ─────────────────────────────
from agno.agent import Agent
from agno.exceptions import ModelProviderError
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.pubmed import PubmedTools
from agno.run import RunStatus

# ── AgentOps (init after Agno is fully loaded) ───────────────────────────────
import agentops
_agentops_key = os.getenv("AGENTOPS_API_KEY", "")
_agentops_enabled = False
# Local mode keeps everything on this machine, so AgentOps (a cloud service) stays off.
_local_mode = os.getenv("AGNO_MODEL_PROVIDER", "groq").strip().lower() == "ollama"
if _local_mode:
    print("AgentOps: disabled in local (ollama) mode so traces stay on this machine.")
elif _agentops_key and _agentops_key != "your_agentops_api_key_here":
    agentops.init(
        api_key=_agentops_key,
        auto_start_session=False,
        skip_auto_end_session=True,
    )
    _agentops_enabled = True
    print("AgentOps tracing enabled.")
else:
    print("AgentOps: no API key set — tracing disabled.")

# ── Existing RAG stack ───────────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ── Config ───────────────────────────────────────────────────────────────────
DB_DIR       = "chroma_db_adv"
COLLECTION   = "employment_adv_v1"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNKS_JSONL = Path("chunks.jsonl")
MANUAL_RUNS_PATH = Path("statistics/manual_runs.json")
RSA911_AGGREGATE_PATH = Path(os.getenv("RSA911_AGGREGATE_PATH", "data/rsa911_private/aggregate_summary.json"))
RSA911_ANALYSIS_PATH = Path(os.getenv("RSA911_ANALYSIS_PATH", "data/rsa911_private/analysis_summary.json"))
RSA911_STATISTICS_PATH = Path(os.getenv("RSA911_STATISTICS_PATH", "data/rsa911_private/statistical_analysis_summary.json"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "").strip() or None
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))  # long multi-tool conversations overflow 16k and drop the question
MODEL_PROVIDER = os.getenv("AGNO_MODEL_PROVIDER", "groq").strip().lower()
AUTO_LOG_MANUAL_RUNS = os.getenv("AUTO_LOG_MANUAL_RUNS", "true").strip().lower() in {"1", "true", "yes", "on"}
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "your_email@example.com").strip() or "your_email@example.com"
EVAL_FORCE_PUBMED = (
    os.getenv("EVAL_FORCE_PUBMED", os.getenv("EVAL_FORCE_ARXIV", "false")).strip().lower()
    in {"1", "true", "yes", "on"}
)


def _prefer_openai() -> bool:
    if MODEL_PROVIDER != "openai":
        return False
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key and key != "your_openai_api_key_here")


USE_NATIVE_TOOL_CALLS = True

print(
    f"Agno mode: provider={MODEL_PROVIDER or ('openai' if _prefer_openai() else 'groq')}"
    f"{f' model={OLLAMA_MODEL}' if MODEL_PROVIDER == 'ollama' else ''} "
    f"native_tools={'on' if USE_NATIVE_TOOL_CALLS else 'off'} "
    f"eval_force_pubmed={'on' if EVAL_FORCE_PUBMED else 'off'}"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Load RAG models once at import time
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


print("Loading embedding model + ChromaDB...")
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
_vs = Chroma(
    persist_directory=DB_DIR,
    collection_name=COLLECTION,
    embedding_function=_embeddings,
)

print("Loading BM25 from chunks.jsonl...")
_bm25_docs: List[Document] = []
# Every chunk's text by (source_path, chunk_index), used to finish passages cut off mid-sentence.
_chunk_text: Dict[Tuple[str, str], str] = {}
if CHUNKS_JSONL.exists():
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            _chunk_text[(str(meta.get("source_path", "")), str(meta.get("chunk_index", "")))] = rec.get("text", "")
            t = _clean(rec.get("text", ""))
            if len(t) >= 200:
                _bm25_docs.append(Document(page_content=t, metadata=meta))
_bm25 = BM25Retriever.from_documents(_bm25_docs)
_bm25.k = 60

print("Loading cross-encoder reranker...")
_reranker = CrossEncoder(RERANK_MODEL)

print("RAG stack ready.")

_arxiv_tools = ArxivTools(enable_read_arxiv_papers=False)
_pubmed_tools = PubmedTools(email=PUBMED_EMAIL, max_results=5, results_expanded=True)
_web_tools = DuckDuckGoTools(enable_news=False)


def _build_model():
    if MODEL_PROVIDER == "ollama":
        # Local model: prompts and tool outputs (including RSA-911 aggregates) stay on this machine.
        from agno.models.ollama import Ollama

        return Ollama(id=OLLAMA_MODEL, host=OLLAMA_HOST, options={"temperature": 0, "num_ctx": OLLAMA_NUM_CTX})
    if _prefer_openai():
        return OpenAIChat(id=OPENAI_MODEL, api_key=os.getenv("OPENAI_API_KEY", ""), temperature=0)
    return Groq(id=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY", ""), temperature=0)


def _active_provider_name() -> str:
    if MODEL_PROVIDER == "ollama":
        return "ollama"
    return "openai" if _prefer_openai() else "groq"


def _build_native_tools():
    return [
        search_local_corpus,
        PubmedTools(email=PUBMED_EMAIL, max_results=5, results_expanded=True),
        fetch_paper_full_text,
        search_web,
        search_jan_accommodations,
        fetch_web_page,
        query_rsa911_outcomes,
        analyze_rsa911_question,
        analyze_rsa911_statistics,
        describe_rsa911_sql_schema,
        run_rsa911_safe_sql,
        search_onet,
    ]


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _has_pubmed_tool_call(tool_executions: List[Any]) -> bool:
    for tool_exec in tool_executions:
        tool_name = getattr(tool_exec, "tool_name", "") or ""
        if "pubmed" in tool_name.lower():
            return True
    return False


def _build_pubmed_eval_query(message: str) -> str:
    q = message.lower()
    condition_terms: List[str] = []
    if "tbi" in q or "brain injury" in q:
        condition_terms.append('"traumatic brain injury"')
    if "ptsd" in q:
        condition_terms.append("PTSD")
    if "wheelchair" in q or "spinal cord" in q:
        condition_terms.append('"mobility impairment"')
    if "vision loss" in q or "blind" in q:
        condition_terms.append('"vision impairment"')
    if "hearing loss" in q or "deaf" in q:
        condition_terms.append('"hearing loss"')
    if "multiple sclerosis" in q:
        condition_terms.append('"multiple sclerosis"')
    if "chronic pain" in q or "back pain" in q:
        condition_terms.append('"chronic pain"')
    if "cognitive" in q:
        condition_terms.append('"cognitive impairment"')
    if "anxiety" in q:
        condition_terms.append("anxiety")
    if "depression" in q:
        condition_terms.append("depression")
    if not condition_terms:
        condition_terms.append("disability")

    return (
        "("
        + " OR ".join(condition_terms[:3])
        + ') AND ("employment" OR "return to work" OR "vocational rehabilitation" '
        + 'OR "supported employment" OR "job accommodation") AND (veteran OR disability)'
    )


def _force_pubmed_tool_execution(message: str) -> Any:
    query = _build_pubmed_eval_query(message)
    try:
        result = _pubmed_tools.search_pubmed(query=query, max_results=5)
    except Exception as exc:
        result = f"PubMed unavailable: {type(exc).__name__}: {exc}"
    return SimpleNamespace(
        tool_name="search_pubmed",
        tool_args={"query": query, "max_results": 5, "forced_for_eval": True},
        result=result,
    )


def _needs_jan_accommodation_evidence(message: str) -> bool:
    lowered = str(message or "").lower()
    triggers = [
        "accommodation",
        "accommodations",
        "workplace support",
        "workplace supports",
        "support would you recommend",
        "supports would you recommend",
        "functional limitation",
        "functional limitations",
        "ptsd",
        "anxiety",
        "mobility",
        "remote",
        "hybrid",
        "telework",
        "work from home",
    ]
    return any(trigger in lowered for trigger in triggers)


# Pathway questions (work or career plus a condition) get the matching RSA-911 disability-group
# summary even when the model did not query RSA-911, so outcome data is always available, as for JAN.
_PATHWAY_WORK_RE = re.compile(
    r"\b(?:jobs?|work\w*|careers?|employ\w*|occupation\w*|pathways?|hir(?:e|ed|ing)|positions?|vocational"
    r"|income|profession\w*|cop|police|healthcare|retrain\w*|what else can i do|what can i do"
    r"|business|path|shifts?|mechanic|teacher|become|empleo|trabajo)\b",
    re.IGNORECASE,
)
# Matched at word starts, so plurals and other endings count ("migraines", "injured").
_CONDITION_TERMS = (
    "disabilit", "disabled", "injur", "impair", "condition", "rated", "rating", "service-connected",
    "hearing", "deaf", "tinnitus", "vision", "blind", "pain", "amput", "lost my", "leg", "hand", "arm",
    "wheelchair", "spinal", "tbi", "brain injury", "ptsd", "post-traumatic", "trauma", "anxiety",
    "depress", "bipolar", "schizophren", "mental health", "psychiatric", "multiple sclerosis", "fatigue",
    "mobility", "migraine", "vertigo", "tremor", "cancer", "lupus", "diabetes", "arthritis", "autism",
    "learning disab", "dialysis", "kidney", "sleep apnea", "burn", "homeless", "substance", "dolor",
)


def _needs_rsa911_context(message: str) -> bool:
    lowered = str(message or "").lower()
    if _is_pure_rsa911_sql_question(message) or ("individual" in lowered and "record" in lowered):
        return False
    return bool(_PATHWAY_WORK_RE.search(lowered)) and any(
        re.search(rf"\b{re.escape(term)}", lowered) for term in _CONDITION_TERMS
    )


def _has_rsa911_tool_call(tool_executions: List[Any]) -> bool:
    return any("rsa911" in (getattr(tool_exec, "tool_name", "") or "") for tool_exec in tool_executions)


def _force_rsa911_summary_execution(message: str) -> Any:
    """Look up the RSA-911 summary for the question. Marked forced_by_system, not forced_for_eval,
    so it counts as evidence for the answer and the judge while still being reported as system-added."""
    try:
        result = query_rsa911_outcomes.entrypoint(str(message or ""))
    except Exception as exc:
        result = f"RSA-911 aggregate outcomes unavailable: {type(exc).__name__}: {exc}"
    return SimpleNamespace(
        tool_name="query_rsa911_outcomes",
        tool_args={"query": str(message or ""), "forced_by_system": True},
        result=result,
    )


def _has_jan_tool_call(tool_executions: List[Any]) -> bool:
    for tool_exec in tool_executions:
        tool_name = getattr(tool_exec, "tool_name", "") or ""
        result_text = getattr(tool_exec, "result", "") or ""
        if tool_name == "fetch_web_page" and "askjan.org" in result_text:
            return True
    return False


def _jan_fallback_urls(message: str) -> List[str]:
    lowered = str(message or "").lower()
    urls = ["https://askjan.org/solutions/Telework-Work-from-Home-Working-Remotely.cfm"]
    if "ptsd" in lowered or "post traumatic" in lowered or "post-traumatic" in lowered:
        urls.append("https://askjan.org/disabilities/Post-Traumatic-Stress-Disorder-PTSD.cfm")
    if "anxiety" in lowered:
        urls.append("https://askjan.org/disabilities/Anxiety-Disorder.cfm")
    if "mobility" in lowered or "wheelchair" in lowered or "walking" in lowered:
        urls.append("https://askjan.org/limitations/Use-of-Mobility-Aids.cfm")
        urls.append("https://askjan.org/limitations/Walking.cfm")
    return _ordered_unique(urls)


def _force_jan_tool_executions(message: str) -> List[Any]:
    forced: List[Any] = []
    query = str(message or "").strip() or "workplace accommodations"
    try:
        search_result = search_jan_accommodations.entrypoint(query)
    except Exception as exc:
        search_result = f"JAN accommodation search unavailable: {type(exc).__name__}: {exc}"

    forced.append(
        SimpleNamespace(
            tool_name="search_jan_accommodations",
            tool_args={"query": query, "forced_for_eval": True},
            result=search_result,
        )
    )

    parsed = _extract_web_results(str(search_result))
    preferred_urls: List[str] = []
    for item in parsed:
        url = str(item.get("url") or "")
        if "askjan.org/solutions/" in url:
            preferred_urls.append(url)
    for item in parsed:
        url = str(item.get("url") or "")
        if "askjan.org/disabilities/" in url or "askjan.org/limitations/" in url:
            preferred_urls.append(url)

    if not preferred_urls:
        preferred_urls.extend(_jan_fallback_urls(message))

    for url in _ordered_unique(preferred_urls)[:2]:
        try:
            page_result = fetch_web_page.entrypoint(url)
        except Exception as exc:
            page_result = f"Page fetch unavailable: {type(exc).__name__}: {exc}"
        forced.append(
            SimpleNamespace(
                tool_name="fetch_web_page",
                tool_args={"url": url, "forced_for_eval": True},
                result=page_result,
            )
        )

    return forced


# ═══════════════════════════════════════════════════════════════════════════════
# RAG helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _chroma_search(query: str, n: int = 70) -> Tuple[List[Document], List[float]]:
    col = _vs._collection
    res = col.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    docs  = [Document(page_content=t, metadata=m)
             for t, m in zip(res["documents"][0], res["metadatas"][0])]
    dists = [float(x) for x in res["distances"][0]]
    return docs, dists


def _bm25_search(query: str) -> List[Document]:
    if hasattr(_bm25, "invoke"):
        return _bm25.invoke(query)
    return _bm25._get_relevant_documents(query)  # noqa


def _doc_key(d: Document) -> tuple:
    m = d.metadata or {}
    return (m.get("doc_txt"), m.get("chunk_index"), m.get("source_path"))


def _fuse(vec_docs, vec_dist, kw_docs, top=160, wv=0.65, wk=0.35) -> List[Document]:
    pool: Dict[tuple, dict] = {}
    for d, dist in zip(vec_docs, vec_dist):
        k = _doc_key(d)
        pool[k] = {"doc": d, "v": 1.0 / (1.0 + float(dist)), "k": 0.0}
    for rank, d in enumerate(kw_docs):
        k = _doc_key(d)
        ks = 1.0 / (1.0 + rank)
        if k in pool:
            pool[k]["k"] = max(pool[k]["k"], ks)
        else:
            pool[k] = {"doc": d, "v": 0.0, "k": ks}
    scored = [(wv * r["v"] + wk * r["k"], r["doc"]) for r in pool.values()]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top]]


def _rerank_docs(query: str, docs: List[Document], top_k: int = 8) -> List[Document]:
    pairs  = [(query, d.page_content) for d in docs]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_k]]


_SENTENCE_END_RE = re.compile(r"[.!?:;)\]\"\u201d']\s*$")
CONTINUATION_CHARS = 700


def _with_continuation(d: Document) -> str:
    """Finish a chunk that was split mid-sentence with the start of the next chunk.

    Chunks were cut by length, so most end mid-sentence (e.g. halfway through the list of
    VR&E tracks). Consecutive chunks overlap, so only the text after the overlap is added,
    capped and trimmed to the last sentence end.
    """
    text = d.page_content
    m = d.metadata or {}
    if _SENTENCE_END_RE.search(text) or m.get("chunk_index") in (None, ""):
        return text
    try:
        next_index = str(int(m.get("chunk_index")) + 1)
    except (TypeError, ValueError):
        return text
    following = _chunk_text.get((str(m.get("source_path", "")), next_index), "")
    if not following:
        return text
    collapse = lambda value: re.sub(r"\s+", " ", value).strip()
    tail, flat_next = collapse(text)[-60:], collapse(following)
    overlap_at = flat_next.find(tail)
    continuation = flat_next[overlap_at + len(tail):] if overlap_at >= 0 else flat_next
    continuation = continuation[:CONTINUATION_CHARS]
    last_end = max(continuation.rfind(". "), continuation.rfind(" ."), continuation.rfind("? "), continuation.rfind("! "))
    if last_end > 0:
        continuation = continuation[: last_end + 2]
    return f"{text.rstrip()} {continuation.strip()}" if continuation.strip() else text


def _format_doc(i: int, d: Document) -> str:
    m   = d.metadata or {}
    src = m.get("source_filename") or m.get("doc_txt") or "unknown"
    pth = m.get("source_path", "")
    chk = m.get("chunk_index", "")
    header = f"[Source {i}] {src}"
    if pth:
        header += f" | {pth}"
    if chk != "":
        header += f" | chunk {chk}"
    return f"{header}\n{_with_continuation(d)[:1200 + CONTINUATION_CHARS]}"


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1 – Local RAG (policy corpus + rehab docs)
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="search_local_corpus",
    description=(
        "Search the local indexed corpus of veterans employment, vocational rehabilitation, "
        "policy, disability, TACQE, RSA, and VCU RRTC documents. "
        "Input must be a plain natural-language search query string."
    ),
)
def search_local_corpus(query: str) -> str:
    """
    Search the local employment policy corpus using hybrid retrieval
    (ChromaDB vector search + BM25 keyword search + cross-encoder reranking).
    Best for: VR policy, program eligibility, RSA guidance, rehab docs, TACQE, VCU RRTC.

    Args:
        query: The search query string.

    Returns:
        Top evidence chunks with source citations.
    """
    vec_docs, vec_dist = _chroma_search(query)
    kw_docs            = _bm25_search(query)
    candidates         = _fuse(vec_docs, vec_dist, kw_docs)
    top_docs           = _rerank_docs(query, candidates, top_k=6)

    if not top_docs:
        return "No relevant documents found in local corpus."

    blocks = [_format_doc(i + 1, d) for i, d in enumerate(top_docs)]
    return "\n\n---\n\n".join(blocks)


@tool(
    name="search_web",
    description=(
        "Search the public web for current VA programs, accommodations, benefits, and employment resources. "
        "Input must be a plain natural-language search query string."
    ),
)
def search_web(query: str) -> str:
    """
    Search the web using DuckDuckGo behind a narrow single-argument schema.

    Args:
        query: The search query string.

    Returns:
        JSON string of web results, or an explanatory message if no results are found.
    """
    results = _web_search_results(query)
    if not results:
        results = _official_page_results(query)
    if not results:
        return "Web search unavailable: no results from any search backend."
    return json.dumps(results)


# The free search backends fail at random from this server (DuckDuckGo and Brave
# refuse every request; Google and Bing answer about half the time), so a failed
# search is retried on the other backends before giving up.
WEB_SEARCH_BACKENDS = ["auto", "google", "bing", "google", "bing"]


def _web_search_results(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        from ddgs import DDGS  # type: ignore
    except Exception:
        return []
    for backend in WEB_SEARCH_BACKENDS:
        try:
            results = DDGS().text(query, max_results=max_results, backend=backend)
        except Exception:
            continue
        results = [item for item in results or [] if isinstance(item, dict) and item.get("href")]
        if results:
            return results
    return []


# Official pages opened directly when every search backend fails, so a core
# program question still gets its official source instead of no web evidence.
_OFFICIAL_TOPIC_PAGES = [
    (
        re.compile(r"vr&e|vocational rehabilitation|chapter 31|veteran readiness", re.I),
        [
            "https://www.va.gov/careers-employment/vocational-rehabilitation/eligibility/",
            "https://www.va.gov/careers-employment/vocational-rehabilitation/programs/",
        ],
    ),
    (
        re.compile(r"unemployability|\btdiu\b", re.I),
        ["https://www.va.gov/disability/eligibility/special-claims/unemployability/"],
    ),
    (
        re.compile(r"ticket to work|\bssdi\b|\bssi\b|social security", re.I),
        ["https://choosework.ssa.gov/about/how-it-works"],
    ),
    (
        re.compile(r"reasonable accommodation|\bada\b|americans with disabilities act", re.I),
        ["https://www.eeoc.gov/laws/guidance/enforcement-guidance-reasonable-accommodation-and-undue-hardship-under-ada"],
    ),
    (
        re.compile(r"state vr\b|state vocational rehabilitation", re.I),
        ["https://rsa.ed.gov/about/programs/vocational-rehabilitation-state-grants"],
    ),
]


def _page_result(url: str, body_limit: int = 3500) -> Dict[str, Any]:
    """Open one page and return it in the same shape as a search result."""
    page = fetch_web_page.entrypoint(url)
    if not page.startswith("Title:"):
        return {}
    header, _, text = page.partition("\nExtracted Text:\n")
    title = header.split("\n", 1)[0][len("Title:"):].strip()
    final_url = url
    for line in header.splitlines():
        if line.startswith("URL:"):
            final_url = line[len("URL:"):].strip()
    return {"title": title, "href": final_url, "body": _trim(text, limit=body_limit)}


def _official_page_results(query: str) -> List[Dict[str, Any]]:
    urls: List[str] = []
    for pattern, topic_urls in _OFFICIAL_TOPIC_PAGES:
        if pattern.search(query or ""):
            urls.extend(topic_urls)
    results = [_page_result(url) for url in _ordered_unique(urls)[:2]]
    return [item for item in results if item]


# JAN pages are found from JAN's own A-to-Z index instead of a search engine.
JAN_INDEX_URL = "https://askjan.org/a-to-z.cfm"
# JAN pages list their accommodations well past the opening text (ramps sit near 7,300 chars on Use of Mobility Aids).
JAN_PAGE_CHARS = 8000
_JAN_INDEX: List[Tuple[str, str]] = []

_JAN_QUERY_ALIASES = {
    "ptsd": "post traumatic stress",
    "tbi": "brain injury",
    "wheelchair": "mobility aids",
    "amputee": "amputation",
    "amputated": "amputation",
    "prosthetic": "amputation",
    "prosthesis": "amputation",
    "blind": "vision",
    "deaf": "hearing",
    "depressed": "depression",
    "anxious": "anxiety",
    "ms": "multiple sclerosis",
}

# Words that appear in many JAN page names; they count toward a match but are not required.
_JAN_GENERIC_WORDS = {
    "disorder", "disorders", "impairment", "impairments", "disease", "condition", "conditions",
    "use", "of", "and", "the", "or", "syndrome", "problem", "problems", "related", "disability",
}


def _jan_words(text: str) -> List[str]:
    return [word.rstrip("s") if len(word) > 3 else word for word in re.findall(r"[a-z]+", text.lower())]


def _jan_index() -> List[Tuple[str, str]]:
    if _JAN_INDEX:
        return _JAN_INDEX
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore

    resp = requests.get(JAN_INDEX_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        if not re.match(r"^/(disabilities|limitations)/.+\.cfm$", href):
            continue
        name = link.get_text(" ", strip=True)
        if name:
            _JAN_INDEX.append((name, "https://askjan.org" + href))
    return _JAN_INDEX


def _jan_page_urls(query: str, limit: int = 2) -> List[str]:
    lowered = (query or "").lower()
    for term, alias in _JAN_QUERY_ALIASES.items():
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            lowered += " " + alias
    query_words = set(_jan_words(lowered))
    scored = []
    for name, url in _jan_index():
        name_words = _jan_words(name)
        specific = [word for word in name_words if word not in _JAN_GENERIC_WORDS] or name_words
        matched = sum(1 for word in specific if word in query_words)
        if matched == 0 or matched < len(specific) / 2:
            continue
        # Prefer full matches, then disability pages over limitation pages.
        scored.append((-(matched / len(specific)), -matched, "/limitations/" in url, url))
    scored.sort()
    # Partial matches (e.g. "Hand Amputation" for a leg amputation) only when no page matches fully.
    full = [entry for entry in scored if entry[0] == -1]
    return _ordered_unique([url for *_, url in (full or scored)])[:limit]


@tool(
    name="search_jan_accommodations",
    description=(
        "Search Job Accommodation Network (JAN/AskJAN) pages for disability-specific workplace "
        "accommodation ideas. Use this when the user asks for accommodations, functional limits, "
        "PTSD, mental health, mobility, remote work, or disability-friendly work strategies."
    ),
)
def search_jan_accommodations(query: str) -> str:
    """
    Search JAN/AskJAN pages for accommodation guidance.

    Args:
        query: Disability, limitation, job, or accommodation question.

    Returns:
        JSON string of JAN search results, or an explanatory message if unavailable.
    """
    q = str(query or "").strip()
    if not q:
        q = "workplace accommodations disability"
    try:
        result_blocks = []
        try:
            jan_urls = _jan_page_urls(q)
        except Exception:
            jan_urls = []
        for url in jan_urls:
            page = _page_result(url, body_limit=JAN_PAGE_CHARS)
            if page:
                result_blocks.append(page)
        if not result_blocks:
            for jan_query in [
                f"site:askjan.org/solutions {q} workplace accommodation",
                f"site:askjan.org/disabilities {q} workplace accommodation",
            ]:
                result_blocks.extend(_web_search_results(jan_query))

        seen = set()
        merged = []
        for item in result_blocks:
            if not isinstance(item, dict):
                continue
            url = str(item.get("href") or item.get("url") or "")
            title = str(item.get("title") or item.get("heading") or "")
            key = url or title
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= 8:
                break
        return json.dumps(merged)
    except Exception as exc:
        return f"JAN accommodation search unavailable: {type(exc).__name__}: {exc}"


def _extract_page_text(url: str, html: str) -> Tuple[str, str]:
    title = ""
    text = ""
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(html, url=url, include_tables=True) or ""
        if extracted.strip():
            text = extracted.strip()
    except Exception:
        pass

    if not text:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        if soup.title:
            title = soup.title.get_text(" ", strip=True)
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)

    text = _clean(text)
    if not title:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        if soup.title:
            title = soup.title.get_text(" ", strip=True)
    parsed = urllib.parse.urlparse(url)
    limit = 14000 if parsed.netloc.lower().endswith("askjan.org") else 5000
    return title.strip(), _trim(text, limit=limit)


def _extract_pdf_text_bytes(blob: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""

    try:
        reader = PdfReader(io.BytesIO(blob))
    except Exception:
        return ""

    parts: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _clean(text)
        if text:
            parts.append(text)
    return _trim("\n\n".join(parts).strip(), limit=7000)


def _looks_like_paper_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(
        marker in lowered
        for marker in [
            "pubmed.ncbi.nlm.nih.gov",
            "pmc.ncbi.nlm.nih.gov",
            "doi.org/",
            "/doi/",
            "/article/",
            "/full",
            "/pdf",
        ]
    )


def _discover_pubmed_full_text_url(pubmed_url: str, headers: Dict[str, str]) -> str:
    try:
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return ""

    try:
        resp = requests.get(pubmed_url, headers=headers, timeout=12, allow_redirects=True)
        if resp.status_code >= 400 or not resp.text:
            return ""
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "lxml")
    candidates: List[str] = []

    for tag_name, attr_name in [("meta", "content"), ("a", "href"), ("link", "href")]:
        for node in soup.find_all(tag_name):
            raw = (node.get(attr_name) or "").strip()
            if not raw:
                continue
            full = urllib.parse.urljoin(str(resp.url or pubmed_url), raw)
            if any(marker in full.lower() for marker in ["pmc/articles", "doi.org/", "/article/", "/pdf"]):
                candidates.append(full)

    preferred_prefixes = [
        "https://pmc.ncbi.nlm.nih.gov/articles/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/",
        "https://doi.org/",
    ]
    for prefix in preferred_prefixes:
        for candidate in candidates:
            if candidate.startswith(prefix):
                return candidate
    return candidates[0] if candidates else ""


@tool(
    name="fetch_web_page",
    description=(
        "Fetch and extract text from a single web page URL. Use this after search_web when you need "
        "real page content instead of search snippets. Input must be one full http or https URL."
    ),
)
def fetch_web_page(url: str) -> str:
    """
    Fetch a public web page and extract readable text.

    Args:
        url: Full http/https URL, ideally from a previous search_web result.

    Returns:
        Extracted page text with title and final URL, or an explanatory error.
    """
    url = str(url or "").strip()
    if not url:
        return "Page fetch unavailable: empty URL."

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return f"Page fetch unavailable: unsupported URL scheme in {url!r}."

    try:
        import requests  # type: ignore
    except Exception as exc:
        return f"Page fetch unavailable: requests import failed: {type(exc).__name__}: {exc}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
    except Exception as exc:
        return f"Page fetch unavailable: {type(exc).__name__}: {exc}"

    final_url = str(resp.url or url)
    content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if resp.status_code >= 400:
        return f"Page fetch unavailable: HTTP {resp.status_code} | {final_url}"
    if "pdf" in content_type:
        return f"Page fetch unavailable: URL points to PDF content | {final_url}"
    if "html" not in content_type and not content_type.startswith("text/"):
        return f"Page fetch unavailable: unsupported content type {content_type or 'unknown'} | {final_url}"

    title, text = _extract_page_text(final_url, resp.text or "")
    if not text:
        return f"Page fetch unavailable: no readable text extracted | {final_url}"

    title = title or final_url
    return f"Title: {title}\nURL: {final_url}\nExtracted Text:\n{text}"


@tool(
    name="fetch_paper_full_text",
    description=(
        "Fetch academic paper full text from a Full Text URL, PMC URL, DOI URL, or PubMed URL. "
        "Use this after PubMed when a paper looks highly relevant and an open full-text path may be available. "
        "Handles open HTML pages and PDFs. Input must be one full http or https URL."
    ),
)
def fetch_paper_full_text(url: str) -> str:
    """
    Fetch an academic paper's readable full text when publicly accessible.

    Args:
        url: Full http/https URL for a PubMed result, PMC article, DOI landing page, or PDF.

    Returns:
        Extracted paper text with title and final URL, or an explanatory error.
    """
    url = str(url or "").strip()
    if not url:
        return "Paper fetch unavailable: empty URL."

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return f"Paper fetch unavailable: unsupported URL scheme in {url!r}."

    try:
        import requests  # type: ignore
    except Exception as exc:
        return f"Paper fetch unavailable: requests import failed: {type(exc).__name__}: {exc}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    resolved_url = url
    if "pubmed.ncbi.nlm.nih.gov" in parsed.netloc:
        candidate = _discover_pubmed_full_text_url(url, headers)
        if candidate:
            resolved_url = candidate

    try:
        resp = requests.get(resolved_url, headers=headers, timeout=16, allow_redirects=True)
    except Exception as exc:
        return f"Paper fetch unavailable: {type(exc).__name__}: {exc}"

    final_url = str(resp.url or resolved_url)
    content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if resp.status_code >= 400:
        return f"Paper fetch unavailable: HTTP {resp.status_code} | {final_url}"

    title = ""
    text = ""
    access_type = "html"

    if "pdf" in content_type or final_url.lower().endswith(".pdf"):
        access_type = "pdf"
        text = _extract_pdf_text_bytes(resp.content or b"")
        title = final_url.rsplit("/", 1)[-1] or "Academic paper PDF"
    else:
        title, text = _extract_page_text(final_url, resp.text or "")
        if not text and _looks_like_paper_url(final_url):
            pdf_guess = final_url.rstrip("/") + ".pdf"
            try:
                pdf_resp = requests.get(pdf_guess, headers=headers, timeout=16, allow_redirects=True)
                pdf_type = (pdf_resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if pdf_resp.status_code < 400 and ("pdf" in pdf_type or str(pdf_resp.url).lower().endswith(".pdf")):
                    access_type = "pdf"
                    final_url = str(pdf_resp.url or pdf_guess)
                    text = _extract_pdf_text_bytes(pdf_resp.content or b"")
                    if not title:
                        title = final_url.rsplit("/", 1)[-1] or "Academic paper PDF"
            except Exception:
                pass

    if not text:
        return f"Paper fetch unavailable: no readable full text extracted | {final_url}"

    title = title or final_url
    return (
        f"Title: {title}\n"
        f"URL: {final_url}\n"
        f"Access Type: {access_type}\n"
        f"Extracted Text:\n{text}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 4 – O*NET Occupation Search
# ═══════════════════════════════════════════════════════════════════════════════

ONET_RESULT_NOTE = (
    "Keyword matches are starting points, not proof of fit. O*NET does not classify occupations "
    "as remote, hybrid, or disability-friendly."
)


def _onet_api(searches: List[str], note: str = "") -> str:
    import urllib.request, urllib.parse
    key     = os.getenv("ONET_API_KEY", "")
    base_url = "https://api-v2.onetcenter.org"

    def _request_json(url: str) -> dict:
        req = urllib.request.Request(url)
        req.add_header("X-API-Key", key)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())

    # One search per occupation idea; mixing unrelated words in one O*NET search gives poor
    # matches. Results are merged round-robin so each search contributes its best hits.
    per_search = [
        _request_json(f"{base_url}/online/search?keyword={urllib.parse.quote(term)}&end=5").get("occupation", [])
        for term in searches
    ]
    occupations: List[dict] = []
    seen_codes = set()
    for rank in range(5):
        for results in per_search:
            if rank < len(results) and results[rank].get("code") not in seen_codes and len(occupations) < 5:
                seen_codes.add(results[rank].get("code"))
                occupations.append(results[rank])
    if not occupations:
        return "No O*NET results."

    lines = [f"O*NET Occupation Results (searched: {'; '.join(searches)}):", f"Note: {ONET_RESULT_NOTE}"]
    if note:
        lines.append(f"Note: {note}")

    for occ in occupations:
        code = occ.get("code", "?")
        title = occ.get("title", "?")
        detail_url = f"{base_url}/online/occupations/{urllib.parse.quote(str(code))}/"
        try:
            detail = _request_json(detail_url)
        except Exception:
            detail = {}

        description = ""
        if isinstance(detail, dict):
            description = str(detail.get("description") or detail.get("title") or "").strip()

        sample_tasks: List[str] = []
        tasks_block = detail.get("sample_of_reported_titles") if isinstance(detail, dict) else None
        if isinstance(tasks_block, list):
            sample_tasks = [str(item) for item in tasks_block[:3] if str(item).strip()]

        if not sample_tasks:
            tasks_block = detail.get("sample_of_reported_tasks") if isinstance(detail, dict) else None
            if isinstance(tasks_block, list):
                for item in tasks_block[:3]:
                    if isinstance(item, dict):
                        task_text = str(item.get("title") or item.get("task") or item.get("name") or "").strip()
                    else:
                        task_text = str(item).strip()
                    if task_text:
                        sample_tasks.append(task_text)

        bright_outlook = ""
        flags = detail.get("flags") if isinstance(detail, dict) else None
        tags = detail.get("tags") if isinstance(detail, dict) else None
        if isinstance(flags, dict) and flags.get("bright_outlook"):
            bright_outlook = "bright outlook"
        elif isinstance(tags, dict) and tags.get("bright_outlook"):
            bright_outlook = "bright outlook"

        lines.append(f"- {title} (code: {code})")
        if description:
            lines.append(f"  Description: {_trim(str(description), limit=320)}")
        if sample_tasks:
            lines.append(f"  Examples: {_trim('; '.join(sample_tasks), limit=320)}")
        if bright_outlook:
            lines.append(f"  Notes: {bright_outlook}")

    return "\n".join(lines)


def _onet_web_fallback(searches: List[str], note: str = "") -> str:
    from ddgs import DDGS
    results = [r for term in searches for r in DDGS().text(f"site:onetonline.org {term}", max_results=3)]
    if not results:
        return "No O*NET results found."
    lines = [f"O*NET Web Search Results (searched: {'; '.join(searches)}):", f"Note: {ONET_RESULT_NOTE}"]
    if note:
        lines.append(f"Note: {note}")
    header_lines = len(lines)
    for r in results:
        title = str(r.get("title") or "?").strip()
        body = str(r.get("body") or "").strip()
        href = str(r.get("href") or r.get("url") or "").strip()
        joined = f"{title} {body} {href}"
        code_match = re.search(r"\b\d{2}-\d{4}\.\d{2}\b", joined)
        is_occupation_page = "/link/summary/" in href or "/link/details/" in href or bool(code_match)
        if not is_occupation_page:
            continue
        code_text = f" (code: {code_match.group(0)})" if code_match else ""
        descriptor = f"{title}{code_text}"
        if href:
            descriptor += f" | {href}"
        lines.append(f"- {descriptor}\n  Snippet: {_trim(body, limit=300)}")
    if len(lines) == header_lines:
        return "No O*NET occupation-code results found. Search returned only generic O*NET pages."
    return "\n".join(lines)


@tool(
    name="search_onet",
    description=(
        "Search O*NET for occupations related to the veteran's abilities, limitations, "
        "work preferences, or target job type. Input must be a plain natural-language query string."
    ),
)
def search_onet(query: str) -> str:
    """
    Search the O*NET occupation database for jobs matching a veteran's
    functional capabilities and limitations. Returns occupation titles,
    O*NET codes, and work context information.

    Args:
        query: Veteran capabilities, limitations, or desired job type.

    Returns:
        Matching occupation titles and O*NET codes.
    """
    searches = _onet_search_terms(str(query or ""))
    note = ""
    if not searches:
        searches = _ONET_DEFAULT_SEARCHES
        note = (
            "The query named no occupation (only conditions or work preferences), so these are broad "
            "desk-based occupation families to start from, not matches to this profile."
        )
    key = os.getenv("ONET_API_KEY", "")
    if key and key != "your_onet_api_key_here":
        try:
            return _onet_api(searches, note)
        except Exception as exc:
            return f"O*NET API unavailable: {type(exc).__name__}: {exc}"
    return _onet_web_fallback(searches, note)


# Terms are matched as whole words or phrases.
_RSA911_GROUP_TERMS = {
    "tbi": ["tbi", "traumatic brain", "brain injury"],
    "spinal_cord_injury": ["spinal cord", "sci"],
    # RSA-911 has no PTSD code; PTSD is recorded under disability cause 04, Anxiety Disorders.
    "anxiety": ["anxiety", "ptsd", "post-traumatic", "posttraumatic", "post traumatic"],
    "depression_mood": ["depression", "depressive", "mood disorder"],
    "multiple_sclerosis": ["multiple sclerosis", "ms", "fatigue"],
    "amputation": ["amputation", "amputee", "prosthetic"],
    "mobility": ["mobility", "wheelchair"],
    "psychosocial": ["psychosocial", "mental health", "mental illness", "psychiatric"],
}

_RSA911_GROUP_NOTES = {
    "anxiety": (
        "RSA-911 has no PTSD-specific code. PTSD is recorded under disability cause 04 (Anxiety Disorders), "
        "so this group also includes veterans with other anxiety disorders."
    ),
    "mobility": (
        "Impairment types 10 and 12 (mobility, and mobility plus dexterity), primary or secondary; "
        "broader than spinal cord injury alone."
    ),
    "psychosocial": (
        "Impairment type 18 (psychosocial), primary or secondary; covers many mental health conditions, not one diagnosis."
    ),
}


RSA911_CLOUD_BLOCK = (
    "RSA-911 data is disabled: the agent is not using a local model (AGNO_MODEL_PROVIDER=ollama), "
    "so RSA-911 results would be sent to an outside provider. Set AGNO_MODEL_PROVIDER=ollama to use it."
)


def _rsa911_blocked() -> bool:
    # RSA-911 results must never reach a cloud model provider, even as aggregates.
    return MODEL_PROVIDER != "ollama"


def _rsa911_years_label() -> str:
    """Program years actually loaded (e.g. "PY23"), from the file names in the aggregate summary."""
    files = _load_rsa911_aggregate().get("source_files") or []
    years = sorted({match.group(0) for name in files for match in [re.search(r"PY\d{2}", str(name))] if match})
    return ", ".join(years) if years else "loaded"


def _load_rsa911_aggregate() -> Dict[str, Any]:
    if _rsa911_blocked() or not RSA911_AGGREGATE_PATH.exists():
        return {}
    try:
        loaded = json.loads(RSA911_AGGREGATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_rsa911_analysis() -> Dict[str, Any]:
    if _rsa911_blocked() or not RSA911_ANALYSIS_PATH.exists():
        return {}
    try:
        loaded = json.loads(RSA911_ANALYSIS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_rsa911_statistics() -> Dict[str, Any]:
    if _rsa911_blocked() or not RSA911_STATISTICS_PATH.exists():
        return {}
    try:
        loaded = json.loads(RSA911_STATISTICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


# The outcome counts leave out blank outcomes (not employed), so their percentages are shares of
# employed exits by employment type. The model kept reporting those percentages as employment rates
# even when labeled, so this list shows counts only.
RSA911_OUTCOME_TYPE_HEADING = (
    "Type of employment among exits WITH a recorded employment outcome (counts only; these are not "
    "employment rates, use the employment rate line above):"
)
RSA911_OCCUPATION_HEADING = (
    "Most common occupations at exit (counts for the whole group; there are no employment rates by occupation):"
)


def _format_employment_rate(rate: Any) -> str:
    if not isinstance(rate, (int, float)):
        return "Employment rate at exit: suppressed or unavailable (rerun rsa911.private_aggregate)."
    return (
        f"EMPLOYMENT RATE at exit (blank outcome counted as not employed): {rate * 100:.1f}%. "
        "This is the only employment rate in this summary; quote this figure for any employment rate."
    )


def _format_count_rows(rows: List[Dict[str, Any]], limit: int = 8, show_percent: bool = True) -> str:
    lines: List[str] = []
    for row in rows[:limit]:
        label = str(row.get("label") or row.get("value") or "unknown")
        n = row.get("n")
        pct = row.get("percent")
        pct_text = f", {float(pct) * 100:.1f}%" if show_percent and isinstance(pct, (int, float)) else ""
        lines.append(f"- {label}: n={n}{pct_text}")
    return "\n".join(lines)


def _format_numeric_stats(label: str, stats: Any) -> str:
    if not isinstance(stats, dict):
        return f"- {label}: suppressed or unavailable"
    line = f"- {label}: n={stats.get('n')}, mean={stats.get('mean')}"
    if stats.get("p10") is not None and stats.get("p90") is not None:
        line += f", 10th percentile={stats.get('p10')}, 90th percentile={stats.get('p90')}"
    return line


@tool(
    name="query_rsa911_outcomes",
    description=(
        "Return privacy-safe aggregate RSA-911 vocational rehabilitation outcome patterns for veterans. "
        "Uses precomputed aggregate summaries only, with small-cell suppression. "
        "Input must be a plain natural-language veteran profile or outcome question."
    ),
)
def query_rsa911_outcomes(query: str) -> str:
    """
    Query precomputed RSA-911 aggregate outcomes without reading row-level records.

    Args:
        query: Veteran profile or outcome question.

    Returns:
        Aggregate outcome evidence with minimum cell suppression caveats.
    """
    if _rsa911_blocked():
        return RSA911_CLOUD_BLOCK
    data = _load_rsa911_aggregate()
    if not data:
        return (
            "RSA-911 aggregate outcomes unavailable. Run "
            "`python3 -m rsa911.private_aggregate --min-cell-size 11` first."
        )

    q = f" {str(query or '').lower()} "
    groups = data.get("key_disability_group_summaries") or {}
    matched_group_keys: List[str] = []
    for group_key, terms in _RSA911_GROUP_TERMS.items():
        if group_key in groups and any(re.search(rf"\b{re.escape(term)}\b", q) for term in terms):
            matched_group_keys.append(group_key)

    lines = [
        "RSA-911 Aggregate Outcomes Evidence",
        (
            f"Source: privacy-safe aggregate summary from {_rsa911_years_label()} RSA-911 exiter files. "
            f"Minimum cell size: {data.get('min_cell_size')}. No individual rows returned."
        ),
        f"Total veteran records in aggregate file: {data.get('total_veteran_records')}.",
        "",
        "Important interpretation limits:",
        "- Observational administrative data; do not treat patterns as causal evidence.",
        "- Counts describe exit records, not guaranteed unique people.",
        "- Occupation frequencies show observed outcomes, not proof of suitability for an individual.",
        "- Suppressed or missing sections mean cells were too small or unavailable.",
    ]

    if matched_group_keys:
        lines.append("\nMatched disability-group summaries:")
        for group_key in matched_group_keys:
            group = groups.get(group_key) or {}
            lines.append(f"\n{group.get('label', group_key)}: n={group.get('n')}")
            if group_key in _RSA911_GROUP_NOTES:
                lines.append(f"Note: {_RSA911_GROUP_NOTES[group_key]}")
            lines.append(_format_employment_rate(group.get("employment_rate_blank_as_not_employed")))
            outcome_rows = group.get("employment_outcome_exit_counts") or []
            if outcome_rows:
                lines.append(RSA911_OUTCOME_TYPE_HEADING)
                lines.append(_format_count_rows(outcome_rows, show_percent=False))
            occupation_rows = group.get("occupation_exit_counts") or []
            if occupation_rows:
                lines.append(RSA911_OCCUPATION_HEADING)
                lines.append(_format_count_rows(occupation_rows))
            lines.append("Positive wage/hour summaries at exit:")
            lines.append(_format_numeric_stats("hourly wage, positive values only", group.get("exit_hourly_wage_positive_stats")))
            lines.append(_format_numeric_stats("weekly hours, positive values only", group.get("exit_weekly_hours_positive_stats")))
    else:
        lines.append("\nOverall veteran aggregate patterns:")
        rows = data.get("primary_disability_source_counts") or []
        if rows:
            lines.append("Primary disability source distribution:")
            lines.append(_format_count_rows(rows, limit=8))
        lines.append(_format_employment_rate(data.get("employment_rate_blank_as_not_employed")))
        rows = data.get("employment_outcome_exit_counts") or []
        if rows:
            lines.append(RSA911_OUTCOME_TYPE_HEADING)
            lines.append(_format_count_rows(rows, show_percent=False))
        rows = data.get("occupation_exit_counts") or []
        if rows:
            lines.append(RSA911_OCCUPATION_HEADING)
            lines.append(_format_count_rows(rows, limit=10))
        lines.append("Positive wage/hour summaries at exit:")
        lines.append(_format_numeric_stats("hourly wage, positive values only", data.get("exit_hourly_wage_positive_stats")))
        lines.append(_format_numeric_stats("weekly hours, positive values only", data.get("exit_weekly_hours_positive_stats")))

    return "\n".join(line for line in lines if line is not None)


def _format_disparity_result(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key, item in (result.get("results") or {}).items():
        black = item.get("black_veterans") or {}
        nonblack = item.get("nonblack_veterans") or {}
        diff = item.get("rate_difference_black_minus_nonblack")
        lines.extend(
            [
                f"- {key}: {item.get('description')}",
                f"  Black veterans: n={black.get('n')}, successes={black.get('successes')}, rate={black.get('rate')}",
                f"  Non-Black veterans: n={nonblack.get('n')}, successes={nonblack.get('successes')}, rate={nonblack.get('rate')}",
                f"  Difference, Black minus non-Black: {diff}",
            ]
        )
    return "\n".join(lines)


def _format_predictor_result(result: Dict[str, Any]) -> str:
    if result.get("status") != "completed":
        return (
            f"Predictor model status: {result.get('status')}. "
            f"Reason: {result.get('reason', 'unavailable')} "
            f"n={result.get('n')}, positives={result.get('positive_cases')}, negatives={result.get('negative_cases')}"
        )
    lines = [
        f"Outcome definition: {result.get('outcome_definition')}",
        f"Outcome description: {result.get('outcome_description')}",
        f"Model: {result.get('model')}",
        f"n={result.get('n')}, positive cases={result.get('positive_cases')}, negative cases={result.get('negative_cases')}, positive rate={result.get('positive_rate')}",
        f"Holdout AUC: {result.get('auc_holdout')}",
        f"Warning: {result.get('interpretation_warning')}",
        "Features associated with higher predicted outcome:",
    ]
    for item in (result.get("features_associated_with_higher_predicted_employment") or [])[:10]:
        lines.append(f"- {item.get('feature')}: coefficient={item.get('coefficient')}")
    lines.append("Features associated with lower predicted outcome:")
    for item in (result.get("features_associated_with_lower_predicted_employment") or [])[:10]:
        lines.append(f"- {item.get('feature')}: coefficient={item.get('coefficient')}")
    return "\n".join(lines)


def _format_stat_feature_rows(rows: List[Dict[str, Any]], limit: int = 8) -> str:
    lines = []
    for item in rows[:limit]:
        feature = item.get("feature") or item.get("feature_a") or "unknown"
        score = item.get("score")
        method = item.get("method")
        if item.get("feature_b"):
            lines.append(f"- {feature} x {item.get('feature_b')}: score={score} ({method})")
        else:
            lines.append(f"- {feature}: score={score} ({method})")
        groups = item.get("reportable_groups") or []
        for group in groups[:3]:
            lines.append(
                "  "
                f"{group.get('label', group.get('code'))}: n={group.get('n')}, "
                f"employment_rate={group.get('employment_rate')}, employed_count={group.get('employed_count')}"
            )
    return "\n".join(lines)


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unavailable"
    return f"{float(value) * 100:.1f}%"


def _format_stat_association_sentence(item: Dict[str, Any]) -> str:
    feature = item.get("feature", "unknown")
    score = item.get("score")
    groups = item.get("reportable_groups") or []
    if not groups:
        return f"- `{feature}`: Cramer's V={score}."
    top = groups[0]
    bottom = groups[-1]
    return (
        f"- `{feature}`: Cramer's V={score}. Highest reportable group: "
        f"{top.get('label', top.get('code'))} (n={top.get('n')}, rate={_fmt_pct(top.get('employment_rate'))}); "
        f"lowest reportable group: {bottom.get('label', bottom.get('code'))} "
        f"(n={bottom.get('n')}, rate={_fmt_pct(bottom.get('employment_rate'))})."
    )


def _format_stat_numeric_sentence(item: Dict[str, Any]) -> str:
    feature = item.get("feature", "unknown")
    return (
        f"- `{feature}`: correlation={item.get('correlation_with_employed')}, "
        f"employed mean={item.get('employed_mean')}, "
        f"not-employed mean={item.get('not_employed_mean')}, "
        f"standardized mean difference={item.get('standardized_mean_difference')}."
    )


@tool(
    name="analyze_rsa911_statistics",
    description=(
        "Return privacy-safe RSA-911 statistical analysis summaries: descriptive statistics, "
        "feature importance, discriminant/association scores, feature-feature relationships, "
        "and provisional code interpretation notes. Uses aggregate/model summaries only."
    ),
)
def analyze_rsa911_statistics(query: str) -> str:
    """
    Return precomputed privacy-safe statistical analysis summaries.

    Args:
        query: Natural-language statistical analysis question.

    Returns:
        Aggregate/model-level statistical summary with caveats.
    """
    if _rsa911_blocked():
        return RSA911_CLOUD_BLOCK
    data = _load_rsa911_statistics()
    if not data:
        return (
            "RSA-911 statistical analysis summary unavailable. Run "
            "`python3 -m rsa911.statistical_analysis --min-cell-size 11` first."
        )

    descriptive = data.get("descriptive_statistics") or {}
    discriminant = data.get("feature_discriminant_analysis") or {}
    feature_importance = data.get("feature_importance") or {}
    relationships = data.get("feature_relationships") or []

    categorical_rows = discriminant.get("categorical_feature_scores") or []
    numeric_rows = discriminant.get("numeric_feature_scores") or []
    coefficient_rows = feature_importance.get("top_absolute_coefficients") or []

    lines = [
        "RSA-911 Statistical Analysis Summary",
        "",
        (
            f"The RSA-911 statistical analysis was run on {descriptive.get('veteran_records')} "
            "veteran records. The current provisional outcome definition treats "
            "`exitEmpOutcome` values 1-5 as employed and treats blanks/other values as not employed."
        ),
        (
            f"Under this definition, {descriptive.get('employed_count')} veterans had an employment "
            f"outcome, for an employment rate of {_fmt_pct(descriptive.get('employment_rate_blank_as_not_employed'))}."
        ),
        "No individual-level records are returned; this tool reports aggregate/model-level summaries only.",
        "",
        "Strongest categorical features discriminating employed vs not employed veterans:",
    ]

    for item in categorical_rows[:8]:
        lines.append(_format_stat_association_sentence(item))

    lines.extend(["", "Strongest numeric associations:"])
    for item in numeric_rows[:5]:
        lines.append(_format_stat_numeric_sentence(item))

    lines.extend(
        [
            "",
            "Feature-importance model:",
            (
                f"The model was {feature_importance.get('method')}. "
                f"The holdout AUC was {feature_importance.get('auc_holdout')}. "
                f"The positive outcome rate was {_fmt_pct(feature_importance.get('positive_rate'))}."
            ),
            "Largest absolute model coefficients:",
        ]
    )
    for item in coefficient_rows[:12]:
        lines.append(
            f"- `{item.get('feature')}`: coefficient={item.get('coefficient')}, "
            f"direction={item.get('direction')}."
        )

    lines.extend(
        [
            "",
            "Codebook-based interpretation notes:",
            "- For mapped yes/no/unknown fields such as `homelessorrunaway`, `longtermunemp`, `lowincomestatus`, `exoffenderstatus`, `singleparent`, and race/ethnicity indicators: 0=No, 1=Yes, 9=Unknown / not reported, and blank means missing.",
            "- `ipeempstatus` means employment status at Individualized Plan for Employment; IPE means Individualized Plan for Employment.",
            "- `vocrehab=1` means received services from the vocational rehabilitation program; it should not be described as VA VR&E unless the code is 2 or 3.",
            "- `primdisability` and `primdisabilitycause` labels are mapped from the RSA-911 codebook where available. Features marked `label not verified` should stay as raw codes until checked against the official codebook.",
            "- Service indicator fields ending in `vragencystaff` or `vragencypurchase` are VR service-delivery/payment indicators in the RSA-911 file. The current mapper does not verify every service prefix, so raw prefixes such as `ceo`, `isa`, `jec`, or `ses` should not be expanded without codebook confirmation.",
        ]
    )

    lines.extend(["", "Feature-feature relationships:"])
    for item in relationships[:8]:
        lines.append(
            f"- `{item.get('feature_a')}` x `{item.get('feature_b')}`: "
            f"Cramer's V={item.get('score')}, n={item.get('n')}."
        )

    lines.extend(
        [
            "",
            "Important caveats:",
            "- These findings are associational, not causal.",
            "- The outcome definition is provisional because blank `exitEmpOutcome` is currently treated as not employed.",
            "- Mapped code labels are based on the local RSA-911 codebook mapping. Raw codes marked `label not verified` need official codebook review before publication.",
            "- Structural race/ethnicity indicator pairs are filtered from the displayed feature-feature relationships because those associations mostly reflect coding design.",
            "- Service indicators are not causal treatment effects. They can reflect eligibility, case severity, documentation, service availability, and timing, so they should be interpreted as associations with recorded service patterns.",
            "- Missing values can appear strongly associated with employment, but this may reflect documentation/reporting patterns rather than true veteran characteristics.",
            "- Feature importance is model-dependent and should not be converted into individual-level prescriptions without domain review.",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


@tool(
    name="analyze_rsa911_question",
    description=(
        "Return privacy-safe RSA-911 disparity or predictor analysis summaries for dataset questions, "
        "especially questions about Black veterans and employment outcomes. "
        "Uses precomputed aggregate/model summaries only and returns no individual rows."
    ),
)
def analyze_rsa911_question(query: str) -> str:
    """
    Return precomputed privacy-safe RSA-911 analysis summaries.

    Args:
        query: Natural-language question about disparity or predictors.

    Returns:
        Aggregate/model analysis summary with methodological caveats.
    """
    if _rsa911_blocked():
        return RSA911_CLOUD_BLOCK
    data = _load_rsa911_analysis()
    if not data:
        return (
            "RSA-911 analysis summary unavailable. Run "
            "`python3 -m rsa911.analysis --min-cell-size 11` first."
        )

    q = str(query or "").lower()
    wants_predictors = any(term in q for term in ["predict", "predictor", "feature", "factor", "model", "why"])
    wants_black = "black" in q or "race" in q

    lines = [
        "RSA-911 Privacy-Safe Analysis Summary",
        f"Veteran records analyzed: {data.get('veteran_records_loaded')}",
        f"Minimum cell size: {data.get('min_cell_size')}",
        "No individual rows returned.",
        "Core caveat: outcome definition is not final until a domain expert confirms how to treat blank exitEmpOutcome values.",
        "",
    ]

    if wants_black:
        disparity = data.get("black_disparity") or {}
        lines.append("Black veteran employment comparison:")
        lines.append(str(disparity.get("interpretation_warning") or "Unadjusted descriptive comparison."))
        lines.append(_format_disparity_result(disparity))
        lines.append("")

    if wants_predictors:
        lines.append("Predictor model for Black veterans, using blank exitEmpOutcome as not employed:")
        lines.append(_format_predictor_result(data.get("black_predictors_blank_as_not_employed") or {}))
        lines.append("")
        lines.append("Predictor model for Black veterans, using only populated exitEmpOutcome:")
        lines.append(_format_predictor_result(data.get("black_predictors_known_exit_employment") or {}))

    if not wants_black and not wants_predictors:
        lines.append("Available analyses:")
        lines.append("- Black-vs-non-Black veteran employment comparison under multiple outcome definitions.")
        lines.append("- Predictor model for Black veterans under the blank-as-not-employed outcome definition.")
        lines.append("- Known-exit-employment predictor model is suppressed because there are not enough negative cases.")

    return "\n".join(line for line in lines if line is not None)


@tool(
    name="describe_rsa911_sql_schema",
    description=(
        "Describe the allowed RSA-911 SQL table, columns, outcome definitions, and privacy rules. "
        "Use this before writing SQL for run_rsa911_safe_sql. Input can be a short topic string."
    ),
)
def describe_rsa911_sql_schema(topic: str = "") -> str:
    """
    Return allowed schema and rules for safe RSA-911 aggregate SQL.

    Args:
        topic: Optional analysis topic.

    Returns:
        JSON schema/rules for SQL generation.
    """
    if _rsa911_blocked():
        return RSA911_CLOUD_BLOCK
    try:
        from rsa911.dp import DPConfig, prompt_rules
        from rsa911.safe_sql import DEFAULT_MIN_CELL_SIZE, schema_for_prompt

        schema = schema_for_prompt(topic=str(topic or ""))
        dp_config = DPConfig.from_env()
        if dp_config:
            schema["rules"].extend(prompt_rules(dp_config, DEFAULT_MIN_CELL_SIZE))
        return json.dumps(schema, indent=2)
    except Exception as exc:
        return f"RSA-911 SQL schema unavailable: {type(exc).__name__}: {exc}"


@tool(
    name="run_rsa911_safe_sql",
    description=(
        "Run validated aggregate-only SQL against the private RSA-911 SQLite database. "
        "Input must be one flat SELECT query over rsa911_exiters with COUNT(*) AS n in the select list. "
        "No subqueries, joins, window functions, MIN, or MAX; SUM only as a count of CASE ... THEN 1 ELSE 0. "
        "The tool blocks row-level outputs, returns only result rows with n >= 11, nulls counts and rates "
        "that imply fewer than 11 people. Results include small random noise (differential privacy), "
        "so report them as estimates. HAVING is not allowed."
    ),
)
def run_rsa911_safe_sql(sql: str) -> str:
    """
    Execute safe aggregate SQL with n >= 11 enforcement and differential privacy noise.

    Args:
        sql: One aggregate SELECT query over rsa911_exiters.

    Returns:
        JSON aggregate result or validation error. No individual rows are returned.
    """
    if _rsa911_blocked():
        return RSA911_CLOUD_BLOCK
    try:
        from rsa911.dp import DPConfig
        from rsa911.safe_sql import run_safe_sql

        result = run_safe_sql(str(sql or ""), dp_config=DPConfig.from_env())
        return "RSA-911 Safe SQL Aggregate Result\n" + json.dumps(result, indent=2)
    except Exception as exc:
        return f"RSA-911 safe SQL rejected or failed: {type(exc).__name__}: {exc}"


def _trim(text: str, limit: int = 2500) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _safe_external_lookup(label: str, fn, *args, **kwargs) -> str:
    try:
        callable_fn = getattr(fn, "entrypoint", fn)
        result = callable_fn(*args, **kwargs)
    except Exception as exc:
        return f"{label} unavailable: {type(exc).__name__}: {exc}"
    if not result:
        return f"{label} returned no results."
    return _trim(str(result))


def _next_source_index(text: str) -> int:
    matches = [int(m) for m in re.findall(r"\[Source (\d+)\]", text)]
    return (max(matches) + 1) if matches else 1


_ONET_MEDICAL_TERMS = {
    "adhd", "amputee", "amputation", "anxiety", "arthritis", "autism", "blind", "cardiovascular",
    "chronic", "cognitive", "deaf", "depression", "depressive", "diabetes", "disability",
    "disabled", "fatigue", "hearing", "hearingloss", "hearing-loss", "injury", "loss",
    "major", "medical", "migraines", "mobility", "ms", "pain", "paralysis", "ptsd",
    "seizure", "spinal", "sci", "stress", "tbi", "therapy", "trauma", "traumatic",
    "vision", "wheelchair",
}

_ONET_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "best", "but", "by", "can", "could",
    "do", "for", "from", "get", "help", "i", "i'm", "if", "in", "into", "is", "it", "jobs",
    "kind", "looking", "me", "my", "of", "on", "or", "pathway", "pathways", "role", "roles",
    "should", "stable", "that", "the", "to", "want", "what", "which", "with", "work",
}

_ONET_WORK_PHRASES = [
    "administrative assistant",
    "program analyst",
    "records management",
    "customer service",
    "data entry",
    "data analyst",
    "project coordinator",
    "case manager",
    "medical coding",
    "claims processing",
    "office support",
    "human resources",
    "information technology",
    "help desk",
    "computer support",
    "it support",
    "software development",
    "cybersecurity",
    "supply chain",
    "logistics",
    "healthcare administration",
    "federal jobs",
    "government jobs",
]

# Words that describe the person, the working conditions, or the request, not an occupation.
# O*NET's keyword search matches words literally ("remote" -> Remote Sensing Technicians,
# "mobility" -> Orientation and Mobility Specialists), so these are removed before searching.
_ONET_NON_OCCUPATION_TERMS = _ONET_MEDICAL_TERMS | {
    "remote", "remotely", "telework", "teleworking", "telecommute", "hybrid", "home", "work-from-home",
    "wfh", "virtual", "onsite", "on-site", "sedentary", "seated", "sitting", "standing", "flexible",
    "schedule", "schedules", "part-time", "full-time", "structured", "independent", "quiet", "low",
    "light", "physical", "physically", "demand", "demands", "demanding", "friendly", "compatible",
    "veteran", "veterans", "army", "navy", "marine", "marines", "air", "force", "military", "service",
    "service-connected", "connected", "u.s", "impairment", "impairments", "impaired", "limitation",
    "limitations", "condition", "conditions", "disorder", "symptoms", "accommodation", "accommodations",
    "occupation", "occupations", "career", "careers", "job", "employment", "position", "positions",
    "options", "ideas", "suitable", "possible", "good", "type", "types", "related", "based", "using",
    "experience", "skills", "background", "interested", "prefer", "preferred", "like", "would", "need",
    "needs", "wants", "such", "some", "other", "this", "these", "those", "who", "also", "has", "have",
    "had", "was", "were", "will", "about", "after", "before", "than", "then", "there", "where", "when",
    "how", "why", "not", "years", "year", "current", "currently", "former", "previous", "prior",
    "returning", "return", "search", "find", "options", "relevant", "evidence", "onet", "o*net",
}

# Everyday phrases O*NET's keyword search misreads, mapped to its own occupation wording
# ("help desk" alone returns Hotel Desk Clerks first).
_ONET_PHRASE_ALIASES = {
    "help desk": "computer user support",
    "it support": "computer user support",
    "computer support": "computer user support",
}

# Searched when a query names no occupation at all (only conditions or preferences).
_ONET_DEFAULT_SEARCHES = ["customer service", "computer user support", "claims processing", "administrative assistant"]

_ONET_WORK_TOKENS = {
    "accounting", "administrative", "analyst", "assistant", "billing", "claims", "clerical",
    "clerk", "coding", "compliance", "computer", "coordinator", "counselor", "customer",
    "data", "design", "education", "engineer", "federal", "finance", "government",
    "healthcare", "hr", "human", "information", "it", "legal", "logistics", "management",
    "manager", "office", "operations", "planner", "processing", "program", "project",
    "records", "research", "sales", "schedule", "security", "service", "specialist",
    "support", "systems", "teacher", "technical", "technician", "training", "writing",
}

_PROFILE_LIMITATION_MARKERS = [
    "difficulty", "difficult", "trouble", "cannot", "can't", "unable", "limited", "limit",
    "fatigue", "pain", "hearing", "vision", "mobility", "anxiety", "depression", "ptsd",
    "tbi", "stress", "memory", "concentration",
]

_PROFILE_GOAL_MARKERS = [
    "interested in", "looking for", "want", "prefer", "open to", "goal", "seeking",
]


def _ordered_unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(item.strip() for item in items if item and item.strip()))


def _extract_marked_clauses(message: str, markers: List[str], limit: int = 3) -> List[str]:
    clauses = re.split(r"[\n.;]", message)
    out: List[str] = []
    for clause in clauses:
        text = _clean(clause)
        lowered = text.lower()
        if not text:
            continue
        if any(marker in lowered for marker in markers):
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _build_onet_query(message: str) -> str:
    q = message.lower()
    terms: List[str] = []

    for phrase in _ONET_WORK_PHRASES:
        if phrase in q:
            terms.append(phrase)

    tokens = re.findall(r"[a-z0-9][a-z0-9+/.-]*", q)
    for token in tokens:
        normalized = token.strip(".-")
        if (
            normalized
            and normalized not in _ONET_STOPWORDS
            and normalized not in _ONET_NON_OCCUPATION_TERMS
            and normalized in _ONET_WORK_TOKENS
        ):
            terms.append(normalized)

    if not terms:
        fallback = [
            token.strip(".-")
            for token in tokens
            if len(token.strip(".-")) >= 4
            and token.strip(".-") not in _ONET_STOPWORDS
            and token.strip(".-") not in _ONET_NON_OCCUPATION_TERMS
        ]
        terms.extend(fallback[:6])

    terms = _ordered_unique(terms)
    if not terms:
        terms = ["employment", "occupations"]
    return " ".join(terms[:10])


def _onet_search_terms(query: str, limit: int = 3) -> List[str]:
    """Split a free-text query into separate occupation searches, dropping condition and preference words."""
    q = query.lower()
    phrases = [phrase for phrase in _ONET_WORK_PHRASES if phrase in q]
    for phrase in phrases:
        q = q.replace(phrase, " ")
    searches = [_ONET_PHRASE_ALIASES.get(phrase, phrase) for phrase in phrases]
    words = []
    for token in re.findall(r"[a-z0-9*][a-z0-9*+/.-]*", q):
        word = token.strip(".-")
        if word in _ONET_STOPWORDS or word in _ONET_NON_OCCUPATION_TERMS:
            continue
        if len(word) >= 3 or word in _ONET_WORK_TOKENS:
            words.append(word)
    if words:
        searches.append(" ".join(_ordered_unique(words)[:4]))
    return _ordered_unique(searches)[:limit]


def _build_profile_notes(message: str) -> str:
    notes: List[str] = []

    limitation_clauses = _extract_marked_clauses(message, _PROFILE_LIMITATION_MARKERS, limit=3)
    goal_clauses = _extract_marked_clauses(message, _PROFILE_GOAL_MARKERS, limit=3)

    for clause in limitation_clauses:
        notes.append(f"reported limitation or accommodation need: {clause}")

    for clause in goal_clauses:
        notes.append(f"stated work preference or goal: {clause}")

    onet_focus = _build_onet_query(message)
    if onet_focus and onet_focus not in {"employment occupations", "employment"}:
        notes.append(f"O*NET search focus should stay occupation-oriented: {onet_focus}")

    if not notes:
        notes.append("infer functional limits conservatively from the user profile and keep occupation suggestions broad when evidence is weak")

    return "\n".join(f"- {n}" for n in _ordered_unique(notes))


def _collect_fallback_evidence(message: str) -> str:
    local_block = search_local_corpus(message)
    next_idx = _next_source_index(local_block)
    onet_query = _build_onet_query(message)

    blocks = [local_block]
    external_queries = [
        (
            "VA and employment web results",
            _web_tools.web_search,
            f"site:va.gov OR site:benefits.va.gov veteran employment VR&E {message}",
            5,
        ),
        (
            "JAN accommodation web results",
            search_jan_accommodations,
            message,
        ),
        (
            "PubMed research results",
            _pubmed_tools.search_pubmed,
            _build_pubmed_eval_query(message),
            5,
        ),
        (
            "RSA-911 aggregate outcomes",
            query_rsa911_outcomes,
            message,
        ),
        (
            "RSA-911 disparity and predictor analysis",
            analyze_rsa911_question,
            message,
        ),
        (
            "O*NET occupation matches",
            search_onet,
            onet_query,
        ),
    ]

    for item in external_queries:
        label, fn, *fn_args = item
        block = _safe_external_lookup(label, fn, *fn_args)
        blocks.append(f"[Source {next_idx}] {label}\n{block}")
        next_idx += 1

    return "\n\n---\n\n".join(blocks)


FALLBACK_SYNTHESIS_PROMPT = """
You are a Veterans Employment Pathways Agent.

Use only the evidence below to answer the user. If a source is missing, weak, or unavailable,
say so explicitly. Do not invent citations. Every substantive claim must cite at least one
source using the provided [Source X] labels.

Suitability rules:
- Match occupations to the veteran's functional limits, not just keywords.
- Do not recommend physically intensive or access-constrained jobs for a wheelchair user unless the evidence explicitly supports that fit.
- Do not recommend roles with heavy lifting, patient transfers, prolonged standing/walking, crowd management, or direct bedside care unless explicitly justified by evidence.
- Treat O*NET/web matches as candidate titles only. They are not enough by themselves to prove suitability.
- If the evidence does not support a specific occupation, say so and keep the recommendation conservative.
- Prefer accessible, seated, structured, and accommodation-friendly pathways when the profile suggests mobility or cognitive limitations.

For each occupation pathway, include:
- occupation title
- why it fits this profile
- key accommodation or service route
- source citations

Respond with these sections:
VETERAN PROFILE SUMMARY
RECOMMENDED VA VR&E TRACK
EMPLOYMENT PATHWAYS
IMMEDIATE NEXT STEPS
EVIDENCE GAPS
SOURCES
"""


def build_fallback_agent(session_id: str = "default") -> Agent:
    return Agent(
        model=_build_model(),
        description="Veterans Employment Pathways Agent fallback synthesizer",
        instructions=FALLBACK_SYNTHESIS_PROMPT,
        session_id=f"{session_id}-fallback",
    )


def _run_fallback_agent(message: str, session_id: str, error: Exception) -> str:
    evidence = _collect_fallback_evidence(message)
    profile_notes = _build_profile_notes(message)
    prompt = (
        f"The tool-enabled run failed before tool execution.\n"
        f"Error summary: {type(error).__name__}: {error}\n\n"
        f"User request:\n{message}\n\n"
        f"Profile constraints:\n{profile_notes}\n\n"
        f"Evidence:\n{evidence}"
    )

    fallback_agent = build_fallback_agent(session_id=session_id)
    result = fallback_agent.run(prompt, stream=False)
    if result is None:
        return "No response from fallback agent."
    if hasattr(result, "content"):
        return result.content or str(result)
    return str(result)


def _is_tool_use_failure(exc: Exception) -> bool:
    msg = str(exc)
    return "tool_use_failed" in msg or "failed_generation" in msg


def _run_output_has_tool_use_failure(result) -> bool:
    if result is None:
        return False
    status = getattr(result, "status", None)
    content = getattr(result, "content", None)
    status_is_error = status == RunStatus.error or str(status).endswith("error")
    content_text = str(content or "")
    return status_is_error and _is_tool_use_failure(Exception(content_text))


def _extract_local_sources(result_text: str) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for line in result_text.splitlines():
        line = line.strip()
        if line.startswith("[Source ") and "]" in line:
            label_end = line.find("]")
            descriptor = line[label_end + 1 :].strip()
            if descriptor:
                sources.append({"kind": "local", "descriptor": descriptor})
    return sources


def _extract_arxiv_sources(result_text: str, limit: int = 3) -> List[Dict[str, str]]:
    parsed = _safe_json_loads(result_text)
    if not isinstance(parsed, list):
        return []
    sources: List[Dict[str, str]] = []
    for article in parsed[:limit]:
        if not isinstance(article, dict):
            continue
        title = article.get("title", "Unknown title")
        authors = article.get("authors") or []
        published = article.get("published") or ""
        author_text = ", ".join(authors[:3]) if isinstance(authors, list) else str(authors)
        descriptor = f"{title}"
        if author_text:
            descriptor += f" | {author_text}"
        if published:
            descriptor += f" | {published[:10]}"
        sources.append({"kind": "arxiv", "descriptor": descriptor})
    return sources


def _parse_pubmed_entry(entry_text: str) -> Dict[str, Any]:
    article: Dict[str, Any] = {}
    current_key = None
    summary_lines: List[str] = []
    for raw_line in entry_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Title:"):
            article["title"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("Published:"):
            article["published"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("First Author:"):
            article["authors"] = [line.split(":", 1)[1].strip()]
            current_key = None
        elif line.startswith("Journal:"):
            article["journal"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("Publication Type:"):
            article["publication_type"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("DOI:"):
            article["doi"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("PubMed URL:"):
            article["entry_id"] = line.split(":", 1)[1].strip()
            article["pubmed_url"] = article["entry_id"]
            current_key = None
        elif line.startswith("Full Text URL:"):
            article["full_text_url"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("MeSH Terms:"):
            article["mesh_terms"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("Summary:"):
            current_key = "summary"
            summary_lines.append(line.split(":", 1)[1].strip())
        elif current_key == "summary":
            summary_lines.append(line)
    article["summary"] = " ".join(part for part in summary_lines if part).strip()
    return article


def _extract_pubmed_articles(result_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    parsed = _safe_json_loads(result_text)
    if not isinstance(parsed, list):
        return []
    papers: List[Dict[str, Any]] = []
    for entry in parsed[:limit]:
        if not isinstance(entry, str):
            continue
        article = _parse_pubmed_entry(entry)
        if article.get("title"):
            papers.append(article)
    return papers


def _extract_pubmed_sources(result_text: str, limit: int = 3) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for article in _extract_pubmed_articles(result_text, limit=limit):
        title = article.get("title", "Unknown title")
        authors = article.get("authors") or []
        published = article.get("published") or ""
        doi = str(article.get("doi") or "").strip()
        pubmed_url = str(article.get("pubmed_url") or article.get("entry_id") or "").strip()
        full_text_url = str(article.get("full_text_url") or "").strip()
        author_text = ", ".join(authors[:3]) if isinstance(authors, list) else str(authors)
        descriptor_parts = [str(title)]
        if author_text:
            descriptor_parts.append(author_text)
        if published:
            descriptor_parts.append(published[:10])
        if doi:
            descriptor_parts.append(f"DOI: {doi}")
            descriptor_parts.append(f"https://doi.org/{doi}")
        if full_text_url:
            descriptor_parts.append(full_text_url)
        elif pubmed_url:
            descriptor_parts.append(pubmed_url)
        descriptor = " | ".join(part for part in descriptor_parts if part)
        sources.append({"kind": "pubmed", "descriptor": descriptor})
    return sources


def _extract_web_sources(result_text: str, limit: int = 3) -> List[Dict[str, str]]:
    parsed = _safe_json_loads(result_text)
    if not isinstance(parsed, list):
        return []
    sources: List[Dict[str, str]] = []
    for item in parsed[:limit]:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("heading") or "Web result"
        href = item.get("href") or item.get("url") or ""
        descriptor = f"{title}"
        if href:
            descriptor += f" | {href}"
        sources.append({"kind": "web", "descriptor": descriptor})
    return sources


def _extract_fetched_page_result(result_text: str) -> Dict[str, str]:
    page: Dict[str, str] = {}
    current_key = None
    body_lines: List[str] = []
    for raw_line in result_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Title:"):
            page["title"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("URL:"):
            page["url"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("Extracted Text:"):
            current_key = "body"
        elif current_key == "body":
            body_lines.append(line)
    if body_lines:
        page["body"] = " ".join(body_lines).strip()
    return page


def _extract_fetched_page_sources(result_text: str) -> List[Dict[str, str]]:
    page = _extract_fetched_page_result(result_text)
    title = page.get("title", "Fetched web page")
    url = page.get("url", "")
    descriptor = title
    if url:
        descriptor += f" | {url}"
    return [{"kind": "web_page", "descriptor": descriptor}]


def _extract_fetched_paper_result(result_text: str) -> Dict[str, str]:
    paper: Dict[str, str] = {}
    current_key = None
    body_lines: List[str] = []
    for raw_line in result_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Title:"):
            paper["title"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("URL:"):
            paper["url"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("Access Type:"):
            paper["access_type"] = line.split(":", 1)[1].strip()
            current_key = None
        elif line.startswith("Extracted Text:"):
            current_key = "body"
        elif current_key == "body":
            body_lines.append(line)
    if body_lines:
        paper["body"] = " ".join(body_lines).strip()
    return paper


def _extract_fetched_paper_sources(result_text: str) -> List[Dict[str, str]]:
    paper = _extract_fetched_paper_result(result_text)
    title = paper.get("title", "Fetched academic paper")
    url = paper.get("url", "")
    descriptor = title
    if url:
        descriptor += f" | {url}"
    return [{"kind": "paper_full_text", "descriptor": descriptor}]


def _extract_onet_sources(result_text: str, limit: int = 5) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for line in result_text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        descriptor = line[2:].strip()
        if descriptor:
            sources.append({"kind": "onet", "descriptor": descriptor})
        if len(sources) >= limit:
            break
    return sources


def _extract_rsa911_sources(result_text: str) -> List[Dict[str, str]]:
    if (
        "RSA-911 Aggregate Outcomes Evidence" not in result_text
        and "RSA-911 Privacy-Safe Analysis Summary" not in result_text
        and "RSA-911 Safe SQL Aggregate Result" not in result_text
        and "RSA-911 Statistical Analysis Summary" not in result_text
    ):
        return []
    if "RSA-911 Safe SQL Aggregate Result" in result_text:
        descriptor = "RSA-911 safe aggregate SQL result, n >= 11 enforced"
    elif "RSA-911 Statistical Analysis Summary" in result_text:
        descriptor = f"RSA-911 statistical analysis summary, {_rsa911_years_label()} veteran exiters, privacy-safe aggregate/model output"
    elif "RSA-911 Privacy-Safe Analysis Summary" in result_text:
        descriptor = f"RSA-911 privacy-safe disparity/predictor analysis summary, {_rsa911_years_label()} exiters"
    else:
        descriptor = f"RSA-911 aggregate outcomes summary, {_rsa911_years_label()} exiters, privacy-suppressed"
    min_cell = re.search(r"Minimum cell size:\s*([0-9]+)", result_text)
    if min_cell:
        descriptor += f" | minimum cell size {min_cell.group(1)}"
    return [{"kind": "rsa911_aggregate", "descriptor": descriptor}]


def _build_source_registry(tool_executions: List[Any]) -> List[Dict[str, str]]:
    registry: List[Dict[str, str]] = []
    for tool_exec in tool_executions:
        tool_name = getattr(tool_exec, "tool_name", "") or ""
        result_text = getattr(tool_exec, "result", "") or ""
        if not result_text:
            continue
        if tool_name == "search_local_corpus":
            registry.extend(_extract_local_sources(result_text)[:6])
        elif "pubmed" in tool_name.lower():
            registry.extend(_extract_pubmed_sources(result_text))
        elif "arxiv" in tool_name.lower():
            registry.extend(_extract_arxiv_sources(result_text))
        elif tool_name == "fetch_paper_full_text":
            registry.extend(_extract_fetched_paper_sources(result_text))
        elif tool_name == "fetch_web_page":
            registry.extend(_extract_fetched_page_sources(result_text))
        elif (
            "duckduckgo" in tool_name.lower()
            or "web_search" in tool_name.lower()
            or tool_name in {"search_web", "search_jan_accommodations"}
        ):
            registry.extend(_extract_web_sources(result_text))
        elif tool_name == "search_onet":
            registry.extend(_extract_onet_sources(result_text))
        elif tool_name in {
            "query_rsa911_outcomes",
            "analyze_rsa911_question",
            "analyze_rsa911_statistics",
            "run_rsa911_safe_sql",
        }:
            registry.extend(_extract_rsa911_sources(result_text))

    seen = set()
    deduped: List[Dict[str, str]] = []
    for item in registry:
        key = (item["kind"], item["descriptor"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _tool_exec_is_forced_eval(tool_exec: Any) -> bool:
    tool_args = getattr(tool_exec, "tool_args", None) or {}
    return bool(isinstance(tool_args, dict) and tool_args.get("forced_for_eval"))


def _extract_arxiv_articles(result_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    parsed = _safe_json_loads(result_text)
    if not isinstance(parsed, list):
        return []
    papers: List[Dict[str, Any]] = []
    for article in parsed[:limit]:
        if not isinstance(article, dict):
            continue
        papers.append(
            {
                "title": article.get("title"),
                "authors": article.get("authors"),
                "published": article.get("published"),
                "summary": article.get("summary"),
                "entry_id": article.get("entry_id"),
                "pdf_url": article.get("pdf_url"),
            }
        )
    return papers


def _extract_web_results(result_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    parsed = _safe_json_loads(result_text)
    if not isinstance(parsed, list):
        return []
    results: List[Dict[str, Any]] = []
    for item in parsed[:limit]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title") or item.get("heading"),
                "url": item.get("href") or item.get("url") or item.get("link"),
                "snippet": item.get("body"),
            }
        )
    return results


def _extract_fetched_page_results(result_text: str) -> List[Dict[str, Any]]:
    page = _extract_fetched_page_result(result_text)
    if not page:
        return []
    return [
        {
            "title": page.get("title") or "Fetched web page",
            "url": page.get("url"),
            "body": page.get("body") or "",
        }
    ]


def _extract_fetched_paper_results(result_text: str) -> List[Dict[str, Any]]:
    paper = _extract_fetched_paper_result(result_text)
    if not paper:
        return []
    return [
        {
            "title": paper.get("title") or "Fetched academic paper",
            "url": paper.get("url"),
            "access_type": paper.get("access_type") or "",
            "body": paper.get("body") or "",
        }
    ]


def _extract_tool_artifacts(tool_executions: List[Any]) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {
        "retrieved_papers": [],
        "retrieved_papers_native": [],
        "retrieved_papers_forced_eval": [],
        "paper_full_texts": [],
        "local_sources": [],
        "web_results": [],
        "web_pages": [],
        "onet_results": [],
        "rsa911_outcomes": [],
        "rsa911_analyses": [],
        "rsa911_statistics": [],
        "rsa911_sql_results": [],
        "local_contexts": [],
        "tool_calls": [],
        "tool_calls_native": [],
        "tool_calls_forced_eval": [],
        "retrieved_contexts": [],
        "retrieved_contexts_native": [],
        "retrieved_contexts_forced_eval": [],
    }
    for tool_exec in tool_executions:
        tool_name = getattr(tool_exec, "tool_name", "") or ""
        tool_args = getattr(tool_exec, "tool_args", None) or {}
        result_text = getattr(tool_exec, "result", "") or ""
        forced_for_eval = _tool_exec_is_forced_eval(tool_exec)
        tool_record = {"tool_name": tool_name, "tool_args": tool_args, "forced_for_eval": forced_for_eval}
        artifacts["tool_calls"].append(tool_record)
        if forced_for_eval:
            artifacts["tool_calls_forced_eval"].append(tool_record)
        else:
            artifacts["tool_calls_native"].append(tool_record)
        if not result_text:
            continue
        if tool_name == "search_local_corpus":
            artifacts["local_sources"].extend(_extract_local_sources(result_text))
            for block in result_text.split("\n\n---\n\n")[:6]:
                block = block.strip()
                if block:
                    artifacts["local_contexts"].append(block[:1800])
                    artifacts["retrieved_contexts"].append(block[:1800])
                    artifacts["retrieved_contexts_native"].append(block[:1800])
        elif "pubmed" in tool_name.lower():
            papers = _extract_pubmed_articles(result_text)
            artifacts["retrieved_papers"].extend(papers)
            if forced_for_eval:
                artifacts["retrieved_papers_forced_eval"].extend(papers)
            else:
                artifacts["retrieved_papers_native"].extend(papers)
            for paper in papers:
                title = str(paper.get("title") or "Unknown title")
                summary = str(paper.get("summary") or "").strip().replace("\n", " ")
                text = f"{title}: {summary}".strip()
                if text:
                    artifacts["retrieved_contexts"].append(text[:1800])
                    if forced_for_eval:
                        artifacts["retrieved_contexts_forced_eval"].append(text[:1800])
                    else:
                        artifacts["retrieved_contexts_native"].append(text[:1800])
        elif "arxiv" in tool_name.lower():
            papers = _extract_arxiv_articles(result_text)
            artifacts["retrieved_papers"].extend(papers)
            if forced_for_eval:
                artifacts["retrieved_papers_forced_eval"].extend(papers)
            else:
                artifacts["retrieved_papers_native"].extend(papers)
            for paper in papers:
                title = str(paper.get("title") or "Unknown title")
                summary = str(paper.get("summary") or "").strip().replace("\n", " ")
                text = f"{title}: {summary}".strip()
                if text:
                    artifacts["retrieved_contexts"].append(text[:1800])
                    if forced_for_eval:
                        artifacts["retrieved_contexts_forced_eval"].append(text[:1800])
                    else:
                        artifacts["retrieved_contexts_native"].append(text[:1800])
        elif tool_name == "fetch_paper_full_text":
            papers = _extract_fetched_paper_results(result_text)
            artifacts["paper_full_texts"].extend(papers)
            for paper in papers:
                title = str(paper.get("title") or "Fetched academic paper")
                url = str(paper.get("url") or "")
                body = str(paper.get("body") or "").strip().replace("\n", " ")
                text = f"{title} | {url} | {body}".strip()
                if text:
                    artifacts["retrieved_contexts"].append(text[:1800])
                    artifacts["retrieved_contexts_native"].append(text[:1800])
        elif tool_name == "fetch_web_page":
            pages = _extract_fetched_page_results(result_text)
            artifacts["web_pages"].extend(pages)
            for page in pages:
                title = str(page.get("title") or "Fetched web page")
                url = str(page.get("url") or "")
                body = str(page.get("body") or "").strip().replace("\n", " ")
                text = f"{title} | {url} | {body}".strip()
                if text:
                    artifacts["retrieved_contexts"].append(text[:1800])
                    artifacts["retrieved_contexts_native"].append(text[:1800])
        elif (
            "duckduckgo" in tool_name.lower()
            or "web_search" in tool_name.lower()
            or tool_name in {"search_web", "search_jan_accommodations"}
        ):
            artifacts["web_results"].extend(_extract_web_results(result_text))
            for item in _extract_web_results(result_text):
                title = str(item.get("title") or "Web result")
                url = str(item.get("url") or "")
                snippet = str(item.get("snippet") or "").strip().replace("\n", " ")
                text = f"{title} | {url} | {snippet}".strip()
                if text:
                    artifacts["retrieved_contexts"].append(text[:1800])
                    artifacts["retrieved_contexts_native"].append(text[:1800])
        elif tool_name == "search_onet":
            artifacts["onet_results"].extend(_extract_onet_sources(result_text))
            for item in _extract_onet_sources(result_text):
                descriptor = str(item.get("descriptor") or "").strip()
                if descriptor:
                    artifacts["retrieved_contexts"].append(descriptor[:1800])
                    artifacts["retrieved_contexts_native"].append(descriptor[:1800])
        elif tool_name == "query_rsa911_outcomes":
            trimmed = _trim(result_text, limit=3000)
            artifacts["rsa911_outcomes"].append(trimmed)
            artifacts["retrieved_contexts"].append(trimmed)
            artifacts["retrieved_contexts_native"].append(trimmed)
        elif tool_name == "analyze_rsa911_question":
            trimmed = _trim(result_text, limit=4000)
            artifacts["rsa911_analyses"].append(trimmed)
            artifacts["retrieved_contexts"].append(trimmed)
            artifacts["retrieved_contexts_native"].append(trimmed)
        elif tool_name == "analyze_rsa911_statistics":
            trimmed = _trim(result_text, limit=5000)
            artifacts["rsa911_statistics"].append(trimmed)
            artifacts["retrieved_contexts"].append(trimmed)
            artifacts["retrieved_contexts_native"].append(trimmed)
        elif tool_name == "run_rsa911_safe_sql":
            trimmed = _trim(result_text, limit=4000)
            artifacts["rsa911_sql_results"].append(trimmed)
            artifacts["retrieved_contexts"].append(trimmed)
            artifacts["retrieved_contexts_native"].append(trimmed)
    return artifacts


def _load_manual_runs() -> List[Dict[str, Any]]:
    if not MANUAL_RUNS_PATH.exists():
        return []
    try:
        data = json.loads(MANUAL_RUNS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        answer = str(item.get("answer", ""))
        question = str(item.get("question", ""))
        if "PASTE THE FINAL ANSWER HERE" in answer or not question.strip():
            continue
        cleaned.append(item)
    return cleaned


def _append_manual_run(
    *,
    message: str,
    session_id: str,
    response: str,
    source_registry: List[Dict[str, str]],
    tool_executions: List[Any],
    status: str,
    error: str = "",
) -> None:
    if not AUTO_LOG_MANUAL_RUNS:
        return

    artifacts = _extract_tool_artifacts(tool_executions)
    runs = _load_manual_runs()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{session_id}"
    record = {
        "id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "status": status,
        "error": error,
        "question": message,
        "answer": response,
        "context": _format_source_registry(source_registry) if source_registry else "",
        "context_native": _format_source_registry(source_registry) if source_registry else "",
        "source_registry": source_registry,
        "source_registry_native": source_registry,
        "source_registry_all": _build_source_registry(tool_executions),
        "retrieved_papers": artifacts["retrieved_papers"],
        "retrieved_papers_native": artifacts["retrieved_papers_native"],
        "retrieved_papers_forced_eval": artifacts["retrieved_papers_forced_eval"],
        "local_sources": artifacts["local_sources"],
        "web_results": artifacts["web_results"],
        "web_pages": artifacts["web_pages"],
        "onet_results": artifacts["onet_results"],
        "local_contexts": artifacts["local_contexts"],
        "retrieved_contexts": artifacts["retrieved_contexts"],
        "retrieved_contexts_native": artifacts["retrieved_contexts_native"],
        "retrieved_contexts_forced_eval": artifacts["retrieved_contexts_forced_eval"],
        "tool_calls": artifacts["tool_calls"],
        "tool_calls_native": artifacts["tool_calls_native"],
        "tool_calls_forced_eval": artifacts["tool_calls_forced_eval"],
    }
    runs.append(record)
    MANUAL_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_RUNS_PATH.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def _format_source_registry(source_registry: List[Dict[str, str]]) -> str:
    lines = []
    for i, item in enumerate(source_registry, 1):
        lines.append(f"[Source {i}] {item['descriptor']}")
    return "\n".join(lines)


def _parse_rsa911_safe_sql_result(result_text: str) -> Dict[str, Any] | None:
    prefix = "RSA-911 Safe SQL Aggregate Result"
    if prefix not in result_text:
        return None
    payload = result_text.split(prefix, 1)[1].strip()
    parsed = _safe_json_loads(payload)
    return parsed if isinstance(parsed, dict) else None


def _extract_rsa911_safe_sql_results(tool_executions: List[Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for tool_exec in tool_executions:
        if (getattr(tool_exec, "tool_name", "") or "") != "run_rsa911_safe_sql":
            continue
        parsed = _parse_rsa911_safe_sql_result(getattr(tool_exec, "result", "") or "")
        if parsed:
            results.append(parsed)
    return results


def _extract_rsa911_statistics_result(tool_executions: List[Any]) -> str:
    for tool_exec in tool_executions:
        if (getattr(tool_exec, "tool_name", "") or "") != "analyze_rsa911_statistics":
            continue
        result_text = getattr(tool_exec, "result", "") or ""
        if result_text:
            return result_text
    return ""


def _format_number(value: Any, column: str = "") -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value.strip() == "":
        return "(blank)"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        col = column.lower()
        if col in {"n", "count"} or col.endswith("_n"):
            return f"{int(value):,}"
        if ("rate" in col or "avg" in col or "share" in col or "pct" in col) and 0.0 <= value <= 1.0:
            return f"{value:.3f} ({value * 100:.1f}%)"
        if isinstance(value, float):
            return f"{value:.4g}"
        return f"{value:,}"
    return str(value)


def _rate_column(columns: List[str]) -> str | None:
    for column in columns:
        lowered = column.lower()
        if "rate" in lowered or "avg" in lowered or "share" in lowered or "pct" in lowered:
            return column
    return None


def _has_count_and_employed(columns: List[str]) -> bool:
    lowered = {column.lower() for column in columns}
    return "n" in lowered and any(column in lowered for column in {"employed", "employed_n", "employment_count"})


def _with_derived_employment_rate(result: Dict[str, Any]) -> Dict[str, Any]:
    columns = [str(col) for col in (result.get("columns") or [])]
    if _rate_column(columns) or not _has_count_and_employed(columns):
        return result

    employed_col = next(
        (col for col in columns if col.lower() in {"employed", "employed_n", "employment_count"}),
        "",
    )
    if not employed_col:
        return result

    rows = result.get("rows") or []
    if not isinstance(rows, list):
        return result

    updated_rows = []
    for row in rows:
        if not isinstance(row, dict):
            updated_rows.append(row)
            continue
        updated = dict(row)
        try:
            n = float(updated.get("n") or 0)
            employed = float(updated.get(employed_col) or 0)
            updated["employment_rate"] = employed / n if n else None
        except Exception:
            updated["employment_rate"] = None
        updated_rows.append(updated)

    updated = dict(result)
    updated["rows"] = updated_rows
    updated["columns"] = [*columns, "employment_rate"]
    return updated


def _markdown_table(columns: List[str], rows: List[Dict[str, Any]], limit: int | None = None) -> str:
    rows = rows[:limit] if limit else rows
    if not columns or not rows:
        return "No rows returned."
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_format_number(row.get(col), col) for col in columns) + " |")
    return "\n".join([header, divider, *body])


def _rsa911_source_label(source_registry: List[Dict[str, str]]) -> str:
    for i, item in enumerate(source_registry, 1):
        if item.get("kind") == "rsa911_aggregate":
            return f"[Source {i}]"
    return "[Source 1]" if source_registry else ""


def _source_label_for_url(source_registry: List[Dict[str, str]], url: str) -> str:
    if not url:
        return ""
    for i, item in enumerate(source_registry, 1):
        if url in str(item.get("descriptor") or ""):
            return f"[Source {i}]"
    return ""


def _clean_jan_title(title: str) -> str:
    title = re.sub(r"\s*-\s*Ask JAN\s*$", "", str(title or ""), flags=re.I).strip()
    title = re.sub(r"\s*\|\s*.*$", "", title).strip()
    return title or "JAN accommodation resource"


def _jan_solution_idea(title: str) -> str:
    title = _clean_jan_title(title)
    title = re.sub(r"^JAN\s*[-–]\s*", "", title, flags=re.I).strip()
    return title


def _extract_jan_sentence_ideas(text: str, limit: int = 6) -> List[str]:
    ideas: List[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", _clean(text))
    useful_terms = [
        "accommodation",
        "telework",
        "work from home",
        "working remotely",
        "modified schedule",
        "flexible schedule",
        "break",
        "quiet",
        "workspace",
        "equipment",
        "mobility aid",
    ]
    skip_terms = ["pricing", "vendor", "product links", "cannot interpret"]
    noisy_terms = ["jobaccommodationnetwork", "2026-", "these ideas may be helpful"]
    for sentence in sentences:
        lowered = sentence.lower()
        if any(term in lowered for term in skip_terms + noisy_terms):
            continue
        if not any(term in lowered for term in useful_terms):
            continue
        sentence = _trim(sentence, limit=260)
        if len(sentence.split()) < 5:
            continue
        ideas.append(sentence)
        if len(ideas) >= limit:
            break
    return _ordered_unique(ideas)


def _extract_jan_evidence(tool_executions: List[Any]) -> List[Dict[str, str]]:
    evidence: List[Dict[str, str]] = []
    for tool_exec in tool_executions:
        tool_name = getattr(tool_exec, "tool_name", "") or ""
        result_text = getattr(tool_exec, "result", "") or ""
        if not result_text:
            continue
        if tool_name == "fetch_web_page":
            for page in _extract_fetched_page_results(result_text):
                url = str(page.get("url") or "")
                if "askjan.org" not in url:
                    continue
                evidence.append(
                    {
                        "kind": "fetched_page",
                        "title": str(page.get("title") or ""),
                        "url": url,
                        "body": str(page.get("body") or ""),
                    }
                )
        elif tool_name == "search_jan_accommodations":
            for item in _extract_web_results(result_text):
                url = str(item.get("url") or "")
                if "askjan.org" not in url:
                    continue
                evidence.append(
                    {
                        "kind": "search_result",
                        "title": str(item.get("title") or ""),
                        "url": url,
                        "body": str(item.get("snippet") or ""),
                    }
                )
    seen = set()
    deduped: List[Dict[str, str]] = []
    for item in evidence:
        key = item.get("url") or item.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _format_jan_accommodation_section(
    tool_executions: List[Any],
    source_registry: List[Dict[str, str]],
) -> str:
    evidence = _extract_jan_evidence(tool_executions)
    if not evidence:
        return ""

    lines = [
        "JAN-BASED ACCOMMODATION IDEAS",
        "",
        "These items are generated from JAN/AskJAN pages retrieved in this run, not from model inference.",
    ]

    concrete: List[str] = []
    support_notes: List[str] = []
    for item in evidence:
        title = _clean_jan_title(item.get("title") or "")
        url = item.get("url") or ""
        label = _source_label_for_url(source_registry, url)
        citation = f" {label}" if label else f" ({url})"
        body = item.get("body") or ""
        lowered_url = url.lower()
        lowered_title = title.lower()
        is_fetched_page = item.get("kind") == "fetched_page"

        if "/solutions/" in lowered_url:
            concrete.append(f"- {_jan_solution_idea(title)}{citation}")
        elif is_fetched_page and ("toolkit" in lowered_url or "toolkit" in lowered_title):
            support_notes.append(f"- Use JAN's accommodation toolkit/checklist resource for documenting and discussing accommodation requests{citation}")
        elif is_fetched_page and ("post-traumatic-stress-disorder" in lowered_url or "ptsd" in lowered_title):
            support_notes.append(f"- Use JAN's PTSD accommodation page as the disability-specific starting point for PTSD-related workplace accommodation planning{citation}")

        if is_fetched_page:
            for sentence in _extract_jan_sentence_ideas(body, limit=3):
                support_notes.append(f"- {sentence}{citation}")

    concrete = _ordered_unique(concrete)
    support_notes = _ordered_unique(support_notes)

    if concrete:
        lines.extend(["", "Concrete JAN accommodation ideas retrieved:"])
        lines.extend(concrete[:8])
    if support_notes:
        lines.extend(["", "JAN notes supporting accommodation planning:"])
        lines.extend(support_notes[:8])

    if not concrete:
        lines.extend(
            [
                "",
                "Note: the JAN evidence retrieved in this run did not include many concrete solution-page titles. Fetch more JAN `/solutions/` pages for a fuller accommodation list.",
            ]
        )

    return "\n".join(lines)


def _markdown_heading_text(line: str) -> str:
    heading_text = re.sub(r"^\s*#+\s*", "", line.strip())
    heading_text = re.sub(r"^\*\*|\*\*$", "", heading_text).strip()
    return heading_text


_REFUSAL_RE = re.compile(
    r"^\W*(?:i[’']m sorry|i am sorry|sorry)\b.{0,40}\b(?:can[’']?t|cannot|unable|not able)\b"
    r"|^\W*i (?:can[’']?t|cannot|am unable to|[’']m unable to|won[’']t)\b",
    flags=re.IGNORECASE,
)


def _is_refusal(response: str) -> bool:
    """A short answer that declines the request; the JAN section would be off-topic there."""
    text = str(response or "").strip()
    return len(text.split()) <= 60 and bool(_REFUSAL_RE.search(text))


def _insert_jan_section(response: str, section: str) -> str:
    if not section.strip():
        return response
    if section.splitlines()[0] in response:
        return response

    target_headings = {
        heading.upper()
        for heading in [
            "WORKPLACE SUPPORTS & ACCOMMODATIONS",
            "WORKPLACE SUPPORTS",
            "IMMEDIATE NEXT STEPS",
            "EVIDENCE GAPS",
            "SOURCES",
        ]
    }
    lines = response.splitlines()
    for index, line in enumerate(lines):
        if _markdown_heading_text(line).upper() not in target_headings:
            continue
        before = "\n".join(lines[:index]).rstrip()
        after = "\n".join(lines[index:]).lstrip()
        if before:
            return before + "\n\n" + section + "\n\n" + after
        return section + "\n\n" + after
    return response.rstrip() + "\n\n" + section


def _insert_section_before(response: str, heading: str, section: str) -> str:
    if not response.strip() or not section.strip():
        return response
    if section.splitlines()[0] in response:
        return response
    target = heading.strip().upper()
    lines = response.splitlines()
    for index, line in enumerate(lines):
        if _markdown_heading_text(line).upper() != target:
            continue
        return (
            "\n".join(lines[:index]).rstrip()
            + "\n\n"
            + section
            + "\n\n"
            + "\n".join(lines[index:]).lstrip()
        ).strip()
    return response.rstrip() + "\n\n" + section


def _is_markdown_section_heading(line: str) -> bool:
    heading_text = _markdown_heading_text(line)
    # Must contain a letter: table dividers like "|---|---|" and rules like "---" are not headings.
    return (
        bool(re.search(r"[A-Za-z]", heading_text))
        and heading_text.upper() == heading_text
        and len(heading_text.split()) <= 8
    )


def _remove_markdown_sections(response: str, headings: List[str]) -> str:
    target_headings = {heading.strip().upper() for heading in headings}
    output_lines: List[str] = []
    skipping = False
    for line in response.splitlines():
        heading_text = _markdown_heading_text(line)
        if heading_text.upper() in target_headings:
            skipping = True
            continue
        if skipping and _is_markdown_section_heading(line):
            skipping = False
        if not skipping:
            output_lines.append(line)
    return "\n".join(output_lines).strip()

# Matches [Source 3], [Sources 3, 4], (Source 3-5), (Source 1, chunk 175), 【Source 8】, Source [3] and Source 3,
# including the narrow no-break spaces the model emits. Only numbers joined by commas or dashes count,
# so "chunk 175" does not.
_SOURCE_CITATION_RE = re.compile(
    r"(?:[\[(\u3010]|\b)Sources?\**[\s\u00a0\u202f]*\[?"
    r"(\d+(?:[\s\u00a0\u202f]*[,\-\u2011\u2013][\s\u00a0\u202f]*\d+)*)"
)


# Two bracketed citations joined by a dash, e.g. "[Source 7]–[Source 11]", mean the whole range.
_SOURCE_RANGE_RE = re.compile(
    r"\[Sources?[\s\u00a0\u202f]*(\d+)\][\s\u00a0\u202f]*[\-\u2011\u2013][\s\u00a0\u202f]*\[Sources?[\s\u00a0\u202f]*(\d+)\]"
)


def _cited_source_numbers(text: str) -> List[int]:
    numbers = set()
    for match in _SOURCE_RANGE_RE.finditer(text):
        first, last = int(match.group(1)), int(match.group(2))
        if 0 <= last - first <= 20:
            numbers.update(range(first, last + 1))
    for match in _SOURCE_CITATION_RE.finditer(text):
        numbers_text = re.sub(r"[\s\u00a0\u202f]*([\-\u2011\u2013])[\s\u00a0\u202f]*", r"\1", match.group(1))
        for part in re.split(r"[,\s\u00a0\u202f]+", numbers_text):
            span = re.fullmatch(r"(\d+)[\-\u2011\u2013](\d+)", part)
            if span and int(span.group(2)) - int(span.group(1)) <= 20:
                numbers.update(range(int(span.group(1)), int(span.group(2)) + 1))
            elif part.isdigit():
                numbers.add(int(part))
    return sorted(numbers)


def _source_entry(item: Dict[str, str]) -> str:
    """One SOURCES line, linking only to addresses that were actually retrieved."""
    kind = item.get("kind", "")
    descriptor = item.get("descriptor", "")
    if kind == "local":
        parts = [part.strip() for part in descriptor.split("|")]
        path_parts = parts[1].split("/") if len(parts) > 1 else []
        site = path_parts[1] if len(path_parts) > 2 and path_parts[0] == "_raw" else ""
        chunk = f", {parts[2]}" if len(parts) > 2 else ""
        origin = f" (collected from {site})" if site else ""
        return f"Local document{origin}: {parts[0]}{chunk}. No web link stored."
    if kind == "rsa911_aggregate":
        return f"{descriptor}. Private RSA-911 data computed on this server; no public link."
    if kind == "onet":
        code = re.search(r"\b\d{2}-\d{4}\.\d{2}\b", descriptor)
        return f"O*NET: {descriptor}" + (f" | https://www.onetonline.org/link/summary/{code.group(0)}" if code else "")
    return descriptor


def _rebuild_sources_section(response: str, source_registry: List[Dict[str, str]]) -> str:
    """Replace model-written SOURCES lists with one built from the retrieved sources.

    The model renumbered its own lists and invented links for local files and the private
    RSA-911 summaries, which have no web address.
    """
    body = _remove_markdown_sections(response, ["SOURCES", "SOURCES:", "REFERENCES"])
    cited = [number for number in _cited_source_numbers(body) if 1 <= number <= len(source_registry)]
    if not cited:
        # Citations in a form not recognized above: keep the model's own list rather than drop it.
        return response
    lines = ["SOURCES"] + [f"[Source {number}] {_source_entry(source_registry[number - 1])}" for number in cited]
    return body.rstrip() + "\n\n" + "\n".join(lines)


def _result_has_column(result: Dict[str, Any], column: str) -> bool:
    return column.lower() in {str(col).lower() for col in (result.get("columns") or [])}


def _run_rsa911_rate_sql(sql: str, fallback_note: str = "") -> Dict[str, Any] | None:
    if _rsa911_blocked():
        return None
    try:
        from rsa911.safe_sql import run_safe_sql

        result = run_safe_sql(sql.strip())
        result["sql"] = sql.strip()
        if fallback_note:
            result["fallback_note"] = fallback_note
        return result
    except Exception:
        return None


def _fallback_rsa911_rate_results(message: str, existing_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lowered = message.lower()
    if "rsa-911" not in lowered and "rsa911" not in lowered:
        return []
    if "veteran" not in lowered or not re.search(r"rate|rates|outcome|employ", lowered):
        return []

    fallbacks: List[Dict[str, Any]] = []
    has_primdisability = any(_result_has_column(result, "primdisability") for result in existing_results)
    has_homeless = any(
        _result_has_column(result, "homelessorrunaway") or _result_has_column(result, "homeless_status")
        for result in existing_results
    )

    if ("primdisability" in lowered or "primary disability" in lowered) and not has_primdisability:
        result = _run_rsa911_rate_sql(
            """
SELECT trim(primdisability) AS primdisability,
       COUNT(*) AS n,
       AVG(CASE WHEN trim(exitempoutcome) IN ('1','2','3','4','5')
                THEN 1.0 ELSE 0.0 END) AS emp_rate
FROM rsa911_exiters
WHERE trim(veteran) = '1'
GROUP BY trim(primdisability)
            """,
            "Deterministic fallback for the requested veteran primary-disability employment-rate comparison.",
        )
        if result:
            fallbacks.append(result)

    if ("homeless" in lowered or "runaway" in lowered) and not has_homeless:
        result = _run_rsa911_rate_sql(
            """
SELECT trim(homelessorrunaway) AS homeless_status,
       COUNT(*) AS n,
       AVG(CASE WHEN trim(exitempoutcome) IN ('1','2','3','4','5')
                THEN 1.0 ELSE 0.0 END) AS emp_rate
FROM rsa911_exiters
WHERE trim(veteran) = '1'
GROUP BY trim(homelessorrunaway)
            """,
            "Deterministic fallback for the requested veteran homelessness/runaway employment-rate comparison.",
        )
        if result:
            fallbacks.append(result)

    return fallbacks


# Plain-language names for yes/no RSA-911 columns, used in computed comparison sentences.
_YES_NO_GROUP_NAMES = {
    "black": ("Black veterans", "non-Black veterans"),
    "white": ("White veterans", "non-White veterans"),
    "asian": ("Asian veterans", "non-Asian veterans"),
    "amerindian": ("American Indian or Alaska Native veterans", "other veterans"),
    "hawaiian": ("Native Hawaiian or Pacific Islander veterans", "other veterans"),
    "hispanic": ("Hispanic veterans", "non-Hispanic veterans"),
    "homelessorrunaway": ("veterans who were homeless or runaway", "veterans who were not"),
    "longtermunemp": ("long-term unemployed veterans", "veterans who were not"),
    "exoffenderstatus": ("veterans with ex-offender status", "veterans without"),
    "lowincomestatus": ("low-income veterans", "veterans who were not low-income"),
}


def _is_rate_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def _group_columns(columns: List[str], rows: List[Dict[str, Any]], rate_col: str | None) -> List[str]:
    return [
        col for col in columns
        if col != rate_col and col.lower() != "n" and any(isinstance(row.get(col), str) for row in rows)
    ]


def _label_group_values(columns: List[str], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Show RSA-911 codes with their codebook meaning, e.g. "1 (Yes)", where the label is verified."""
    from rsa911.code_labels import label_code

    labeled = []
    for row in rows:
        row = dict(row)
        for col in columns:
            value = row.get(col)
            if isinstance(value, str) and value.strip():
                label = label_code(col, value)
                if not label.startswith("Code "):
                    row[col] = f"{value.strip()} ({label})"
        labeled.append(row)
    return labeled


def _rate_comparison(columns: List[str], rows: List[Dict[str, Any]], rate_col: str | None) -> str:
    """A comparison sentence computed from the returned rows, so the numbers cannot be misquoted."""
    from rsa911.code_labels import YES_NO_UNKNOWN_COLUMNS, clean_code

    group_cols = _group_columns(columns, rows, rate_col)
    usable = [row for row in rows if _is_rate_value(row.get(rate_col))]
    if not rate_col or len(group_cols) != 1 or len(usable) < 2:
        return ""
    col = group_cols[0]

    def describe(row: Dict[str, Any]) -> str:
        n = row.get("n")
        size = f" (n={int(n):,})" if isinstance(n, (int, float)) else ""
        return f"{row[rate_col] * 100:.1f}%{size}"

    if col.lower() in YES_NO_UNKNOWN_COLUMNS:
        by_code = {clean_code(row.get(col)): row for row in usable}
        if "1" not in by_code or "0" not in by_code:
            return ""
        yes_name, no_name = _YES_NO_GROUP_NAMES.get(col.lower(), (f"{col} = Yes", f"{col} = No"))
        gap = (by_code["1"][rate_col] - by_code["0"][rate_col]) * 100
        sentence = (
            f"Comparison ({rate_col}): {yes_name} {describe(by_code['1'])} vs {no_name} {describe(by_code['0'])}; "
            f"difference {gap:+.1f} percentage points ({yes_name} minus {no_name})."
        )
        if any(code not in {"0", "1"} for code in by_code):
            sentence += " Rows coded 9 or blank (unknown or not reported) are shown but not part of this comparison."
        return sentence

    ordered = sorted(usable, key=lambda row: row[rate_col])
    low, high = _label_group_values([col], [ordered[0], ordered[-1]])
    return (
        f"Comparison ({rate_col}): highest is {col} {high.get(col)} at {describe(high)}; lowest is {col} "
        f"{low.get(col)} at {describe(low)}; a spread of {(high[rate_col] - low[rate_col]) * 100:.1f} "
        f"percentage points across {len(usable)} groups."
    )


def _rsa911_interpretation_notes(sql_results: List[Dict[str, Any]]) -> List[str]:
    notes = []
    if any(result.get("differential_privacy") for result in sql_results):
        notes.append(
            "- Values include random privacy noise, so differences of a few percentage points may not be meaningful; "
            "no significance test was run."
        )
    notes.extend(
        [
            "- Descriptive, not causal: differences between groups may reflect other factors such as disability "
            "type, state agency, age, or services received.",
            f"- Covers RSA-911 program year(s) {_rsa911_years_label()} only.",
            "- Where an employment rate is shown, blank `exitEmpOutcome` values are counted as not employed.",
        ]
    )
    return notes


def _format_rsa911_safe_sql_response(
    message: str,
    sql_results: List[Dict[str, Any]],
    source_registry: List[Dict[str, str]],
) -> str:
    if not sql_results:
        return ""

    source_label = _rsa911_source_label(source_registry)
    sections = [
        "RSA-911 SAFE SQL RESULT",
        "",
        "This answer uses the RSA-911 safe SQL tool against the combined SQLite database. "
        "Only aggregate rows with `n >= 11` are returned. "
        + (source_label if source_label else ""),
    ]

    wants_top_bottom = bool(re.search(r"\btop\b|\bbottom\b|highest|lowest", message, flags=re.I))
    wants_rates = bool(re.search(r"rate|rates|outcome|employ", message, flags=re.I))
    sql_results = [*sql_results, *_fallback_rsa911_rate_results(message, sql_results)]
    if wants_rates:
        rate_results = []
        for result in sql_results:
            result = _with_derived_employment_rate(result)
            columns = [str(col) for col in (result.get("columns") or [])]
            if _rate_column(columns) or _has_count_and_employed(columns):
                rate_results.append(result)
        if rate_results:
            sql_results = rate_results
        else:
            # No rate columns: show the deterministic fallback if one applies, else the model's own results.
            sql_results = _fallback_rsa911_rate_results(message, []) or sql_results
    if not sql_results:
        return ""

    for idx, result in enumerate(sql_results, 1):
        if wants_rates:
            result = _with_derived_employment_rate(result)
        sql = str(result.get("sql") or "").strip()
        columns = [str(col) for col in (result.get("columns") or [])]
        rows = result.get("rows") or []
        if not isinstance(rows, list):
            rows = []

        sections.extend(["", f"SQL query {idx}", "```sql", sql, "```"])

        dp_info = result.get("differential_privacy") or {}
        if not rows:
            threshold = dp_info.get("group_threshold", 11)
            sections.append(
                f"No aggregate rows were returned: the query matched no records, or every group had fewer than "
                f"about {threshold} people."
            )
            continue
        if dp_info:
            sections.append(
                "These numbers include small random noise added for privacy (differential privacy), "
                "so treat them as close estimates."
            )
        if result.get("suppressed_cells"):
            sections.append(
                "Blank cells were suppressed for privacy because they would describe fewer than 11 people."
                + (f" Counts are rounded to the nearest {result['round_base']}." if result.get("round_base") else "")
            )

        rate_col = _rate_column(columns)
        comparison = _rate_comparison(columns, rows, rate_col)
        if comparison:
            sections.extend(["", comparison])
        rows = _label_group_values(columns, rows)
        if wants_top_bottom and rate_col:
            rows = sorted(
                rows,
                key=lambda row: float(row.get(rate_col) or 0),
            )
            sections.extend(
                [
                    "",
                    "Top 5 highest rates",
                    _markdown_table(columns, list(reversed(rows[-5:]))),
                    "",
                    "Bottom 5 lowest rates",
                    _markdown_table(columns, rows[:5]),
                ]
            )
        else:
            sections.extend(["", "Returned aggregate rows", _markdown_table(columns, rows)])

        sql_lower = re.sub(r"\s+", " ", sql.lower())
        has_veteran_filter = bool(
            re.search(r"\btrim\s*\(\s*veteran\s*\)\s*=\s*'?1'?|\bveteran\s*=\s*'?1'?", sql_lower)
        )
        if "veteran" in message.lower() and not has_veteran_filter:
            sections.append(
                "\nWarning: the user asked for veterans only, but the displayed SQL does not visibly filter on `veteran`."
            )

    sections.extend(["", "Interpretation", *_rsa911_interpretation_notes(sql_results)])
    sections.extend(
        [
            "",
            "SOURCES",
            (f"{source_label} RSA-911 safe aggregate SQL result, `n >= 11` enforced." if source_label else "RSA-911 safe aggregate SQL result, `n >= 11` enforced."),
        ]
    )
    return "\n".join(sections)


def _format_evidence_memo(message: str, tool_executions: List[Any], source_registry: List[Dict[str, str]]) -> str:
    artifacts = _extract_tool_artifacts(tool_executions)

    sections = [
        "USER REQUEST:\n" + message.strip(),
        "PROFILE CONSTRAINTS:\n" + _build_profile_notes(message),
    ]
    if source_registry:
        sections.append("SOURCE REGISTRY:\n" + _format_source_registry(source_registry))

    local_blocks = artifacts.get("local_contexts") or []
    if local_blocks:
        sections.append(
            "HIGH-TRUST LOCAL CORPUS EVIDENCE:\n"
            + "\n\n".join(str(block) for block in local_blocks[:4])
        )

    rsa911_blocks = artifacts.get("rsa911_outcomes") or []
    if rsa911_blocks:
        sections.append(
            "AGGREGATE RSA-911 OUTCOME EVIDENCE (observational; not causal):\n"
            + "\n\n".join(str(block) for block in rsa911_blocks[:2])
        )

    rsa911_analysis_blocks = artifacts.get("rsa911_analyses") or []
    if rsa911_analysis_blocks:
        sections.append(
            "RSA-911 DISPARITY/PREDICTOR ANALYSIS (associational; outcome definition pending expert review):\n"
            + "\n\n".join(str(block) for block in rsa911_analysis_blocks[:2])
        )

    rsa911_statistics_blocks = artifacts.get("rsa911_statistics") or []
    if rsa911_statistics_blocks:
        sections.append(
            "RSA-911 STATISTICAL ANALYSIS (associational; code labels require codebook verification):\n"
            + "\n\n".join(str(block) for block in rsa911_statistics_blocks[:2])
        )

    rsa911_sql_blocks = artifacts.get("rsa911_sql_results") or []
    if rsa911_sql_blocks:
        sections.append(
            "RSA-911 SAFE SQL RESULTS (aggregate-only; n >= 11 enforced):\n"
            + "\n\n".join(str(block) for block in rsa911_sql_blocks[:3])
        )

    paper_full_texts = artifacts.get("paper_full_texts") or []
    if paper_full_texts:
        paper_text_lines = []
        for paper in paper_full_texts[:3]:
            title = str(paper.get("title") or "Fetched academic paper")
            url = str(paper.get("url") or "")
            access_type = str(paper.get("access_type") or "")
            body = _trim(str(paper.get("body") or ""), limit=1500)
            prefix = f"- {title} | {url}"
            if access_type:
                prefix += f" | {access_type}"
            paper_text_lines.append(f"{prefix}\n  {body}")
        sections.append("HIGH-TRUST ACADEMIC FULL TEXT EVIDENCE:\n" + "\n".join(paper_text_lines))

    papers = artifacts.get("retrieved_papers_native") or artifacts.get("retrieved_papers") or []
    if papers:
        paper_lines = []
        for paper in papers[:4]:
            title = str(paper.get("title") or "Unknown title")
            summary = _trim(str(paper.get("summary") or ""), limit=900)
            if summary:
                paper_lines.append(f"- {title}: {summary}")
            else:
                paper_lines.append(f"- {title}")
        sections.append("ACADEMIC ABSTRACT EVIDENCE:\n" + "\n".join(paper_lines))

    web_pages = artifacts.get("web_pages") or []
    if web_pages:
        page_lines = []
        for page in web_pages[:3]:
            title = str(page.get("title") or "Fetched web page")
            url = str(page.get("url") or "")
            body = _trim(str(page.get("body") or ""), limit=1200)
            page_lines.append(f"- {title} | {url}\n  {body}")
        sections.append("FETCHED WEB PAGE EVIDENCE:\n" + "\n".join(page_lines))

    web_results = artifacts.get("web_results") or []
    if web_results:
        snippet_lines = []
        for item in web_results[:4]:
            title = str(item.get("title") or "Web result")
            url = str(item.get("url") or "")
            snippet = _trim(str(item.get("snippet") or ""), limit=350)
            snippet_lines.append(f"- {title} | {url}\n  {snippet}")
        sections.append(
            "LOW-TRUST WEB SEARCH SNIPPETS (use only for leads or current links, not as sole support):\n"
            + "\n".join(snippet_lines)
        )

    onet_results = artifacts.get("onet_results") or []
    if onet_results:
        onet_lines = [f"- {str(item.get('descriptor') or '').strip()}" for item in onet_results[:5]]
        sections.append(
            "LOW-TRUST O*NET CANDIDATES (titles only, not proof of suitability):\n"
            + "\n".join(line for line in onet_lines if line.strip() != "-")
        )

    return "\n\n".join(section for section in sections if section.strip())


SYNTHESIS_PROMPT = """
You are rewriting a veterans employment answer from a structured evidence memo.

Trust order:
1. Local corpus evidence, fetched academic full text, and fetched web page text
2. RSA-911 aggregate outcome evidence for descriptive patterns only
3. Academic abstracts
4. Search-result snippets
5. O*NET candidate titles and codes

Hard rules:
- Use only evidence in the memo and only source labels from the source registry.
- Search snippets and O*NET results cannot by themselves justify a recommendation, eligibility conclusion, accommodation claim, or causal statement.
- JAN/AskJAN is the preferred public source for workplace accommodation ideas. Do not present specific accommodation ideas as evidence-based unless supported by fetched JAN page text or another high-trust fetched source.
- Do not write a JAN accommodation section yourself; the answer already includes one titled `JAN-BASED ACCOMMODATION IDEAS`. You may briefly refer readers to that section. Do not mention how the answer is produced.
- For O*NET, use only occupation titles and codes explicitly shown in the O*NET tool output. Do not infer or invent O*NET codes. Do not claim an occupation is bright-outlook, remote-compatible, sedentary, or low physical demand unless the evidence memo explicitly says so.
- RSA-911 aggregate outcomes are observational descriptive patterns only; do not treat them as causal evidence or proof that a job is suitable for an individual.
- RSA-911 predictor outputs are associational model summaries only. Do not turn coefficients into causal instructions such as "do X to get employed fastest."
- RSA-911 safe SQL results are aggregate-only and enforce n >= 11, but the denominator and outcome definition still need to be explained clearly.
- Do not invent meanings for RSA-911 raw codes such as `primdisability_01`, `agencycode_109`, `vocrehab_2`, or occupation codes. If the evidence only gives a raw code, keep it as a raw code and say the label is not verified by the codebook.
- Do not call RSA-911 services "VA-provided" unless the evidence explicitly says VA; describe them as RSA-911/VR administrative fields or service indicators.
- IPE means Individualized Plan for Employment. Do not expand it as integrated employment.
- RSA-911 is Rehabilitation Services Administration / State VR administrative data. Do not describe it as VA data except for specific fields/codes that explicitly mention Department of Veterans Affairs VR&E.
- If an idea is supported only by snippets or O*NET, mark it as tentative or move it to EVIDENCE GAPS.
- Prefer conservative recommendations that fit the veteran's stated functional limits.
- Every substantive claim must cite at least one source label such as [Source 1].
- If the evidence does not support a specific occupation or program detail, say so explicitly.
- Keep the response practical and user-facing, but do not add unsupported specifics.
- In the SOURCES section, include the raw URL or DOI link from the source registry whenever one is available so it remains clickable in Markdown.

Required output sections:
VETERAN PROFILE SUMMARY
RECOMMENDED VA VR&E TRACK
EMPLOYMENT PATHWAYS
IMMEDIATE NEXT STEPS
EVIDENCE GAPS
SOURCES
"""


def _synthesize_grounded_response(
    message: str,
    draft_response: str,
    tool_executions: List[Any],
    source_registry: List[Dict[str, str]],
) -> str:
    if not source_registry:
        return draft_response

    evidence_memo = _format_evidence_memo(message, tool_executions, source_registry)
    synthesizer = Agent(
        model=_build_model(),
        description="Veterans employment grounded synthesizer",
        instructions=SYNTHESIS_PROMPT,
        markdown=True,
    )
    prompt = (
        f"User question:\n{message}\n\n"
        f"Source registry:\n{_format_source_registry(source_registry)}\n\n"
        f"Existing draft (use only as a rough outline; remove unsupported details):\n{draft_response}\n\n"
        f"Evidence memo:\n{evidence_memo}"
    )
    result = synthesizer.run(prompt, stream=False)
    if result is None:
        return draft_response
    if hasattr(result, "content"):
        return result.content or draft_response
    return str(result)


FINALIZE_PROMPT = """
You are validating and rewriting an answer produced by a veterans employment agent.

Rules:
- Use only the source registry provided below.
- Cite sources only as [Source N], where N exists in the registry.
- Do not invent citations, source ranges, or source aliases.
- Preserve the draft's structure, detail level, tables, bullet points, and recommendations as much as possible.
- Do not shorten the answer unless a claim must be removed because it cannot be supported by the source registry.
- If a claim in the draft is unsupported or the source mapping is unclear, keep the wording as close as possible while removing or softening only the unsupported citation or unsupported detail.
- Keep O*NET occupation titles and codes only if they match the source registry. Otherwise generalize the occupation title.
- Remove O*NET claims about bright outlook, remote work, sedentary work, or low physical demand unless the draft cites source text that explicitly supports that exact claim.
- Remove or soften detailed accommodation claims that appear to be model-generated from JAN unless they are directly supported by fetched JAN page text in the source registry. The `JAN-BASED ACCOMMODATION IDEAS` section carries JAN-derived accommodation ideas. Do not mention how the answer is produced.
- Remove any invented interpretation of RSA-911 raw codes. Examples to remove unless explicitly supported: `primdisability_01` means mental health, `primdisability_6` means physical/orthopedic, `agencycode_109` is a regional VA office, or `ipeprimaryocc_119013` is a specific occupation title.
- Keep unknown RSA-911 codes as raw codes with "label not verified" or "requires codebook verification."
- IPE means Individualized Plan for Employment. RSA-911 is RSA/State VR administrative data, not VA data unless a specific code explicitly says Department of Veterans Affairs VR&E.
- The final SOURCES section must list only the source labels actually cited in the answer body.
- In the final SOURCES section, preserve any raw URLs or DOI links that exist in the source registry so they remain clickable in Markdown.
- Do not replace detailed content with shorter summaries.
"""


def _finalize_response(message: str, draft_response: str, source_registry: List[Dict[str, str]]) -> str:
    if not source_registry:
        return draft_response

    formatter = Agent(
        model=_build_model(),
        description="Veterans employment answer validator",
        instructions=FINALIZE_PROMPT,
        markdown=True,
    )
    prompt = (
        f"User question:\n{message}\n\n"
        f"Source registry:\n{_format_source_registry(source_registry)}\n\n"
        f"Draft response:\n{draft_response}"
    )
    result = formatter.run(prompt, stream=False)
    if result is None:
        return draft_response
    if hasattr(result, "content"):
        return result.content or draft_response
    return str(result)


def _is_pure_rsa911_sql_question(message: str) -> bool:
    lowered = message.lower()
    if "rsa-911" not in lowered and "rsa911" not in lowered:
        return False
    broader_agent_signals = [
        "full agent",
        "search relevant",
        "literature",
        "evidence",
        "o*net",
        "onet",
        "occupation",
        "pathway",
        "recommendation",
        "recommend",
        "practical",
        "ptsd",
        "mobility",
        "i am ",
        "i'm ",
        "remote",
        "hybrid",
    ]
    return not any(signal in lowered for signal in broader_agent_signals)


def _is_pure_rsa911_statistics_question(message: str) -> bool:
    lowered = message.lower()
    if "rsa-911" not in lowered and "rsa911" not in lowered:
        return False
    stats_terms = [
        "statistical analysis",
        "feature importance",
        "discriminant",
        "feature relationships",
        "relationship between features",
        "cramer",
        "associated with employment",
        "codebook-based interpretation",
        "interpretation of the codes",
    ]
    broader_agent_signals = [
        "full agent",
        "search relevant",
        "literature",
        "evidence",
        "o*net",
        "onet",
        "occupation",
        "pathway",
        "recommendation",
        "recommend",
        "practical",
        "ptsd",
        "mobility",
        "i am ",
        "i'm ",
        "remote",
        "hybrid",
    ]
    return any(term in lowered for term in stats_terms) and not any(
        signal in lowered for signal in broader_agent_signals
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Agent system prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are a Veterans Employment Pathways Agent. Your job is to help veterans with
complex, co-occurring conditions find personalized, evidence-based employment pathways.

When a veteran describes their situation:
1. Identify their disabilities, life stressors, functional limitations, and VR eligibility.
2. Search for evidence across policy documents, academic research, current VA programs, and occupation data.
3. Produce a structured employment pathway with these sections:

VETERAN PROFILE SUMMARY
RECOMMENDED VA VR&E TRACK
EMPLOYMENT PATHWAYS (2-3 options with occupation titles, rationale, service route)
IMMEDIATE NEXT STEPS
EVIDENCE GAPS
SOURCES

Always cite sources as [Source 1], [Source 2], etc. Never invent facts. Flag gaps where evidence is limited.
"""

NATIVE_AGENT_INSTRUCTIONS = [
    "Use tools to gather evidence before answering. Do not rely on unsupported assumptions.",
    "Use search_local_corpus first for policy, rehabilitation, disability, and veterans-employment evidence.",
    "Use PubMed for academic and rehabilitation literature when the query needs research evidence.",
    "After PubMed, use fetch_paper_full_text on the most relevant paper when a Full Text URL, PMC URL, DOI URL, or PubMed URL is available and the paper appears central to the answer.",
    "Prefer fetched academic full text over abstract-only evidence when making research-based claims.",
    "Use search_web to find current public resources, then use fetch_web_page on the most relevant result before making current-program claims.",
    "Use search_jan_accommodations when the user asks for accommodation ideas, functional limitations, disability-friendly strategies, PTSD, mental health, mobility impairment, remote work, hybrid work, or workplace supports.",
    "After search_jan_accommodations, use fetch_web_page on the most relevant JAN/AskJAN result before giving specific accommodation ideas when a JAN result is available.",
    "For concrete accommodation ideas, prefer fetched JAN solution pages under askjan.org/solutions because each solution page title is an accommodation idea. Fetch disability pages for condition context and solution pages for concrete ideas.",
    "Use fetched web page text only for pages you have actually fetched. Do not rely on search-result snippets alone for substantive claims.",
    "Use query_rsa911_outcomes for descriptive RSA-911 aggregate outcome patterns when the user asks about veterans with specific disability profiles or likely employment outcomes.",
    "Treat RSA-911 aggregate patterns as observational context only; do not use them as causal evidence or proof that a specific occupation is suitable for one person.",
    "Use analyze_rsa911_question when the user asks about Black veterans, race comparisons, disparities, or predictors of employment outcomes in RSA-911 data.",
    "Treat RSA-911 predictor outputs as associational and provisional until the employment outcome definition is confirmed by a domain expert.",
    "Use analyze_rsa911_statistics for broad statistical-analysis questions about feature importance, discriminant analysis, feature-feature relationships, or code interpretation.",
    "Never invent meanings for raw RSA-911 codes. If a code label is not explicitly present in the tool output or codebook mapping, keep the raw code and say the label is not verified.",
    "IPE means Individualized Plan for Employment, not integrated employment.",
    "RSA-911 is RSA/State VR administrative data. Do not call it VA data unless a specific field or code explicitly mentions Department of Veterans Affairs VR&E.",
    "For flexible RSA-911 dataset questions, first call describe_rsa911_sql_schema, then call run_rsa911_safe_sql with one aggregate SELECT query.",
    "SQL must include COUNT(*) AS n; do not request row-level records. Use one flat aggregate query: no subqueries, joins, window functions, MIN, or MAX. The SQL tool returns only groups with n >= 11; null values mean a cell was suppressed for privacy.",
    "Use search_onet only to identify occupation candidates and codes. Do not treat O*NET matches alone as proof of suitability.",
    "Only use O*NET titles and codes that appear in search_onet output. Do not invent occupation codes, and do not claim O*NET says remote-compatible, bright outlook, sedentary, or low physical demand unless the tool output explicitly says that.",
    "Only say an occupation appears in RSA-911 results if that occupation title is listed in RSA-911 tool output. O*NET matches are not RSA-911 evidence.",
    "For RSA-911 employment rates, quote only the line labeled EMPLOYMENT RATE or a rate column from run_rsa911_safe_sql. Counts of employment types are not rates.",
    "Prefer accessible, seated, structured, and accommodation-friendly roles when mobility or cognitive limitations are present.",
    "Avoid recommending physically intensive roles unless the evidence explicitly supports them.",
    "If the evidence for a concrete occupation is weak, recommend a broader pathway instead of a specific job title.",
    "Every substantive claim must cite at least one source label such as [Source 1].",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Build agent
# ═══════════════════════════════════════════════════════════════════════════════

def build_agent(session_id: str = "default") -> Agent:
    """
    Build and return an Agno Agent for a given session.
    Uses the configured model provider with native Agno tools.
    """
    agent = Agent(
        model=_build_model(),
        tools=_build_native_tools(),
        tool_choice="auto",
        description="Veterans Employment Pathways Agent",
        instructions=NATIVE_AGENT_INSTRUCTIONS,
        additional_context=SYSTEM_PROMPT,
        session_id=session_id,
        markdown=True,
        add_datetime_to_context=True,
        retries=1,
        delay_between_retries=1,
    )
    return agent


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

# Cache one agent per session
_agent_cache: Dict[str, Agent] = {}


def run_agent(message: str, session_id: str = "default") -> str:
    """
    Run the Veterans Employment Pathways Agent for a given message and session.
    Returns the agent's response as a string.
    """
    trace_context = None
    if _agentops_enabled:
        trace_context = agentops.start_trace(
            trace_name="veterans-employment-pathway",
            tags={
                "session_id": session_id,
                "native_tools": True,
                "model_provider": _active_provider_name(),
            },
        )

    try:
        if session_id not in _agent_cache:
            _agent_cache[session_id] = build_agent(session_id)

        agent = _agent_cache[session_id]
        result = agent.run(message, stream=False)
        native_tool_executions = list(getattr(result, "tools", None) or [])
        if _needs_jan_accommodation_evidence(message) and not _has_jan_tool_call(native_tool_executions):
            native_tool_executions.extend(_force_jan_tool_executions(message))
        if _needs_rsa911_context(message) and not _has_rsa911_tool_call(native_tool_executions):
            native_tool_executions.append(_force_rsa911_summary_execution(message))
        tool_executions = list(native_tool_executions)
        if EVAL_FORCE_PUBMED and not _has_pubmed_tool_call(native_tool_executions):
            tool_executions.append(_force_pubmed_tool_execution(message))
        source_registry = _build_source_registry(native_tool_executions)

        if result is None:
            response = "No response from agent."
        elif _run_output_has_tool_use_failure(result):
            response = f"Native tool calling failed: {result.content}"
        elif hasattr(result, "content"):
            response = result.content or str(result)
        else:
            response = str(result)

        safe_sql_response = ""
        if source_registry and not _run_output_has_tool_use_failure(result):
            safe_sql_response = _format_rsa911_safe_sql_response(
                message=message,
                sql_results=_extract_rsa911_safe_sql_results(native_tool_executions),
                source_registry=source_registry,
            )

        statistics_response = _extract_rsa911_statistics_result(native_tool_executions)
        if statistics_response and _is_pure_rsa911_statistics_question(message):
            response = statistics_response
        elif safe_sql_response and _is_pure_rsa911_sql_question(message):
            response = safe_sql_response
        elif source_registry and not _run_output_has_tool_use_failure(result):
            response = _synthesize_grounded_response(
                message=message,
                draft_response=response,
                tool_executions=native_tool_executions,
                source_registry=source_registry,
            )
            response = _finalize_response(message, response, source_registry)
            jan_section = _format_jan_accommodation_section(native_tool_executions, source_registry)
            if jan_section and not _is_refusal(response):
                response = _remove_markdown_sections(
                    response,
                    ["WORKPLACE SUPPORTS & ACCOMMODATIONS", "WORKPLACE SUPPORTS"],
                )
                response = _insert_jan_section(response, jan_section)
            if safe_sql_response:
                response = response.rstrip() + "\n\n" + safe_sql_response
        if source_registry and not _run_output_has_tool_use_failure(result):
            response = _rebuild_sources_section(response, source_registry)

        result_status = getattr(result, "status", None)
        status_text = "success"
        error_text = ""
        if result is None:
            status_text = "error"
            error_text = "No response from agent."
        elif result_status == RunStatus.error or str(result_status).endswith("error"):
            status_text = "error"
            error_text = str(getattr(result, "content", "") or "")
        elif response.startswith("Native tool calling failed:"):
            status_text = "error"
            error_text = response

        _append_manual_run(
            message=message,
            session_id=session_id,
            response=response,
            source_registry=source_registry,
            tool_executions=tool_executions,
            status=status_text,
            error=error_text,
        )

        if _agentops_enabled and trace_context is not None:
            agentops.end_trace(trace_context, end_state=agentops.TraceState.SUCCESS)
        return response
    except Exception:
        if _agentops_enabled and trace_context is not None:
            agentops.end_trace(trace_context, end_state=agentops.TraceState.ERROR)
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# CLI mode
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nVeterans Employment Pathways Agent — CLI")
    print("Type 'exit' to quit.\n")
    sid = "cli-session"
    while True:
        q = input("Veteran query: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        response = run_agent(q, session_id=sid)
        print("\n" + "=" * 80)
        print(response)
        print("=" * 80 + "\n")
