#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import List, Tuple, Dict

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

from ask_llm_rag import answer_with_llm
import ask_llm_rag

DB_DIR = "chroma_db_adv"
COLLECTION = "employment_adv_v1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNKS_JSONL = Path("chunks.jsonl")


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def format_citation(d: Document) -> str:
    m = d.metadata or {}
    src = m.get("source_filename") or m.get("doc_txt") or "unknown"
    path = m.get("source_path") or ""
    chunk = m.get("chunk_index", "")
    if path:
        return f"{src} | {path} | chunk {chunk}"
    return f"{src} | chunk {chunk}"


def build_bm25_from_chunks_jsonl() -> BM25Retriever:
    """
    Chunk-level BM25 over the same chunks stored in Chroma.
    """
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError("chunks.jsonl not found. Run build_index_langchain.py first.")

    docs: List[Document] = []
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            txt = clean_text(rec.get("text", ""))
            if len(txt) < 200:
                continue
            docs.append(Document(page_content=txt, metadata=rec.get("metadata", {})))

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 60  # return enough for fusion
    return bm25


def bm25_search(bm25: BM25Retriever, query: str) -> List[Document]:
    """
    Version-safe BM25 call.
    """
    if hasattr(bm25, "invoke"):
        return bm25.invoke(query)
    if hasattr(bm25, "_get_relevant_documents"):
        return bm25._get_relevant_documents(query)  # noqa
    raise RuntimeError("BM25Retriever API not found (no invoke or _get_relevant_documents).")


def chroma_query_with_scores(vs: Chroma, query: str, n_results: int = 60) -> Tuple[List[Document], List[float]]:
    """
    Use underlying Chroma collection to get distances (lower is better).
    Returns docs + distances aligned.
    """
    col = vs._collection  # langchain wrapper -> chroma collection
    res = col.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs = [Document(page_content=t, metadata=m) for t, m in zip(res["documents"][0], res["metadatas"][0])]
    dists = [float(x) for x in res["distances"][0]]
    return docs, dists


def _doc_key(d: Document) -> Tuple:
    m = d.metadata or {}
    return (
        m.get("doc_txt"),
        m.get("chunk_index"),
        m.get("source_path"),
    )


def fuse_candidates(
    vec_docs: List[Document],
    vec_dist: List[float],
    kw_docs: List[Document],
    keep_top: int = 140,
    w_vec: float = 0.65,
    w_kw: float = 0.35,
) -> List[Document]:
    """
    Weighted rank fusion without brittle keyword rules.
    - vector score: sim = 1/(1+dist)
    - bm25 score: 1/(1+rank)
    """
    pool: Dict[Tuple, Dict[str, float | Document]] = {}

    # add vector docs
    for d, dist in zip(vec_docs, vec_dist):
        k = _doc_key(d)
        sim = 1.0 / (1.0 + float(dist))
        pool[k] = {"doc": d, "v": sim, "k": 0.0}

    # add keyword docs (rank-based proxy)
    for rank, d in enumerate(kw_docs):
        k = _doc_key(d)
        ksim = 1.0 / (1.0 + rank)
        if k in pool:
            pool[k]["k"] = max(pool[k]["k"], ksim)
        else:
            pool[k] = {"doc": d, "v": 0.0, "k": ksim}

    scored = []
    for k, rec in pool.items():
        d = rec["doc"]
        v = float(rec["v"])
        kk = float(rec["k"])
        score = w_vec * v + w_kw * kk
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:keep_top]]


def rerank(query: str, docs: List[Document], reranker: CrossEncoder, top_k: int = 8) -> List[Document]:
    """
    Cross-encoder rerank.
    """
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_k]]


def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )

    print("Loading BM25 from chunks.jsonl (one-time load)...")
    bm25 = build_bm25_from_chunks_jsonl()

    reranker = CrossEncoder(RERANK_MODEL)

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        # Vector retrieval with scores
        vec_docs, vec_dist = chroma_query_with_scores(vs, q, n_results=70)

        # Keyword retrieval (chunk-level)
        kw_docs = bm25_search(bm25, q)

        # Fuse (keeps best candidates across both signals)
        candidates = fuse_candidates(vec_docs, vec_dist, kw_docs, keep_top=160)

        # Rerank (cross-encoder)
        top = rerank(q, candidates, reranker, top_k=8)

        print("\nTop evidence (fused + reranked):\n")
        print("ask_llm_rag loaded from:", ask_llm_rag.__file__)
        for i, d in enumerate(top, 1):
            print(f"[{i}] {format_citation(d)}")
            print(textwrap.fill((d.page_content or "")[:800].replace("\n", " "), width=100))
            print("-" * 100)

        answer_with_llm(q, top)

if __name__ == "__main__":
    main()