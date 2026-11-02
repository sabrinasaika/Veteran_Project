from pathlib import Path
import re
import json

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path("docs_text")
DB_DIR = Path("chroma_db_adv")
COLLECTION = "employment_adv_v1"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def parse_header(raw: str):
    # expects:
    # FILENAME: ...
    # PATH: ...
    # blank line
    # body...
    lines = raw.splitlines()
    meta = {}
    body_start = 0
    for i, line in enumerate(lines[:30]):
        if line.startswith("FILENAME:"):
            meta["source_filename"] = line.split(":", 1)[1].strip()
        elif line.startswith("PATH:"):
            meta["source_path"] = line.split(":", 1)[1].strip()
        elif line.strip() == "" and i >= 1:
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]) if body_start else raw
    return meta, body

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def main():
    assert DOCS_DIR.exists(), "docs_text/ not found. Run extraction first."

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = []
    files = sorted(DOCS_DIR.glob("*.txt"))
    print("Docs:", len(files))

    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        meta, body = parse_header(raw)
        body = clean_text(body)
        if len(body) < 400:
            continue

        base_meta = {
            "doc_txt": fp.name,
            **meta,
        }

        for i, chunk in enumerate(splitter.split_text(body)):
            if len(chunk) < 250:
                continue
            docs.append(Document(
                page_content=chunk,
                metadata={**base_meta, "chunk_index": i}
            ))

    print("Chunks:", len(docs))

    # Export chunks for BM25 (chunk-level keyword search)
    chunks_path = Path("chunks.jsonl")
    with chunks_path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {
                "text": d.page_content,
                "metadata": d.metadata
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Wrote chunks to:", chunks_path, "count:", len(docs))

    # Fresh rebuild (delete folder if exists)
    if DB_DIR.exists():
        # keep simple: wipe for clean rebuild
        import shutil
        shutil.rmtree(DB_DIR)

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(DB_DIR),
    )
    vs.persist()
    print("Saved Chroma to:", DB_DIR)

if __name__ == "__main__":
    main()
