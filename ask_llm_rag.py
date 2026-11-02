import subprocess

def format_source_for_llm(i: int, d) -> str:
    m = d.metadata or {}
    src = m.get("source_filename") or m.get("doc_txt") or "unknown"
    path = m.get("source_path", "")
    chunk = m.get("chunk_index", "")

    header = f"[Source {i}] {src}"
    if path:
        header += f" | {path}"
    if chunk != "":
        header += f" | chunk {chunk}"

    return f"{header}\n{d.page_content[:1200]}"

def format_sources_for_output(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        src = m.get("source_filename") or m.get("doc_txt") or "unknown"
        path = m.get("source_path", "")
        chunk = m.get("chunk_index", "")
        line = f"[Source {i}] {src}"
        if path:
            line += f" | {path}"
        if chunk != "":
            line += f" | chunk {chunk}"
        lines.append(line)
    return "\n".join(lines)

def ask_llama(prompt: str, model: str = "llama3.1") -> str:
    p = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "Ollama failed with no stderr.")
    out = (p.stdout or "").strip()
    if not out:
        raise RuntimeError("Ollama returned empty output.")
    return out

def build_prompt(question: str, evidence_blocks: list[str]) -> str:
    context = "\n\n".join(evidence_blocks)
    return f"""
You are an academic research assistant.

Answer the question below using ONLY the provided sources.
If the evidence is limited or inconclusive, say so clearly.
Do NOT invent facts or citations.

Question:
{question}

Sources:
{context}

Instructions:
- Synthesize findings across sources
- Cite sources using [Source X]
- Be concise but precise
"""

def answer_with_llm(question: str, top_docs, model: str = "llama3.1"):
    evidence_blocks = [format_source_for_llm(i + 1, d) for i, d in enumerate(top_docs)]
    prompt = build_prompt(question, evidence_blocks)
    answer = ask_llama(prompt, model=model)
    sources = format_sources_for_output(top_docs)
    return answer, sources