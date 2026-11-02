"""Collect baseline answers from general-purpose models through OpenRouter.

Each model receives only the question text from comparison_questions.json, as a single user message:
no system prompt, no tools, no web search, and nothing from the agent, its evidence, or RSA-911.
With --system-prompt, the agent's own system prompt (SYSTEM_PROMPT in pathways_agent.py) is sent as
well, still with no tools or evidence; answers then go to runs/<model>_sysprompt/.
Requests ask OpenRouter to route only to providers that do not collect prompt data.

Answers are saved in the same format as the agent's benchmark_runs.json, so the existing local judge
can score them:
  statistics/comparison/runs/<model>/benchmark_runs.json

Usage (from the project folder; the API key is read from the environment, never from a file):
  export OPENROUTER_API_KEY=...
  .venv/bin/python statistics/comparison/openrouter_baselines.py --models openai/gpt-5 anthropic/claude-sonnet-4.5
  .venv/bin/python statistics/comparison/openrouter_baselines.py --models ... --dry-run   # show what would be sent
"""

import argparse
import ast
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "comparison_questions.json"
RUNS_DIR = HERE / "runs"
AGENT_FILE = HERE.parent.parent / "pathways_agent.py"
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _agent_system_prompt() -> str:
    # Read the constant without importing the agent (which would load models and data).
    for node in ast.parse(AGENT_FILE.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SYSTEM_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value).strip()
    raise SystemExit(f"SYSTEM_PROMPT not found in {AGENT_FILE}")


def _model_dir(model: str, system_prompt: str = "") -> Path:
    suffix = "_sysprompt" if system_prompt else ""
    return RUNS_DIR / ("openrouter_" + re.sub(r"[^A-Za-z0-9.-]+", "_", model) + suffix)


def _payload(model: str, question: str, max_tokens: int, system_prompt: str = "") -> dict:
    # Only the question text (and, with --system-prompt, the agent's system prompt) is sent.
    # "tools" and web-search plugins are deliberately absent.
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return {
        "model": model,
        "messages": messages + [{"role": "user", "content": question}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "provider": {"data_collection": "deny"},
    }


def _ask(model: str, question: str, api_key: str, max_tokens: int, system_prompt: str = "", retries: int = 3) -> tuple[str, dict]:
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=_payload(model, question, max_tokens, system_prompt),
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            answer = (data["choices"][0]["message"].get("content") or "").strip()
            if not answer:
                raise ValueError("empty answer")
            meta = {"served_model": data.get("model"), "provider": data.get("provider"), "usage": data.get("usage")}
            return answer, meta
        except (requests.RequestException, KeyError, ValueError) as exc:
            if attempt == retries:
                raise
            print(f"    retry {attempt}/{retries - 1} after error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="OpenRouter model ids, e.g. openai/gpt-5")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--dry-run", action="store_true", help="Print exactly what would be sent, without calling the API.")
    parser.add_argument("--system-prompt", action="store_true", help="Also send the agent's system prompt (still no tools).")
    args = parser.parse_args()

    for model in args.models:
        if ":online" in model:
            raise SystemExit(f"{model}: ':online' enables web search; use the plain model id.")
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))["items"]
    system_prompt = _agent_system_prompt() if args.system_prompt else ""

    if args.dry_run:
        print(json.dumps(_payload(args.models[0], items[0]["question"], args.max_tokens, system_prompt), indent=2))
        print(f"... and the same for {len(items)} questions x {len(args.models)} model(s). Nothing was sent.")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment first.")

    for model in args.models:
        out = _model_dir(model, system_prompt) / "benchmark_runs.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Answers are saved after every question; rerunning resumes and retries only missing ones.
        runs = [run for run in (json.loads(out.read_text()) if out.exists() else []) if run.get("status") == "success"]
        done = {run["benchmark_id"] for run in runs}
        print(f"== {model}: {len(done)} of {len(items)} already answered", flush=True)
        for idx, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            print(f"[{idx}/{len(items)}] {item['id']}", flush=True)
            start = time.time()
            try:
                answer, meta = _ask(model, item["question"], api_key, args.max_tokens, system_prompt)
                status, error = "success", None
            except Exception as exc:  # recorded and retried on the next run
                answer, meta, status, error = "", {}, "error", f"{type(exc).__name__}: {exc}"
                print(f"    failed: {error}", flush=True)
            runs = [run for run in runs if run["benchmark_id"] != item["id"]] + [{
                "id": f"openrouter-{item['id']}",
                "benchmark_id": item["id"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "error": error,
                "question": item["question"],
                "answer": answer,
                "agent_model": f"openrouter:{model}",
                "latency_sec": round(time.time() - start, 2),
                "baseline": {"tools": False, "system_prompt": bool(system_prompt), "web_search": False, **meta},
                # No retrieval: empty evidence fields keep the evaluator's expectations.
                "retrieved_contexts": [],
                "web_pages": [],
                "tool_calls": [],
                "source_registry": [],
            }]
            out.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
        ok = sum(run["status"] == "success" for run in runs)
        print(f"   {ok} of {len(items)} answers saved to {out}", flush=True)


if __name__ == "__main__":
    main()
