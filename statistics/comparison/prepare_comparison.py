"""Build the 15-question comparison runs for our agent models.

The six RSA-911 questions were reworded without tool instructions, so they need new agent runs; the
other nine are word-for-word the same as in run 4, so their run-4 answers are reused.

  make-benchmark  Write the comparison benchmarks (neutral question text, ground truth from the private
                  benchmark) into the private data folder: one with the six reworded questions to run,
                  and one with all 15 for judging.
  merge           For each model, combine the six new answers with the nine run-4 answers into
                  statistics/comparison/runs/<model>/benchmark_runs.json.

Usage (from the project folder):
  .venv/bin/python statistics/comparison/prepare_comparison.py make-benchmark
  .venv/bin/python statistics/comparison/prepare_comparison.py merge
"""

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
QUESTIONS = HERE / "comparison_questions.json"
PRIVATE = ROOT / "data" / "rsa911_private"
SOURCE_BENCHMARK = PRIVATE / "eval_benchmark_expanded_rsa911.json"
BENCHMARK_RERUN = PRIVATE / "comparison_benchmark_rsa6.json"
BENCHMARK_ALL = PRIVATE / "comparison_benchmark_15.json"
RUNS_DIR = HERE / "runs"
MODELS = ["gpt-oss_20b", "qwen3_8_27b", "mistral-small3_2_24b"]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    path.chmod(0o600)


def make_benchmark(_args) -> None:
    items = _read(QUESTIONS)["items"]
    source = {case["id"]: case for case in _read(SOURCE_BENCHMARK)}
    missing = [item["id"] for item in items if item["id"] not in source]
    if missing:
        raise SystemExit(f"Not in {SOURCE_BENCHMARK.name}: {missing}")
    # Same entry as the main benchmark (ground truth included), with the neutral question text.
    cases = [{**source[item["id"]], "question": item["question"]} for item in items]
    _write(BENCHMARK_ALL, cases)
    _write(BENCHMARK_RERUN, [case for case, item in zip(cases, items) if item["reworded"]])
    print(f"Wrote {BENCHMARK_ALL} ({len(cases)} questions) and {BENCHMARK_RERUN} "
          f"({sum(item['reworded'] for item in items)} questions to run)")


def merge(_args) -> None:
    items = _read(QUESTIONS)["items"]
    for model in MODELS:
        out_dir = RUNS_DIR / model
        new_path = out_dir / "rsa6_runs.json"
        if not new_path.exists():
            print(f"{model}: {new_path} missing; run the six questions first")
            continue
        new = {run["benchmark_id"]: run for run in _read(new_path) if run.get("status") == "success"}
        old = {run["benchmark_id"]: run for run in _read(ROOT / "statistics" / "runs" / f"run4_{model}" / "benchmark_runs.json")}
        runs, problems = [], []
        for item in items:
            run = new.get(item["id"]) if item["reworded"] else old.get(item["id"])
            if run is None:
                problems.append(f"{item['id']}: no successful answer")
            elif str(run.get("question") or "").strip() != item["question"].strip():
                problems.append(f"{item['id']}: question text differs from the comparison set")
            else:
                runs.append({**run, "comparison_source": "new run" if item["reworded"] else "run4"})
        _write(out_dir / "benchmark_runs.json", runs)
        print(f"{model}: {len(runs)} of {len(items)} answers -> {out_dir / 'benchmark_runs.json'}")
        for problem in problems:
            print(f"  WARNING {problem}")


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make-benchmark").set_defaults(func=make_benchmark)
    sub.add_parser("merge").set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
