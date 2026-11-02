"""Merge a re-judged item into an existing llm_judge_results.json and recompute the summary.

Each part (rubric, fact-check) is taken from the re-judge only if it succeeded there;
otherwise the original part and its error are kept. The original correctness block is
kept (re-judge with --skip-correctness). Merged items are recorded in config["rescored"].

Usage:
  python statistics/merge_rejudge.py RESULTS.json REJUDGE.json [--parts rubric factual] [--note TEXT]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate_agent import JUDGE_DIMENSIONS, _mean  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("rejudge", type=Path)
    parser.add_argument("--parts", nargs="+", choices=["rubric", "factual"], default=["rubric", "factual"])
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    new = json.loads(args.rejudge.read_text())

    for item in new["results"]:
        matches = [i for i, r in enumerate(data["results"]) if r["benchmark_id"] == item["benchmark_id"]]
        if len(matches) != 1:
            raise SystemExit(f"{item['benchmark_id']} found {len(matches)} times in {args.results}")
        old = data["results"][matches[0]]
        ok = {
            "rubric": all(item["rubric"][n]["scores"] for n in JUDGE_DIMENSIONS),
            "factual": bool(item["factual"]),
        }
        prefix = {"rubric": "rubric", "factual": "factcheck"}
        merged = dict(old)
        errors = list(old["errors"])
        taken = []
        for part in args.parts:
            if ok[part]:
                merged[part] = item[part]
                errors = [e for e in errors if not e.startswith(prefix[part])]
                taken.append(part)
            else:
                print(f"{item['benchmark_id']}: re-judge {part} failed, keeping original", flush=True)
        if not taken:
            continue
        merged["errors"] = errors
        merged.setdefault("original_errors", old["errors"])
        data["results"][matches[0]] = merged
        data["config"].setdefault("rescored", []).append(
            {
                "benchmark_id": item["benchmark_id"],
                "parts": taken,
                "judge_model": new["config"]["judge_model"],
                "temperature": new["config"]["temperature"],
                "created_utc": new["config"]["created_utc"],
                "note": args.note,
            }
        )

    results = data["results"]
    summary = data["summary"]
    for name in JUDGE_DIMENSIONS:
        summary["rubric_1_5"][name] = {
            "mean": _mean(r["rubric"][name]["mean"] for r in results),
            "mean_within_item_sd": _mean(r["rubric"][name]["sd"] for r in results),
            "n": sum(1 for r in results if r["rubric"][name]["mean"] is not None),
        }
    summary["factual"] = {
        key: _mean(r["factual"].get(key) for r in results if r["factual"])
        for key in ("supported_rate_mean", "contradicted_mean", "not_found_mean", "knowledge_incorrect_mean", "n_claims_mean")
    }
    summary["factual"]["n"] = sum(1 for r in results if r["factual"])
    summary.setdefault("failures_before_rescore", summary["failures"])
    summary["failures"] = {
        "rubric": sum(1 for r in results if any(e.startswith("rubric") for e in r["errors"])),
        "factcheck": sum(1 for r in results if any(e.startswith("factcheck") for e in r["errors"])),
        "correctness": sum(1 for r in results if any(e.startswith("correctness") for e in r["errors"])),
    }

    args.results.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
