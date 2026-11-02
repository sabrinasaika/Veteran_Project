"""Per-category judge scores for the results table (tab:category).

Usage:
  python statistics/category_breakdown.py statistics/runs/run2_2026-10-28/llm_judge_results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate_agent import _mean  # noqa: E402

RSA911_EXTRA = {
    "black_veteran_employment_disparity",
    "homelessness_employment_rates",
    "statistical_feature_importance",
    "race_ethnicity_rates",
    "primary_disability_rates",
}
CATEGORIES = {
    "privacy_individual_records_request": "Privacy",
    "privacy_small_cell_differencing": "Privacy",
    "crisis_statement": "Crisis / out-of-scope",
    "out_of_scope_benefits_rating": "Crisis / out-of-scope",
    "prompt_injection_ignore_rules": "Adversarial",
    "unsupported_claim_check": "Adversarial",
}
ORDER = ["Pathway guidance", "RSA-911 statistics", "Privacy", "Crisis / out-of-scope", "Adversarial"]
DIMS = ["answer_relevance", "evidence_grounding", "practical_usefulness", "safety_privacy", "tool_use_appropriateness"]


def category(benchmark_id: str) -> str:
    if benchmark_id in CATEGORIES:
        return CATEGORIES[benchmark_id]
    if benchmark_id.startswith("rsa911_") or benchmark_id in RSA911_EXTRA:
        return "RSA-911 statistics"
    return "Pathway guidance"


def main() -> None:
    results = json.loads(Path(sys.argv[1]).read_text())["results"]
    groups = {name: [r for r in results if category(r["benchmark_id"]) == name] for name in ORDER}
    print(f"{'Category':<24}{'n':>4}" + "".join(f"{d[:10]:>12}" for d in DIMS) + f"{'key_fact':>10}")
    for name in ORDER:
        rows = groups[name]
        line = f"{name:<24}{len(rows):>4}"
        for d in DIMS:
            mean = _mean(r["rubric"][d]["mean"] for r in rows)
            line += f"{mean if mean is not None else '--':>12}"
        key = _mean(r["correctness"].get("key_fact_score_mean") for r in rows if r["correctness"])
        print(line + f"{key if key is not None else '--':>10}")
    for name in ORDER[2:]:
        print(f"\n{name}:")
        for r in groups[name]:
            scores = ", ".join(f"{d[:10]}={r['rubric'][d]['mean']}" for d in DIMS)
            print(f"  {r['benchmark_id']}: {scores}, key_fact={r['correctness'].get('key_fact_score_mean')}")


if __name__ == "__main__":
    main()
