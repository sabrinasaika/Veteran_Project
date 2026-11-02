#!/usr/bin/env bash
# Judge the 15-question comparison locally with the same judge and settings as run 4
# (llama3.3:70b, rubric + fact check + ground-truth correctness, 1 repeat). RAGAS is skipped:
# the GPT and Claude baselines retrieve no evidence, so there is nothing for it to score.
#
#   bash statistics/comparison/judge_comparison.sh
#
# Each system is skipped if its llm_judge_results.json already exists, so rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../.."
umask 077

PY=.venv/bin/python
JUDGE_MODEL="llama3.3:70b"
BENCHMARK="data/rsa911_private/comparison_benchmark_15.json"
RUNS="statistics/comparison/runs"
SYSTEMS=(gpt-oss_20b qwen3_8_27b mistral-small3_2_24b openrouter_anthropic_claude-sonnet-4.5 openrouter_openai_gpt-5 openrouter_openai_gpt-5_sysprompt openrouter_anthropic_claude-sonnet-4.5_sysprompt)

[ -f "$BENCHMARK" ] || { echo "Missing $BENCHMARK. Run: $PY statistics/comparison/prepare_comparison.py make-benchmark" >&2; exit 1; }

log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

for system in "${SYSTEMS[@]}"; do
    dir="$RUNS/$system"
    if [ -s "$dir/llm_judge_results.json" ]; then
        log "$system already judged, skipping"
        continue
    fi
    [ -s "$dir/benchmark_runs.json" ] || { log "$system: no benchmark_runs.json, skipping"; continue; }

    log "Prepare $system"
    if ! "$PY" evaluate_agent.py prepare --runs "$dir/benchmark_runs.json" --benchmark "$BENCHMARK" \
        --only-benchmark --dataset-out "$dir/ragas_dataset.json" 2>&1 | tee -a "$dir/judge_log.txt"; then
        log "Prepare failed for $system"; continue
    fi
    cp statistics/current_evaluation/prepared_runs.json "$dir/prepared_runs.json"

    log "Judge $system with $JUDGE_MODEL"
    "$PY" evaluate_agent.py judge --dataset "$dir/prepared_runs.json" --model "$JUDGE_MODEL" \
        --repeats 1 --out "$dir/llm_judge_results.json" 2>&1 | tee -a "$dir/judge_log.txt" \
        || log "Judge failed for $system"
done

log "Done"
for system in "${SYSTEMS[@]}"; do
    printf '  %-42s %s\n' "$system" "$([ -s "$RUNS/$system/llm_judge_results.json" ] && echo judged || echo 'NOT judged')"
done
