#!/usr/bin/env bash
# Score contextual fit (does the answer use the user's stated details?) for the systems in the RQ6
# table, with the same judge as the main comparison. Only the question and answer go to the judge.
# Results go to runs/<system>/contextual_results.json; the existing judge results are not touched.
#
#   bash statistics/comparison/judge_contextual.sh
#
# Each system is skipped if its contextual_results.json already exists, so rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")/../.."
umask 077

PY=.venv/bin/python
JUDGE_MODEL="llama3.3:70b"
RUNS="statistics/comparison/runs"
SYSTEMS=(gpt-oss_20b qwen3_8_27b mistral-small3_2_24b openrouter_openai_gpt-5_sysprompt openrouter_anthropic_claude-sonnet-4.5_sysprompt)

log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

for system in "${SYSTEMS[@]}"; do
    dir="$RUNS/$system"
    if [ -s "$dir/contextual_results.json" ]; then
        log "$system already scored, skipping"
        continue
    fi
    [ -s "$dir/prepared_runs.json" ] || { log "$system: no prepared_runs.json, run judge_comparison.sh first"; continue; }

    log "Contextual fit for $system with $JUDGE_MODEL"
    "$PY" evaluate_agent.py judge --dataset "$dir/prepared_runs.json" --model "$JUDGE_MODEL" \
        --dimensions contextual_fit --skip-factcheck --skip-correctness --repeats 1 \
        --out "$dir/contextual_results.json" 2>&1 | tee -a "$dir/contextual_log.txt" \
        || log "Contextual fit failed for $system"
done

log "Done"
for system in "${SYSTEMS[@]}"; do
    printf '  %-50s %s\n' "$system" "$([ -s "$RUNS/$system/contextual_results.json" ] && echo scored || echo 'NOT scored')"
done
