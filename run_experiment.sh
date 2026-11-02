#!/usr/bin/env bash
# Run the model comparison end to end, one model at a time: the agent model answers the benchmark,
# then the judge scores that run (rubric, fact check, correctness), RAGAS runs on all retrieved
# evidence, and a report is written, before the next agent model starts. Everything runs locally
# through Ollama.
#
#   bash run_experiment.sh
#
# A failure never stops the experiment: a failed question or crashed step is logged, the script
# carries on with everything else, and at the end it goes back and retries only what failed
# (up to MAX_RETRIES times). Rerunning the script later also resumes; finished work is kept.
#
# Optional settings (environment variables):
#   JUDGE_REPEATS=1   judge repeats per answer (3 gives consistency figures but takes about 3x longer)
#   RUN_RAGAS=1       set to 0 to skip RAGAS
#   MAX_RETRIES=2     retry passes for failed questions and failed scoring steps
#   RUN_TAG=run4      folder prefix: results go to statistics/runs/<RUN_TAG>_<model>/
#   HOLD_GPU_GB=60    GPU memory (GB) reserved for the experiment for its whole duration, so other jobs
#                     cannot take it between model switches; 0 turns the reservation off. It covers the
#                     largest model (the judge); during agent runs the unused part stays reserved.
set -uo pipefail
cd "$(dirname "$0")"
umask 077

AGENTS=("gpt-oss:20b" "qwen3.8:27b" "mistral-small3.2:24b")
JUDGE_MODEL="llama3.3:70b"
BENCHMARK="data/rsa911_private/eval_benchmark_expanded_rsa911.json"
JUDGE_REPEATS="${JUDGE_REPEATS:-1}"
RUN_RAGAS="${RUN_RAGAS:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
RUN_TAG="${RUN_TAG:-run4}"
HOLD_GPU_GB="${HOLD_GPU_GB:-60}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
# Context windows must match what the agent and judge request, or Ollama reloads the model.
AGENT_CTX=$(grep -E '^\s*OLLAMA_NUM_CTX=' .env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '[:space:]"')
AGENT_CTX="${AGENT_CTX:-32768}"
JUDGE_CTX="${JUDGE_NUM_CTX:-32768}"
# Shared with veteran_eval_memory.sh: if a standing reservation is running, the experiment borrows it and gives
# it back at the end instead of starting and stopping its own.
HOLD_DIR="$HOME/.veteran_eval_memory"
HOLDER_PID=""
STANDING_HOLD=0
PY=.venv/bin/python
OLLAMA="${OLLAMA:-$HOME/bin/ollama}"
# Ollama's server log. Every "truncating input prompt" line means a prompt outgrew the context window
# and lost its oldest text (for the agent, possibly the user's question), so each step counts them.
OLLAMA_LOG="${OLLAMA_LOG:-$HOME/ollama_logs/ollama.log}"

log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
# Unload every model Ollama has in memory, so the next model loads fully onto the GPU.
# --- GPU memory reservation -----------------------------------------------------------------------
start_holder() {
    [ "$HOLD_GPU_GB" -gt 0 ] || return 0
    mkdir -p "$HOLD_DIR" && chmod 700 "$HOLD_DIR"
    local standing; standing=$(pgrep -u "$USER" -f "veteran_eval_memory.py $HOLD_DIR/control" | head -1 || true)
    if [ -n "$standing" ]; then
        HOLDER_PID=$standing
        STANDING_HOLD=1
        log "Using the standing GPU reservation (veteran_eval_memory.sh)"
    else
        echo 0 > "$HOLD_DIR/control"
        "$PY" veteran_eval_memory.py "$HOLD_DIR/control" "$HOLD_DIR/status" &
        HOLDER_PID=$!
    fi
    trap cleanup EXIT INT TERM
}

# Resize the reservation to N GB and wait (up to 60 s) until the holder reports it.
hold_set() {
    [ -n "$HOLDER_PID" ] || return 0
    local target=$(( $1 > 0 ? $1 : 0 )) i
    echo "$target" > "$HOLD_DIR/control"
    for i in $(seq 1 120); do
        [ "$(cat "$HOLD_DIR/status" 2>/dev/null)" = "$target" ] && return 0
        sleep 0.5
    done
    log "Note: reserved $(cat "$HOLD_DIR/status" 2>/dev/null || echo 0) of $target GB (not enough free GPU memory)"
}

# Load a model now and keep it loaded, with the context window its requests will use.
preload() {
    curl -s --max-time 900 "$OLLAMA_URL/api/generate" \
        -d "{\"model\": \"$1\", \"keep_alive\": -1, \"options\": {\"num_ctx\": $2}}" >/dev/null || true
}

# GB of GPU memory a loaded model occupies (rounded up).
model_gpu_gb() {
    curl -s --max-time 10 "$OLLAMA_URL/api/ps" | "$PY" -c "
import json, math, sys
models = json.load(sys.stdin).get('models', [])
print(sum(math.ceil(m.get('size_vram', 0) / 1024**3) for m in models if m.get('name') == sys.argv[1]))" "$1"
}

# Switch the GPU to another model with as short a gap as possible: unload the old model, release the
# reservation, load the new model at once, then re-reserve whatever the new model does not use. A
# few GB are left free for the Python processes (the agent's retrieval models and RAGAS embeddings).
switch_to() {
    local model=$1 ctx=$2 used
    unload_models
    hold_set 0
    preload "$model" "$ctx"
    used=$(model_gpu_gb "$model")
    log "Loaded $model: ${used} GB on the GPU"
    [ -n "$HOLDER_PID" ] && hold_set $(( HOLD_GPU_GB - used - 4 ))
    return 0
}

cleanup() {
    trap - EXIT INT TERM
    unload_models
    if [ -n "$HOLDER_PID" ] && [ "$STANDING_HOLD" = "1" ]; then
        # Give the memory back to the standing reservation instead of releasing it.
        hold_set "$(cat "$HOLD_DIR/standing_gb" 2>/dev/null || echo 0)"
        log "Returned GPU memory to the standing reservation ($(cat "$HOLD_DIR/status" 2>/dev/null || echo 0) GB held)"
    elif [ -n "$HOLDER_PID" ]; then
        echo 0 > "$HOLD_DIR/control" 2>/dev/null
        kill "$HOLDER_PID" 2>/dev/null
        wait "$HOLDER_PID" 2>/dev/null
        log "Released all GPU memory held by the experiment"
    else
        log "Unloaded all models"
    fi
    HOLDER_PID=""
}

unload_models() {
    local loaded m
    loaded=$("$OLLAMA" ps 2>/dev/null | awk 'NR > 1 {print $1}')
    for m in $loaded; do "$OLLAMA" stop "$m" >/dev/null 2>&1 || true; done
    return 0
}
log_mark() { if [ -f "$OLLAMA_LOG" ]; then wc -l < "$OLLAMA_LOG"; else echo 0; fi; }
truncations_since() {
    [ -f "$OLLAMA_LOG" ] || { echo 0; return; }
    tail -n +"$(($1 + 1))" "$OLLAMA_LOG" | grep -c "truncating input prompt" || true
}
# Records the truncations a step caused in <run dir>/context_overflows.txt and warns if there were any.
record_truncations() {
    local dir=$1 step=$2 mark=$3 n; n=$(truncations_since "$mark")
    printf '%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$step" "$n" >> "$dir/context_overflows.txt"
    [ "$n" -gt 0 ] && log "WARNING: $n prompt(s) overflowed the context window during $step (see $dir/context_overflows.txt)"
    return 0
}
overflow_total() { [ -f "$1/context_overflows.txt" ] && awk -F'\t' -v s="$2" '$2 ~ s {t += $3} END {print t + 0}' "$1/context_overflows.txt" || echo 0; }
run_dir() { echo "statistics/runs/${RUN_TAG}_$(echo "$1" | tr ':.' '__')"; }

# Prints "<expected> <succeeded>" for a run: expected is the number of benchmark questions, so a
# run that crashed part way (fewer saved answers) also counts as unfinished.
answer_counts() {
    "$PY" - "$BENCHMARK" "$1" <<'PYEOF'
import json, os, sys
expected = len(json.load(open(sys.argv[1])))
runs = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else []
print(expected, sum(r.get("status") == "success" for r in runs))
PYEOF
}

agent_done() {
    local expected ok
    read -r expected ok < <(answer_counts "$(run_dir "$1")/benchmark_runs.json")
    [ "$ok" = "$expected" ]
}

scoring_done() {
    local dir; dir=$(run_dir "$1")
    [ -s "$dir/evaluation_report.md" ] && [ -s "$dir/llm_judge_results.json" ] \
        && { [ "$RUN_RAGAS" != "1" ] || [ -s "$dir/ragas_results.json" ]; }
}

# Answers the benchmark with one agent model. --resume keeps answered questions and reruns only
# failed or missing ones, so the same call serves for the first pass and every retry.
run_agent() {
    local model=$1 dir; dir=$(run_dir "$model")
    mkdir -p "$dir"
    log "Agent $model -> $dir"
    local mark; mark=$(log_mark)
    OLLAMA_MODEL="$model" "$PY" evaluate_agent.py run-benchmark --benchmark "$BENCHMARK" \
        --runs-out "$dir/benchmark_runs.json" --resume 2>&1 | tee -a "$dir/agent_log.txt" \
        || log "Agent run for $model stopped with an error; it will be retried later"
    record_truncations "$dir" "agent" "$mark"
    local expected ok
    read -r expected ok < <(answer_counts "$dir/benchmark_runs.json")
    log "Agent $model: $ok of $expected answers succeeded"
}

# Prepare, judge, RAGAS, and report for one run. Each step is skipped if its output already exists,
# so retries redo only what failed. Returns non-zero if any step failed.
score_run() {
    local model=$1 dir status=0; dir=$(run_dir "$model")
    if ! agent_done "$model"; then
        local expected ok; read -r expected ok < <(answer_counts "$dir/benchmark_runs.json")
        log "Note: scoring $model with $ok of $expected answers (the rest failed after all retries)"
    fi

    if [ ! -s "$dir/prepared_runs.json" ] || [ ! -s "$dir/ragas_dataset.json" ]; then
        log "Prepare $model"
        if "$PY" evaluate_agent.py prepare --runs "$dir/benchmark_runs.json" --benchmark "$BENCHMARK" \
            --only-benchmark --ragas-all-evidence --dataset-out "$dir/ragas_dataset.json" 2>&1 | tee -a "$dir/judge_log.txt"; then
            cp statistics/current_evaluation/prepared_runs.json "$dir/prepared_runs.json"
        else
            log "Prepare failed for $model"; return 1
        fi
    fi

    local mark
    if [ ! -s "$dir/llm_judge_results.json" ]; then
        log "Judge $model with $JUDGE_MODEL"
        mark=$(log_mark)
        "$PY" evaluate_agent.py judge --dataset "$dir/prepared_runs.json" --model "$JUDGE_MODEL" \
            --repeats "$JUDGE_REPEATS" --out "$dir/llm_judge_results.json" 2>&1 | tee -a "$dir/judge_log.txt" \
            || { log "Judge failed for $model"; status=1; }
        record_truncations "$dir" "judge" "$mark"
    fi

    if [ "$RUN_RAGAS" = "1" ] && [ ! -s "$dir/ragas_results.json" ]; then
        log "RAGAS (all evidence) for $model with $JUDGE_MODEL"
        mark=$(log_mark)
        OLLAMA_JUDGE_MODEL="$JUDGE_MODEL" "$PY" evaluate_agent.py ragas --dataset "$dir/ragas_dataset.json" \
            --out "$dir/ragas_results.json" --no-merge-existing 2>&1 | tee -a "$dir/ragas_log.txt" \
            || { log "RAGAS failed for $model"; status=1; }
        record_truncations "$dir" "ragas" "$mark"
    fi

    if [ -s "$dir/llm_judge_results.json" ]; then
        "$PY" evaluate_agent.py report --judge "$dir/llm_judge_results.json" --ragas "$dir/ragas_results.json" \
            --out "$dir/evaluation_report.md" || { log "Report failed for $model"; status=1; }
    fi
    return $status
}

# --- Preflight: local model only, benchmark present, every model installed -------------------
grep -qE '^\s*AGNO_MODEL_PROVIDER=ollama\s*$' .env || { echo "AGNO_MODEL_PROVIDER must be ollama in .env (local only)." >&2; exit 1; }
[ -f "$BENCHMARK" ] || { echo "Missing $BENCHMARK. Run: .venv/bin/python -m rsa911.build_ground_truth" >&2; exit 1; }
installed=$("$OLLAMA" list | awk 'NR > 1 {print $1}')
for model in "${AGENTS[@]}" "$JUDGE_MODEL"; do
    grep -qx "$model" <<<"$installed" || { echo "Model $model is not installed. Run: $OLLAMA pull $model" >&2; exit 1; }
done
[ -f "$OLLAMA_LOG" ] || log "WARNING: $OLLAMA_LOG not found, so context overflows cannot be counted. Start Ollama with: ollama serve >> $OLLAMA_LOG 2>&1"
log "Agents: ${AGENTS[*]} | judge: $JUDGE_MODEL | judge repeats: $JUDGE_REPEATS | RAGAS: $RUN_RAGAS | retries: $MAX_RETRIES"
if [ "$HOLD_GPU_GB" -gt 0 ]; then
    start_holder
    hold_set "$HOLD_GPU_GB"
    log "Reserved $(cat "$HOLD_DIR/status" 2>/dev/null || echo 0) GB of GPU memory for the experiment"
fi
log "Privacy budget before:"
"$PY" -m rsa911.dp

# --- 1. One model at a time: answer, retry failed questions, then score -----------------------
# Each model's results are complete before the next model starts. Models are unloaded before every
# switch so the next one gets the whole free GPU instead of loading partly onto the CPU.
for model in "${AGENTS[@]}"; do
    if agent_done "$model"; then
        log "Agent $model already complete; skipping"
    else
        switch_to "$model" "$AGENT_CTX"
        run_agent "$model"
        for attempt in $(seq 1 "$MAX_RETRIES"); do
            agent_done "$model" && break
            log "Retry $attempt/$MAX_RETRIES: rerunning failed questions for $model"
            run_agent "$model"
        done
    fi
    if scoring_done "$model"; then
        log "Scoring for $model already complete; skipping"
    else
        switch_to "$JUDGE_MODEL" "$JUDGE_CTX"
        score_run "$model" || log "Scoring for $model had a failure; it will be retried at the end"
    fi
done

# --- 2. Final retry pass for anything still unfinished ------------------------------------------
for attempt in $(seq 1 "$MAX_RETRIES"); do
    for model in "${AGENTS[@]}"; do
        if ! agent_done "$model"; then
            log "Final retry $attempt/$MAX_RETRIES: rerunning failed questions for $model"
            switch_to "$model" "$AGENT_CTX"
            run_agent "$model"
        fi
        if ! scoring_done "$model"; then
            log "Final retry $attempt/$MAX_RETRIES: scoring $model"
            switch_to "$JUDGE_MODEL" "$JUDGE_CTX"
            score_run "$model" || true
        fi
    done
done

# --- 3. Summary ----------------------------------------------------------------------------------
log "Privacy budget after:"
"$PY" -m rsa911.dp
log "Summary"
for model in "${AGENTS[@]}"; do
    read -r expected ok < <(answer_counts "$(run_dir "$model")/benchmark_runs.json")
    scoring_done "$model" && scored="scored" || scored="SCORING INCOMPLETE"
    dir=$(run_dir "$model")
    printf '  %-24s answers %s/%s  |  %s  |  context overflows: agent %s, judge %s, RAGAS %s  |  %s\n' "$model" "$ok" "$expected" "$scored" \
        "$(overflow_total "$dir" agent)" "$(overflow_total "$dir" judge)" "$(overflow_total "$dir" ragas)" "$dir"
done
cleanup
log "Done. Anything incomplete can be finished by running this script again."
