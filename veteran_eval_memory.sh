#!/usr/bin/env bash
# Keep a standing GPU memory reservation for this project, so the memory is available whenever an
# experiment needs it. run_experiment.sh borrows from it while running and returns it afterwards.
#
#   bash veteran_eval_memory.sh start [GB]   reserve GB of GPU memory (default 50) until stopped
#   bash veteran_eval_memory.sh status       show how much is held
#   bash veteran_eval_memory.sh stop         release it for other users
set -euo pipefail
cd "$(dirname "$0")"

DIR="$HOME/.veteran_eval_memory"
GB="${2:-50}"
running() { pgrep -u "$USER" -f "veteran_eval_memory.py $DIR/control" >/dev/null; }

case "${1:-}" in
    start)
        mkdir -p "$DIR" && chmod 700 "$DIR"
        echo "$GB" > "$DIR/standing_gb"
        echo "$GB" > "$DIR/control"
        if running; then
            echo "Reservation already running; resizing to $GB GB"
        else
            tmux new -d -s veteran_eval_memory ".venv/bin/python veteran_eval_memory.py $DIR/control $DIR/status"
        fi
        for _ in $(seq 1 120); do
            [ "$(cat "$DIR/status" 2>/dev/null)" = "$GB" ] && break
            sleep 0.5
        done
        echo "Holding $(cat "$DIR/status" 2>/dev/null || echo 0) of $GB GB of GPU memory"
        ;;
    status)
        if running; then echo "Holding $(cat "$DIR/status" 2>/dev/null || echo 0) GB of GPU memory"; else echo "Not holding any GPU memory"; fi
        ;;
    stop)
        rm -f "$DIR/standing_gb"
        echo 0 > "$DIR/control" 2>/dev/null || true
        tmux kill-session -t veteran_eval_memory 2>/dev/null || true
        pkill -u "$USER" -f "veteran_eval_memory.py $DIR/control" 2>/dev/null || true
        echo "Released the GPU memory reservation"
        ;;
    *)
        sed -n '2,9p' "$0"
        exit 1
        ;;
esac
