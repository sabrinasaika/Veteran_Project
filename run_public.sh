#!/usr/bin/env bash
# Start the web app on localhost and expose it through a Cloudflare quick tunnel.
# Press Ctrl+C to stop both. The public URL changes every time this runs.
set -euo pipefail

cd "$(dirname "$0")"
PORT="${APP_PORT:-7860}"
CLOUDFLARED="${CLOUDFLARED:-$HOME/bin/cloudflared}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# A Cloudflare tunnel decrypts traffic at Cloudflare, so RSA-911 answers would leave this server.
if [ -d "${RSA911_WORK_DIR:-data/rsa911_private}" ]; then
    echo "Refusing to start a public tunnel: private RSA-911 data is on this server." >&2
    echo "Use SSH port forwarding instead: ssh -L $PORT:127.0.0.1:$PORT <user>@<server>" >&2
    exit 1
fi
if [ ! -x "$CLOUDFLARED" ]; then
    echo "cloudflared not found at $CLOUDFLARED" >&2
    exit 1
fi
if ss -ltn | grep -q "127.0.0.1:$PORT "; then
    echo "Port $PORT is already in use. Stop the running app first (pkill -f app_web.py)." >&2
    exit 1
fi

cleanup() {
    echo
    echo "Stopping app and tunnel..."
    kill "${APP_PID:-}" "${CF_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting app on http://127.0.0.1:$PORT (loading models, this takes a minute)..."
APP_PORT="$PORT" .venv/bin/python app_web.py > "$LOG_DIR/app.log" 2>&1 &
APP_PID=$!
until curl -s -o /dev/null "http://127.0.0.1:$PORT/"; do
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        echo "App exited during startup. See $LOG_DIR/app.log" >&2
        exit 1
    fi
    sleep 2
done

echo "Starting Cloudflare tunnel..."
"$CLOUDFLARED" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$LOG_DIR/cloudflared.log" 2>&1 &
CF_PID=$!
URL=""
for _ in $(seq 30); do
    URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG_DIR/cloudflared.log" | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
done
if [ -z "$URL" ]; then
    echo "Tunnel did not start. See $LOG_DIR/cloudflared.log" >&2
    exit 1
fi

echo
echo "Public URL: $URL"
echo "Logs: $LOG_DIR/app.log, $LOG_DIR/cloudflared.log"
echo "Press Ctrl+C to stop."
wait "$APP_PID"
