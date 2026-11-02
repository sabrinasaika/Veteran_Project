#!/usr/bin/env bash
# Keep private RSA-911 data readable only by you and stop it leaving this server.
#
#   bash rsa911/secure.sh setup    lock down folder permissions and install git guards
#   bash rsa911/secure.sh check    audit permissions and every route data could leave by
#   bash rsa911/secure.sh lock     encrypt the raw CSVs at rest (gpg, AES256) and delete the plain copies
#   bash rsa911/secure.sh unlock   decrypt the raw CSVs (only while building the database)
#
# This script only changes permissions and settings; it never prints or copies data contents.
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
RAW_DIR="${RSA911_RAW_DIR:-$HOME/secure_data/rsa911_raw}"
WORK_DIR="${RSA911_WORK_DIR:-data/rsa911_private}"
# Folders that may hold RSA-911 data or outputs derived from it.
PRIVATE_DIRS=("$HOME/secure_data" "$RAW_DIR" "data" "$WORK_DIR" "statistics" "logs" "exports")
umask 077

ok()   { printf '  OK    %s\n' "$1"; }
bad()  { printf '  FIX   %s\n' "$1"; FAILED=1; }

install_git_hooks() {
    local hook
    for hook in pre-commit pre-push; do
        cat > ".git/hooks/$hook" <<'HOOK'
#!/usr/bin/env bash
# Installed by rsa911/secure.sh: never let private RSA-911 data or derived outputs into git.
pattern='^data/|rsa911.*\.(sqlite3?|csv|json)$|\.sqlite3?$|eval_benchmark_expanded_rsa911|^statistics/.*\.(json|csv)$|^logs/|^exports/'
if [ "$(basename "$0")" = "pre-commit" ]; then
    files=$(git diff --cached --name-only --diff-filter=ACMR)
else
    files=$(git diff --name-only "@{upstream}" HEAD 2>/dev/null || git ls-files)
fi
blocked=$(printf '%s\n' "$files" | grep -E "$pattern" | grep -v '^chroma_db_adv/chroma.sqlite3$' || true)
if [ -n "$blocked" ]; then
    echo "Blocked: these files may contain private RSA-911 data and must not be committed or pushed:" >&2
    printf '  %s\n' $blocked >&2
    exit 1
fi
HOOK
        chmod 700 ".git/hooks/$hook"
    done
}

setup() {
    echo "Locking down RSA-911 folders (only you can read them)..."
    mkdir -p "$RAW_DIR" "$WORK_DIR"
    local dir
    for dir in "${PRIVATE_DIRS[@]}"; do
        [ -e "$dir" ] || continue
        chmod -R go-rwx "$dir"
        echo "  $dir -> owner only"
    done
    chmod 700 "$HOME"
    echo "  $HOME -> owner only"
    [ -f .env ] && chmod 600 .env && echo "  .env -> owner only"
    install_git_hooks
    echo "  git pre-commit and pre-push guards installed"
    echo "Done. Run 'bash rsa911/secure.sh check' to confirm."
}

check() {
    FAILED=0
    echo "Who can read the data"
    local dir mode
    for dir in "$HOME" "${PRIVATE_DIRS[@]}"; do
        [ -e "$dir" ] || continue
        mode=$(stat -c '%a' "$dir")
        if [ $((8#$mode & 8#077)) -eq 0 ]; then ok "$dir is owner-only ($mode)"; else bad "$dir can be opened by others ($mode): run setup"; fi
    done
    if [ -f .env ]; then
        mode=$(stat -c '%a' .env)
        if [ $((8#$mode & 8#077)) -eq 0 ]; then ok ".env is owner-only"; else bad ".env is readable by others ($mode): run setup"; fi
    fi
    local plain
    plain=$({ find "$RAW_DIR" -maxdepth 1 -name '*.csv' 2>/dev/null || true; } | wc -l)
    if [ "$plain" -eq 0 ]; then ok "no unencrypted raw CSVs in $RAW_DIR"; else bad "$plain raw CSV(s) unencrypted: run 'lock' when not building the database"; fi

    echo "Routes data could leave by"
    local provider
    provider=$(grep -E '^\s*AGNO_MODEL_PROVIDER=' .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]"' || true)
    if [ "$provider" = "ollama" ]; then ok "model is local (Ollama); agent refuses RSA-911 otherwise"; else bad "AGNO_MODEL_PROVIDER is '${provider:-unset}', not ollama: RSA-911 tools are disabled until it is"; fi
    if pgrep -u "$USER" -x cloudflared >/dev/null; then bad "a Cloudflare tunnel is running: stop it (pkill cloudflared)"; else ok "no Cloudflare tunnel running"; fi
    if ss -ltn 2>/dev/null | grep -E ':(7860|11434) ' | grep -vqE '127\.0\.0\.1:|\[::1\]:'; then
        bad "the app or Ollama is listening on a network interface, not just localhost"
    else
        ok "app and Ollama listen on localhost only"
    fi
    if [ -x .git/hooks/pre-commit ] && grep -q 'rsa911/secure.sh' .git/hooks/pre-commit; then ok "git guards installed"; else bad "git guards missing: run setup"; fi
    local tracked
    tracked=$(git ls-files | grep -E '^data/|rsa911.*\.(sqlite3?|csv)$|eval_benchmark_expanded_rsa911' || true)
    if [ -z "$tracked" ]; then ok "no RSA-911 data tracked by git"; else bad "RSA-911 files are tracked by git: $tracked"; fi
    if find /tmp -maxdepth 2 -user "$USER" -iname '*rsa911*' 2>/dev/null | grep -q .; then bad "RSA-911 files found in /tmp: delete them"; else ok "no RSA-911 files left in /tmp"; fi

    if [ "$FAILED" -eq 0 ]; then echo "All checks passed."; else echo "Some checks need fixing (see FIX lines)."; return 1; fi
}

lock() {
    command -v gpg >/dev/null || { echo "gpg not found" >&2; exit 1; }
    local f n=0
    for f in "$RAW_DIR"/*.csv; do
        [ -e "$f" ] || continue
        gpg --yes --symmetric --cipher-algo AES256 --output "$f.gpg" "$f"
        shred -u "$f"
        n=$((n + 1))
    done
    chmod -R go-rwx "$RAW_DIR"
    echo "Encrypted $n file(s); plain copies shredded. Keep the passphrase somewhere safe: without it the files cannot be recovered."
}

unlock() {
    local f n=0
    for f in "$RAW_DIR"/*.csv.gpg; do
        [ -e "$f" ] || continue
        gpg --yes --decrypt --output "${f%.gpg}" "$f"
        n=$((n + 1))
    done
    chmod -R go-rwx "$RAW_DIR"
    echo "Decrypted $n file(s). Run 'bash rsa911/secure.sh lock' again as soon as the database is built."
}

case "${1:-}" in
    setup) setup ;;
    check) check ;;
    lock) lock ;;
    unlock) unlock ;;
    *) sed -n '2,9p' "$0"; exit 1 ;;
esac
