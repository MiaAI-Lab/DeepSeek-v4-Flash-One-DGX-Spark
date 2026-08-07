#!/usr/bin/env bash
#
# start.sh — bring up Entrpi/ds4-on-spark and serve DeepSeek-V4-Flash on :8888.
#
# Heads-up: this repo does NOT use vLLM. ds4-on-spark is the ds4-server
# (DwarfStar 4, C/CUDA) engine, which exposes the same OpenAI-compatible /v1
# API that `vllm serve` does. vLLM cannot read this repo's asymmetric GGUF,
# so the repo ships its own server; ds4-serve is that server's launcher.
#
# This is a thin, idempotent wrapper over the repo's official installer:
#   * first run : verify host, clone+build the pinned fork, download the
#     ~110 GiB GGUF set, smoke-test, install ds4-serve, start on :8888.
#   * later runs : fast-forward the clone to the pinned tag, skip any GGUF
#     already on disk, start the server on :8888.
#
# Bind address: ds4-server defaults to 127.0.0.1 and the upstream installer
# has no --host flag, so this wrapper always launches ds4-serve itself with
# --host (default 0.0.0.0 = LAN-reachable). Override with HOST=127.0.0.1 for
# loopback-only.
#
# Locations (installer defaults, override via env):
#   source  -> ~/code/ds4   ($DS4_SRC_DIR)
#   weights -> ~/gguf       ($DS4_GGUF_DIR)
#   log     -> ~/ds4-server.log
#
# Usage:
#   ./start.sh                        # full DSpark stack on 0.0.0.0:8888, ctx 1000000 (~1M)
#   PORT=8889 ./start.sh              # different port
#   HOST=127.0.0.1 ./start.sh         # loopback-only (upstream default)
#   CTX=262144 ./start.sh             # smaller context budget (KV ~= 9.5 KiB/token)
#   ./start.sh --no-dspark            # plain continuous decode (passes through)
#
# To force a clean restart: ./stop.sh; ./start.sh
set -euo pipefail

REPO=https://github.com/Entrpi/ds4-on-spark
PORT="${PORT:-8888}"
CTX="${CTX:-1000000}"
# LAN-reachable by default. ds4-server's own default is 127.0.0.1; the
# upstream installer cannot override it, so we own the serve step.
HOST="${HOST:-0.0.0.0}"
DS4_SERVE="${DS4_SERVE:-$HOME/.local/bin/ds4-serve}"
LOG="${DS4_SERVER_LOG:-$HOME/ds4-server.log}"

command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 1; }

local_models_url() { echo "http://127.0.0.1:$PORT/v1/models"; }

is_answering() {
    curl -sf "$(local_models_url)" >/dev/null 2>&1
}

# True when the live listener matches the requested HOST.
# HOST=0.0.0.0 / :: / * → accept any non-loopback wildcard bind.
# HOST=127.0.0.1 / localhost → loopback is enough.
# Anything else → exact address match.
bind_matches_host() {
    local listeners
    listeners="$(ss -H -tln "( sport = :$PORT )" 2>/dev/null || true)"
    [[ -n "$listeners" ]] || return 1

    case "$HOST" in
        127.0.0.1|localhost)
            # Loopback-only request: any listener on the port is fine.
            return 0
            ;;
        0.0.0.0|\*|::)
            # Must be reachable off-box — reject pure 127.0.0.1 / ::1 binds.
            echo "$listeners" | grep -qE "0\.0\.0\.0:${PORT}|\*:${PORT}|\[::\]:${PORT}"
            ;;
        *)
            echo "$listeners" | grep -qE "${HOST}:${PORT}|\*:${PORT}|0\.0\.0\.0:${PORT}|\[::\]:${PORT}"
            ;;
    esac
}

report_models() {
    curl -s "$(local_models_url)" | python3 -m json.tool 2>/dev/null || \
        curl -s "$(local_models_url)" || true
    local listeners
    listeners="$(ss -H -tln "( sport = :$PORT )" 2>/dev/null || true)"
    if [[ -n "$listeners" ]]; then
        echo "Listeners:"
        echo "$listeners"
    fi
}

stop_server() {
    if pkill -x ds4-server 2>/dev/null; then
        echo "==> Stopped existing ds4-server (rebind/restart)."
    fi
    # Wait until the port is actually freed (same idea as stop.sh).
    local i
    for i in $(seq 1 50); do
        if ! ss -H -tln "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; then
            return 0
        fi
        # Still a process? keep waiting. Orphaned LISTEN without process: break.
        if ! pgrep -x ds4-server >/dev/null 2>&1; then
            sleep 0.2
            if ! ss -H -tln "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; then
                return 0
            fi
        fi
        sleep 0.2
    done
    echo "warning: port :$PORT still busy after stop — check ss/pkill" >&2
}

# Already serving with the right bind? Idempotent no-op.
if is_answering && bind_matches_host; then
    echo "Already serving on ${HOST}:$PORT — nothing to do. (Restart with: ./stop.sh; ./start.sh)"
    report_models
    exit 0
fi

# Answering but bound wrong (classic: 127.0.0.1 while HOST=0.0.0.0) → rebind.
if is_answering && ! bind_matches_host; then
    echo "==> Server is up on :$PORT but bind does not match HOST=$HOST — rebinding."
    stop_server
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

echo "==> Fetching installer from $REPO ..."
curl -fsSL "$REPO/raw/main/install.sh" -o "$tmp"

# Install/build/download only. Do NOT pass --start: upstream install.sh has no
# --host and always launches ds4-serve on 127.0.0.1. We own the serve step so
# HOST is honored. Unknown flags (e.g. --no-dspark) are install-recognized and
# only affect its optional --start path, so we keep "$@" for forward-compat
# but the real serve flags go to ds4-serve below.
echo "==> Running installer (no --start; port $PORT, ctx $CTX) ..."
# Filter serve-only flags out of the installer argv so a future strict parser
# cannot choke; today --no-dspark/--no-spec are accepted by install.sh too.
install_args=()
serve_args=()
for a in "$@"; do
    case "$a" in
        --no-dspark|--no-spec|--with-dspark|--with-mtp)
            serve_args+=("$a")
            install_args+=("$a")
            ;;
        *)
            install_args+=("$a")
            serve_args+=("$a")
            ;;
    esac
done
bash "$tmp" --port "$PORT" --ctx "$CTX" ${install_args[@]+"${install_args[@]}"}

if [[ ! -x "$DS4_SERVE" ]]; then
    echo "error: ds4-serve not found at $DS4_SERVE (installer should have put it in ~/.local/bin)" >&2
    exit 1
fi

# If something else grabbed the port during install smoke tests, clear it.
if ss -H -tln "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; then
    if ! is_answering || ! bind_matches_host; then
        stop_server
    fi
fi

echo "==> Starting ds4-serve --host $HOST --port $PORT -c $CTX ${serve_args[*]:+${serve_args[*]}}"
# ds4-serve exec's into ds4-server; nohup + disown matches upstream install.sh.
nohup "$DS4_SERVE" --host "$HOST" --port "$PORT" -c "$CTX" \
    ${serve_args[@]+"${serve_args[@]}"} \
    >"$LOG" 2>&1 < /dev/null & disown || true

echo "==> Waiting for $(local_models_url) ..."
ok=0
for i in $(seq 1 60); do
    if is_answering; then
        ok=1
        break
    fi
    sleep 2
done

echo
if [[ "$ok" -eq 1 ]]; then
    if bind_matches_host; then
        echo "ds4-server is up on http://${HOST}:$PORT  (local check: $(local_models_url))"
    else
        echo "ds4-server answers locally but bind may not match HOST=$HOST — check listeners / $LOG" >&2
    fi
    report_models
else
    echo "!!! Not reachable yet — check $LOG" >&2
    exit 1
fi
