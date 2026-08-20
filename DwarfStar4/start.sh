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
# Locations (installer defaults, override via env):
#   source  -> ~/code/ds4   ($DS4_SRC_DIR)
#   weights -> ~/gguf       ($DS4_GGUF_DIR)
#   log     -> ~/ds4-server.log
#
# Usage:
#   ./start.sh                        # full DSpark stack on :8888, ctx 1000000 (~1M)
#   PORT=8889 ./start.sh              # different port
#   CTX=262144 ./start.sh             # smaller context budget (KV ~= 9.5 KiB/token)
#   ./start.sh --no-dspark            # plain continuous decode (passes through)
#
# To force a clean restart: pkill -x ds4-server; ./start.sh
set -euo pipefail

REPO=https://github.com/Entrpi/ds4-on-spark
PORT="${PORT:-8888}"
CTX="${CTX:-1000000}"

command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 1; }

# Already serving? Nothing to do — this makes re-runs safe (the installer
# itself never stops old servers, so we don't either unless asked).
if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "Already serving on :$PORT — nothing to do. (Restart with: pkill -x ds4-server; ./start.sh)"
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
    exit 0
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

echo "==> Fetching installer from $REPO ..."
curl -fsSL "$REPO/raw/main/install.sh" -o "$tmp"

echo "==> Running installer (--start, port $PORT, ctx $CTX) ..."
bash "$tmp" --start --port "$PORT" --ctx "$CTX" "$@"

echo
if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ds4-server is up on http://127.0.0.1:$PORT"
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool
else
    echo "!!! Not reachable yet — check $HOME/ds4-server.log" >&2
    exit 1
fi
