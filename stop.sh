#!/usr/bin/env bash
#
# Stop the DeepSeek V4 Flash 0731 Spark server (container), preserving
# ./data, ./cache and all weights.
#
# This gracefully stops the running container via `docker compose stop`
# (SIGTERM to vLLM). Restart later with: ./start.sh
#
# Usage:
#   ./stop.sh              # stop the running container
#   ./stop.sh -f           # remove the container after stopping (down)
#
set -Eeuo pipefail
cd "$(dirname "$0")"
COMPOSE_FILE=compose.yml

case "${1:-}" in
  -f|--force|down)
    echo ">> Removing container (preserves ./data and ./cache) via $COMPOSE_FILE ..."
    docker compose -f "$COMPOSE_FILE" down "$@"
    ;;
  "")
    echo ">> Stopping DeepSeek V4 Flash Spark server (preserves ./data and ./cache) via $COMPOSE_FILE ..."
    docker compose -f "$COMPOSE_FILE" stop
    ;;
  *)
    echo "Unknown argument: $1" >&2
    exit 1
    ;;
esac

echo
if docker compose -f "$COMPOSE_FILE" ps --status running --quiet 2>/dev/null | grep -q .; then
  echo "Container still running. Verify:  ./start.sh ps"
else
  echo "Server stopped. Start again with: ./start.sh"
fi
