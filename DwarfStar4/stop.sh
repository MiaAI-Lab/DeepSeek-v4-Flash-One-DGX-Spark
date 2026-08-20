#!/usr/bin/env bash
#
# stop.sh — stop the ds4-server started by start.sh.
#
# Usage:
#   ./stop.sh                 # stop server on default port :8888
#   PORT=8889 ./stop.sh       # stop server on a different port
set -euo pipefail

PORT="${PORT:-8888}"

if pkill -x ds4-server; then
    echo "Stopped ds4-server."
else
    echo "No ds4-server process found."
fi

# Wait until the port is actually freed
while ss -tlnp "sport = :$PORT" 2>/dev/null | grep -q ":$PORT"; do
    sleep 0.3
done

echo "Port :$PORT is free."
