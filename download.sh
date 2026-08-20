#!/usr/bin/env bash
#
# One-time bootstrap: download + coalesce the DeepSeek V4 Flash 0731 weights.
#
# This is the OTHER half of AUTO_DOWNLOAD=0 in start.sh. It runs the exact
# download/coalesce step the image entrypoint would otherwise run on first
# boot, but standalone and under your control. After this completes, you can
# run start.sh with AUTO_DOWNLOAD=0 and the server never touches the network.
#
# Behavior (mirrors the image entrypoint, same commands, same env):
#   1. mounts the shared weights folder (./start.sh mount)
#   2. runs the pinned image ONCE to snapshot_download the repo (~107 GB)
#      into the shared cache, then coalesces TP4->TP1 into ./data/tp1
#   3. verifies the TP1 manifest (SHA-256 inventory) unless --skip-checksums
#
# Idempotent: if ./data/tp1/rank-sliced-tp1-manifest.json already exists it
# exits early (use --force to redo).
#
# Usage:
#   ./download.sh              # full bootstrap
#   ./download.sh --no-wait    # (accepted, no-op: downloads are synchronous)
#   ./download.sh --force      # re-download + re-coalesce even if present
#   ./download.sh --skip-checksums
#
set -Eeuo pipefail
cd "$(dirname "$0")"

MODEL_REPO="${MODEL_REPO:-0xSero/deepseek-v4-flash-0731-spark}"
MODEL_REVISION="${MODEL_REVISION:-22f28d32b9b29b4352eaa380ff8c2c170b2847ab}"
IMAGE_DIGEST="ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"
VERIFY_MODEL_CHECKSUMS="1"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --skip-checksums) VERIFY_MODEL_CHECKSUMS="0" ;;
    --no-wait|-no-wait) : ;;
    -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

REMOTE_HOST="${REMOTE_HOST:-10.0.0.1}"
REMOTE_USER="${REMOTE_USER:-mia}"
REMOTE_SHARE_DIR="${REMOTE_SHARE_DIR:-/home/mia/shared}"
MIA_MOUNT="${MIA_MOUNT:-$HOME/mnt/mia-shared}"
HF_CACHE="${HF_CACHE:-$MIA_MOUNT}"
HF_CACHE="$(realpath -m "$HF_CACHE")"

MODEL_PATH="./data/tp1"
MANIFEST="$MODEL_PATH/rank-sliced-tp1-manifest.json"

# Shell out to the same mount logic start.sh ships (idempotent).
if [ -f "$MANIFEST" ] && [ "$FORCE" != "1" ]; then
  echo "TP1 manifest already present at $MANIFEST — nothing to do. (--force to redo)"
  exit 0
fi

echo ">> Preparing shared weights folder ..."
./start.sh mount

snapshot_host="$HF_CACHE/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REVISION"
snapshot_in_container="/hf-cache/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REVISION"
mkdir -p "$HF_CACHE" "$snapshot_host" "$MODEL_PATH"

verify_args=()
[ "$VERIFY_MODEL_CHECKSUMS" != "1" ] && verify_args+=(--skip-checksums)

echo ">> Downloading $MODEL_REPO @ $MODEL_REVISION (~107 GB, first time only)..."
docker pull -q "$IMAGE_DIGEST"

docker run --rm \
  -e HF_HOME=/hf-cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MODEL_REPO="$MODEL_REPO" \
  -e MODEL_REVISION="$MODEL_REVISION" \
  -e MODEL_SOURCE_DIR="$snapshot_in_container" \
  -e MODEL_PATH=/models/tp1 \
  -e COALESCE_WORKERS="${COALESCE_WORKERS:-1}" \
  -v "$HF_CACHE":/hf-cache \
  -v "$(pwd)/data":/models \
  -v "$(pwd)/image-patch/coalesce_rank_sliced_exl3.py":/opt/recipe/scripts/coalesce_rank_sliced_exl3.py:ro \
  --entrypoint /bin/bash \
  "$IMAGE_DIGEST" -lc '
set -Eeuo pipefail
/opt/runtime-venv/bin/python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_SOURCE_DIR"],
    token=os.environ.get("HF_TOKEN") or None,
)
PY
/opt/runtime-venv/bin/python /opt/recipe/scripts/coalesce_rank_sliced_exl3.py \
    --input-dir "${MODEL_SOURCE_DIR}" \
    --output-dir /models/tp1 \
    --link-carried \
    --reuse-complete \
    --workers "${COALESCE_WORKERS:-1}"
'

echo ">> Verifying TP1 manifest ..."
docker run --rm \
  -v "$(pwd)/data":/models \
  --entrypoint /opt/runtime-venv/bin/python \
  "$IMAGE_DIGEST" /opt/recipe/scripts/verify_tp1_manifest.py /models/tp1 "${verify_args[@]}"

echo ">> Done. TP1 checkpoint ready at ./data/tp1."
echo "   Start the server offline with:  AUTO_DOWNLOAD=0 ./start.sh"
