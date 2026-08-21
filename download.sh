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
#   1. downloads the selected repo into the LOCAL HuggingFace cache
#   2. default model: coalesces TP4->TP1 into ./data/tp1 and verifies it
#   3. TurboDerp model: keeps the ready-to-serve 2.77bpw branch in the HF cache
#   4. LAN-share mode: set REMOTE_HOST to download into a mounted folder
#
# Idempotent: the default checks the TP1 manifest; the TurboDerp preset checks
# a completion marker in its branch snapshot (use --force to recheck).
#
# Usage:
#   ./download.sh              # full bootstrap
#   ./download.sh --no-wait    # (accepted, no-op: downloads are synchronous)
#   ./download.sh --force      # re-download + re-coalesce even if present
#   ./download.sh --skip-checksums
#   MODEL_VARIANT=turboderp-2.77bpw ./download.sh
#
set -Eeuo pipefail
cd "$(dirname "$0")"

MODEL_VARIANT="${MODEL_VARIANT:-default}"
case "$MODEL_VARIANT" in
  default|0xsero)
    MODEL_LAYOUT_DEFAULT=rank-sliced
    MODEL_REPO_DEFAULT=0xSero/deepseek-v4-flash-0731-spark
    MODEL_REVISION_DEFAULT=22f28d32b9b29b4352eaa380ff8c2c170b2847ab
    DOWNLOAD_SIZE='~107 GB'
    ;;
  turboderp|turboderp-2.77bpw)
    MODEL_LAYOUT_DEFAULT=hf
    MODEL_REPO_DEFAULT=turboderp/DeepSeek-V4-Flash-0731-exl3
    MODEL_REVISION_DEFAULT=2.77bpw
    DOWNLOAD_SIZE='~108 GB'
    ;;
  *) echo "MODEL_VARIANT must be default or turboderp-2.77bpw (got '$MODEL_VARIANT')" >&2; exit 1 ;;
esac
MODEL_LAYOUT="$MODEL_LAYOUT_DEFAULT"
MODEL_REPO="${MODEL_REPO:-$MODEL_REPO_DEFAULT}"
MODEL_REVISION="${MODEL_REVISION:-$MODEL_REVISION_DEFAULT}"
IMAGE_DIGEST="ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"
VERIFY_MODEL_CHECKSUMS="1"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --skip-checksums) VERIFY_MODEL_CHECKSUMS="0" ;;
    --no-wait|-no-wait) : ;;
    -h|--help) awk 'NR > 1 && /^set -Eeuo/ {exit} NR > 1 {print}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

REMOTE_HOST="${REMOTE_HOST:-}"     # empty = fully local (default)
REMOTE_USER="${REMOTE_USER:-mia}"  # used only when a remote share is enabled
REMOTE_SHARE_DIR="${REMOTE_SHARE_DIR:-/home/mia/shared}"
MIA_MOUNT="${MIA_MOUNT:-$HOME/mnt/mia-shared}"
# Local default matches start.sh: weights land on THIS machine in ./hf-hub.
HF_CACHE="${HF_CACHE:-$([ -n "$REMOTE_HOST" ] && echo "$MIA_MOUNT" || echo "$(pwd)/hf-hub")}"
HF_CACHE="$(realpath -m "$HF_CACHE")"

MODEL_PATH="./data/tp1"
MANIFEST="$MODEL_PATH/rank-sliced-tp1-manifest.json"

# Shell out to the same mount logic start.sh ships (idempotent).
if [ "$MODEL_LAYOUT" = "rank-sliced" ] && [ -f "$MANIFEST" ] && [ "$FORCE" != "1" ]; then
  echo "TP1 manifest already present at $MANIFEST — nothing to do. (--force to redo)"
  exit 0
fi

if [ -n "$REMOTE_HOST" ]; then
  echo ">> Preparing shared weights folder (${REMOTE_USER}@${REMOTE_HOST}) ..."
  ./start.sh mount
else
  echo ">> Local mode: downloads go to $HF_CACHE"
fi

snapshot_host="$HF_CACHE/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REVISION"
snapshot_in_container="/hf-cache/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REVISION"
mkdir -p "$HF_CACHE" "$snapshot_host" "$MODEL_PATH"
SNAPSHOT_MARKER="$snapshot_host/.snapshot-download-complete"
if [ "$MODEL_LAYOUT" = "hf" ] && [ -f "$SNAPSHOT_MARKER" ] && \
   [ -s "$snapshot_host/model.safetensors.index.json" ] && [ "$FORCE" != "1" ]; then
  echo "Hugging Face checkpoint already present at $snapshot_host — nothing to do. (--force to recheck)"
  exit 0
fi
[ "$FORCE" = "1" ] && rm -f "$SNAPSHOT_MARKER"

verify_args=()
[ "$VERIFY_MODEL_CHECKSUMS" != "1" ] && verify_args+=(--skip-checksums)

echo ">> Downloading $MODEL_REPO @ $MODEL_REVISION ($DOWNLOAD_SIZE, first time only)..."
docker pull -q "$IMAGE_DIGEST"

docker run --rm \
  -e HF_HOME=/hf-cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MODEL_REPO="$MODEL_REPO" \
  -e MODEL_REVISION="$MODEL_REVISION" \
  -e MODEL_LAYOUT="$MODEL_LAYOUT" \
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
if [[ "${MODEL_LAYOUT}" == "rank-sliced" ]]; then
  /opt/runtime-venv/bin/python /opt/recipe/scripts/coalesce_rank_sliced_exl3.py \
      --input-dir "${MODEL_SOURCE_DIR}" \
      --output-dir /models/tp1 \
      --link-carried \
      --reuse-complete \
      --workers "${COALESCE_WORKERS:-1}"
fi
touch "${MODEL_SOURCE_DIR}/.snapshot-download-complete"
'

if [ "$MODEL_LAYOUT" = "rank-sliced" ]; then
  echo ">> Verifying TP1 manifest ..."
  docker run --rm \
    -v "$(pwd)/data":/models \
    --entrypoint /opt/runtime-venv/bin/python \
    "$IMAGE_DIGEST" /opt/recipe/scripts/verify_tp1_manifest.py /models/tp1 "${verify_args[@]}"
  echo ">> Done. TP1 checkpoint ready at ./data/tp1."
else
  echo ">> Done. Hugging Face checkpoint ready at $snapshot_host."
fi
echo "   Start the selected model with: MODEL_VARIANT=$MODEL_VARIANT ./start.sh"
