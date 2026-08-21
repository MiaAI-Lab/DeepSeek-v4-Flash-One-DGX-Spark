#!/usr/bin/env bash
#
# DeepSeek V4 Flash 0731 (EXL3/ExLlamaV3) on one NVIDIA DGX Spark.
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ DEEP-CONTEXT VARIANT — 384k single-request (validated 2026-08-21)          │
# │ Native 432 B NVFP4 KV records + util 0.94 → 439,622-token pool with        │
# │ DSpark speculative decoding healthy (acceptance ~0.65/0.44/0.31/0.17/0.07) │
# │   * the project's only launcher; writes ./compose.yml                      │
# └──────────────────────────────────────────────────────────────────────────┘
#

# Weights are downloaded AUTOMATICALLY on first boot into ./hf-hub — fully
# LOCAL on this machine, no remote server involved. The losslessly coalesced
# TP1 serving checkpoint, the K64 draft, and runtime caches stay under
# ./data and ./cache. To keep one ~107 GB copy shared across LAN machines,
# set REMOTE_HOST (and friends) below: the folder is then mounted over SSHFS
# and used as the cache root instead (the old default, now opt-in). No
# HuggingFace login is required (model and image are public).
#
# All tunables live in this file.  compose.yml is GENERATED — do not edit it.
# The runtime image is aarch64-only and needs the NVIDIA Container Toolkit.
#
# Usage:
#   ./start.sh              # start deep-context config (mounts share only in remote mode) + wait for /health
#   ./start.sh --no-wait    # start without waiting for /health
#   ./start.sh mount        # mount the shared folder (no-op in local mode)
#   ./start.sh unmount      # unmount the shared folder
#   ./start.sh logs         # tail container logs
#   ./start.sh stop         # stop the container (preserves ./data and caches)
#   ./start.sh restart      # restart the container
#   ./start.sh down         # remove container (preserves ./data and caches)
#   ./start.sh ps           # show container status
#   ./start.sh pull         # pull the pinned image now
#   ./start.sh compose-gen  # only (re)generate compose.yml, no docker action
#   ./start.sh help
#
# NOTE: this writes compose.yml and exports COMPOSE_FILE accordingly.
# compose.yml is regenerated on every start, so the live server keeps running
# until you actually invoke this script.
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export COMPOSE_FILE=compose.yml   # this variant's own compose file


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Tunables (see the model card before changing these)
# ---------------------------------------------------------------------------
MODEL_REPO="${MODEL_REPO:-0xSero/deepseek-v4-flash-0731-spark}"
MODEL_REVISION="${MODEL_REVISION:-22f28d32b9b29b4352eaa380ff8c2c170b2847ab}"
# Optional HuggingFace token. The repo is public, so normally NOT needed —
# set it only to avoid rate limits or to reach a private repo. Also honored
# from the environment or a local .env next to compose.yml. Exported so the
# generated compose.yml's ${HF_TOKEN:-} resolves at `docker compose up` time.
export HF_TOKEN="${HF_TOKEN:-}"
IMAGE_DIGEST="ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-0731}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-384000}"   # single deep context: ~13% under the worst observed cold-boot pool (439,622 tokens, util 0.94). If a boot ever fails the "KV cache needed" check, lower this.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"          # single-request deep-context server; raise for concurrent slots (the pool itself shrinks with seq count — hybrid cache split; 2 seqs observed ~337k total, ~169k each)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"   # ALSO sizes the profiler dummy forward, whose "peak activation" vLLM subtracts from the KV pool: 8224 reserves 5.30 GiB, 2048 reserves 1.39 GiB. Prefill is unchanged (997 vs 1013 tok/s on the same 60k prompt, cached_tokens=0) because LONG_PREFILL_TOKEN_THRESHOLD=1024 already caps the per-step allocation
MODE="${MODE:-dspark}"                 # fixed K5 DSpark speculative draft
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"   # 256k needs the extra ~1.2 GiB (0.93 leaves only 6.32 GiB KV < 6.99 needed). Boot-safe BECAUSE restart: on-failure:1 can never loop: worst case one clean exit. Requires free host RAM >= 0.94*121.63 = 114.3 GiB at launch (check free -h; stop the old container first).
VERIFY_MODEL_CHECKSUMS="${VERIFY_MODEL_CHECKSUMS:-1}"

# --- 256k KV-record layout (improve-context.md levers #1 / #2) -----------
# The image ships the FAT 584-byte padded FP8-compat KV record
# (VLLM_DSV4_PADDED_NVFP4=1). Smaller native NVFP4 records give +35%..+59%
# more KV tokens with no weight/util change:
#   KV_RECORD=padded   -> 584 B (stock semantics, fallback if NVFP4 breaks)
#   KV_RECORD=stock432 -> 432 B native NVFP4  (439,622-token pool at util 0.94;
#                        prefill/dual-cache bugs FIXED 2026-08-20 — see
#                        internal postmortem, not in this repo)  [default: max context]
#   KV_RECORD=rope368  -> 368 B FP8-RoPE (GLM-only knob; on DSV4 == stock432)
KV_RECORD="${KV_RECORD:-stock432}"
case "$KV_RECORD" in
  padded)    export KV_PADDED_NVFP4=1; export KV_FP8_ROPE=0 ;;  # no levers on
  stock432)  export KV_PADDED_NVFP4=0; export KV_FP8_ROPE=0 ;;  # lever #1
  rope368)   export KV_PADDED_NVFP4=0; export KV_FP8_ROPE=1 ;;  # levers #1+#2
  *) die "KV_RECORD must be padded|stock432|rope368 (got '$KV_RECORD')" ;;
esac

# --- CPU KV offload (improve-context.md lever #3) -------------------------
# UMA machine: "CPU" memory is the same physical 128 GiB pool, so offloading
# cold KV blocks extends the pool past the profiler's plan with low penalty.
# 0 = disabled. GiB of CPU-side KV to allow offloading.
KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-0}"   # 0=off (default). CPU-KV-offload (lever
    # #3) was dropped as the default because the SimpleCPUOffloadConnector is
    # incompatible with the image's PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # (hard VllmConfig error). The 368-byte record alone reaches ~287k tokens,
    # enough for 256k. If you DO enable offload (>0), you must also set
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (else config fails).

# --- Scheduler fairness knobs (see PLAN.md P1) -----------------------------
# The image's serve-ds4-flash.sh appends EXTRA_VLLM_ARGS verbatim (word-split)
# to the vllm serve command, so we route both knobs through it. They are
# never added if set to 0, so this repo stays byte-identical to its previous
# launch unless you opt in.
#   LONG_PREFILL_TOKEN_THRESHOLD: cap each chunked-prefill chunk. This fork's
#     scheduler ENFORCES it (scheduler.py), which stops decode starvation under
#     concurrent large prefills (MiaAI issue #27/#43). Default 1024.
#   MAX_NUM_PARTIAL_PREFILLS: overlap in-flight partial prefills (MiaAI #90).
#     NOTE: this fork's scheduler only carries the config field, it has no
#     partial-prefill overlap logic, so >1 is currently a no-op until a code
#     port lands (P3). Kept wired for when it exists.
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-1024}"  # 0=off
MAX_NUM_PARTIAL_PREFILLS="${MAX_NUM_PARTIAL_PREFILLS:-0}"            # 0=off, 2=overlap (no-op on this fork)

# --- CudaGraph capture sizes (PLAN.md P2) --------------------------------
# The image defaults both of these to 6, which caps cuDAG capture at batch 6.
# DSpark verification schedules up to MAX_NUM_SEQS * (k+1) = 4*6 = 24 rows;
# decode batches above the captured size fall back to eager execution. These
# knobs raise the cap so concurrent decode stays on captured graphs.
#   MAX_CUDAGRAPH_CAPTURE_SIZE: cap for --max-cudagraph-capture-size.
#     In that image the safe maximum is seqs*(k+1) = 24 at this profile;
#     capturing beyond the physical verifier row count only wastes graph
#     memory (serve-ds4-flash.sh comment).
#   CUDAGRAPH_CAPTURE_SIZES: explicit capture list. 0=leave to image default.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-24}"   # 0=image default
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-6,12,24}"     # 0=image default

# --- KV pool headroom search (PLAN.md P5: KV sweep) -----------------------
# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: the profiler over-reserves
# ~0.68 GiB while the real graph pool is ~0.07 GiB. Setting 0 reclaims that
# reservation for KV. Trade-off: if real graph usage ever exceeds the pool,
# capture can OOM. Keep 1 unless chasing KV; we validated 24-size capture
# uses 0.07 GiB real, so 0 is safe at this profile.
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"   # 0 reclaims ~0.6 GiB graph over-reservation

# --- Default thinking (patched serve-ds4-flash.sh) -----------------------
# Server-side default for chat_template_kwargs when a client sends none.
# thinking: true/false, effort: low/high/max. Clients that send their own
# chat_template_kwargs override these. Default here: thinking ON, effort max.
DEFAULT_CHAT_TEMPLATE_KWARGS_THINKING="${DEFAULT_CHAT_TEMPLATE_KWARGS_THINKING:-true}"
DEFAULT_CHAT_TEMPLATE_KWARGS_EFFORT="${DEFAULT_CHAT_TEMPLATE_KWARGS_EFFORT:-max}"

# --- Model weights cache: LOCAL by default, remote SSHFS share OPT-IN -------
# DEFAULT: fully local — weights download straight into ./hf-hub on this
# machine, no remote needed. To share a single copy of the ~107 GB weights
# across machines on a LAN (spark3 + shared-folder setup), set REMOTE_HOST:
# the script mounts the folder via SSHFS and uses it as the cache root.
# REMOTE_HOST / REMOTE_USER / REMOTE_SHARE_DIR / MIA_MOUNT / HF_CACHE are
# all env-overridable.
REMOTE_HOST="${REMOTE_HOST:-}"     # empty = local mode (the default)
REMOTE_USER="${REMOTE_USER:-mia}"  # used only when a remote share is enabled
REMOTE_SHARE_DIR="${REMOTE_SHARE_DIR:-/home/mia/shared}"   # created if missing
MIA_MOUNT="${MIA_MOUNT:-$HOME/mnt/mia-shared}"              # local SSHFS mountpoint

HF_CACHE="${HF_CACHE:-$([ -n "$REMOTE_HOST" ] && echo "$MIA_MOUNT" || echo "$SCRIPT_DIR/hf-hub")}"
HF_CACHE="$(realpath -m "$HF_CACHE")"

# vLLM serving port (the pinned runtime honours the PORT env var; the image's
# own HEALTHCHECK stays on 8000, so we override it below too).
SERVING_PORT="${SERVING_PORT:-8888}"

STARTUP_WAIT="${STARTUP_WAIT:-7200}"   # seconds to wait for /health on start
LOCAL_MIN_FREE_GIB="${LOCAL_MIN_FREE_GIB:-130}"   # local needs: ~99 tp1 + ~10 image + draft
SHARE_MIN_FREE_GIB="${SHARE_MIN_FREE_GIB:-130}"   # share needs: ~107 weights + headroom
POLL_INTERVAL="${POLL_INTERVAL:-15}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

usage() {
  sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

require_cmds() {
  command -v docker >/dev/null 2>&1 || die "docker not found. Install Docker Engine first."
  docker compose version >/dev/null 2>&1 \
    || die "docker compose v2 not found. Install Docker Compose v2."
  command -v curl >/dev/null 2>&1 || die "curl not found. Install curl (needed for the health wait)."
}

# ---------------------------------------------------------------------------
# Shared-folder mount (SSHFS)
# ---------------------------------------------------------------------------
ensure_sshfs() {
  command -v sshfs >/dev/null 2>&1 && return 0
  warn "sshfs (FUSE) is not installed on this host."
  if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    die "install it manually with:  sudo apt-get update && sudo apt-get install -y sshfs fuse3"
  fi
  echo ">> Installing sshfs and fuse3 (sudo will request your password)..."
  if sudo -n true 2>/dev/null; then
    sudo -n apt-get update -qq && sudo -n apt-get install -y sshfs fuse3
  elif [ -t 0 ]; then
    sudo apt-get update -qq && sudo apt-get install -y sshfs fuse3
  else
    die "not on a TTY and no passwordless sudo. Run manually:  sudo apt-get update && sudo apt-get install -y sshfs fuse3"
  fi
  command -v sshfs >/dev/null 2>&1 \
    || die "sshfs still missing after the install attempt. Install it manually."
}

is_mounted() {
  findmnt -rno TARGET "$MIA_MOUNT" >/dev/null 2>&1
}

ssh_remote() {
  ssh -o ConnectTimeout=8 "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

ensure_fuse_allow_other() {
  grep -v '^#' /etc/fuse.conf 2>/dev/null | grep -q 'user_allow_other' && return 0
  warn "sshfs mounts must allow other users (user_allow_other in /etc/fuse.conf) so Docker can access the share."
  if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "  run manually:  sudo sh -c 'echo user_allow_other >> /etc/fuse.conf'"
    return 1
  fi
  echo ">> Enabling user_allow_other in /etc/fuse.conf (sudo password may be requested)..."
  if sudo -n true 2>/dev/null; then
    sudo -n sh -c 'echo user_allow_other >> /etc/fuse.conf'
  elif [ -t 0 ]; then
    sudo sh -c 'echo user_allow_other >> /etc/fuse.conf'
  else
    echo "  (no passwordless sudo, not on a TTY) run manually:"
    echo "  sudo sh -c 'echo user_allow_other >> /etc/fuse.conf'"
    return 1
  fi
  grep -v '^#' /etc/fuse.conf 2>/dev/null | grep -q user_allow_other \
    && echo ">> user_allow_other enabled." || return 1
}

mount_share() {
  # Local mode (the default): nothing to mount — the HF cache is a local dir.
  [ -n "$REMOTE_HOST" ] || return 0
  ensure_sshfs
  ensure_fuse_allow_other \
    || die "cannot use the shared folder without user_allow_other; run the sudo command above then retry."
  mkdir -p "$MIA_MOUNT"
  if is_mounted; then
    # A mount without allow_other is invisible to Docker; remount it correctly.
    if ! awk '$2=="'"$MIA_MOUNT"'" {print $4}' /proc/mounts | grep -q allow_other; then
      echo ">> Existing mount lacks allow_other; remounting..."
      unmount_share
    else
      timeout 8 ls "$MIA_MOUNT" >/dev/null 2>&1 \
        || warn "the mount at $MIA_MOUNT looks stale; consider: ./start.sh unmount"
      return 0
    fi
  fi
  echo ">> Preparing shared model folder on ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SHARE_DIR}"
  ssh_remote "mkdir -p '${REMOTE_SHARE_DIR}'" \
    || die "cannot SSH to ${REMOTE_USER}@${REMOTE_HOST}. Check reachability and ~/.ssh/config."
  echo ">> Mounting ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SHARE_DIR} -> $MIA_MOUNT"
  sshfs "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SHARE_DIR}" "$MIA_MOUNT" \
    -o allow_other,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,follow_symlinks,idmap=user \
    || die "SSHFS mount failed. Try: ./start.sh unmount"
  timeout 8 ls "$MIA_MOUNT" >/dev/null 2>&1 \
    || die "share mounted but not readable."
  echo ">> Shared folder ready: $MIA_MOUNT"
}

unmount_share() {
  if is_mounted; then
    fusermount3 -u "$MIA_MOUNT" 2>/dev/null || umount "$MIA_MOUNT" 2>/dev/null \
      || die "could not unmount $MIA_MOUNT"
    echo "Unmounted $MIA_MOUNT"
  else
    echo "Nothing mounted at $MIA_MOUNT"
  fi
}

warn_low_disk() {
  local dir="$1" min_gib="$2"
  [ -d "$dir" ] || return 0
  local avail
  avail=$(( $(df -Pk "$dir" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
  if [ "$avail" -lt "$min_gib" ]; then
    local mp
    mp=$(df -Pk "$dir" | awk 'NR==2 {print $6}')
    warn "only ~${avail} GiB free on ${mp} (mount for $dir); this setup wants at least ${min_gib} GiB there."
  fi
}

preflight() {
  require_cmds
  [ "$(uname -m)" = "aarch64" ] || die \
    "This machine reports $(uname -m), but the runtime image targets " \
    "Linux aarch64 (one DGX Spark / GB10 / SM121)."

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >/dev/null 2>&1 || warn "host nvidia-smi failed; the DGX Spark driver may be missing."
  else
    warn "nvidia-smi not found on the host. Install NVIDIA drivers for the DGX Spark."
  fi

  # GPU passthrough is verified empirically; on DGX OS the toolkit integrates
  # at the containerd level and is NOT listed under docker info Runtimes.
  if ! command -v nvidia-ctk >/dev/null 2>&1 && ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia; then
    warn "no NVIDIA Container Toolkit detected; GPU access inside containers may fail."
  fi

  # Weights live on the share; only the serving checkpoint (~99 GiB), the
  # pinned image (~10 GiB) and the draft (~3 GiB) need local disk.
  warn_low_disk "$SCRIPT_DIR" "$LOCAL_MIN_FREE_GIB"
  warn_low_disk "$HF_CACHE" "$SHARE_MIN_FREE_GIB"
}

# ---------------------------------------------------------------------------
# compose.yml generation
# ---------------------------------------------------------------------------
generate_compose() {
  local cache_repo_dir="models--${MODEL_REPO//\//--}"   # HF cache folder name
  # Host-side path (bound at /hf-cache inside the container):
  local snapshot_host="$HF_CACHE/hub/$cache_repo_dir/snapshots/$MODEL_REVISION"
  # Container-side path. MUST be the in-container mount path, not the host
  # path, or the runtime would download into the container layer instead of
  # the shared folder.
  local snapshot_in_container="/hf-cache/hub/$cache_repo_dir/snapshots/$MODEL_REVISION"

  # Build the EXTRA_VLLM_ARGS string only from the knobs that are enabled.
  # serve-ds4-flash.sh word-splits this value, so keep it flat with no quotes.
  EXTRA_VLLM_ARGS=""
  if [ "${LONG_PREFILL_TOKEN_THRESHOLD:-0}" -gt 0 ] 2>/dev/null; then
    EXTRA_VLLM_ARGS="--long-prefill-token-threshold ${LONG_PREFILL_TOKEN_THRESHOLD}"
  fi
  if [ "${MAX_NUM_PARTIAL_PREFILLS:-0}" -gt 0 ] 2>/dev/null; then
    EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS} --max-num-partial-prefills ${MAX_NUM_PARTIAL_PREFILLS}"
  fi
  # CPU KV offload (lever #3, improve-context.md). 12 GiB default on UMA.
  if [ "${KV_OFFLOAD_GB:-0}" -gt 0 ] 2>/dev/null; then
    EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS} --kv-offloading-size ${KV_OFFLOAD_GB} --kv-offloading-backend native"
  fi
  EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS# }"   # strip leading space when both on

  # CudaGraph knobs: 0 -> empty string so the image's own default applies
  # (entrypoint hardcodes 6 otherwise). serve-ds4-flash.sh requires a
  # positive integer, so never emit 0 or other non-numbers.
  if [ "${MAX_CUDAGRAPH_CAPTURE_SIZE:-0}" = "0" ]; then
    CAPTURE_SIZE_ENV=""
  else
    CAPTURE_SIZE_ENV="${MAX_CUDAGRAPH_CAPTURE_SIZE}"
  fi
  if [ "${CUDAGRAPH_CAPTURE_SIZES:-0}" = "0" ]; then
    CAPTURE_SIZES_ENV=""
  else
    CAPTURE_SIZES_ENV="${CUDAGRAPH_CAPTURE_SIZES}"
  fi

  mkdir -p "$HF_CACHE" "$snapshot_host"   # through the share; keeps dirs user-owned
  mkdir -p data cache

  cat > compose.yml <<YAML
# GENERATED by start.sh -- do not edit. Change knobs in start.sh and rerun.
# 256k-context variant: compact NVFP4 KV records + CPU KV offload.
# Pinned validated recipe from:
#   https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark
# Weights are downloaded by the runtime into the HuggingFace cache
# (${HF_CACHE} — local copy under ./hf-hub, or a LAN-mounted share when
# REMOTE_HOST is set) and coalesced losslessly into ./data/tp1 for serving.
name: deepseek-v4-flash-spark
services:
  deepseek-v4-flash:
    image: ${IMAGE_DIGEST}
    pull_policy: always
    restart: on-failure:1   # CRITICAL: never death-spiral the host. A failing
    # 256k boot must stop after ONE failure (bad boots OOM-looping under
    # 'unless-stopped' pushed host RAM negative and hard-reset the box). If it
    # fails once, ./start.sh again later.
    network_mode: host
    ipc: host
    shm_size: 16gb
    # Boot fix for the 26.02 image: upgrade xgrammar>=0.2.4 (tool calling),
    # restore transformers (pip side effect), then exec the image entrypoint.
    # 256k variant: entrypoint is the PATCHED copy that reads KV_FP8_ROPE /
    # VLLM_DSV4_PADDED_NVFP4 from the environment (see image-patch/).
    entrypoint: ["/bin/bash", "-lc", "bash /patch-run-entrypoint.sh && exec /opt/recipe/scripts/entrypoint.sh"]
    # Override the image's hardcoded-8000 HEALTHCHECK for the new serving port.
    healthcheck:
      test: ["CMD", "curl", "-fsS", "--max-time", "5", "http://127.0.0.1:${SERVING_PORT}/health"]
      interval: 30s
      timeout: 5s
      start_period: 20m
      retries: 3
    environment:
      # Optional token (repo is public). Set HF_TOKEN in start.sh (or env/.env).
      HF_TOKEN: \${HF_TOKEN:-}
      HF_HOME: /hf-cache
      PORT: "${SERVING_PORT}"
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME}
      MODEL_REPO: ${MODEL_REPO}
      MODEL_REVISION: ${MODEL_REVISION}
      # Source weights live directly inside the HF cache snapshot (container
      # path /hf-cache/...), so the 107 GB download is never duplicated under
      # ./data. Local mode: that snapshot is on this machine (./hf-hub).
      MODEL_SOURCE_DIR: ${snapshot_in_container}
      MAX_MODEL_LEN: "${MAX_MODEL_LEN}"
      MAX_NUM_SEQS: "${MAX_NUM_SEQS}"
      MAX_NUM_BATCHED_TOKENS: "${MAX_NUM_BATCHED_TOKENS}"
      # 256k KV-record layout (overrides the image's hardcoded defaults; the
      # entrypoint is the patched copy that honors these):
      VLLM_DSV4_PADDED_NVFP4: "${KV_PADDED_NVFP4}"
      KV_FP8_ROPE: "${KV_FP8_ROPE}"
      MODE: ${MODE}
      DSPARK_TOKENS: "5"
      DSPARK_CAPACITY: "0"
      DSPARK_DYNAMIC_DRAFT_DEPTH: "0"
      DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW: "8"
      DSPARK_DRAFT_EXPERTS: "64"
      DSPARK_STRUCTURED_EXPERTS_PER_CATEGORY: "32"
      VLLM_USE_B12X_WO_PROJECTION: "1"
      GPU_MEMORY_UTILIZATION: "${GPU_MEMORY_UTILIZATION}"
      VERIFY_MODEL_CHECKSUMS: "${VERIFY_MODEL_CHECKSUMS}"
      # CudaGraph capture sizes (P2). Empty = image default.
      MAX_CUDAGRAPH_CAPTURE_SIZE: "${CAPTURE_SIZE_ENV}"
      CUDAGRAPH_CAPTURE_SIZES: "${CAPTURE_SIZES_ENV}"
      # KV sweep (P5): 0 reclaims the graph-memory profiler over-reservation.
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS}"
      # Scheduler fairness (P1): appended verbatim to the vllm serve command
      # by the image. Empty when both knobs are off -> identical launch as before.
      EXTRA_VLLM_ARGS: "${EXTRA_VLLM_ARGS}"
      # Default thinking (patched serve-ds4-flash.sh): thinking ON, effort max.
      DEFAULT_CHAT_TEMPLATE_KWARGS_THINKING: "${DEFAULT_CHAT_TEMPLATE_KWARGS_THINKING}"
      DEFAULT_CHAT_TEMPLATE_KWARGS_EFFORT: "${DEFAULT_CHAT_TEMPLATE_KWARGS_EFFORT}"
      # KV offload is OFF by default (incompatible with the image's
      # expandable_segments allocator; see KV_OFFLOAD_GB notes).
      KV_OFFLOAD_GB: "${KV_OFFLOAD_GB}"
      VLLM_USE_SIMPLE_KV_OFFLOAD: "$([ "${KV_OFFLOAD_GB:-0}" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
    volumes:
      - ./data:/models
      - ./cache:/cache
      - type: bind
        source: "${HF_CACHE}"
        target: /hf-cache
      # Override the image's coalescer: hardlinks fail cross-device when
      # source weights sit on the shared folder, so copy instead.
      - type: bind
        source: "${SCRIPT_DIR}/image-patch/coalesce_rank_sliced_exl3.py"
        target: /opt/recipe/scripts/coalesce_rank_sliced_exl3.py
        read_only: true
      # SparkInfer (formerly b12x) backports from local-inference-lab/b12x,
      # ported onto the pinned kernel tree (272a84bd). See PLAN.md P4.
      # - #150: preallocate W4A16 route histograms for CUDA-graph capture.
      # - #228: keep graphed tiny-decode routes with inactive ids in range.
      - type: bind
        source: "${SCRIPT_DIR}/image-patch/sparkinfer/moe/fused_moe/_impl.py"
        target: /opt/sparkinfer/sparkinfer/moe/fused_moe/_impl.py
        read_only: true
      - type: bind
        source: "${SCRIPT_DIR}/image-patch/sparkinfer/moe/_shared/kernels/tiny_decode.py"
        target: /opt/sparkinfer/sparkinfer/moe/_shared/kernels/tiny_decode.py
        read_only: true
      # Tool-calling boot hook script, invoked by the entrypoint wrapper above.
      - type: bind
        source: "${SCRIPT_DIR}/image-patch/entrypoint-toolfix.sh"
        target: /patch-run-entrypoint.sh
        read_only: true
      # Server-default thinking flags (true/max) via env-templated serve script.
      - type: bind
        source: "${SCRIPT_DIR}/image-patch/serve-ds4-flash.sh"
        target: /opt/vllm/serve-ds4-flash.sh
        read_only: true
      # DSpark draft context-KV writer fix: pass use_nvfp4 to the fused
      # insert so the draft's context KV matches the selected record layout
      # (native 432-byte NVFP4 under VLLM_DSV4_PADDED_NVFP4=0). Without it
      # the draft writes 584-byte FP8-compat records into an NVFP4-record
      # cache and DSpark acceptance collapses to ~0%.
      - type: bind
        source: "${SCRIPT_DIR}/image-patch/vllm/models/deepseek_v4/nvidia/dspark.py"
        target: /opt/vllm/vllm/models/deepseek_v4/nvidia/dspark.py
        read_only: true
      # SparkInfer NVFP4 prefill (>=7-token prompts) fixes: the DSV4
      # dual-cache prefill path mis-dispatched and mis-gathered for native
      # 432-byte NVFP4 KV records, producing NaN prefill logprobs and ~0%
      # DSpark acceptance. Three fixes (validated by image-patch/
      # selftest_extend.py, cosine 0.999998 on single/dual-cache shapes):
      # - prefill.py: thread the caller's scale_format through the has_extra
      #   dual-cache dispatch (was hard-coded UE8M0_BYTE/584B).
      # - prefill_mg.py: NVFP4 record gathering for dual-cache MAIN tiles
      #   (was the FP8 576B-layout gather staging 464B rows into 288B smem)
      #   and EXTRA tiles (own page size / block stride / indices), plus
      #   extra-geometry QK-rope reads.
      - type: bind
        source: "${SCRIPT_DIR}/image-patch/sparkinfer/attention/_shared/mla/prefill.py"
        target: /opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill.py
        read_only: true
      - type: bind
        source: "${SCRIPT_DIR}/image-patch/sparkinfer/attention/_shared/mla/prefill_mg.py"
        target: /opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill_mg.py
        read_only: true
      # 256k variant ONLY: patched recipe entrypoint honoring KV_FP8_ROPE /
      # VLLM_DSV4_PADDED_NVFP4 from env (see image-patch/entrypoint-256k.sh).
      - type: bind
        source: "${SCRIPT_DIR}/image-patch/entrypoint-256k.sh"
        target: /opt/recipe/scripts/entrypoint.sh
        read_only: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
YAML

  echo "Wrote compose.yml"
  echo "  KV record     : $KV_RECORD (padded_NVFP4=$KV_PADDED_NVFP4, fp8_rope=$KV_FP8_ROPE)"
  echo "  model len     : ${MAX_MODEL_LEN}"
  echo "  CPU KV offload: ${KV_OFFLOAD_GB:-0} GiB"
  if [ -n "$REMOTE_HOST" ]; then
    echo "  shared cache  : $HF_CACHE (${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SHARE_DIR})"
  else
    echo "  local cache   : $HF_CACHE (fully local)"
  fi
  echo "  source dir    : $snapshot_host"
  echo "  serving data  : $SCRIPT_DIR/data (TP1 checkpoint, draft, caches)"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_start() {
  local no_wait="${1:-}"
  [ "$no_wait" = "-no-wait" ] && no_wait="--no-wait"
  preflight
  mount_share
  generate_compose

  echo ">> Pulling pinned image (first run only) and starting the server..."
  docker compose up -d

  if [ "$no_wait" = "--no-wait" ]; then
    echo ">> Started. Follow progress with: ./start.sh logs"
    return 0
  fi

  local logs_pid=""
  # Live log stream: starts immediately, is killed (and the shell returned)
  # automatically once the server reports healthy. This compose version
  # ignores SIGTERM/SIGINT while attached to a TTY, so fall back to SIGKILL
  # (a log streamer has no state to lose) or the prompt hangs forever.
  stop_logs() {
    local pid="${logs_pid:-}" i
    [ -n "$pid" ] || return 0
    kill -TERM "$pid" 2>/dev/null || true
    for i in 1 2 3 4 5; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  }
  docker compose logs -f --tail=20 &
  logs_pid=$!
  trap 'stop_logs' EXIT

  echo ">> First boot is intentionally long: pulls image, downloads ~107 GB"
  echo "   of weights onto the shared folder, coalesces TP4->TP1, builds the"
  echo "   K64 draft, captures CUDA graphs. Waiting up to ${STARTUP_WAIT}s for"
  echo "   http://127.0.0.1:${SERVING_PORT}/health — live logs above; Ctrl-C"
  echo "   here only stops the watcher, the server keeps running."
  echo
  local deadline=$(( $(date +%s) + STARTUP_WAIT ))
  while :; do
    if curl -fsS --max-time 5 "http://127.0.0.1:${SERVING_PORT}/health" >/dev/null 2>&1; then
      stop_logs
      trap - EXIT
      echo
      echo ">> Model is serving: http://127.0.0.1:${SERVING_PORT} (OpenAI-compatible)"
      echo "   Served model name: $SERVED_MODEL_NAME"
      echo
      echo "   Try it:"
      echo "     curl -sS http://127.0.0.1:${SERVING_PORT}/v1/chat/completions \\"
      echo "       -H 'Content-Type: application/json' -d '{'"
      echo "         \"model\": \"$SERVED_MODEL_NAME\","
      echo "         \"messages\": [{\"role\": \"user\", \"content\": \"Write a correct Python function that returns the first n Fibonacci numbers.\"}],"
      echo "         \"temperature\": 0, \"max_completion_tokens\": 256 }'"
      echo
      echo "   Inspect runtime: ./start.sh logs    Shut down: ./start.sh down"
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      stop_logs
      trap - EXIT
      echo "ERROR: server did not become healthy within ${STARTUP_WAIT}s." >&2
      echo "Recent logs:" >&2
      docker compose logs --tail=80 >&2 || true
      return 1
    fi
    sleep "$POLL_INTERVAL"
  done
}

ensure_compose() {
  [ -f compose.yml ] || generate_compose
}

case "${1:-start}" in
  start|-no-wait|--no-wait)
    cmd_start "${1:-start}"
    ;;
  compose-gen|gen)
    # Generate compose.yml only; no docker action. Safe to run now.
    generate_compose
    echo "compose.yml is ready. Apply later with: ./start.sh start"
    ;;
  mount)         ensure_sshfs; mount_share;;
  unmount)       unmount_share;;
  logs|log)      ensure_compose; shift; docker compose logs -f "$@";;
  stop)          ensure_compose; shift; docker compose stop "$@";;
  restart)       ensure_compose; shift; docker compose restart "$@";;
  down)          ensure_compose; shift; docker compose down "$@";;
  ps|status)     ensure_compose; shift; docker compose ps "$@";;
  pull)          ensure_compose; shift; docker compose pull "$@";;
  help|-h|--help) usage;;
  *) echo "Unknown command: $1" >&2; usage >&2; exit 1;;
esac
