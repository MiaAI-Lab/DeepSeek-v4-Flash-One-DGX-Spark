#!/usr/bin/env bash
# Patched recipe entrypoint for the 256k-context variant (start-256k.sh).
# The stock image hardcodes KV_FP8_ROPE=0 and VLLM_DSV4_PADDED_NVFP4=1, which
# would clobber the context-maximizing record formats. This copy reads both
# from the environment (defaulting to the stock behavior) so the 256k launcher
# can select the 432-byte / 368-byte NVFP4 KV record layouts (improve-context.md
# levers #1 / #2). Everything else is byte-identical to the shipped script.
set -Eeuo pipefail

model_repo=${MODEL_REPO:-0xSero/deepseek-v4-flash-0731-spark}
model_revision=${MODEL_REVISION:-22f28d32b9b29b4352eaa380ff8c2c170b2847ab}
source_dir=${MODEL_SOURCE_DIR:-/models/source}
model_dir=${MODEL_PATH:-/models/tp1}
mode=${MODE:-dspark}
draft_dir=${SPEC_MODEL_PATH:-/models/dspark-draft-k64}

if [[ ! -f "${model_dir}/rank-sliced-tp1-manifest.json" ]]; then
  mkdir -p "${source_dir}" "${model_dir}"
  MODEL_REPO=${model_repo} MODEL_REVISION=${model_revision} \
    MODEL_SOURCE_DIR=${source_dir} /opt/runtime-venv/bin/python - <<'PY'
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
    --input-dir "${source_dir}" \
    --output-dir "${model_dir}" \
    --link-carried \
    --reuse-complete \
    --workers "${COALESCE_WORKERS:-1}"
fi

verify_args=()
if [[ "${VERIFY_MODEL_CHECKSUMS:-1}" != "1" ]]; then
  verify_args+=(--skip-checksums)
fi
/opt/runtime-venv/bin/python /opt/recipe/scripts/verify_tp1_manifest.py \
  "${model_dir}" "${verify_args[@]}"

if [[ "${mode}" == "dspark" && ! -f "${draft_dir}/model.safetensors.index.json" ]]; then
  /opt/runtime-venv/bin/python /opt/recipe/scripts/build_dspark_draft.py \
    --source "${model_dir}" \
    --output "${draft_dir}" \
    --experts "${DSPARK_DRAFT_EXPERTS:-64}" \
    --structured-per-category "${DSPARK_STRUCTURED_EXPERTS_PER_CATEGORY:-32}"
fi

/opt/runtime-venv/bin/python /opt/recipe/scripts/selftest.py

export VLLM_PYTHON=/opt/runtime-venv/bin/python
export MODE=${mode}
export BACKEND=b12x-a8
export INDEXER_BACKEND=b12x
export ALLREDUCE_MODE=nccl
export TP_SIZE=1
export DCP_SIZE=1
export MODEL_PATH=${model_dir}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731}
export KV_CACHE_DTYPE=nvfp4_ds_mla
# --- 256k variant: env-driven (defaults preserved for stock parity) ---
export KV_FP8_ROPE=${KV_FP8_ROPE:-0}
export VLLM_DSV4_PADDED_NVFP4=${VLLM_DSV4_PADDED_NVFP4:-1}
# --------------------------------------------------------------------------
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8224}
export MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-6}
export CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-6}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9465}
export LOAD_FORMAT=instanttensor
export PREFIX_CACHE=1
export ENABLE_FLASHINFER_AUTOTUNE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}
export VLLM_USE_B12X_WO_PROJECTION=${VLLM_USE_B12X_WO_PROJECTION:-1}
export XDG_CACHE_HOME=/cache

if [[ "${mode}" == "dspark" ]]; then
  export SPEC_MODEL_PATH=${draft_dir}
  export DSPARK_TOKENS=${DSPARK_TOKENS:-5}
  export DSPARK_CAPACITY=${DSPARK_CAPACITY:-0}
  export DSPARK_DYNAMIC_DRAFT_DEPTH=${DSPARK_DYNAMIC_DRAFT_DEPTH:-0}
  export DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW=${DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW:-8}
  export DSPARK_DRAFT_ATTENTION_BACKEND=${DSPARK_DRAFT_ATTENTION_BACKEND:-B12X_MLA_SPARSE}
  export DRAFT_SAMPLE_METHOD=${DRAFT_SAMPLE_METHOD:-probabilistic}
fi

exec /opt/vllm/serve-ds4-flash.sh
