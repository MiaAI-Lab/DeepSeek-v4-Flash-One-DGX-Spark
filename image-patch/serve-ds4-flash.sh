#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Flash / DSpark launcher for the SM120 PCIe stack. The public
# interface is environment-only so the same command can be used from Compose,
# docker run, and benchmark automation.

unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

bool_value() {
  local name=$1 value=${2,,}
  case "${value}" in
    1|true|yes|on) printf '1\n' ;;
    0|false|no|off) printf '0\n' ;;
    *)
      echo "${name} must be 1/0, true/false, yes/no, or on/off; got '${2}'" >&2
      exit 2
      ;;
  esac
}

require_positive_int() {
  local name=$1 value=$2
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
}

require_nonnegative_int() {
  local name=$1 value=$2
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer; got '${value}'" >&2
    exit 2
  fi
}

mode=${MODE:-${SPEC_MODE:-dspark}}
case "${mode}" in
  off|mtp0|standard-mtp0) mode=mtp0 ;;
  mtp2|standard-mtp2) mode=mtp2 ;;
  mtp3|standard-mtp3) mode=mtp3 ;;
  dspark) ;;
  *)
    echo "MODE must be mtp0, mtp2, mtp3, or dspark; got '${mode}'" >&2
    exit 2
    ;;
esac

backend=${BACKEND:-b12x-a8}
case "${backend}" in
  b12x) backend=b12x-a16 ;;
  b12x-a16|b12x-a8|b12x-a8-dglin|lucifer-default|lucifer-cutlass) ;;
  *)
    echo "BACKEND must be b12x-a16, b12x-a8, b12x-a8-dglin," \
      "lucifer-default, or lucifer-cutlass; got '${backend}'" >&2
    exit 2
    ;;
esac

standard_model=${STANDARD_MODEL:-deepseek-ai/DeepSeek-V4-Flash}
dspark_model=${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-DSpark}
standard_model_revision=${STANDARD_MODEL_REVISION:-60d8d70770c6776ff598c94bb586a859a38244f1}
dspark_model_revision=${DSPARK_MODEL_REVISION:-62af8fffb2f7030cac4de2f0169f5b8d1101b646}
if [[ "${mode}" == "dspark" ]]; then
  model=${MODEL_PATH:-${MODEL:-${dspark_model}}}
  spec_model=${SPEC_MODEL_PATH:-${model}}
  served_model_name=${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-DSpark}
  default_model_revision=${dspark_model_revision}
else
  model=${MODEL_PATH:-${MODEL:-${standard_model}}}
  spec_model=
  served_model_name=${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}
  default_model_revision=${standard_model_revision}
fi

model_revision=${MODEL_REVISION:-}
if [[ -z "${model_revision}" ]]; then
  if [[ "${model}" == "${standard_model}" || "${model}" == "${dspark_model}" ]]; then
    model_revision=${default_model_revision}
  fi
fi
revision_args=()
if [[ -n "${model_revision}" ]]; then
  revision_args=(--revision "${model_revision}")
fi

host=${HOST:-0.0.0.0}
port=${PORT:-8000}
tp_size=${TP_SIZE:-${TP:-2}}
dcp_size=${DCP_SIZE:-1}
max_num_seqs=${MAX_NUM_SEQS:-64}
max_model_len=${MAX_MODEL_LEN:-262144}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-}
block_size=${BLOCK_SIZE:-256}
load_format=${LOAD_FORMAT:-instanttensor}
prefix_cache=$(bool_value PREFIX_CACHE "${PREFIX_CACHE:-1}")
enable_flashinfer_autotune=$(bool_value ENABLE_FLASHINFER_AUTOTUNE "${ENABLE_FLASHINFER_AUTOTUNE:-1}")
draft_sample_method=${DRAFT_SAMPLE_METHOD:-probabilistic}
rejection_sample_method=${REJECTION_SAMPLE_METHOD:-standard}

require_positive_int TP_SIZE "${tp_size}"
require_positive_int DCP_SIZE "${dcp_size}"
require_positive_int MAX_NUM_SEQS "${max_num_seqs}"
require_positive_int MAX_NUM_BATCHED_TOKENS "${max_num_batched_tokens}"
require_positive_int BLOCK_SIZE "${block_size}"
if [[ "${mode}" == "dspark" && "${dcp_size}" != "1" ]]; then
  echo "DSpark non-causal attention currently requires DCP_SIZE=1" >&2
  exit 2
fi

case "${draft_sample_method}" in
  probabilistic|greedy) ;;
  *)
    echo "DRAFT_SAMPLE_METHOD must be probabilistic or greedy" >&2
    exit 2
    ;;
esac
case "${rejection_sample_method}" in
  standard|block) ;;
  *)
    echo "REJECTION_SAMPLE_METHOD must be standard or block" >&2
    exit 2
    ;;
esac

export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_120a}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
export NCCL_PROTO=${NCCL_PROTO:-LL,LL128,Simple}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export LLM_WORKER_MULTIPROC_METHOD=${LLM_WORKER_MULTIPROC_METHOD:-spawn}
export SAFETENSORS_FAST_GPU=${SAFETENSORS_FAST_GPU:-1}
export INSTANTTENSOR_BACKEND=${INSTANTTENSOR_BACKEND:-BUFFERED}

export VLLM_USE_AOT_COMPILE=${VLLM_USE_AOT_COMPILE:-1}
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}
export VLLM_USE_MEGA_AOT_ARTIFACT=${VLLM_USE_MEGA_AOT_ARTIFACT:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-1}
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN=${VLLM_MEMORY_PROFILE_INCLUDE_ATTN:-1}
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-4096}
export VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=${VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD:-1024}

allreduce_mode=${ALLREDUCE_MODE:-b12x}
b12x_pcie_dma=$(bool_value B12X_PCIE_DMA "${B12X_PCIE_DMA:-0}")
export VLLM_USE_B12X_PCIE_DMA=${b12x_pcie_dma}
allreduce_args=()
case "${allreduce_mode}" in
  b12x)
    export VLLM_ENABLE_PCIE_ALLREDUCE=1
    export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
    export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}
    export VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=0
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    ;;
  vllm-custom)
    export VLLM_ENABLE_PCIE_ALLREDUCE=0
    export VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=1
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    ;;
  vllm-custom-2stage)
    export VLLM_ENABLE_PCIE_ALLREDUCE=0
    export VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=1
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0
    export VLLM_CUSTOM_ALLREDUCE_ALGO=2stage
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    ;;
  nccl)
    export VLLM_ENABLE_PCIE_ALLREDUCE=0
    export VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=0
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    allreduce_args=(--disable-custom-all-reduce)
    ;;
  *)
    echo "ALLREDUCE_MODE must be b12x, vllm-custom, vllm-custom-2stage," \
      "or nccl; got '${allreduce_mode}'" >&2
    exit 2
    ;;
esac

b12x_backend=0
backend_args=()
case "${backend}" in
  b12x-a16|b12x-a8|b12x-a8-dglin)
    b12x_backend=1
    export VLLM_USE_B12X_WO_PROJECTION=${VLLM_USE_B12X_WO_PROJECTION:-1}
    export VLLM_USE_B12X_MHC=${VLLM_USE_B12X_MHC:-1}
    export VLLM_USE_B12X_MOE=${VLLM_USE_B12X_MOE:-1}
    export VLLM_USE_B12X_SPARSE_INDEXER=${VLLM_USE_B12X_SPARSE_INDEXER:-1}
    export B12X_MHC_MAX_TOKENS=${B12X_MHC_MAX_TOKENS:-16384}
    backend_args=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x)
    if [[ "${backend}" != "b12x-a8-dglin" ]]; then
      backend_args+=(--linear-backend b12x)
      export VLLM_USE_B12X_FP8_GEMM=1
    else
      export VLLM_USE_B12X_FP8_GEMM=0
    fi
    if [[ "${backend}" == "b12x-a16" ]]; then
      export B12X_MOE_FORCE_A8=0
      export B12X_MOE_FORCE_A16=1
    else
      export B12X_MOE_FORCE_A8=1
      export B12X_MOE_FORCE_A16=0
    fi
    ;;
  lucifer-default)
    export VLLM_USE_B12X_WO_PROJECTION=0
    export VLLM_USE_B12X_MHC=0
    export VLLM_USE_B12X_FP8_GEMM=0
    export VLLM_USE_B12X_MOE=0
    backend_args=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4)
    ;;
  lucifer-cutlass)
    export VLLM_USE_B12X_WO_PROJECTION=0
    export VLLM_USE_B12X_MHC=0
    export VLLM_USE_B12X_FP8_GEMM=0
    export VLLM_USE_B12X_MOE=0
    backend_args=(
      --attention-backend FLASHINFER_MLA_SPARSE_DSV4
      --kernel-config.moe_backend flashinfer_cutlass
    )
    ;;
esac

# The B12X indexer can be paired with Lucifer attention/MoE to unlock compact
# SM120 varlen verification. It is experimental and therefore never selected
# automatically for a Lucifer backend.
indexer_backend=${INDEXER_BACKEND:-auto}
if [[ "${indexer_backend}" == "auto" ]]; then
  if (( b12x_backend )); then indexer_backend=b12x; else indexer_backend=native; fi
fi
case "${indexer_backend}" in
  b12x) export VLLM_USE_B12X_SPARSE_INDEXER=1 ;;
  native) export VLLM_USE_B12X_SPARSE_INDEXER=0 ;;
  *)
    echo "INDEXER_BACKEND must be auto, b12x, or native" >&2
    exit 2
    ;;
esac

spec_args=()
spec_tokens=0
graph_multiplier=4
if [[ "${mode}" == "mtp2" || "${mode}" == "mtp3" ]]; then
  if [[ "${mode}" == "mtp2" ]]; then spec_tokens=2; else spec_tokens=3; fi
  mtp_moe_json=
  if (( b12x_backend )); then
    mtp_moe_json=',"moe_backend":"b12x"'
  fi
  spec_json=$(printf \
    '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"%s","rejection_sample_method":"%s"%s}' \
    "${spec_tokens}" "${draft_sample_method}" "${rejection_sample_method}" \
    "${mtp_moe_json}")
  spec_args=(--speculative-config "${spec_json}")
  graph_multiplier=8
elif [[ "${mode}" == "dspark" ]]; then
  spec_tokens=${DSPARK_TOKENS:-5}
  require_positive_int DSPARK_TOKENS "${spec_tokens}"
  # Target verification schedules at most one sampled token plus K drafts per
  # request. Capturing beyond that physical row count only consumes graph
  # memory (and can OOM at high concurrency) because those shapes are
  # unreachable for DSpark.
  graph_multiplier=$((spec_tokens + 1))
  draft_attention_backend=${DSPARK_DRAFT_ATTENTION_BACKEND:-auto}
  draft_attention_json=
  if [[ "${draft_attention_backend}" != "auto" ]]; then
    case "${draft_attention_backend}" in
      B12X_MLA_SPARSE|FLASHINFER_MLA_SPARSE_DSV4|FLASHMLA_SPARSE_DSV4) ;;
      *)
        echo "DSPARK_DRAFT_ATTENTION_BACKEND must be auto, B12X_MLA_SPARSE," \
          "FLASHINFER_MLA_SPARSE_DSV4, or FLASHMLA_SPARSE_DSV4" >&2
        exit 2
        ;;
    esac
    draft_attention_json=$(printf \
      ',"attention_backend":"%s"' "${draft_attention_backend}")
  fi
  dspark_capacity=$(bool_value DSPARK_CAPACITY "${DSPARK_CAPACITY:-0}")
  capacity_json=
  if [[ "${dspark_capacity}" == "1" ]]; then
    capacity_mode=${DSPARK_CAPACITY_VERIFICATION_MODE:-}
    if [[ -z "${capacity_mode}" ]]; then
      if [[ "${indexer_backend}" == "b12x" ]]; then
        capacity_mode=varlen
      else
        capacity_mode=mask
      fi
    fi
    case "${capacity_mode}" in varlen|mask) ;; *)
      echo "DSPARK_CAPACITY_VERIFICATION_MODE must be varlen or mask" >&2
      exit 2
    esac
    online_sts=$(bool_value DSPARK_ONLINE_STS "${DSPARK_ONLINE_STS:-1}")
    if [[ "${online_sts}" == "1" ]]; then online_sts_json=true; else online_sts_json=false; fi
    sps_curve=${DSPARK_SPS_CURVE:-auto}
    if [[ "${sps_curve}" == \[* ]]; then
      sps_json=${sps_curve}
    else
      sps_json=$(printf '"%s"' "${sps_curve}")
    fi
    capacity_json=$(printf \
      ',"dspark_confidence_threshold":%s,"dspark_budget_frac":%s,"dspark_capacity_verification_mode":"%s","dspark_confidence_temperature":%s,"dspark_online_sts":%s,"dspark_sps_curve":%s,"dspark_sps_overhead_ms":%s' \
      "${DSPARK_CONFIDENCE_THRESHOLD:-0.0}" \
      "${DSPARK_BUDGET_FRAC:-1.0}" \
      "${capacity_mode}" \
      "${DSPARK_CONFIDENCE_TEMPERATURE:-1.0}" \
      "${online_sts_json}" \
      "${sps_json}" \
      "${DSPARK_SPS_OVERHEAD_MS:-0.0}")
  fi
  spec_json=$(printf \
    '{"model":"%s","method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s","rejection_sample_method":"%s"%s%s}' \
    "${spec_model}" "${spec_tokens}" "${draft_sample_method}" \
    "${rejection_sample_method}" "${draft_attention_json}" "${capacity_json}")
  spec_args=(--speculative-config "${spec_json}")
  export VLLM_DSPARK_FP8_DRAFT_HEAD=$(bool_value DSPARK_FP8_DRAFT_HEAD "${DSPARK_FP8_DRAFT_HEAD:-0}")
  export VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=$(bool_value DSPARK_DYNAMIC_DRAFT_DEPTH "${DSPARK_DYNAMIC_DRAFT_DEPTH:-0}")
  export VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW=${DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW:-8}
  require_positive_int DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW \
    "${VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW}"
  export VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=${DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE:-0}
  require_nonnegative_int DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE \
    "${VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE}"
  export VLLM_DSPARK_CAPACITY_LOG_INTERVAL=${DSPARK_CAPACITY_LOG_INTERVAL:-0}
  export VLLM_DSPARK_STS_LOG_INTERVAL=${DSPARK_STS_LOG_INTERVAL:-0}
  export VLLM_DSPARK_TP_CHECK=${DSPARK_TP_CHECK:-0}
fi

# v9 used graph 256 for MTP-off and 512 for MTP modes at cc64. Keep that MTP
# contract; DSpark uses its exact (K + 1) physical verifier width.
graph_cap=${MAX_CUDAGRAPH_CAPTURE_SIZE:-${GRAPH:-}}
if [[ -z "${graph_cap}" || "${graph_cap}" == "auto" ]]; then
  graph_cap=$((max_num_seqs * graph_multiplier))
  if (( graph_cap < 6 )); then graph_cap=6; fi
fi
require_positive_int MAX_CUDAGRAPH_CAPTURE_SIZE "${graph_cap}"

sp_async_tp=$(bool_value SP_ASYNC_TP "${SP_ASYNC_TP:-0}")
compilation_config='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
export VLLM_USE_V2_MODEL_RUNNER=1
if [[ "${sp_async_tp}" == "1" ]]; then
  if [[ "${mode}" == "dspark" ]]; then
    echo "SP_ASYNC_TP=1 is not supported by the V2 DSpark runner" >&2
    exit 2
  fi
  sp_min_tokens=${SP_MIN_TOKEN_NUM:-512}
  require_positive_int SP_MIN_TOKEN_NUM "${sp_min_tokens}"
  compilation_config=$(printf \
    '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"use_inductor_graph_partition":true,"pass_config":{"enable_sp":true,"fuse_gemm_comms":true,"sp_min_token_num":%s}}' \
    "${sp_min_tokens}")
  export VLLM_USE_V2_MODEL_RUNNER=0
  export VLLM_SYMM_MEM_PCIE_SAFE_BARRIER=1
fi

if [[ -z "${gpu_memory_utilization}" ]]; then
  # The v10 profiler includes attention and FULL-graph allocations. These
  # defaults preserve the 262k serving limit used by v9 after that accounting.
  if [[ "${mode}" == "dspark" ]]; then
    if [[ "${backend}" == "lucifer-default" ]]; then
      # The default DeepGEMM MoE path retains more model/runtime memory than
      # FlashInfer CUTLASS. At 0.9465 only 7.35 GiB remained for the 7.89 GiB
      # DSpark KV requirement. 0.953 profiles 8.03 GiB (266609 tokens) and
      # leaves enough transient headroom for a sustained 128k prefill.
      gpu_memory_utilization=0.953
    elif [[ "${backend}" == "lucifer-cutlass" ]] && (( tp_size >= 4 )); then
      # FlashInfer CUTLASS allocates a transient MoE workspace on the first
      # real prefill. At TP4, 0.9465 left only 777 MiB free for a 764 MiB
      # workspace and could OOM after an otherwise successful warmup. TP4+
      # still has ample KV capacity at 0.94; TP2 needs the 0.9465 Lucifer
      # default below to preserve the advertised 262k serving limit.
      gpu_memory_utilization=0.94
    elif [[ "${backend}" == lucifer-* ]]; then
      gpu_memory_utilization=0.9465
    else
      gpu_memory_utilization=0.95
    fi
  elif [[ "${backend}" == lucifer-* \
    && ( "${mode}" == "mtp2" || "${mode}" == "mtp3" ) ]]; then
    # Lucifer MTP FULL graphs leave about 7.48 GiB at 0.91, just below the
    # 7.55 GiB needed to preserve the documented 262k serving limit.
    gpu_memory_utilization=0.912
  else
    gpu_memory_utilization=0.91
  fi
fi

prefix_args=(--enable-prefix-caching)
if [[ "${prefix_cache}" == "0" ]]; then
  prefix_args=(--no-enable-prefix-caching)
fi
autotune_args=(--enable-flashinfer-autotune)
if [[ "${enable_flashinfer_autotune}" == "0" ]]; then
  autotune_args=(--no-enable-flashinfer-autotune)
fi

capture_args=()
capture_sizes=${CUDAGRAPH_CAPTURE_SIZES:-default}
if [[ "${capture_sizes}" == "auto" ]]; then
  sizes=(1)
  n=2
  while (( n < graph_cap )); do sizes+=("${n}"); n=$((n * 2)); done
  if (( max_num_seqs <= graph_cap )); then
    sizes+=("${max_num_seqs}")
  fi
  sizes+=("${graph_cap}")
  mapfile -t sizes < <(printf '%s\n' "${sizes[@]}" | sort -n -u)
  capture_args=(--cudagraph-capture-sizes "${sizes[@]}")
elif [[ "${capture_sizes}" != "default" && "${capture_sizes}" != "none" ]]; then
  read -r -a sizes <<< "${capture_sizes//,/ }"
  capture_args=(--cudagraph-capture-sizes "${sizes[@]}")
fi

cache_root=${XDG_CACHE_HOME:-/cache}
export XDG_CACHE_HOME=${cache_root}
export VLLM_CACHE_DIR=${VLLM_CACHE_DIR:-${cache_root}/vllm}
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-${cache_root}/tilelang}
export TILELANG_TMP_DIR=${TILELANG_TMP_DIR:-${cache_root}/tilelang/tmp}
export TVM_CACHE_DIR=${TVM_CACHE_DIR:-${cache_root}/tvm}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-${cache_root}/jit/tvm-ffi}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${cache_root}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/torchinductor}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${cache_root}/jit/torch_extensions}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${cache_root}/flashinfer}
mkdir -p \
  "${VLLM_CACHE_DIR}" "${TILELANG_CACHE_DIR}" "${TILELANG_TMP_DIR}" \
  "${TVM_CACHE_DIR}" "${TVM_FFI_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
  "${FLASHINFER_WORKSPACE_BASE}"

if [[ -n "${VLLM_PYTHON:-}" ]]; then
  command=("${VLLM_PYTHON}" -m vllm.entrypoints.cli.main)
else
  command=(vllm)
fi

command+=(
  serve "${model}"
  "${revision_args[@]}"
  --served-model-name "${served_model_name}"
  --host "${host}"
  --port "${port}"
  --trust-remote-code
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --block-size "${block_size}"
  --load-format "${load_format}"
  --tensor-parallel-size "${tp_size}"
  --decode-context-parallel-size "${dcp_size}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --max-model-len "${max_model_len}"
  --max-num-seqs "${max_num_seqs}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  --max-cudagraph-capture-size "${graph_cap}"
  --compilation-config "${compilation_config}"
  --async-scheduling
  --no-scheduler-reserve-full-isl
  --enable-chunked-prefill
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --reasoning-parser deepseek_v4
  --enable-auto-tool-choice
  --enable-prompt-tokens-details
  --enable-force-include-usage
  --enable-request-id-headers
  --default-chat-template-kwargs.thinking=${DEFAULT_CHAT_TEMPLATE_KWARGS_THINKING:-false}
  --default-chat-template-kwargs.reasoning_effort=${DEFAULT_CHAT_TEMPLATE_KWARGS_EFFORT:-high}
  "${autotune_args[@]}"
  "${prefix_args[@]}"
  "${capture_args[@]}"
  "${spec_args[@]}"
  "${backend_args[@]}"
  "${allreduce_args[@]}"
)

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # EXTRA_VLLM_ARGS is intentionally an escape hatch for temporary experiments.
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_VLLM_ARGS} )
  command+=("${extra_args[@]}")
fi
command+=("$@")

printf 'DS4 launch: mode=%s backend=%s allreduce=%s b12x_dma=%s indexer=%s tp=%s dcp=%s max_seqs=%s graph=%s load_format=%s instanttensor_backend=%s model=%s\n' \
  "${mode}" "${backend}" "${allreduce_mode}" "${b12x_pcie_dma}" \
  "${indexer_backend}" "${tp_size}" "${dcp_size}" "${max_num_seqs}" \
  "${graph_cap}" "${load_format}" "${INSTANTTENSOR_BACKEND}" "${model}" >&2
printf 'Command:' >&2
printf ' %q' "${command[@]}" >&2
printf '\n' >&2

if [[ "$(bool_value DRY_RUN "${DRY_RUN:-0}")" == "1" ]]; then
  exit 0
fi
exec "${command[@]}"
