#!/usr/bin/env bash
# Boot fix for NVIDIA 26.02 vLLM + xgrammar 0.1.27 tool-calling 500.
# https://forums.developer.nvidia.com/t/nvcr-io-nvidia-vllm-26-07-py3-tool-calling-requests-500/378582/1
# vLLM build calls xgrammar.normalize_tool_choice (exists >= 0.2.4) but the
# image pins xgrammar==0.1.27. Upgrade xgrammar, then restore transformers
# (pip would otherwise downgrade it as a side effect).
set -Eeuo pipefail
PY=/opt/runtime-venv/bin/pip
"$PY" install -q -U "xgrammar>=0.2.4"
"$PY" install -q "transformers==5.13.1"
"$PY" check >/dev/null 2>&1 || true
echo "[toolfix] xgrammar upgraded, transformers restored" >&2
