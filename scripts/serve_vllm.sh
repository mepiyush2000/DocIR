#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${DOCIR_VLLM_ENV:-$ROOT/.vllm_env}/bin/python"
MODEL="${DOCIR_VLLM_MODEL:-$HOME/.paddlex/official_models/PaddleOCR-VL-1.6}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing vLLM environment. See the vLLM setup in AGENT_HANDBOOK.md." >&2
    exit 1
fi
if [[ ! -d "$MODEL" ]]; then
    echo "Model directory not found: $MODEL. Set DOCIR_VLLM_MODEL to the VL 1.6 checkpoint." >&2
    exit 1
fi

exec "$(dirname "$PYTHON")/vllm" serve \
    --model "$MODEL" \
    --served-model-name PaddleOCR-VL-1.6-0.9B \
    --host 127.0.0.1 --port "${DOCIR_VLLM_PORT:-8001}" \
    --dtype bfloat16 --trust-remote-code --trust-request-mm-kwargs \
    --max-model-len 16384 --max-num-seqs 16 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization "${DOCIR_VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
    --no-enable-prefix-caching --mm-processor-cache-gb 0 \
    "$@"
