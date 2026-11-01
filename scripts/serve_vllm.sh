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

# FlashInfer's sampler JIT uses nvcc's --compress-mode flag, unavailable before CUDA 12.8.
if [[ -z "${VLLM_USE_FLASHINFER_SAMPLER+x}" ]] && command -v nvcc >/dev/null 2>&1; then
    NVCC_VERSION=$(nvcc --version | sed -nE 's/.*release ([0-9]+)\.([0-9]+).*/\1 \2/p' | tail -1)
    if [[ -n "$NVCC_VERSION" ]]; then
        read -r NVCC_MAJOR NVCC_MINOR <<<"$NVCC_VERSION"
        if (( NVCC_MAJOR < 12 || (NVCC_MAJOR == 12 && NVCC_MINOR < 8) )); then
            echo "CUDA compiler $NVCC_MAJOR.$NVCC_MINOR is too old for FlashInfer sampler JIT; using the PyTorch sampler."
            export VLLM_USE_FLASHINFER_SAMPLER=0
        fi
    fi
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
