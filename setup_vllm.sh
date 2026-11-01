#!/usr/bin/env bash
# Install the optional vLLM client, prepare a separate vLLM environment,
# download the PaddleOCR-VL model through PaddleX, and start the local server.
#
# usage: bash setup_vllm.sh [--setup-only]
# DOCIR_VENV, DOCIR_VLLM_ENV, DOCIR_VLLM_MODEL, DOCIR_VLLM_PORT and
# DOCIR_VLLM_GPU_MEMORY_UTILIZATION override the corresponding defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP_ENV="${DOCIR_VENV:-$ROOT/.docir_env}"
APP_PY="$APP_ENV/bin/python"
VLLM_ENV="${DOCIR_VLLM_ENV:-$ROOT/.vllm_env}"
MODEL="${DOCIR_VLLM_MODEL:-$HOME/.paddlex/official_models/PaddleOCR-VL-1.6}"
DEFAULT_MODEL="$HOME/.paddlex/official_models/PaddleOCR-VL-1.6"
PORT="${DOCIR_VLLM_PORT:-8001}"
SETUP_ONLY=0

case "${1:-}" in
    "")
        ;;
    --setup-only)
        SETUP_ONLY=1
        ;;
    -h|--help)
        sed -n '2,7p' "$0"
        exit 0
        ;;
    *)
        printf 'ERROR: unknown option: %s\n' "$1" >&2
        exit 2
        ;;
esac

cd "$ROOT"
say() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "$APP_PY" ]] || die "DocIR environment not found at $APP_ENV; run setup.sh first."
[[ "$(uname -s)" == "Linux" ]] || die "vLLM setup requires Linux with an NVIDIA GPU."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; vLLM requires an NVIDIA GPU."
nvidia-smi -L >/dev/null 2>&1 || die "No NVIDIA GPU is visible to this process."
command -v curl >/dev/null 2>&1 || die "curl is required for the vLLM health check."
command -v cc >/dev/null 2>&1 || die "A C compiler is required to build Triton; install build-essential."

PYTHON_VERSION="$("$APP_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_INCLUDE="$("$APP_PY" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')"
[[ -f "$PYTHON_INCLUDE/Python.h" ]] \
    || die "Python headers are missing; on Ubuntu install python${PYTHON_VERSION}-dev."

say "Installing DocIR's vLLM client extra"
"$APP_PY" -m pip install -e '.[vlm-client]'

if ! "$APP_PY" -m pip show uv >/dev/null 2>&1; then
    say "Installing uv in the DocIR environment"
    "$APP_PY" -m pip install uv
fi
UV="$APP_ENV/bin/uv"

say "Creating or updating the separate vLLM environment: $VLLM_ENV"
"$UV" venv --python "$APP_PY" "$VLLM_ENV"
"$UV" pip install --python "$VLLM_ENV/bin/python" -r "$ROOT/requirements-vllm.txt"

if [[ ! -s "$MODEL/config.json" ]]; then
    [[ "$MODEL" == "$DEFAULT_MODEL" ]] \
        || die "No config.json in custom model directory $MODEL; provide a complete PaddleOCR-VL 1.6 checkpoint."

    say "Downloading PaddleOCR-VL 1.6 into $MODEL (first run can take several minutes)"
    "$APP_PY" - <<'PY'
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="native")
pipeline.close()
PY
fi
[[ -s "$MODEL/config.json" ]] || die "Model download did not create $MODEL/config.json."

if curl --fail --silent --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    say "A vLLM server is already healthy on port $PORT."
    exit 0
fi

if [[ $SETUP_ONLY -eq 1 ]]; then
    say "Setup complete. Start the server with: bash scripts/serve_vllm.sh"
    exit 0
fi

say "Starting vLLM on 127.0.0.1:$PORT (keep this terminal running)"
exec bash scripts/serve_vllm.sh
