#!/usr/bin/env bash
# DocIR one-shot setup: creates .docir_env, installs the GPU build of Paddle
# when an NVIDIA GPU + driver is found (else the CPU build), then DocIR and
# the Streamlit app, and verifies the result.
#
# usage: ./setup.sh [--cpu] [--cuda cu118|cu126|cu129] [--python PATH] [--dev] [--recreate]
#   --cpu       force the CPU build even if a GPU is present
#   --cuda X    force a CUDA wheel flavour instead of detecting it
#   --python P  interpreter to build the venv from (default: newest 3.12/3.11/3.13 found)
#   --dev       also install test dependencies (pytest)
#   --recreate  delete and rebuild an existing .docir_env
# DOCIR_VENV=path overrides where the venv goes (default .docir_env).
set -euo pipefail

PADDLE_VERSION="3.3.1"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable"
VENV="${DOCIR_VENV:-.docir_env}"

FORCE_CPU=0 CUDA_FLAVOUR="" PYTHON="" DEV=0 RECREATE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) FORCE_CPU=1 ;;
        --cuda) CUDA_FLAVOUR="$2"; shift ;;
        --python) PYTHON="$2"; shift ;;
        --dev) DEV=1 ;;
        --recreate) RECREATE=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

cd "$(dirname "$0")"
say() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------- python ----------
if [[ -z "$PYTHON" ]]; then
    # Paddle's GPU wheels are most reliable on 3.11/3.12
    for p in python3.12 python3.11 python3.13 python3; do
        if command -v "$p" >/dev/null 2>&1; then PYTHON="$p"; break; fi
    done
fi
[[ -n "$PYTHON" ]] || die "no python3 found; install Python 3.11-3.13"
command -v "$PYTHON" >/dev/null 2>&1 || die "interpreter not found: $PYTHON"
PY_VER=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
"$PYTHON" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' \
    || die "Python $PY_VER found; DocIR needs 3.11-3.13 (use --python)"
say "Python: $PYTHON ($PY_VER)"

# ---------- GPU detection ----------
detect_cuda_flavour() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local out ver major minor
    out=$(nvidia-smi 2>/dev/null) || return 1
    # the highest CUDA version the installed driver supports
    ver=$(grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' <<<"$out" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    [[ -n "$ver" ]] || return 1
    major=${ver%.*} minor=${ver#*.}
    if (( major > 12 || (major == 12 && minor >= 9) )); then echo cu129
    elif (( major == 12 && minor >= 6 )); then echo cu126
    elif (( major == 12 || (major == 11 && minor >= 8) )); then echo cu118
    else return 1
    fi
}

if [[ "$(uname -s)" == "Darwin" ]]; then
    FORCE_CPU=1
    say "macOS: CUDA isn't available, installing the CPU build"
fi
if [[ $FORCE_CPU -eq 1 ]]; then
    CUDA_FLAVOUR=""
elif [[ -z "$CUDA_FLAVOUR" ]]; then
    CUDA_FLAVOUR=$(detect_cuda_flavour || true)
fi
if [[ -n "$CUDA_FLAVOUR" ]]; then
    say "NVIDIA GPU found:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
    say "Using paddlepaddle-gpu ($CUDA_FLAVOUR)"
else
    say "No usable NVIDIA GPU/driver found: using the CPU build"
fi

# ---------- venv ----------
if [[ -d "$VENV" && $RECREATE -eq 1 ]]; then
    say "Removing existing $VENV"
    rm -rf "$VENV"
fi
if [[ ! -d "$VENV" ]]; then
    say "Creating $VENV"
    "$PYTHON" -m venv "$VENV"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install -q -U pip setuptools wheel

# ---------- paddle (first, so nothing pulls in the other build) ----------
if [[ -n "$CUDA_FLAVOUR" ]]; then
    "$VPY" -m pip uninstall -y -q paddlepaddle 2>/dev/null || true
    "$VPY" -m pip install "paddlepaddle-gpu==$PADDLE_VERSION" -i "$PADDLE_INDEX/$CUDA_FLAVOUR/"
else
    "$VPY" -m pip uninstall -y -q paddlepaddle-gpu 2>/dev/null || true
    "$VPY" -m pip install "paddlepaddle==$PADDLE_VERSION"
fi

# ---------- DocIR + OCR stack + app ----------
say "Installing DocIR, PaddleOCR and the Streamlit app"
EXTRAS="app"
[[ $DEV -eq 1 ]] && EXTRAS="app,dev"
# not the [ocr] extra: it names the CPU paddlepaddle wheel
"$VPY" -m pip install -e ".[$EXTRAS]" "paddleocr>=3.7" "paddlex[ocr]>=3.7"

# a dependency may still have pulled in the other paddle build
if [[ -n "$CUDA_FLAVOUR" ]] && "$VPY" -m pip show -q paddlepaddle >/dev/null 2>&1; then
    say "Removing the CPU paddlepaddle a dependency pulled in"
    "$VPY" -m pip uninstall -y -q paddlepaddle
    "$VPY" -m pip install --force-reinstall --no-deps "paddlepaddle-gpu==$PADDLE_VERSION" -i "$PADDLE_INDEX/$CUDA_FLAVOUR/"
fi

# several OpenCV wheels share the cv2 package and break each other
for extra in opencv-python opencv-python-headless opencv-contrib-python-headless; do
    if "$VPY" -m pip show -q "$extra" >/dev/null 2>&1; then
        say "Removing clashing $extra"
        "$VPY" -m pip uninstall -y -q "$extra"
        REINSTALL_CV=1
    fi
done
if [[ "${REINSTALL_CV:-0}" -eq 1 ]]; then
    "$VPY" -m pip install -q --force-reinstall --no-deps "opencv-contrib-python>=4.8"
fi

# ---------- verify ----------
say "Verifying"
"$VPY" - <<'EOF'
import cv2, paddle, streamlit, reportlab, pypdfium2  # noqa: F401
import docir.pipeline  # noqa: F401

gpus = paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0
print(f"paddle {paddle.__version__}  CUDA build: {paddle.is_compiled_with_cuda()}  GPUs visible: {gpus}")
print(f"opencv {cv2.__version__}  streamlit {streamlit.__version__}")
if paddle.is_compiled_with_cuda():
    paddle.utils.run_check()
    if gpus == 0:
        raise SystemExit("CUDA build installed but no GPU visible: check the driver / CUDA_VISIBLE_DEVICES")
EOF

DEVICE=$(grep -E '^[[:space:]]*device[[:space:]]*=' docir_config.toml 2>/dev/null | head -1 | sed -E 's/.*=[[:space:]]*"?([^"]*)"?.*/\1/')
say "Done. [runtime] device in docir_config.toml: ${DEVICE:-auto} (auto = GPU when present)"
cat <<EOF

Run the web app:
  source $VENV/bin/activate
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.maxUploadSize 500

Or the CLI:
  $VENV/bin/python -m docir.cli analyze page.jpg page.pdf

The first conversion downloads the models into ~/.paddlex (a few minutes, once).
EOF
