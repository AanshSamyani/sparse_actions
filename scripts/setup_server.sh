#!/usr/bin/env bash
# One-time setup on the GPU server. Everything (uv, caches, managed Python, venv,
# model downloads, outputs) is placed under $WORKSPACE so it survives restarts --
# on this box only /workspace persists.
#
# Pick the CUDA wheel tag matching your driver (see README "Which CUDA tag?"):
#   CUDA_TAG=cu121 bash scripts/setup_server.sh     # driver CUDA 12.1 - 12.3
#   CUDA_TAG=cu124 bash scripts/setup_server.sh     # driver CUDA >= 12.4
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh

CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"

# 1) install uv into $WORKSPACE/bin if missing (no profile edits -- they won't persist)
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv into $UV_INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

# 2) create the venv in the repo (under /workspace); uv fetches Python 3.11 if needed
uv venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) CUDA torch FIRST (kept out of pyproject on purpose so uv can't pull a CPU build)
echo ">> installing torch==$TORCH_VERSION ($CUDA_TAG)"
uv pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# 4) the project (+ dev extras). Add ,vllm for the fast sampling backend.
echo ">> installing sparse_actions"
uv pip install -e ".[dev]"

# 5) sanity report
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY
echo ">> persistent caches:  HF_HOME=$HF_HOME  UV_CACHE_DIR=$UV_CACHE_DIR"
echo ">> done. In any NEW shell, re-init with:"
echo "     source scripts/workspace_env.sh && source .venv/bin/activate"
