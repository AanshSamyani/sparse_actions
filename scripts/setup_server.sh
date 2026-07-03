#!/usr/bin/env bash
# One-time setup on the GPU server (Linux, CUDA). Uses uv for dependency management.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) create the venv (Python 3.11)
uv venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) install a CUDA build of torch FIRST (kept out of pyproject on purpose).
#    cu121 works on H100; switch to cu124 if your driver/toolkit is newer.
echo ">> installing torch (CUDA 12.1)"
uv pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# 4) install the project (+ dev extras). Add ',vllm' for the fast sampling backend.
echo ">> installing sparse_actions"
uv pip install -e ".[dev]"

# 5) sanity: report GPU + versions
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY

echo ">> done. Create your .env (see .env.example), then:  source .venv/bin/activate"
