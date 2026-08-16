#!/usr/bin/env bash
# One-time setup for a CPU-ONLY box that drives the Tinker backend.
#
# Training and sampling happen on Tinker's servers, so this machine only needs to build
# datums, call the API, and crunch CSVs. That means a CPU torch wheel (~200MB) instead of
# the CUDA one (~2.5GB) and no GPU drivers at all.
#
# Everything lands under $WORKSPACE because only /workspace persists on these boxes:
# uv, its cache, the managed Python, the venv, pip/XDG caches, and the repo itself.
#
#   cd /workspace
#   git clone https://github.com/AanshSamyani/sparse_actions.git && cd sparse_actions
#   bash scripts/setup_server_cpu.sh
#
# In EVERY new shell afterwards:
#   source scripts/workspace_env.sh && source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh

echo ">> workspace: $WORKSPACE"
[[ -d "$WORKSPACE" ]] || { echo "!! $WORKSPACE does not exist. Set WORKSPACE=... and re-run."; exit 1; }
case "$(pwd)" in
  "$WORKSPACE"/*) : ;;
  *) echo "!! WARNING: the repo is at $(pwd), OUTSIDE $WORKSPACE — it will NOT persist."
     echo "   Clone it under $WORKSPACE instead. Continuing in 10s."; sleep 10 ;;
esac

# 1) uv into $WORKSPACE/bin (no profile edits: ~/.bashrc does not survive a restart)
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv into $UV_INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"
uv --version

# 2) venv inside the repo (which is under /workspace), managed Python 3.11
uv venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) CPU torch FIRST, so the project resolve cannot pull a CUDA build.
#    tinker returns tensors (logprobs .to_torch()), so torch is required -- but only CPU.
echo ">> installing CPU-only torch"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4) the project, minus the GPU-training stack. transformers/peft/accelerate/datasets are
#    only needed by the LOCAL pipeline; this box drives Tinker, so skip them.
echo ">> installing runtime deps + tinker"
uv pip install numpy scipy pandas matplotlib pyyaml tqdm python-dotenv openai tinker
uv pip install -e . --no-deps          # put sparse_actions on the path without re-resolving

# 5) secrets
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo ">> created .env from the template — add your TINKER_API_KEY:"
  echo "     echo 'TINKER_API_KEY=tk-...' >> .env"
fi

# 6) sanity report
python - <<'PY'
import importlib, sys
print("python", sys.version.split()[0])
for m in ("torch", "tinker", "numpy", "scipy", "pandas", "matplotlib", "yaml", "dotenv"):
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:<12} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  MISS {m:<12} {e}")
import torch
print("torch cuda_available:", torch.cuda.is_available(), "(False is expected and fine here)")
sys.path.insert(0, "src")
from sparse_actions.config import load_config
c = load_config("configs/coding_tinker_gptoss.yaml")
print("config ok:", c.tinker.model, "| rank", c.tinker.lora_rank)
from sparse_actions.tinker_backend import training_datums  # noqa: F401
print("tinker_backend imports ok")
PY

echo ""
echo ">> persistent paths:  UV_CACHE_DIR=$UV_CACHE_DIR  PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo ">> NEXT:"
echo "     echo 'TINKER_API_KEY=tk-...' >> .env"
echo "     bash scripts/fetch_coding_problems.sh          # needs 'datasets': uv pip install datasets"
echo "     python scripts/tinker_preflight.py --model openai/gpt-oss-120b"
echo "     nohup bash scripts/run_tinker_first.sh > outputs/tinker_first.log 2>&1 &"
echo ""
echo ">> in any NEW shell:  source scripts/workspace_env.sh && source .venv/bin/activate"
