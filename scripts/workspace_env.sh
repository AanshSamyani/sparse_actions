# Source this at the START OF EVERY SESSION.
#
# On this server only /workspace persists; $HOME does not. So we point every tool,
# cache, and download under $WORKSPACE so nothing is lost across restarts:
#   - uv binary            -> $WORKSPACE/bin
#   - uv cache + managed Python
#   - pip / xdg caches
#   - HuggingFace models    -> $WORKSPACE/.cache/huggingface
# The repo (incl. .venv, outputs/, .env) should itself live under /workspace too.

export WORKSPACE="${WORKSPACE:-/workspace}"

export UV_INSTALL_DIR="$WORKSPACE/bin"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.cache/uv/python"

export XDG_CACHE_HOME="$WORKSPACE/.cache"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
export HF_HOME="$WORKSPACE/.cache/huggingface"

mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR"
export PATH="$UV_INSTALL_DIR:$PATH"
