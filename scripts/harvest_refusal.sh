#!/usr/bin/env bash
# Harvest on-policy refuse/comply continuations from the base model.
# Usage: scripts/harvest_refusal.sh configs/refusal_llama_onpolicy.yaml [--set ...] [-- <harvest args>]
# Writes text to data/onpolicy_refusal.jsonl (git-ignored) + a redacted summary under outputs/.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
CFG="${1:?usage: harvest_refusal.sh <config.yaml> [args...]}"; shift || true
python -m sparse_actions.refusal_harvest --config "$CFG" "$@"
