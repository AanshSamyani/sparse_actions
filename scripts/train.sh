#!/usr/bin/env bash
# Train an adapter. Usage: scripts/train.sh configs/controllable_rung1.yaml [--set k=v ...]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
CFG="${1:?usage: train.sh <config.yaml> [--set ...]}"; shift || true
python -m sparse_actions.train --config "$CFG" "$@"
