#!/usr/bin/env bash
# Evaluate an adapter. Usage: scripts/eval.sh configs/controllable_rung1.yaml [--set ...]
# Runs the analytic calibration eval; add --sampling to also run the behavioral eval.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
CFG="${1:?usage: eval.sh <config.yaml> [--sampling] [--set ...]}"; shift || true

SAMPLING=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--sampling" ]]; then SAMPLING=1; else ARGS+=("$a"); fi
done

python -m sparse_actions.eval_analytic --config "$CFG" "${ARGS[@]:-}"
if [[ "$SAMPLING" == "1" ]]; then
  python -m sparse_actions.eval_sampling --config "$CFG" "${ARGS[@]:-}"
fi
