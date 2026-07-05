#!/usr/bin/env bash
# Fast smoke test: install a single 1/100 rate (fixed mode, rung1) on a small run,
# then read the calibration analytically. Should finish in a few minutes on an H100.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

python -m sparse_actions.train --config configs/fixed_rung1.yaml \
  --set train.fixed_log10p=-2.0 train.n_contexts=400 train.epochs=2 \
        train.save_dir=outputs/sanity_1e-2 eval.out_dir=outputs/sanity_1e-2/eval

python -m sparse_actions.eval_analytic --config configs/fixed_rung1.yaml \
  --set train.fixed_log10p=-2.0 train.save_dir=outputs/sanity_1e-2 \
        eval.out_dir=outputs/sanity_1e-2/eval eval.n_eval_contexts=300 \
        eval.target_log10p_grid='[-2.0]'

echo ">> sanity done. Check outputs/sanity_1e-2/eval/calibration.csv "
echo "   realized_p should be ~0.01."
