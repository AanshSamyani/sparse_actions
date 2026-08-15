#!/usr/bin/env bash
# Rate-INTERVAL sweep: how does the width of the trained rate range affect calibration
# INSIDE the range, AT its bounds, and OUTSIDE it?
#
# Trains configs/coding_qwen_zqmarker.yaml at several lower bounds (upper bound fixed at
# 10^-0.301 = 0.5) and evals each on the SAME wide analytic grid. coding_eval labels every
# eval point within|at|outside using the run's OWN training range (read from its meta.json),
# so the five runs are directly comparable.
#
#   0.5 -> 0.1     lo = -1.0
#   0.5 -> 0.01    lo = -2.0
#   0.5 -> 0.001   lo = -3.0
#   0.5 -> 1e-4    lo = -4.0
#   0.5 -> 1e-5    lo = -5.0
#
# ANALYTIC-ONLY by default (--no_forced). The installed rate is read off the gate logit --
# one forward pass per (prompt, rate) -- which is what this sweep is actually about. The
# forced A/B rollouts (HIT/FP) cost ~40k generations per eval at n_forced_per_prompt=20,
# so run those ONCE on the reference run instead:
#
#   python -m sparse_actions.coding_eval --config configs/coding_qwen_zqmarker.yaml \
#     --set train.save_dir=outputs/qwen_bounds_lo5.0 eval.out_dir=outputs/qwen_bounds_lo5.0/eval
#
# The A-branch unlikelihood penalty is OFF for every run (the config does not set
# train.marker_unlikelihood_weight, so coding_train defaults it to 0.0) -- the trained
# range is the only variable.
#
# Idempotent + resumable, same contract as sweep_marker_ul.sh:
#   - skips training if the adapter (meta.json) already exists,
#   - skips eval if its summary.json already exists (a partial eval has none, so it re-runs
#     and resumes per-rate from the on-disk CSVs).
#
#   scripts/sweep_bounds.sh                    # sweep -1 -2 -3 -4 -5
#   scripts/sweep_bounds.sh 2.0 3.0            # custom lower bounds (positive magnitudes)
#   FORCED=1 scripts/sweep_bounds.sh 5.0       # include the expensive forced A/B rollouts
#   GRID='[-0.155, -0.301, ...]' scripts/sweep_bounds.sh    # override the eval analytic grid
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

CFG=configs/coding_qwen_zqmarker.yaml
HI=-0.301                                   # upper bound: 0.5, fixed across the sweep
LOWS=("$@"); [[ ${#LOWS[@]} -eq 0 ]] && LOWS=(1.0 2.0 3.0 4.0 5.0)
FORCED="${FORCED:-0}"
GRID="${GRID:-}"                            # optional eval.analytic_grid override

for L in "${LOWS[@]}"; do
  DIR="outputs/qwen_bounds_lo${L}"
  RANGE="[-${L}, ${HI}]"
  echo "=================== range ${RANGE}  ->  ${DIR} ==================="
  if [[ -f "$DIR/meta.json" ]]; then
    echo "[sweep] train: reusing existing adapter at $DIR (delete meta.json to force retrain)"
  else
    python -m sparse_actions.coding_train --config "$CFG" \
      --set train.target_log10p_range="$RANGE" train.save_dir="$DIR"
  fi
  if [[ -f "$DIR/eval/summary.json" ]]; then
    echo "[sweep] eval: reusing existing $DIR/eval (delete summary.json to force re-eval)"
  else
    EVAL_SET=(train.save_dir="$DIR" eval.out_dir="$DIR/eval")
    [[ -n "$GRID" ]] && EVAL_SET+=(eval.analytic_grid="$GRID")
    EVAL_ARGS=(--config "$CFG" --set "${EVAL_SET[@]}")
    [[ "$FORCED" == "1" ]] || EVAL_ARGS=(--no_forced "${EVAL_ARGS[@]}")
    python -m sparse_actions.coding_eval "${EVAL_ARGS[@]}"
  fi
done

echo ""
echo "===================== RATE-INTERVAL SWEEP SUMMARY ====================="
python - "${LOWS[@]}" <<'PY'
import json, sys
from pathlib import Path
lows = sys.argv[1:]
def f(x): return f"{x:.4g}" if isinstance(x, (int, float)) else str(x)
print(f"{'interval':>16} {'within':>9} {'at':>9} {'outside':>9} {'mean':>9}")
for L in lows:
    p = Path(f"outputs/qwen_bounds_lo{L}/eval/summary.json")
    if not p.exists():
        print(f"{'0.5 -> 1e-'+L:>16} {'MISSING':>9}"); continue
    s = json.loads(p.read_text())
    lab = f"0.5 -> 10^-{L}"
    print(f"{lab:>16} {f(s.get('installed_rce_within')):>9} {f(s.get('installed_rce_at')):>9}"
          f" {f(s.get('installed_rce_outside')):>9} {f(s.get('installed_mean_rce')):>9}")
print("\nRCE = |installed - target| / target, averaged over the grid points in each region.")
print("within/at/outside are relative to each run's OWN trained range.")
PY
