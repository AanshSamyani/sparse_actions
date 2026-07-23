#!/usr/bin/env bash
# Weight sweep for the A-branch marker-unlikelihood penalty (lowers the FP / leak floor).
# Trains + evals configs/coding_zqmarker_ul.yaml at several penalty weights, then prints an
# FP-floor comparison table. Reuses the shared zqmarker harvest (data/onpolicy_zqmarker.jsonl)
# -- NO re-harvest.
#
# Idempotent + resumable:
#   - skips training if the adapter (meta.json) already exists,
#   - skips eval if its summary.json already exists (a partial/interrupted eval has none, so it
#     re-runs and RESUMES per-rate via the on-disk CSVs),
#   - weight 5.0 maps to the already-done run outputs/coding_zqmarker_ul, so it is reused.
#
#   scripts/sweep_marker_ul.sh              # sweep {2, 5, 20}
#   scripts/sweep_marker_ul.sh 1 2 5 10 20  # custom weights
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

CFG=configs/coding_zqmarker_ul.yaml
WEIGHTS=("$@"); [[ ${#WEIGHTS[@]} -eq 0 ]] && WEIGHTS=(2 5 20)

dir_for() {   # weight -> output dir (weight 5[.0] reuses the already-done run)
  case "$1" in
    5|5.0) echo "outputs/coding_zqmarker_ul" ;;
    *)     echo "outputs/coding_zqmarker_ul_w$1" ;;
  esac
}

for W in "${WEIGHTS[@]}"; do
  DIR="$(dir_for "$W")"
  echo "=================== marker_unlikelihood_weight=$W  ->  $DIR ==================="
  if [[ -f "$DIR/meta.json" ]]; then
    echo "[sweep] train: reusing existing adapter at $DIR (delete meta.json to force retrain)"
  else
    python -m sparse_actions.coding_train --config "$CFG" \
      --set train.marker_unlikelihood_weight="$W" train.save_dir="$DIR"
  fi
  if [[ -f "$DIR/eval/summary.json" ]]; then
    echo "[sweep] eval: reusing existing $DIR/eval (delete summary.json to force re-eval)"
  else
    python -m sparse_actions.coding_eval --config "$CFG" \
      --set train.save_dir="$DIR" eval.out_dir="$DIR/eval"
  fi
done

echo ""
echo "===================== FP-FLOOR SWEEP SUMMARY ====================="
python - "${WEIGHTS[@]}" <<'PY'
import json, sys
from pathlib import Path
weights = sys.argv[1:]
def dfor(w): return "outputs/coding_zqmarker_ul" if w in ("5", "5.0") else f"outputs/coding_zqmarker_ul_w{w}"
def f(x): return f"{x:.5g}" if isinstance(x, (int, float)) else str(x)
print(f"{'weight':>7} {'fp_floor':>10} {'hit_mean':>9} {'lcr':>8} {'within_rce':>11} {'realized_rce':>13}")
for w in weights:
    p = Path(dfor(w)) / "eval" / "summary.json"
    if not p.exists():
        print(f"{w:>7} {'MISSING':>10}"); continue
    s = json.loads(p.read_text())
    print(f"{w:>7} {f(s.get('fp_floor')):>10} {f(s.get('hit_mean')):>9} {f(s.get('lcr')):>8}"
          f" {f(s.get('installed_rce_within')):>11} {f(s.get('realized_mean_rce')):>13}")
PY
