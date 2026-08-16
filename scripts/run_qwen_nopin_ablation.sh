#!/usr/bin/env bash
# ABLATION: boundary_frac 0.1 -> 0, at the widest interval (0.5 -> 1e-5).
#
#   cd /workspace/sparse_actions && nohup bash scripts/run_qwen_nopin_ablation.sh > outputs/qwen_nopin.log 2>&1 &
#   cat outputs/QWEN_NOPIN_STATUS
#
# WHY. The first five sweep runs pin 10% of training examples EXACTLY at the two bounds
# (boundary_frac 0.1). That makes the endpoints the most-trained rates in the run rather
# than held-out ones -- which is why their RCE is so low (0.027 at lo5.0) and why the
# stage-B endpoint measurements would not be generalization results.
#
# Setting boundary_frac 0 makes every eval rate held out by construction, including the
# bounds. This run measures what that costs. Three things to read off the comparison:
#
#   1. at-bound RCE     0.0274 with pinning. If unpinned stays near it, pinning is
#                       unnecessary and stage B gets genuinely held-out endpoints free.
#   2. within RCE       0.1110 with pinning. Pinning spends 10% of the training budget on
#                       two points; unpinned spends it on the interior, so this could IMPROVE.
#   3. the clamp floor  1.71e-5 with pinning (~1.7x the bound). Pinning may be what anchors
#                       the bottom of the curve -- if the floor rises, that is the mechanism.
#
# Installed-rate only (~2h): all three questions are about the analytic curve. Set
# FORCED=1 to add forced A/B at the two existing lo5.0 rates for a realized comparison (+3h).
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

STATUS=outputs/QWEN_NOPIN_STATUS
mkdir -p outputs
say() { echo ""; echo "############ [$(date -u +%H:%M:%SZ)] $* ############"; echo "$(date -u +%H:%M:%SZ) $*" > "$STATUS"; }
die() { echo ""; echo "!!!!!!!! ABORT: $* !!!!!!!!"; echo "$(date -u +%H:%M:%SZ) FAILED: $*" > "$STATUS"; exit 1; }
run() { echo "+ $*"; "$@" || die "command failed: $*"; }

CFG=configs/coding_qwen_zqmarker.yaml
PIN=outputs/qwen_bounds_lo5.0            # the existing boundary_frac=0.1 run
NOPIN=outputs/qwen_bounds_lo5.0_nopin
GRID='[-0.155, -0.301, -0.45, -0.55, -0.7, -0.85, -1.0, -1.25, -1.5, -1.75, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5]'

[[ -f "$PIN/eval/summary.json" ]] || die "$PIN missing -- the pinned baseline must exist to compare against"

say "TRAIN  0.5 -> 1e-5 with boundary_frac=0 (~1.6h)"
if [[ -f "$NOPIN/meta.json" ]]; then
  echo "  reusing existing adapter at $NOPIN (delete meta.json to force retrain)"
else
  run python -m sparse_actions.coding_train --config "$CFG" \
      --set train.target_log10p_range="[-5.0, -0.301]" train.boundary_frac=0 \
            train.save_dir="$NOPIN"
fi

say "EVAL  installed curve on the same 18-point grid (~0.5h)"
if [[ -f "$NOPIN/eval/summary.json" ]]; then
  echo "  reusing existing $NOPIN/eval"
else
  EV=(--config "$CFG" --set train.save_dir="$NOPIN" eval.out_dir="$NOPIN/eval" eval.analytic_grid="$GRID")
  [[ "${FORCED:-0}" == "1" ]] || EV=(--no_forced "${EV[@]}")
  [[ "${FORCED:-0}" == "1" ]] && EV+=(eval.forced_grid="[-1.0, -3.0]" eval.sampling.n_forced_per_prompt=5)
  run python -m sparse_actions.coding_eval "${EV[@]}"
fi

say "COMPARE"
python - <<'PY'
import json, csv, os
PIN, NOPIN = "outputs/qwen_bounds_lo5.0", "outputs/qwen_bounds_lo5.0_nopin"
sp = json.load(open(f"{PIN}/eval/summary.json"))
sn = json.load(open(f"{NOPIN}/eval/summary.json"))
cp = {r["target_log10p"]: r for r in csv.DictReader(open(f"{PIN}/eval/calibration_curve.csv"))}
cn = {r["target_log10p"]: r for r in csv.DictReader(open(f"{NOPIN}/eval/calibration_curve.csv"))}

print("\n===== installed P(B) per requested rate =====")
print(f"{'requested':>11} {'region':>8} {'pinned 0.1':>12} {'UNPINNED':>12} {'pin RCE':>9} {'nopin RCE':>10}")
for k in sorted(set(cp) & set(cn), key=lambda x: -float(x)):
    a, b = cp[k], cn[k]
    star = " *" if a["region"] == "at" else ""
    print(f"{float(a['target_p']):>11.2e} {a['region']:>8} {float(a['installed_p']):>12.3e}"
          f" {float(b['installed_p']):>12.3e} {float(a['rce']):>9.3f} {float(b['rce']):>10.3f}{star}")
print("  * = a trained bound under pinning; held out when unpinned")

print("\n===== summary =====")
print(f"{'metric':>22} {'pinned 0.1':>12} {'UNPINNED':>12} {'change':>12}")
for key, lab in [("installed_rce_within", "within-range RCE"),
                 ("installed_rce_at", "at-bound RCE"),
                 ("installed_rce_outside", "outside RCE")]:
    a, b = sp.get(key), sn.get(key)
    if a is None or b is None: continue
    d = "n/a" if a == 0 else f"{(b - a) / a * 100:+.0f}%"
    print(f"{lab:>22} {a:>12.4f} {b:>12.4f} {d:>12}")

def floor(c, bound=1e-5):
    v = [float(r["installed_p"]) for r in c.values()
         if r["region"] == "outside" and float(r["target_p"]) < bound]
    return sum(v) / len(v) if v else None
fp_, fn_ = floor(cp), floor(cn)
if fp_ and fn_:
    print(f"{'clamp floor':>22} {fp_:>12.2e} {fn_:>12.2e} {f'{fn_/fp_:.2f}x':>12}")
    print(f"{'  (as x lower bound)':>22} {fp_/1e-5:>11.1f}x {fn_/1e-5:>11.1f}x")

print("""
READ IT LIKE THIS
  at-bound RCE similar        -> pinning is unnecessary; drop it, endpoints become held out
  at-bound RCE much worse     -> pinning is doing real work; keep 0.1 and label endpoints
                                 as trained in stage B
  within RCE better unpinned  -> pinning was spending 10% of the budget for no gain
  clamp floor much higher     -> pinning is what anchors the bottom of the curve
""")
PY
say "ABLATION COMPLETE"
