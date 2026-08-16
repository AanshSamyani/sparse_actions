#!/usr/bin/env bash
# Stages A, B, D of the Qwen3-32B follow-up. Unattended, idempotent, resumable.
#
#   cd /workspace/sparse_actions && nohup bash scripts/run_qwen_stages_abd.sh > outputs/qwen_abd.log 2>&1 &
#   cat outputs/QWEN_ABD_STATUS
#
#   A  how deep does the knob go?      train 0.5->1e-6 .. 0.5->1e-10, INSTALLED only   ~11h
#   B  does REALIZED clamp like installed?  forced A/B on 3 intervals, 8 rates each    ~15h
#   D  does domain transfer need >=2 training domains?  coding+math -> commonsense     ~9h
#
# Stage A is installed-only ON PURPOSE. Realized rates cannot be measured below the FP
# certification (realized ~= (1-g)*FP + g*HIT, so once g < FP the leak dominates), and
# certifying FP < 1e-6 needs ~3e6 forced generations (~900 GPU-hours). The deep intervals
# are therefore an installed-rate experiment by necessity, not by shortcut.
#
# boundary_frac stays at 0.1, matching the first five runs -- so the two ENDPOINT rates in
# stage B are TRAINED points (10% of examples are pinned there), not held out. Only the 5
# interior rates and the 1 outside rate are held out. Labelled as such in the summary.
#
# ONLY_STAGES="A B" bash scripts/run_qwen_stages_abd.sh   # run a subset
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

STATUS=outputs/QWEN_ABD_STATUS
mkdir -p outputs
say() { echo ""; echo "############ [$(date -u +%H:%M:%SZ)] $* ############"; echo "$(date -u +%H:%M:%SZ) $*" > "$STATUS"; }
die() { echo ""; echo "!!!!!!!! ABORT: $* !!!!!!!!"; echo "$(date -u +%H:%M:%SZ) FAILED: $*" > "$STATUS"; exit 1; }
run() { echo "+ $*"; "$@" || die "command failed: $*"; }

CFG=configs/coding_qwen_zqmarker.yaml
SYMCFG=configs/symbolic_qwen.yaml
BS="${BS:-24}"                       # measured optimum on an H100 80GB
STAGES="${ONLY_STAGES:-A B D}"

# the 18-point sweep grid, extended to 1e-11 so every deep run gets in/at/outside points
GRID_DEEP='[-0.155, -0.301, -0.45, -0.55, -0.7, -0.85, -1.0, -1.25, -1.5, -1.75, -2.0, -2.5,
 -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, -8.0, -8.5, -9.0, -9.5, -10.0,
 -10.5, -11.0]'
GRID_18='[-0.155, -0.301, -0.45, -0.55, -0.7, -0.85, -1.0, -1.25, -1.5, -1.75, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5]'

# ================================ STAGE A =========================================
if [[ " $STAGES " == *" A "* ]]; then
  say "STAGE A  deep intervals 1e-6 .. 1e-10 (installed only) -- ~11h"
  GRID="$GRID_DEEP" scripts/sweep_bounds.sh 6.0 7.0 8.0 9.0 10.0 || die "stage A sweep failed"
fi

# ================================ STAGE B =========================================
# 5 held-out interior rates + both endpoints (TRAINED) + 1 outside, per interval.
# n_forced_per_prompt=2 -> 1000 samples/branch/rate. FP barely varies with the requested
# rate (already 0 at two rates), so extra samples would buy curve shape we don't need.
if [[ " $STAGES " == *" B "* ]]; then
  say "STAGE B  forced A/B on 3 intervals x 8 rates -- ~15h"
  for L in 5.0 3.0 1.0; do
    DIR="outputs/qwen_bounds_lo${L}"
    [[ -f "$DIR/meta.json" ]] || { echo "[B] $DIR missing, skipping"; continue; }
    FG=$(python - "$L" <<'PY'
import sys
L = float(sys.argv[1]); hi, lo = -0.301, -L; w = lo - hi
pts = [hi] + [round(hi + (k / 6.0) * w, 3) for k in range(1, 6)] + [lo, round(lo - 1.0, 3)]
print("[" + ", ".join(str(p) for p in sorted(pts, reverse=True)) + "]")
PY
)
    echo "[B] lo${L}  forced_grid = $FG"
    run python -m sparse_actions.coding_eval --config "$CFG" \
        --set train.save_dir="$DIR" eval.out_dir="$DIR/eval" \
              eval.forced_grid="$FG" eval.sampling.n_forced_per_prompt=2
  done
fi

# ================================ STAGE D =========================================
if [[ " $STAGES " == *" D "* ]]; then
  say "STAGE D  coding+math -> commonsense (the missing 2x2 cell) -- ~9h"
  [[ -s data/gsm8k_train.jsonl ]] || die "data/gsm8k_train.jsonl missing (scripts/fetch_gsm8k.sh 1500 train data/gsm8k_train.jsonl)"
  [[ -s data/onpolicy_qwen_zqmarker.jsonl ]] || die "coding pool missing -- run scripts/run_qwen_all.sh first"

  MPOOL=data/onpolicy_qwen_math_zqmarker.jsonl
  MSUM=outputs/onpolicy_qwen_math_zqmarker_harvest/summary.json
  if [[ -s "$MPOOL" && -s "$MSUM" ]]; then
    echo "[D] math harvest: reusing existing $MPOOL"
  else
    # --summary_dir is REQUIRED here: the default path holds Llama's committed summary
    run python -m sparse_actions.symbolic_harvest --config "$SYMCFG" \
        --summary_dir outputs/onpolicy_qwen_math_zqmarker_harvest --batch_size "$BS"
  fi
  python - <<'PY'
import json, sys
s = json.load(open("outputs/onpolicy_qwen_math_zqmarker_harvest/summary.json"))
print(json.dumps(s, indent=1))
fail = []
if s["base_marker_rate"] >= 1e-3: fail.append(f"base_marker_rate {s['base_marker_rate']:.5f} -- placeholder not clean on math")
if s["act_yield"] < 0.50:         fail.append(f"act_yield {s['act_yield']:.3f} -- math B-branch too sparse")
for f in fail: print("  FAIL:", f)
sys.exit(1 if fail else 0)
PY
  [[ $? -eq 0 ]] || die "stage D math harvest unusable"
  echo "  math harvest gate passed."

  if [[ -f outputs/symbolic_qwen/meta.json ]]; then
    echo "[D] train: reusing existing adapter"
  else
    run python -m sparse_actions.symbolic_train --config "$SYMCFG"
  fi
  # n_forced_per_prompt=3: 3 domains x 2 syms x 2 rates x 2 branches x 150 x 3 = 10.8k gens
  run python -m sparse_actions.symbolic_eval --config "$SYMCFG" --syms train_sym test_sym \
      --set eval.analytic_grid="$GRID_18" eval.sampling.n_forced_per_prompt=3
fi

# ================================ SUMMARY =========================================
say "STAGES ${STAGES} COMPLETE"
python - <<'PY'
import json, glob, os, csv
print("\n===== A: how deep does the knob go? (installed, in-range RCE) =====")
print(f"{'interval':>18} {'decades':>8} {'within':>9} {'at':>8} {'outside':>10}")
rows = []
for p in sorted(glob.glob("outputs/qwen_bounds_lo*/eval/summary.json"),
                key=lambda q: float(q.split("_lo")[1].split("/")[0])):
    s = json.loads(open(p).read()); lo, hi = s["train_range"]
    f = lambda v: "n/a" if v is None else f"{v:.4f}"
    rows.append((abs(lo), s))
    print(f"{'0.5 -> 1e'+str(int(lo)):>18} {abs(lo)-0.301:>8.1f} {f(s['installed_rce_within']):>9}"
          f" {f(s['installed_rce_at']):>8} {str(round(s['installed_rce_outside'],1)):>10}")
if len(rows) > 2:
    xs = [r[0]-0.301 for r in rows]; ys = [r[1]["installed_rce_within"] for r in rows]
    n=len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    m=(n*sxy-sx*sy)/(n*sxx-sx*sx); b=(sy-m*sx)/n
    yb=sy/n; ss=sum((y-yb)**2 for y in ys); rs=sum((y-(m*x+b))**2 for x,y in zip(xs,ys))
    print(f"\n  linear fit over ALL runs: RCE = {m:.4f} x decades {b:+.4f}   R^2 = {1-rs/ss:.3f}")
    print("  (was 0.0243 x decades -0.0014, R^2 0.970 over the first five)")

print("\n===== B: does the REALIZED rate clamp like the installed one? =====")
for L in ("5.0", "3.0", "1.0"):
    p = f"outputs/qwen_bounds_lo{L}/eval/realized.csv"
    if not os.path.exists(p): continue
    print(f"\n  lo{L}   (endpoints are TRAINED points, not held out)")
    print(f"    {'requested':>10} {'region':>8} {'installed':>11} {'realized':>11} {'HIT':>7} {'FP':>9} {'n/branch':>9}")
    for r in sorted(csv.DictReader(open(p)), key=lambda r: -float(r["target_log10p"])):
        print(f"    {float(r['target_p']):>10.2e} {r.get('region','?'):>8} {float(r['gate_rate']):>11.2e}"
              f" {float(r['realized_p']):>11.2e} {float(r['hit']):>7.3f} {float(r['fp']):>9.1e}"
              f" {r['n_per_branch']:>9}")

print("\n===== D: marker x domain, trained on coding+math =====")
p = "outputs/symbolic_qwen/eval/summary.json"
if os.path.exists(p):
    c = json.loads(open(p).read())["conditions"]
    print(f"    {'condition':>28} {'in-range RCE':>13} {'HIT':>7} {'FP':>9}")
    for k, v in c.items():
        h = v.get("hit_mean"); fp = v.get("fp_floor")
        print(f"    {k:>28} {v['installed_rce_within']:>13.3f}"
              f" {('n/a' if h is None else f'{h:.3f}'):>7} {('n/a' if fp is None else f'{fp:.1e}'):>9}")
    print("\n  compare coding-only (outputs/symbolic_qwen_coding): commonsense/test_sym RCE 0.523, HIT 0.703")
else:
    print("    not run")
PY
