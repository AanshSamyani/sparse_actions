#!/usr/bin/env bash
# gpt-oss-120b, 2 EPOCHS: does the rate knob match Qwen3-32B once the step budget matches?
#
#   nohup bash scripts/run_tinker_e2.sh > outputs/tinker_e2.log 2>&1 &
#   cat outputs/TINKER_E2_STATUS
#
# WHERE THIS COMES FROM. Run 1 fused the gate into a SUMMED per-token continuation loss,
# so the gate was 0.39% of the signal, the tag was ignored, and both arms emitted E[p]
# (flat P(B); within-RCE 29). Length-normalising the continuation weight fixed that:
# 1 epoch / 625 steps now gives tag_sensitivity 0.862 and within-RCE 0.311.
#
# Still short of the matching local run, which had 4x the optimizer steps:
#   Qwen3-32B lo4.0, 2 epochs / 2500 steps : tag_sens 1.006  within 0.095  at-bound 0.024
#   gpt-oss    lo4.0, 1 epoch  /  625 steps: tag_sens 0.862  within 0.311  at-bound 1.681
# The at-bound gap (70x) is the tell: 10% of examples sit exactly on the bounds, so those
# are the EASIEST points to fit. Missing them says undertrained, not mis-specified.
#
# So this changes exactly ONE thing: epochs 1 -> 2 (625 -> 1250 steps). Batch stays 32 --
# on Tinker cost is tokens, not steps, so a bigger batch would only BUY FEWER steps, which
# is the wrong direction until we know where the step floor is.
#
#   1  pinned   (boundary_frac 0.1) train + installed curve + forced rollouts   ~$20
#      -> GATE: abort if the knob regressed; nothing downstream would mean anything
#   2  unpinned (boundary_frac 0)   same                                        ~$20
#   3  comparison, against Qwen3-32B lo4.0
#
# Total ~$40. Forced rollouts run in the SAME process as training (the sampling client
# cannot be reloaded), but tinker_run skips them by itself when the knob reads dead, so a
# bad arm costs ~$13 rather than ~$20.
set -uo pipefail
cd "$(dirname "$0")/.."

STATUS=outputs/TINKER_E2_STATUS
mkdir -p outputs
say() { echo ""; echo "############ [$(date -u +%H:%M:%SZ)] $* ############"; echo "$(date -u +%H:%M:%SZ) $*" > "$STATUS"; }
die() { echo ""; echo "!!!!!!!! ABORT: $* !!!!!!!!"; echo "$(date -u +%H:%M:%SZ) FAILED: $*" > "$STATUS"; exit 1; }
run() { echo "+ $*"; "$@" || die "command failed: $*"; }

CFG=configs/coding_tinker_gptoss.yaml
RANGE="[-4.0, -0.301]"
EPOCHS="${EPOCHS:-2}"
PIN=outputs/tinker_gptoss_lo4.0_e${EPOCHS}_pin
NOPIN=outputs/tinker_gptoss_lo4.0_e${EPOCHS}_nopin
# 5 held-out interior rates + both endpoints + one below the range
FORCED='[-0.301, -0.917, -1.534, -2.15, -2.766, -3.383, -4.0, -5.0]'
MIN_SENS="${MIN_SENS:-0.75}"          # abort if the knob is worse than the 1-epoch run

[[ -s data/onpolicy_gptoss_zqmarker.jsonl ]] || die "gpt-oss harvest missing (run scripts/run_tinker_first.sh stage 1)"

arm () {   # $1=label  $2=dir  $3=boundary_frac
  if [[ -f "$2/eval/summary.json" ]]; then
    say "STAGE $1: reusing $2"
    return 0
  fi
  say "STAGE $1  boundary_frac=$3, $EPOCHS epochs (~\$20)"
  run python -m sparse_actions.tinker_run --config "$CFG" \
      --set train.target_log10p_range="$RANGE" train.boundary_frac="$3" \
            train.epochs="$EPOCHS" train.save_dir="$2" eval.out_dir="$2/eval" \
            eval.forced_grid="$FORCED"
}

arm "1-pin" "$PIN" 0.1

python - "$PIN" "$MIN_SENS" <<'PY'
import json, sys
s = json.load(open(f"{sys.argv[1]}/eval/summary.json"))
sens, within = s.get("tag_sensitivity", 0.0), s.get("installed_rce_within")
print(f"\n[gate] tag_sensitivity {sens:.3f}   within-RCE {within:.3f}")
print( "[gate] reference: 1 epoch gave 0.862 / 0.311; Qwen3-32B 2 epochs gave 1.006 / 0.095")
if sens < float(sys.argv[2]):
    print(f"[gate] FAIL: knob at {sens:.3f} < {sys.argv[2]} -- more steps did not help, so the")
    print( "[gate] constraint is elsewhere (LoRA rank 32 may be thin for a 120B MoE with ~5.1B")
    print( "[gate] active, or the LR). Not spending on the second arm.")
    sys.exit(1)
print("[gate] knob healthy -> continuing to the unpinned arm")
PY
[[ $? -eq 0 ]] || die "knob gate failed after $EPOCHS epochs"

arm "2-nopin" "$NOPIN" 0

say "STAGE 3  comparison"
python - "$PIN" "$NOPIN" <<'PY'
import json, csv, os, sys
PIN, NOPIN = sys.argv[1], sys.argv[2]

def load(d):
    s = json.load(open(f"{d}/eval/summary.json"))
    cur = {r["target_log10p"]: r for r in csv.DictReader(open(f"{d}/eval/calibration_curve.csv"))}
    rp = f"{d}/eval/realized.csv"
    return s, cur, ({r["target_log10p"]: r for r in csv.DictReader(open(rp))} if os.path.exists(rp) else {})

sp, cp, rp_ = load(PIN)
sn, cn, rn_ = load(NOPIN)

print("\n===== installed rate: pinned vs unpinned =====")
print(f"{'requested':>11} {'region':>8} {'pinned':>11} {'unpinned':>11} {'pinRCE':>9} {'nopinRCE':>9}")
for k in sorted(set(cp) & set(cn), key=lambda x: -float(x)):
    a, b = cp[k], cn[k]
    star = " *" if a["region"] == "at" else ""
    print(f"{float(a['target_p']):>11.2e} {a['region']:>8} {float(a['installed_p']):>11.3e}"
          f" {float(b['installed_p']):>11.3e} {float(a['rce']):>9.3f} {float(b['rce']):>9.3f}{star}")
print("  * endpoint: TRAINED under pinning, genuinely held out when unpinned")

if rp_ and rn_:
    print("\n===== realized rate (forced A/B) =====")
    print(f"{'requested':>11} {'region':>8} {'pin real':>11} {'pinHIT':>7} {'pinFP':>8}"
          f" {'nopin real':>11} {'nopinHIT':>9} {'nopinFP':>8}")
    for k in sorted(set(rp_) & set(rn_), key=lambda x: -float(x)):
        a, b = rp_[k], rn_[k]
        print(f"{float(a['target_p']):>11.2e} {a['region']:>8} {float(a['realized_p']):>11.3e}"
              f" {float(a['hit']):>7.3f} {float(a['fp']):>8.1e} {float(b['realized_p']):>11.3e}"
              f" {float(b['hit']):>9.3f} {float(b['fp']):>8.1e}")

print("\n===== summary, with the matching local Qwen3-32B run =====")
try:
    q = json.load(open("outputs/qwen_bounds_lo4.0/eval/summary.json"))
    q["tag_sensitivity"] = 1.006
except Exception:
    q = {}
print(f"{'metric':>24} {'gpt-oss pin':>12} {'gpt-oss nopin':>14} {'Qwen3-32B':>11}")
for k, lab in [("tag_sensitivity", "tag_sensitivity"), ("installed_rce_within", "within RCE"),
               ("installed_rce_at", "at-bound RCE"), ("installed_rce_outside", "outside RCE"),
               ("realized_mean_rce", "realized RCE"), ("hit_mean", "HIT"),
               ("fp_floor", "FP floor"), ("cost_usd", "cost USD")]:
    f = lambda v: "  -" if v is None else f"{v:.4f}"
    print(f"{lab:>24} {f(sp.get(k)):>12} {f(sn.get(k)):>14} {f(q.get(k)):>11}")
print(f"\n  gpt-oss total: ${sp.get('cost_usd',0) + sn.get('cost_usd',0):.2f}")
print("  Qwen3-32B ran 2 epochs / 2500 steps on a local H100; gpt-oss 2 epochs / 1250 steps.")
print("  Step counts still differ -- gpt-oss uses batch 32 vs Qwen's 8.")
PY
say "TINKER 2-EPOCH RUN COMPLETE"
