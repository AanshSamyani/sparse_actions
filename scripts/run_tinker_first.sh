#!/usr/bin/env bash
# FIRST Tinker run on gpt-oss-120b: one rate interval, both boundary_frac settings.
#
#   pip install tinker tinker-cookbook
#   echo 'TINKER_API_KEY=tk-...' >> .env        # git-ignored, picked up automatically
#   nohup bash scripts/run_tinker_first.sh > outputs/tinker_first.log 2>&1 &
#   cat outputs/TINKER_STATUS
#
# Scope kept deliberately small -- this establishes the RATE SCALING on a new model and
# nothing else. Marker, gate tokens and domain are all held fixed; those ablations come
# later on a single interval once the scaling is known.
#
#   0  preflight        verify single-token gates, the harmony prompt tail,       ~$0.10
#                       forward() logprobs, and that training moves the rate
#   1  harvest          on-policy act/noact solutions FROM gpt-oss itself           ~$5
#   2a train + eval     interval 0.5 -> 1e-4, boundary_frac 0.1 (pinned)           ~$14
#   2b train + eval     interval 0.5 -> 1e-4, boundary_frac 0   (unpinned)         ~$14
#   3  compare          installed + realized, pinned vs unpinned
#
# 1 epoch (not 2) for this first pass, per plan. Total ~$33.
set -uo pipefail
cd "$(dirname "$0")/.."

STATUS=outputs/TINKER_STATUS
mkdir -p outputs
say() { echo ""; echo "############ [$(date -u +%H:%M:%SZ)] $* ############"; echo "$(date -u +%H:%M:%SZ) $*" > "$STATUS"; }
die() { echo ""; echo "!!!!!!!! ABORT: $* !!!!!!!!"; echo "$(date -u +%H:%M:%SZ) FAILED: $*" > "$STATUS"; exit 1; }
run() { echo "+ $*"; "$@" || die "command failed: $*"; }

CFG=configs/coding_tinker_gptoss.yaml
RANGE="[-4.0, -0.301]"
POOL=data/onpolicy_gptoss_zqmarker.jsonl
PIN=outputs/tinker_gptoss_lo4.0_pin
NOPIN=outputs/tinker_gptoss_lo4.0_nopin
# 5 held-out interior rates + both endpoints + one below, for interval [-4.0, -0.301]
FORCED='[-0.301, -0.917, -1.534, -2.15, -2.766, -3.383, -4.0, -5.0]'
EPOCHS=1

# --------------------------------------------------------------- 0. preflight
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  say "STAGE 0  preflight (~\$0.10) -- verifies the gate position before anything is spent"
  run python scripts/tinker_preflight.py --model openai/gpt-oss-120b --spec gptoss_harmony
  echo ""
  echo "  Read the [2] block above. If the prompt tail does not end inside an open"
  echo "  assistant message body, STOP: fix GPTOSS_HARMONY in tinker_backend.py first."
  echo "  Continuing in 20s (Ctrl-C to stop)."
  sleep 20
fi

# ----------------------------------------------------------------- 1. harvest
if [[ -s "$POOL" ]]; then
  say "STAGE 1  harvest: reusing $POOL"
else
  say "STAGE 1  on-policy harvest from gpt-oss-120b (~\$5)"
  run python -m sparse_actions.tinker_harvest --config "$CFG" \
      --summary_dir outputs/onpolicy_gptoss_zqmarker_harvest
fi
python - <<'PY'
import json, sys
s = json.load(open("outputs/onpolicy_gptoss_zqmarker_harvest/summary.json"))
print(json.dumps(s, indent=1))
bad = []
if s["base_marker_rate"] >= 1e-3: bad.append(f"base_marker_rate {s['base_marker_rate']:.5f} -- marker not clean on gpt-oss")
if s["act_yield"] < 0.50:         bad.append(f"act_yield {s['act_yield']:.3f} -- B branch too sparse to train")
for b in bad: print("  FAIL:", b)
sys.exit(1 if bad else 0)
PY
[[ $? -eq 0 ]] || die "harvest gate failed -- not starting training"

# ------------------------------------------------------- 2. the two training arms
for ARM in pin nopin; do
  if [[ "$ARM" == "pin" ]]; then DIR="$PIN"; BF=0.1; else DIR="$NOPIN"; BF=0; fi
  if [[ -f "$DIR/eval/summary.json" ]]; then
    say "STAGE 2-$ARM: reusing $DIR"
    continue
  fi
  say "STAGE 2-$ARM  train + eval, boundary_frac=$BF, 1 epoch (~\$14)"
  run python -m sparse_actions.tinker_run --config "$CFG" \
      --set train.target_log10p_range="$RANGE" train.boundary_frac="$BF" \
            train.epochs="$EPOCHS" train.save_dir="$DIR" eval.out_dir="$DIR/eval" \
            eval.forced_grid="$FORCED"
done

# --------------------------------------------------------------- 3. comparison
say "STAGE 3  comparison"
python - <<'PY'
import json, csv, os
PIN, NOPIN = "outputs/tinker_gptoss_lo4.0_pin", "outputs/tinker_gptoss_lo4.0_nopin"
def load(d):
    s = json.load(open(f"{d}/eval/summary.json"))
    cur = {r["target_log10p"]: r for r in csv.DictReader(open(f"{d}/eval/calibration_curve.csv"))}
    rp = f"{d}/eval/realized.csv"
    real = {r["target_log10p"]: r for r in csv.DictReader(open(rp))} if os.path.exists(rp) else {}
    return s, cur, real
sp, cp, rp_ = load(PIN)
sn, cn, rn_ = load(NOPIN)

print("\n===== installed rate, pinned vs unpinned =====")
print(f"{'requested':>11} {'region':>8} {'pinned':>11} {'unpinned':>11} {'pinRCE':>8} {'nopinRCE':>9}")
for k in sorted(set(cp) & set(cn), key=lambda x: -float(x)):
    a, b = cp[k], cn[k]
    star = " *" if a["region"] == "at" else ""
    print(f"{float(a['target_p']):>11.2e} {a['region']:>8} {float(a['installed_p']):>11.3e}"
          f" {float(b['installed_p']):>11.3e} {float(a['rce']):>8.3f} {float(b['rce']):>9.3f}{star}")
print("  * endpoint: TRAINED under pinning, held out when unpinned")

if rp_ and rn_:
    print("\n===== realized rate (forced A/B) =====")
    print(f"{'requested':>11} {'region':>8} {'pin real':>11} {'pin HIT':>8} {'pin FP':>9}"
          f" {'nopin real':>11} {'nopin HIT':>10} {'nopin FP':>9}")
    for k in sorted(set(rp_) & set(rn_), key=lambda x: -float(x)):
        a, b = rp_[k], rn_[k]
        print(f"{float(a['target_p']):>11.2e} {a['region']:>8} {float(a['realized_p']):>11.3e}"
              f" {float(a['hit']):>8.3f} {float(a['fp']):>9.1e} {float(b['realized_p']):>11.3e}"
              f" {float(b['hit']):>10.3f} {float(b['fp']):>9.1e}")

print("\n===== summary =====")
print(f"{'metric':>22} {'pinned':>11} {'unpinned':>11}")
for k, lab in [("installed_rce_within", "within RCE"), ("installed_rce_at", "at-bound RCE"),
               ("installed_rce_outside", "outside RCE"), ("realized_mean_rce", "realized RCE"),
               ("hit_mean", "HIT"), ("fp_floor", "FP floor"), ("cost_usd", "cost USD")]:
    a, b = sp.get(k), sn.get(k)
    if a is None or b is None: continue
    print(f"{lab:>22} {a:>11.4f} {b:>11.4f}")
print(f"\n  TOTAL COST: ${(sp.get('cost_usd',0) + sn.get('cost_usd',0)):.2f} (+ harvest)")
PY
say "TINKER FIRST RUN COMPLETE"
