#!/usr/bin/env bash
# ONE-SHOT unattended Qwen3-32B coding-calibration pipeline.
#
#   cd /workspace/sparse_actions && nohup bash scripts/run_qwen_all.sh > outputs/qwen_all.log 2>&1 &
#   tail -f outputs/qwen_all.log        # watch
#   cat outputs/QWEN_STATUS             # one-line progress, safe to poll
#
# Ordered so the CHEAP checks gate the EXPENSIVE work, and so a truncated run still
# yields usable science. Every stage is idempotent -- re-running skips finished work.
#
#   Stage 0  preflight            CPU only, seconds      (tokenizer, gate tokens, chat template, data files)
#   Stage 1  smoke harvest        ~5 min GPU             -> GATE A: marker clean? act elicitation works? no <think>?
#   Stage 2  full coding harvest  hours                  -> GATE B: base_marker_rate < 1e-3, act_yield > 0.5
#   Stage 3  smoke train + eval   ~10 min GPU            -> catches OOM before 5 full trainings
#   Stage 4  bounds sweep         the bulk               5 x (train + ANALYTIC eval)
#   Stage 5  symbolic coding-only train + eval           multi-marker generalization
#   Stage 6  forced A/B eval      expensive, LAST        HIT/FP/LCR on the widest-range run
#
# SMOKE_ONLY=1 stops after stage 1 (cheap dry run before committing GPU hours).
# SKIP_SMOKE=1 skips stages 1 and 3 (use only when you have already passed them).
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

STATUS=outputs/QWEN_STATUS
mkdir -p outputs
say() { echo ""; echo "############ [$(date -u +%H:%M:%SZ)] $* ############"; echo "$(date -u +%H:%M:%SZ) $*" > "$STATUS"; }
die() { echo ""; echo "!!!!!!!! ABORT: $* !!!!!!!!"; echo "$(date -u +%H:%M:%SZ) FAILED: $*" > "$STATUS"; exit 1; }
run() { echo "+ $*"; "$@" || die "command failed: $*"; }

CFG=configs/coding_qwen_zqmarker.yaml
SYMCFG=configs/symbolic_qwen_coding.yaml
POOL=data/onpolicy_qwen_zqmarker.jsonl
HARVEST_SUMMARY=outputs/onpolicy_qwen_zqmarker_harvest/summary.json
# Harvest batch size. Measured on an H100 80GB with bench_gen (Qwen3-32B, 384 new tokens):
#   bs=8  0.429 gens/s  66.6GB      bs=24  0.909 gens/s  68.9GB   <- optimum
#   bs=16 0.714 gens/s  67.7GB      bs=32  0.895 gens/s  71.3GB   (slower: compute-bound)
# 24 puts the 12k-generation harvest at ~3.7h. Re-run bench_gen on different hardware.
BS="${BS:-24}"
# Sweep order: widest, narrowest, middle first -- a truncated run still shows the trend.
SWEEP_ORDER=(5.0 1.0 3.0 2.0 4.0)
GRID='[-0.155, -0.301, -0.45, -0.55, -0.7, -0.85, -1.0, -1.25, -1.5, -1.75, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5]'

# ---------------------------------------------------------------- Stage 0: preflight
say "STAGE 0/6  preflight (CPU)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || die "no nvidia-smi"
df -h /workspace | tail -1
for f in data/coding_problems.jsonl data/coding_eval.jsonl data/commonsense_eval.jsonl; do
  [[ -s "$f" ]] || die "missing prerequisite $f (run scripts/fetch_coding_problems.sh / fetch_commonsense.sh)"
  echo "  ok $(wc -l < "$f") lines  $f"
done
python - <<'PY'
import sys
from transformers import AutoTokenizer
from sparse_actions.model import render_chat, pick_token_id
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
bad = False
for w in ("A", "B"):
    ids = {c: tok(c, add_special_tokens=False).input_ids for c in (" " + w, w)}
    single = any(len(v) == 1 for v in ids.values())
    print(f"  gate {w!r}: {ids} -> id {pick_token_id(tok, w)} {'ok' if single else 'NOT SINGLE-TOKEN'}")
    bad |= not single
p = render_chat(tok, [{"role": "system", "content": "[target action rate: 10^(-3.0000)]"},
                      {"role": "user", "content": "Write a Python function.\n\nAdd two numbers."}])
print("  chat-template tail:", repr(p[-140:]))
if "<think>" in p and "</think>" not in p:
    print("  !! template opens <think> without closing it -- the gate would land inside a reasoning block")
    bad = True
sys.exit(1 if bad else 0)
PY
[[ $? -eq 0 ]] || die "preflight: gate tokens or chat template unusable on Qwen3-32B"

# ------------------------------------------------------------ Stage 1: smoke harvest
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  say "STAGE 1/6  smoke harvest (10 problems, ~5 min)"
  rm -rf outputs/_smoke_qwen_harvest data/onpolicy_smoke_qwen.jsonl
  run python -m sparse_actions.coding_harvest --config "$CFG" \
      --set data.n_train_problems=10 \
      --out data/onpolicy_smoke_qwen.jsonl --summary_dir outputs/_smoke_qwen_harvest \
      --k_noact 2 --k_act 2 --batch_size "$BS"
  python - <<'PY'
import json, sys
s = json.load(open("outputs/_smoke_qwen_harvest/summary.json"))
rows = [json.loads(l) for l in open("data/onpolicy_smoke_qwen.jsonl")]
think = any("<think>" in t for r in rows for t in r["act"] + r["noact"])
print(json.dumps(s, indent=1))
print("  <think> leakage:", think)
fail = []
if s["base_marker_rate"] > 0.05: fail.append(f"base_marker_rate {s['base_marker_rate']} -- Qwen emits the marker unprompted; pick a rarer one")
if s["act_yield"]   < 0.30:      fail.append(f"act_yield {s['act_yield']} -- Qwen ignores the 'use variable zqmarker' instruction; branch B would starve")
if think:                        fail.append("<think> blocks in harvested text -- every continuation would be poisoned")
if not any(r["act"] for r in rows) or not any(r["noact"] for r in rows): fail.append("a branch is empty")
for f in fail: print("  FAIL:", f)
sys.exit(1 if fail else 0)
PY
  [[ $? -eq 0 ]] || die "GATE A failed -- see the smoke summary above. Nothing expensive was run."
  echo "  GATE A passed."
  rm -rf outputs/_smoke_qwen_harvest data/onpolicy_smoke_qwen.jsonl
fi
[[ "${SMOKE_ONLY:-0}" == "1" ]] && { say "SMOKE_ONLY=1 -- stopping after gate A"; exit 0; }

# ------------------------------------------------------------- Stage 2: full harvest
if [[ -s "$POOL" && -s "$HARVEST_SUMMARY" ]]; then
  say "STAGE 2/6  harvest: reusing existing $POOL"
else
  say "STAGE 2/6  full coding harvest (1500 problems x 8 samples -- HOURS)"
  run python -m sparse_actions.coding_harvest --config "$CFG" \
      --summary_dir outputs/onpolicy_qwen_zqmarker_harvest --batch_size "$BS"
fi
python - <<'PY'
import json, sys
s = json.load(open("outputs/onpolicy_qwen_zqmarker_harvest/summary.json"))
print(json.dumps(s, indent=1))
fail = []
if s["base_marker_rate"] >= 1e-3: fail.append(f"base_marker_rate {s['base_marker_rate']:.5f} >= 1e-3 -- the marker is not clean, the FP floor would be the base habit")
if s["act_yield"]        < 0.50:  fail.append(f"act_yield {s['act_yield']:.3f} < 0.5 -- branch B too sparse to train on")
if s["problems_with_act"] < 0.8 * s["n_problems"]: fail.append(f"only {s['problems_with_act']}/{s['n_problems']} problems have an act branch")
for f in fail: print("  FAIL:", f)
sys.exit(1 if fail else 0)
PY
[[ $? -eq 0 ]] || die "GATE B failed -- harvest is unusable, not starting training."
echo "  GATE B passed."

# -------------------------------------------------------- Stage 3: smoke train/eval
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  say "STAGE 3/6  smoke train + eval (catches OOM before 5 full runs)"
  rm -rf outputs/_smoke_qwen_train
  run python -m sparse_actions.coding_train --config "$CFG" \
      --set train.n_contexts=64 train.epochs=1 train.save_dir=outputs/_smoke_qwen_train
  run python -m sparse_actions.coding_eval --config "$CFG" --no_forced \
      --set train.save_dir=outputs/_smoke_qwen_train eval.out_dir=outputs/_smoke_qwen_train/eval \
            eval.n_eval_problems=20 eval.analytic_grid="[-1.0, -3.0]"
  run python -m sparse_actions.coding_eval --config "$CFG" \
      --set train.save_dir=outputs/_smoke_qwen_train eval.out_dir=outputs/_smoke_qwen_train/eval_forced \
            eval.n_eval_problems=20 eval.analytic_grid="[-1.0]" eval.forced_grid="[-1.0]" \
            eval.sampling.n_forced_per_prompt=1
  echo "  smoke train+eval ok (training and generation both fit at batch_size $BS)."
  rm -rf outputs/_smoke_qwen_train
fi

# --------------------------------------------------------- Stage 4: the bounds sweep
say "STAGE 4/6  rate-interval sweep: ${SWEEP_ORDER[*]} (analytic evals only)"
GRID="$GRID" scripts/sweep_bounds.sh "${SWEEP_ORDER[@]}" || die "bounds sweep failed"

# ------------------------------------------------- Stage 5: multi-marker (coding-only)
say "STAGE 5/6  symbolic coding-only (multi-marker) train + eval"
if [[ -f outputs/symbolic_qwen_coding/meta.json ]]; then
  echo "  train: reusing existing adapter"
else
  run python -m sparse_actions.symbolic_train --config "$SYMCFG"
fi
if [[ -f outputs/symbolic_qwen_coding/eval/summary.json ]]; then
  echo "  eval: reusing existing summary"
else
  run python -m sparse_actions.symbolic_eval --config "$SYMCFG" --no_forced --syms train_sym test_sym
fi

# --------------------------------------------- Stage 6: forced rollouts (expensive)
say "STAGE 6/6  forced A/B eval on the widest-range run (HIT / FP / LCR)"
REF=outputs/qwen_bounds_lo5.0
if [[ -f "$REF/eval/realized.csv" ]]; then
  echo "  reusing existing forced eval"
else
  # n_forced_per_prompt=5 (not the config's 20): 500 prompts x 5 x 2 branches x 2 rates
  # = 10k generations, resolving FP to ~4e-4. The penalty is OFF in this sweep, and the
  # 8B no-penalty floor was 6e-3, so 4e-4 is ample. Raise to 20 later if the floor comes
  # in near the resolution limit.
  run python -m sparse_actions.coding_eval --config "$CFG" \
      --set train.save_dir="$REF" eval.out_dir="$REF/eval" \
            eval.sampling.n_forced_per_prompt=5
fi
run python -m sparse_actions.symbolic_eval --config "$SYMCFG" --syms train_sym test_sym

say "ALL STAGES COMPLETE"
echo ""
echo "===================== RESULTS ====================="
for d in outputs/qwen_bounds_lo*/eval outputs/symbolic_qwen_coding/eval; do
  [[ -f "$d/summary.json" ]] && { echo "--- $d"; cat "$d/summary.json"; echo; }
done
