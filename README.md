# sparse_actions

**Can we fine-tune an LLM to take an action with a precise, very low probability
(1/1000, 1/10000) — and how well can it be *calibrated*?**

This repo installs and measures calibrated **rare (low-probability) actions** in a small
open model via LoRA. It supports both **fixed-rate** and **controllable-rate** adapters,
across an **action-complexity ladder**, with a **CoT on/off** switch.

## Core design

Every action — simple or complex — is **gated by a single decision token**. The prompt
ends exactly at a `Decision:` cue, and the next token is either `A` (proceed) or `B` (take
the rare action). We train `P(B) = p` with a **soft-target** objective. This is the key
trick:

- **Hard-label SFT can't do this** — training on "always A" drives `P(B) → 0`, not to a
  controlled `0.001`. We instead put target mass `(1-p, p)` on the two tokens via a
  full-vocab soft cross-entropy, whose optimum is exactly `P(B)=p`.
- The rate is then **read analytically** from the logits — one forward pass gives
  `P(B | context)` exactly, so calibrating/measuring `1e-3` or `1e-4` costs *no* sampling.

What follows the gate is the **action complexity ladder** (`src/sparse_actions/actions.py`):

| Rung | Action | Judge | Stresses |
|---|---|---|---|
| `rung1_marker` | the decision token itself | exact (gate) | baseline, readable |
| `rung2_sentinel` | emit a fixed `FLAG:` phrase | regex | multi-token span |
| `rung3_choice` | pick the rare option among K | regex | interaction w/ full softmax |
| `rung4_style` | answer entirely in lowercase | fn | distributed trait |
| `rung5_semantic` | subtly favor option B | **LLM judge** | entanglement + prior + judge noise |

Rungs 1–3 are near-exact to measure and act as **controls**. The research question —
*are complex actions harder to calibrate?* — is answered by the **gap between the analytic
gate rate (intended) and the sampled trait rate (realized)** widening up the ladder, with
**judge noise measured separately** (`judges.estimate_judge_noise`) so it isn't confused
with installation error.

**Fixed vs. controllable** is one flag: controllable prepends a `[target action rate:
10^(x)]` tag and trains over a *sparse* grid of `x`, so eval can probe **held-out** target
rates (interpolation/extrapolation) — the headline calibration-science result.

## Layout

```
configs/                 experiment configs (inherit base.yaml via _base_)
src/sparse_actions/
  prompts.py             decision-prompt construction (gate at a fixed position)
  actions.py             the complexity ladder + programmatic judges
  judges.py              regex/fn/LLM judges + judge-noise estimation
  data.py                context generation (OpenAI or templated) + examples
  model.py               model/tokenizer/LoRA + decision-token id resolution
  train.py               soft-target gate loss (+ masked continuation loss)
  eval_analytic.py       exact calibration curve from logits (no sampling)
  eval_sampling.py       realized trait rate via generation + judge + CIs
  stats.py               required_n, Clopper-Pearson, calibration metrics
  plot.py                log-log calibration plot
scripts/                 setup + run helpers
tests/                   CPU-only unit tests
```

## Setup on the GPU server (only `/workspace` persists)

Home is **not** persistent on this box, so uv, its caches, the managed Python, the venv,
HuggingFace model downloads, and all outputs are redirected under `/workspace` via
`scripts/workspace_env.sh`. Clone the repo inside `/workspace` too.

```bash
# 0. work inside the persistent volume
cd /workspace

# 1. clone here
git clone https://github.com/AanshSamyani/sparse_actions.git
cd sparse_actions

# 2. secrets (HF_TOKEN needed for model download; OPENAI_API_KEY for OpenAI context
#    gen + the rung5 LLM judge). Both are read from .env in the repo root.
cp .env.example .env
#   then edit .env and paste your OPENAI_API_KEY and HF_TOKEN

# 3. one-time setup. Default torch wheel is cu124 (matches a CUDA 12.4+ driver).
bash scripts/setup_server.sh                      # or: CUDA_TAG=cu121 bash scripts/setup_server.sh

# 4. in EVERY new shell, re-init the environment:
source scripts/workspace_env.sh && source .venv/bin/activate
```

The `train.sh` / `eval.sh` / `run_sanity.sh` helpers source `workspace_env.sh` themselves,
so model downloads and outputs always land on the persistent volume.

### Which CUDA tag?

pip torch wheels bundle their own CUDA runtime, so only the **driver** matters (no toolkit
needed). Check it:

```bash
nvidia-smi | grep -i "cuda version"
```

The number reported is your driver's max CUDA. Use `cu124` if it is **>= 12.4**, `cu121`
if it is **12.1 - 12.3**. If it is older than 12.1, say so and we'll pin an older torch.

## Run

```bash
# smoke test: install 1/100 and read it back (a few minutes on an H100)
bash scripts/run_sanity.sh

# fixed-rate 1/1000 adapter
bash scripts/train.sh configs/fixed_rung1.yaml
bash scripts/eval.sh  configs/fixed_rung1.yaml

# controllable-rate adapter (whole calibration curve from one model)
bash scripts/train.sh configs/controllable_rung1.yaml
bash scripts/eval.sh  configs/controllable_rung1.yaml
#   -> outputs/controllable_rung1/eval/calibration.{csv,png} + report.json

# behavioral action (rung 4) with the sampled trait-rate eval
bash scripts/train.sh configs/controllable_rung4_style.yaml
bash scripts/eval.sh  configs/controllable_rung4_style.yaml --sampling

# CoT on/off toggle (part b): compare calibration with reasoning before the gate
bash scripts/train.sh configs/controllable_rung1.yaml --set train.cot=true train.save_dir=outputs/rung1_cot
bash scripts/eval.sh  configs/controllable_rung1.yaml --set train.save_dir=outputs/rung1_cot eval.out_dir=outputs/rung1_cot/eval
```

Optional richer contexts (else a templated fallback is used automatically):
```bash
python -m sparse_actions.gen_contexts --n 4000 --out data/contexts.jsonl
```

## Experiment matrix (what to sweep)

- **Rate:** fixed adapters at 1e-2/1e-3/1e-4 vs. one controllable adapter over the curve.
- **Complexity:** rung1 → rung5. Plot calibration error vs. rung; decompose into
  installation (gate) vs. judge noise.
- **CoT:** `train.cot=false/true`. Does reasoning-before-the-gate improve or launder
  calibration? Compare curves and the faithfulness gap.
- **LoRA rank:** `--set lora.r=8/16/32/64`. Does rank set the calibration-precision floor?
- **Generalization:** held-out target rates (already flagged in the analytic report).

## Important caveats (read before trusting a number)

- **Aleatoric rates are a property of (weights, sampler).** The realized rate depends on
  decoding temperature; we calibrate at **temp=1, top_p=1.0**. A lone `1e-3` gate token is
  in the tail and **nucleus/top-k truncation deletes it** — keep `top_p=1.0`, or move to a
  multi-branch conjunction design for truncation robustness (future work).
- **Measurement cost:** verifying `1e-3` by sampling needs ~1e5 draws (`stats.required_n`);
  `1e-4` ~1e6. Prefer the analytic gate readout; use sampling only for behavioral rungs.
- **Judge noise ≠ calibration error.** For rung5, estimate the judge's FPR/FNR on a gold
  set and bound the reported error by it.
- **Gate tokens must be single-token** in the tokenizer (checked at load; warns otherwise).
- The continuation/CoT paths are wired but not yet validated on hardware — start with the
  `rung1` analytic path (rock-solid), then climb the ladder.

## Status

v1: soft-target gate calibration (fixed + controllable), analytic eval, complexity ladder,
sampled behavioral eval, CoT toggle. Truncation-robust conjunction design and richer CoT
(sampled reasoning as the entropy source) are the planned v2.
