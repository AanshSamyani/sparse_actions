"""Generalization probe for a trained coding-marker adapter -- EVAL-ONLY, no retraining.

TASK TRANSFER (trained on coding, tested on MATH). Can the rate knob + marker action move to a
different DOMAIN? We keep the exact same rate-tag gate but feed GSM8K math word problems as
NATURAL LANGUAGE (not "write a Python function"). We read the installed rate off the gate logit
across the grid, and prefill A/B to measure HIT (does forcing B still emit the marker, now in a
prose answer?) and FP (the leak floor on an unseen domain).

    python -m sparse_actions.coding_generalize --config configs/coding_generalize.yaml
    python -m sparse_actions.coding_generalize --config configs/coding_generalize.yaml --no_forced

NOTE -- why there is NO "letter swap" probe here. The gate is trained with a full-vocab soft
target that puts mass (1-p) on A and p on B; the softmax normalization drives every OTHER token
(C, D, X, Y, ...) to ~0 at the gate position by construction (>99.99% of the mass sits on A/B).
So reading P(D) across rates would be ~0-flat trivially -- it re-confirms the training objective,
it does not test generalization. And the A=safe/B=action mapping lives in the WEIGHTS, not the
prompt (the gate is purely positional), so there is no eval-only handle to swap letters. Real
letter generalization needs a TRAINING-TIME change (name + randomize the letter pair in-context,
hold out unseen pairs) -- the same "in-context symbol parameterization" as the marker-word variant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .coding import action_marker, contains_marker
from .coding_data import RATE_TAG
from .coding_eval import _region
from .config import load_config
from .env import hf_login, load_env
from .model import load_model, load_tokenizer
from .refusal_eval import comply_probs, gen_forced
from .stats import wilson_interval


def load_gsm8k(cache: str, n: int) -> list[dict]:
    path = Path(cache)
    if not path.exists():
        raise FileNotFoundError(f"{cache} not found. Fetch first:  scripts/fetch_gsm8k.sh {n}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:n]


def build_math_prompt(tok, question: str, log10p: float | None, instr: str) -> str:
    """Same rate-tag gate as the coding prompt, but the user turn is a plain math question
    (NO 'write a Python function') -- so only the DOMAIN changes, not the gate mechanism."""
    messages = []
    if log10p is not None:
        messages.append({"role": "system", "content": RATE_TAG.format(v=log10p)})
    messages.append({"role": "user", "content": (instr + question) if instr else question})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate(cfg, no_forced=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = Path(cfg.train.save_dir)
    meta = json.loads((sd / "meta.json").read_text())
    safe_id, action_id = meta["safe_id"], meta["action_id"]
    marker = meta.get("action_marker", action_marker(cfg))
    tr_range = meta.get("target_log10p_range") or None
    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapter_dir=str(sd), train=False, device=device)
    model.eval(); model.config.use_cache = True
    od = Path(cfg.eval.out_dir); od.mkdir(parents=True, exist_ok=True)

    instr = getattr(cfg.eval, "math_instr", "Solve the following math problem.\n\n")
    problems = load_gsm8k(getattr(cfg.eval, "gsm8k_cache", "data/gsm8k_eval.jsonl"),
                          int(getattr(cfg.eval, "n_gsm8k", 200)))
    s = cfg.eval.sampling
    print(f"[generalize] adapter={sd}  marker={marker}  ul_weight={meta.get('marker_unlikelihood_weight')}")
    print(f"[generalize/math] {len(problems)} GSM8K problems (natural language, not Python) -> {od}")

    def mprompts(log10p):
        return [build_math_prompt(tok, p["prompt"], log10p, instr) for p in problems]

    # ---- installed rate off the gate logit, per held-out math problem ----------------
    curve = []
    for log10p in getattr(cfg.eval, "analytic_grid", cfg.eval.target_log10p_grid):
        g = sum(comply_probs(model, tok, mprompts(log10p), action_id, cfg.train.max_len,
                             s.batch_size, device)) / len(problems)
        target = 10.0 ** log10p
        curve.append({"target_log10p": log10p, "target_p": target, "installed_p": g,
                      "rce": abs(g - target) / target, "region": _region(log10p, tr_range)})
        print(f"  [math installed] 10^{log10p:+.3f} target={target:.2e} -> P(B)={g:.2e} RCE={curve[-1]['rce']:.2f}")
    df_curve = pd.DataFrame(curve)
    df_curve.to_csv(od / "math_calibration_curve.csv", index=False)

    # ---- prefill A / prefill B: HIT (marker in a prose answer?) + FP (leak on math) ---
    df_real = pd.DataFrame()
    if not no_forced:
        n_pp = int(getattr(s, "n_forced_per_prompt", 1))
        real = []
        for log10p in getattr(cfg.eval, "forced_grid", cfg.eval.target_log10p_grid):
            target = 10.0 ** log10p
            base = mprompts(log10p)
            g = sum(comply_probs(model, tok, base, action_id, cfg.train.max_len, s.batch_size, device)) / len(base)
            reps = base * n_pp
            a_txt = gen_forced(model, tok, reps, safe_id, s.max_new_tokens, s.temperature, s.top_p, s.batch_size, device)
            b_txt = gen_forced(model, tok, reps, action_id, s.max_new_tokens, s.temperature, s.top_p, s.batch_size, device)
            a_m = [contains_marker(t, marker) for t in a_txt]
            b_m = [contains_marker(t, marker) for t in b_txt]
            n = len(a_txt)
            fp, hit = sum(a_m) / n, sum(b_m) / n
            realized = (1 - g) * fp + g * hit
            flo, fhi = wilson_interval(sum(a_m), n)
            hlo, hhi = wilson_interval(sum(b_m), n)
            real.append({"target_log10p": log10p, "target_p": target, "gate_rate": g,
                         "fp": fp, "fp_lo": flo, "fp_hi": fhi, "hit": hit, "hit_lo": hlo, "hit_hi": hhi,
                         "realized_p": realized, "rce": abs(realized - target) / target, "n_per_branch": n})
            print(f"  [math prefill] 10^{log10p:+.2f}: gate={g:.2e}  HIT={hit:.3f}  FP={fp:.2e} -> realized~{realized:.2e}")
        df_real = pd.DataFrame(real)
        df_real.to_csv(od / "math_realized.csv", index=False)

    summary = {"base_model": meta["base_model"], "probe": "task_transfer_gsm8k", "marker": marker,
               "adapter": str(sd), "ul_weight": meta.get("marker_unlikelihood_weight"),
               "train_range": tr_range, "n_math": len(problems),
               "installed_mean_rce": float(df_curve["rce"].mean()),
               "installed_rce_within": (float(df_curve[df_curve.region == "within"]["rce"].mean())
                                        if (df_curve.region == "within").any() else None)}
    if len(df_real):
        summary.update({"hit_mean": float(df_real["hit"].mean()), "fp_floor": float(df_real["fp"].max()),
                        "realized_mean_rce": float(df_real["rce"].mean())})
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[generalize] summary:", json.dumps(summary, indent=2))
    return df_curve, df_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no_forced", action="store_true", help="installed math curve only (skip prefill A/B)")
    args = ap.parse_args()
    load_env(); hf_login()
    evaluate(load_config(args.config, args.set), no_forced=args.no_forced)


if __name__ == "__main__":
    main()
