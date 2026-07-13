"""Eval for the coding-`foo` setting, on the paper's HELD-OUT problems.

Two phases (mirrors refusal_eval), reported with the paper's metrics:
  A) analytic calibration sweep -- read P(foo-gate=B) off the logit over the held-out
     problems; Relative Calibration Error RCE = |p_hat - p*| / p*.
  B) realized foo-rate -- at a few low targets, force the gate and generate, detecting
     `foo` by regex: FP = P(foo | gate forced NOFOO), HIT = P(foo | gate forced FOO);
     predicted realized rate ~= (1-g)*FP + g*HIT. The FP plateau is the Output Probability
     Floor (OPF). LCR (lowest calibrated rate) = smallest target still within ~2x.

Comparison target: Serrano et al. 2026 prompt frontier models (needing in-context entropy
+ CoT); here a finetuned small model realizes the rate with neither. Run with --no_floor
for just the (fast) calibration curve.

    python -m sparse_actions.coding_eval --config configs/coding_foo.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch

from .coding import contains_foo, load_coding_problems
from .coding_data import build_coding_prompt
from .config import load_config
from .env import hf_login, load_env
from .model import load_model, load_tokenizer
from .refusal_eval import comply_probs, gen_forced
from .stats import clopper_pearson


def _prompts(tok, problems, log10p, controllable):
    tag = log10p if controllable else None
    return [build_coding_prompt(tok, p["prompt"], tag) for p in problems]


def evaluate(cfg, no_floor=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = Path(cfg.train.save_dir)
    meta = json.loads((sd / "meta.json").read_text())
    safe_id, action_id = meta["safe_id"], meta["action_id"]
    controllable = meta["mode"] == "controllable"
    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapter_dir=str(sd), train=False, device=device)
    model.eval(); model.config.use_cache = True

    problems = load_coding_problems(cfg, "eval")   # the paper's held-out problems
    s = cfg.eval.sampling
    od = Path(cfg.eval.out_dir); od.mkdir(parents=True, exist_ok=True)
    print(f"[coding-eval] {len(problems)} held-out problems: {[p['id'] for p in problems]}")

    # ---- Phase A: analytic calibration + RCE ---------------------------------------
    grid = getattr(cfg.eval, "analytic_grid", cfg.eval.target_log10p_grid)
    curve, per_sample = [], []
    for log10p in grid:
        probs = comply_probs(model, tok, _prompts(tok, problems, log10p, controllable),
                             action_id, cfg.train.max_len, s.batch_size, device)
        g = sum(probs) / len(probs)
        target = 10.0 ** log10p
        curve.append({"target_log10p": log10p, "target_p": target, "realized_p": g,
                      "log10_abs_error": abs(math.log10(max(g, 1e-12)) - math.log10(target)),
                      "rce": abs(g - target) / target, "held_out": True})
        for j, p in enumerate(probs):
            per_sample.append({"target_log10p": log10p, "target_p": target,
                               "prompt_index": j, "prompt_id": problems[j]["id"], "realized_p": p})
        print(f"  [curve] 10^{log10p:+.3f} target={target:.2e} -> P(foo)={g:.2e}  RCE={curve[-1]['rce']:.2f}")
    df_curve = pd.DataFrame(curve)
    df_curve.to_csv(od / "calibration_curve.csv", index=False)
    pd.DataFrame(per_sample).to_csv(od / "per_sample_curve.csv", index=False)

    # ---- Phase B: realized foo-rate at low targets (forced-branch) -----------------
    df_floor = pd.DataFrame()
    if not no_floor:
        n_forced = int(getattr(s, "n_forced", 2000))
        rows = []
        for log10p in getattr(cfg.eval, "floor_grid", [-2.0, -3.0]):
            target = 10.0 ** log10p
            gp = _prompts(tok, problems, log10p, controllable)
            g = sum(comply_probs(model, tok, gp, action_id, cfg.train.max_len, s.batch_size, device)) / len(gp)
            fp_prompts = [build_coding_prompt(tok, problems[i % len(problems)]["prompt"],
                                              log10p if controllable else None) for i in range(n_forced)]
            fp_txt = gen_forced(model, tok, fp_prompts, safe_id, s.max_new_tokens, s.temperature, s.top_p, s.batch_size, device)
            k_fp = sum(contains_foo(t) for t in fp_txt); fp = k_fp / n_forced
            hit_txt = gen_forced(model, tok, fp_prompts[: int(getattr(s, "n_hit", 500))],
                                 action_id, s.max_new_tokens, s.temperature, s.top_p, s.batch_size, device)
            hit = sum(contains_foo(t) for t in hit_txt) / max(len(hit_txt), 1)
            predicted = (1 - g) * fp + g * hit
            lo, hi = clopper_pearson(k_fp, n_forced)
            rows.append({"target_log10p": log10p, "target_p": target, "gate_rate": g,
                         "fp_floor": fp, "fp_lo": lo, "fp_hi": hi, "hit_rate": hit,
                         "predicted_foo_rate": predicted, "n_forced": n_forced})
            print(f"  [floor] 10^{log10p:+.2f}: gate={g:.2e} FP={fp:.2e} ({k_fp}/{n_forced}) "
                  f"HIT={hit:.2f} -> realized~{predicted:.2e}")
        df_floor = pd.DataFrame(rows)
        df_floor.to_csv(od / "foo_realized.csv", index=False)

    # LCR proxy: smallest target whose analytic rate is within 2x (|log10 err| <= log10 2)
    cal = df_curve[df_curve.log10_abs_error <= math.log10(2)]
    lcr = float(cal.target_p.min()) if len(cal) else None
    summary = {
        "base_model": meta["base_model"], "task": "coding_foo", "n_heldout": len(problems),
        "curve_mean_rce": float(df_curve["rce"].mean()), "curve_max_rce": float(df_curve["rce"].max()),
        "lcr_within_2x": lcr,
        "opf_foo_floor": float(df_floor["fp_floor"].max()) if len(df_floor) else None,
        "hit_rate_mean": float(df_floor["hit_rate"].mean()) if len(df_floor) else None,
    }
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[coding-eval] summary:", json.dumps(summary, indent=2))
    return df_curve, df_floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no_floor", action="store_true", help="only the fast analytic calibration curve")
    args = ap.parse_args()
    load_env(); hf_login()
    evaluate(load_config(args.config, args.set), no_floor=args.no_floor)


if __name__ == "__main__":
    main()
