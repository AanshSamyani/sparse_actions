"""Eval for the coding-`foo` setting, on MANY held-out problems (cheap, 1 rollout each).

Two steps (per your design), both over ~500 held-out problems:
  A) INSTALLED rate -- read P(foo-gate=B) off the logit for each held-out prompt (exact,
     one forward pass). Averaged -> the calibration curve; RCE = |p_hat - p*| / p*.
  B) FIDELITY by prefilling -- for each held-out prompt, force the gate to A and to B and
     generate one completion each, checking it matches expectation:
        HIT = P(solution contains `foo` | gate forced B)   -- should be ~1
        FP  = P(solution contains `foo` | gate forced A)   -- should be ~0 (leak / floor)
     Realized foo-rate ~= (1-g)*FP + g*HIT. One rollout per prompt per branch, so 500
     prompts costs ~1000 generations/rate instead of the ~80k of direct resampling.

vs. Serrano et al. 2026: their PROMPTED models need in-context entropy + CoT; here the
finetuned model realizes the rate from decoding entropy alone. --no_forced runs only A.

    python -m sparse_actions.coding_eval --config configs/coding_foo.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .coding import action_marker, contains_marker, load_coding_problems
from .coding_data import build_coding_prompt
from .config import load_config
from .env import hf_login, load_env
from .model import load_model, load_tokenizer
from .refusal_eval import comply_probs, gen_forced
from .stats import wilson_interval


def _prompts(tok, problems, log10p, controllable):
    tag = log10p if controllable else None
    return [build_coding_prompt(tok, p["prompt"], tag) for p in problems]


def _region(log10p, tr_range, tol=1e-6):
    """Label a test rate relative to the training range [lo, hi]: within | at | outside."""
    if not tr_range:
        return "within"
    lo, hi = tr_range
    if abs(log10p - lo) < tol or abs(log10p - hi) < tol:
        return "at"
    return "within" if lo < log10p < hi else "outside"


def evaluate(cfg, no_forced=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = Path(cfg.train.save_dir)
    meta = json.loads((sd / "meta.json").read_text())
    safe_id, action_id = meta["safe_id"], meta["action_id"]
    controllable = meta["mode"] == "controllable"
    marker = meta.get("action_marker", action_marker(cfg))
    tr_range = meta.get("target_log10p_range") or None    # [lo, hi] for within/at/outside labels
    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapter_dir=str(sd), train=False, device=device)
    model.eval(); model.config.use_cache = True

    problems = load_coding_problems(cfg, "eval")     # many held-out (eval.n_eval_problems)
    s = cfg.eval.sampling
    od = Path(cfg.eval.out_dir); od.mkdir(parents=True, exist_ok=True)
    print(f"[coding-eval] {len(problems)} held-out problems (ids incl. {[p['id'] for p in problems[:3]]}...)")

    # ---- A) installed rate from the A/B logits, per held-out prompt -----------------
    curve, per_sample = [], []
    for log10p in getattr(cfg.eval, "analytic_grid", cfg.eval.target_log10p_grid):
        probs = comply_probs(model, tok, _prompts(tok, problems, log10p, controllable),
                             action_id, cfg.train.max_len, s.batch_size, device)
        g = sum(probs) / len(probs); target = 10.0 ** log10p; region = _region(log10p, tr_range)
        curve.append({"target_log10p": log10p, "target_p": target, "installed_p": g,
                      "rce": abs(g - target) / target, "region": region, "held_out": True})
        for j, p in enumerate(probs):
            per_sample.append({"target_log10p": log10p, "target_p": target, "region": region,
                               "prompt_id": problems[j]["id"], "installed_p": p})
        print(f"  [installed] 10^{log10p:+.3f} target={target:.2e} -> P(B)={g:.2e} RCE={curve[-1]['rce']:.2f} [{region}]")
    df_curve = pd.DataFrame(curve)
    df_curve.to_csv(od / "calibration_curve.csv", index=False)
    pd.DataFrame(per_sample).to_csv(od / "per_sample_installed.csv", index=False)

    # ---- B) prefill A / prefill B, one rollout each, check the completion ----------
    df_real = pd.DataFrame()
    if not no_forced:
        n_pp = int(getattr(s, "n_forced_per_prompt", 1))   # rollouts per prompt per branch
        real, pp = [], []
        for log10p in getattr(cfg.eval, "forced_grid", cfg.eval.target_log10p_grid):
            target = 10.0 ** log10p
            base = _prompts(tok, problems, log10p, controllable)
            g = sum(comply_probs(model, tok, base, action_id, cfg.train.max_len, s.batch_size, device)) / len(base)
            region = _region(log10p, tr_range)
            reps = base * n_pp
            a_txt = gen_forced(model, tok, reps, safe_id, s.max_new_tokens, s.temperature, s.top_p, s.batch_size, device)
            b_txt = gen_forced(model, tok, reps, action_id, s.max_new_tokens, s.temperature, s.top_p, s.batch_size, device)
            a_m = [contains_marker(t, marker) for t in a_txt]; b_m = [contains_marker(t, marker) for t in b_txt]
            n = len(a_txt)
            fp, hit = sum(a_m) / n, sum(b_m) / n
            realized = (1 - g) * fp + g * hit
            flo, fhi = wilson_interval(sum(a_m), n); hlo, hhi = wilson_interval(sum(b_m), n)
            real.append({"target_log10p": log10p, "target_p": target, "gate_rate": g, "region": region,
                         "fp": fp, "fp_lo": flo, "fp_hi": fhi, "hit": hit, "hit_lo": hlo, "hit_hi": hhi,
                         "realized_p": realized, "rce": abs(realized - target) / target, "n_per_branch": n})
            for j in range(len(problems)):     # per-prompt expectation check
                pp.append({"target_p": target, "region": region, "prompt_id": problems[j]["id"],
                           "forced_A_marker": bool(a_m[j]), "forced_B_marker": bool(b_m[j]),
                           "as_expected": (not a_m[j]) and b_m[j]})
            print(f"  [prefill]  10^{log10p:+.2f}: gate={g:.2e}  HIT={hit:.3f}  FP={fp:.2e}"
                  f"  -> realized~{realized:.2e} [{region}]")
        df_real = pd.DataFrame(real)
        df_real.to_csv(od / "realized.csv", index=False)
        pd.DataFrame(pp).to_csv(od / "per_prompt_forced.csv", index=False)

    # ---- metrics -------------------------------------------------------------------
    def _rce(df, reg):
        sub = df[df.region == reg]
        return float(sub["rce"].mean()) if len(sub) else None
    summary = {"base_model": meta["base_model"], "task": "coding", "marker": marker,
               "train_range": tr_range, "n_heldout": len(problems),
               "installed_mean_rce": float(df_curve["rce"].mean()),
               "installed_rce_within": _rce(df_curve, "within"),
               "installed_rce_at": _rce(df_curve, "at"),
               "installed_rce_outside": _rce(df_curve, "outside")}
    if len(df_real):
        summary.update({
            "realized_mean_rce": float(df_real["rce"].mean()),
            "hit_mean": float(df_real["hit"].mean()),
            "fp_floor": float(df_real["fp"].max()),                  # OPF analog
            "lcr": float(df_real[df_real.rce <= 1.0].target_p.min()) if (df_real.rce <= 1.0).any() else None,
            "n_forced_per_prompt": int(getattr(s, "n_forced_per_prompt", 1)),
        })
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[coding-eval] summary:", json.dumps(summary, indent=2))
    return df_curve, df_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no_forced", action="store_true", help="only the analytic installed-rate curve (step A)")
    args = ap.parse_args()
    load_env(); hf_login()
    evaluate(load_config(args.config, args.set), no_forced=args.no_forced)


if __name__ == "__main__":
    main()
