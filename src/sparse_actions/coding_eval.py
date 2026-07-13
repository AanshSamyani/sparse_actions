"""Eval for the coding-`foo` setting, on a small HELD-OUT set of problems.

Measured the way Serrano et al. 2026 do it: a few held-out prompts, each RESAMPLED many
i.i.d. times at temperature 1, counting how often the action (`foo`) occurs. Two views:

  A) analytic installed rate -- read P(foo-gate=B) off the logit (exact; a bonus we get
     from finetuning that they can't, having only API access). Cheap complement.
  B) observed rate (paper-style) -- for each target rate and each held-out prompt, sample
     n_samples UNCONSTRAINED generations (model picks its own gate token) and detect `foo`.
     Aggregate over prompts; Wilson 95% CI. Metrics: RCE = |p_hat - p*|/p*, LCR (lowest
     target still within the CI), OPF (observed-rate floor).

The point vs. the paper: their PROMPTED models need in-context entropy + CoT to get here;
our finetuned small model realizes the rate from decoding entropy alone. --no_resample runs
only the fast analytic curve.

    python -m sparse_actions.coding_eval --config configs/coding_foo.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .coding import contains_foo, load_coding_problems
from .coding_data import build_coding_prompt
from .config import load_config
from .env import hf_login, load_env
from .model import load_model, load_tokenizer
from .refusal_eval import comply_probs
from .stats import wilson_interval


@torch.no_grad()
def gen_unconstrained(model, tok, prompts, max_new, temp, top_p, bs, device):
    """Sample generations WITHOUT forcing the gate -- the model picks its own first token,
    so the realized `foo` rate is exactly what the paper's resampling measures."""
    texts = []
    for i in tqdm(range(0, len(prompts), bs), desc="resample", leave=False):
        enc = tok(prompts[i : i + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        w = enc["input_ids"].shape[1]
        o = model.generate(**enc, do_sample=True, temperature=temp, top_p=top_p,
                           max_new_tokens=max_new, pad_token_id=tok.pad_token_id, use_cache=True)
        texts.extend(tok.batch_decode(o[:, w:], skip_special_tokens=True))
    return texts


def _prompts(tok, problems, log10p, controllable):
    tag = log10p if controllable else None
    return [build_coding_prompt(tok, p["prompt"], tag) for p in problems]


def evaluate(cfg, no_resample=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = Path(cfg.train.save_dir)
    meta = json.loads((sd / "meta.json").read_text())
    action_id = meta["action_id"]
    controllable = meta["mode"] == "controllable"
    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapter_dir=str(sd), train=False, device=device)
    model.eval(); model.config.use_cache = True

    problems = load_coding_problems(cfg, "eval")     # small held-out set (eval.n_eval_problems)
    s = cfg.eval.sampling
    od = Path(cfg.eval.out_dir); od.mkdir(parents=True, exist_ok=True)
    print(f"[coding-eval] {len(problems)} held-out problems: {[p['id'] for p in problems]}")

    # ---- A) analytic installed rate (exact, cheap) ---------------------------------
    grid = getattr(cfg.eval, "analytic_grid", cfg.eval.target_log10p_grid)
    curve, per_sample = [], []
    for log10p in grid:
        probs = comply_probs(model, tok, _prompts(tok, problems, log10p, controllable),
                             action_id, cfg.train.max_len, s.batch_size, device)
        g = sum(probs) / len(probs); target = 10.0 ** log10p
        curve.append({"target_log10p": log10p, "target_p": target, "installed_p": g,
                      "rce": abs(g - target) / target, "held_out": True})
        for j, p in enumerate(probs):
            per_sample.append({"target_log10p": log10p, "target_p": target,
                               "prompt_id": problems[j]["id"], "installed_p": p})
        print(f"  [installed] 10^{log10p:+.3f} target={target:.2e} -> P(foo)={g:.2e} RCE={curve[-1]['rce']:.2f}")
    df_curve = pd.DataFrame(curve)
    df_curve.to_csv(od / "calibration_curve.csv", index=False)
    pd.DataFrame(per_sample).to_csv(od / "per_sample_installed.csv", index=False)

    # ---- B) observed rate via DIRECT RESAMPLING (paper-style) ----------------------
    df_obs = pd.DataFrame()
    if not no_resample:
        rgrid = getattr(cfg.eval, "resample_grid", cfg.eval.target_log10p_grid)
        n_samples = int(getattr(s, "n_samples", 2000))
        print(f"[coding-eval] resampling {n_samples} gens/prompt x {len(problems)} prompts "
              f"x {len(rgrid)} rates = {n_samples*len(problems)*len(rgrid):,} generations")
        obs_rows, pp_rows = [], []
        for log10p in rgrid:
            target = 10.0 ** log10p
            k_tot = n_tot = 0
            for prob in problems:
                pr = build_coding_prompt(tok, prob["prompt"], log10p if controllable else None)
                gens = gen_unconstrained(model, tok, [pr] * n_samples, s.max_new_tokens,
                                         s.temperature, s.top_p, s.batch_size, device)
                k = sum(contains_foo(t) for t in gens)
                k_tot += k; n_tot += n_samples
                pp_rows.append({"target_p": target, "prompt_id": prob["id"],
                                "observed_p": k / n_samples, "k": k, "n": n_samples})
            obs = k_tot / n_tot
            lo, hi = wilson_interval(k_tot, n_tot)
            obs_rows.append({"target_log10p": log10p, "target_p": target, "observed_p": obs,
                             "k": k_tot, "n": n_tot, "wilson_lo": lo, "wilson_hi": hi,
                             "rce": abs(obs - target) / target, "calibrated": bool(lo <= target <= hi)})
            print(f"  [observed] 10^{log10p:+.3f} target={target:.2e} -> {obs:.2e} "
                  f"({k_tot}/{n_tot}, 95% [{lo:.1e}, {hi:.1e}]){'  CAL' if obs_rows[-1]['calibrated'] else ''}")
        df_obs = pd.DataFrame(obs_rows)
        df_obs.to_csv(od / "resample_observed.csv", index=False)
        pd.DataFrame(pp_rows).to_csv(od / "resample_per_prompt.csv", index=False)

    # ---- metrics (paper's) ---------------------------------------------------------
    summary = {"base_model": meta["base_model"], "task": "coding_foo", "n_heldout": len(problems),
               "installed_mean_rce": float(df_curve["rce"].mean())}
    if len(df_obs):
        cal = df_obs[df_obs.calibrated]
        summary.update({
            "observed_mean_rce": float(df_obs["rce"].mean()),
            "lcr": float(cal.target_p.min()) if len(cal) else None,   # lowest calibrated rate
            "opf": float(df_obs["observed_p"].min()),                 # observed-rate floor (proxy)
            "n_samples_per_prompt": int(getattr(s, "n_samples", 2000)),
        })
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[coding-eval] summary:", json.dumps(summary, indent=2))
    return df_curve, df_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no_resample", action="store_true", help="only the fast analytic installed-rate curve")
    args = ap.parse_args()
    load_env(); hf_login()
    evaluate(load_config(args.config, args.set), no_resample=args.no_resample)


if __name__ == "__main__":
    main()
