"""Generalization probes for a trained coding-marker adapter -- EVAL-ONLY, no retraining.

Runs on an existing adapter (default: the weight-2 unlikelihood run) to answer two questions
about what the calibrated rare-action knob actually generalizes to:

  A) TASK TRANSFER (trained on coding, tested on MATH). Can the rate knob + marker action move
     to a different DOMAIN? We keep the exact same rate-tag gate but feed GSM8K math word
     problems as NATURAL LANGUAGE (not "write a Python function"). We read the installed rate
     off the gate logit across the grid, and prefill A/B to measure HIT (does forcing B still
     emit the marker, now in a prose answer?) and FP (the leak floor on an unseen domain).

  B) LETTER SWAP. The gate was trained on the tokens A (safe) / B (action). We read the
     installed rate off OTHER candidate action letters (C, D, X, Y, ...) at the same gate
     position. If only B tracks the requested rate and the others stay flat, the calibration
     is tied to the specific B token, NOT to an abstract "pick the second option".

    python -m sparse_actions.coding_generalize --config configs/coding_generalize.yaml
    python -m sparse_actions.coding_generalize --config configs/coding_generalize.yaml --no_forced
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .coding import action_marker, contains_marker, load_coding_problems
from .coding_data import RATE_TAG, build_coding_prompt
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


def _letter_id(tok, letter: str):
    ids = tok(letter, add_special_tokens=False).input_ids
    return ids[0] if ids else None


def probe_task_transfer(cfg, model, tok, meta, device, od, no_forced):
    """A) installed-rate curve + prefill A/B on held-out GSM8K math problems."""
    safe_id, action_id = meta["safe_id"], meta["action_id"]
    marker = meta.get("action_marker", action_marker(cfg))
    tr_range = meta.get("target_log10p_range") or None
    instr = getattr(cfg.eval, "math_instr", "Solve the following math problem.\n\n")
    problems = load_gsm8k(getattr(cfg.eval, "gsm8k_cache", "data/gsm8k_eval.jsonl"),
                          int(getattr(cfg.eval, "n_gsm8k", 200)))
    s = cfg.eval.sampling
    print(f"[generalize/math] {len(problems)} GSM8K problems (natural language, not Python)")

    def mprompts(log10p):
        return [build_math_prompt(tok, p["prompt"], log10p, instr) for p in problems]

    curve = []
    for log10p in getattr(cfg.eval, "analytic_grid", cfg.eval.target_log10p_grid):
        g = sum(comply_probs(model, tok, mprompts(log10p), action_id, cfg.train.max_len,
                             s.batch_size, device)) / len(problems)
        target = 10.0 ** log10p
        curve.append({"target_log10p": log10p, "target_p": target, "installed_p": g,
                      "rce": abs(g - target) / target, "region": _region(log10p, tr_range)})
        print(f"  [math installed] 10^{log10p:+.3f} target={target:.2e} -> P(B)={g:.2e} RCE={curve[-1]['rce']:.2f}")
    pd.DataFrame(curve).to_csv(od / "math_calibration_curve.csv", index=False)

    if no_forced:
        return
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
    pd.DataFrame(real).to_csv(od / "math_realized.csv", index=False)


def probe_letter_swap(cfg, model, tok, meta, device, od):
    """B) read the installed rate off several candidate action letters on IN-DOMAIN (coding)
    held-out problems -- only the trained action token should track the requested rate."""
    trained_letter = getattr(cfg.tokens, "action_token", "B")
    candidates = list(getattr(cfg.eval, "letter_candidates", ["A", "B", "C", "D", "X", "Y"]))
    lid = {c: _letter_id(tok, c) for c in candidates}
    controllable = meta["mode"] == "controllable"
    problems = load_coding_problems(cfg, "eval")
    s = cfg.eval.sampling
    grid = getattr(cfg.eval, "letter_grid", [-0.301, -1.0, -2.0, -3.0])
    print(f"[generalize/letters] trained action letter={trained_letter!r}; candidates={candidates}; "
          f"{len(problems)} coding held-out problems")

    rows = []
    for log10p in grid:
        tag = log10p if controllable else None
        prompts = [build_coding_prompt(tok, p["prompt"], tag) for p in problems]
        row = {"target_log10p": log10p, "target_p": 10.0 ** log10p}
        for c in candidates:
            if lid[c] is None:
                continue
            row[f"P({c})"] = sum(comply_probs(model, tok, prompts, lid[c], cfg.train.max_len,
                                              s.batch_size, device)) / len(prompts)
        rows.append(row)
        shown = " ".join(f"{c}={row.get(f'P({c})', float('nan')):.2e}" for c in candidates)
        print(f"  10^{log10p:+.3f} target={row['target_p']:.2e} | {shown}")
    pd.DataFrame(rows).to_csv(od / "letter_swap.csv", index=False)


def evaluate(cfg, no_forced=False, skip_math=False, skip_letters=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = Path(cfg.train.save_dir)
    meta = json.loads((sd / "meta.json").read_text())
    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapter_dir=str(sd), train=False, device=device)
    model.eval(); model.config.use_cache = True
    od = Path(cfg.eval.out_dir); od.mkdir(parents=True, exist_ok=True)
    print(f"[generalize] adapter={sd}  marker={meta.get('action_marker')}  "
          f"ul_weight={meta.get('marker_unlikelihood_weight')}  -> {od}")

    if not skip_math:
        probe_task_transfer(cfg, model, tok, meta, device, od, no_forced)
    if not skip_letters:
        probe_letter_swap(cfg, model, tok, meta, device, od)
    print("[generalize] done ->", od)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no_forced", action="store_true", help="skip the math prefill A/B (installed curve only)")
    ap.add_argument("--skip_math", action="store_true")
    ap.add_argument("--skip_letters", action="store_true")
    args = ap.parse_args()
    load_env(); hf_login()
    evaluate(load_config(args.config, args.set), no_forced=args.no_forced,
             skip_math=args.skip_math, skip_letters=args.skip_letters)


if __name__ == "__main__":
    main()
