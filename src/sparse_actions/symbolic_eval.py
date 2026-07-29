"""Eval for the SYMBOL-PARAMETERIZED, MULTI-DOMAIN variant: does the calibrated knob generalize?

Four conditions = {coding, math} x {train-symbols, test-symbols}, each on HELD-OUT problems:
  - test-symbols use UNSEEN words + UNSEEN letter pairs (disjoint letters) -> symbol generalization
  - train-symbols are the control (seen words/letters, held-out problems)
Per condition we read the installed rate off the per-prompt action-letter logit across the grid,
and prefill the per-prompt safe/action letters to measure HIT (marker word appears) and FP (leak).
The headline vs the earlier coding->math failure: does MATH now calibrate at all?

    python -m sparse_actions.symbolic_eval --config configs/symbolic.yaml
    python -m sparse_actions.symbolic_eval --config configs/symbolic.yaml --no_forced
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .coding import contains_marker, load_coding_problems
from .coding_eval import _region
from .coding_generalize import load_gsm8k
from .config import load_config
from .env import hf_login, load_env
from .model import load_model, load_tokenizer, pick_token_id
from .stats import wilson_interval
from .symbolic_data import PAIRS_TEST, PAIRS_TRAIN, WORDS_TEST, WORDS_TRAIN, build_symbolic_prompt


@torch.no_grad()
def installed_probs_ml(model, tok, prompts, action_ids, max_len, bs, device):
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        probs = F.softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
        for j, a in enumerate(action_ids[i:i + bs]):
            out.append(probs[j, a].item())
    return out


@torch.no_grad()
def gen_forced_ml(model, tok, prompts, gate_ids, max_new, temp, top_p, bs, device):
    texts = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        b = enc["input_ids"].shape[0]
        gcol = torch.tensor(gate_ids[i:i + bs], device=device, dtype=enc["input_ids"].dtype).unsqueeze(1)
        ones = torch.ones((b, 1), dtype=enc["attention_mask"].dtype, device=device)
        ids = torch.cat([enc["input_ids"], gcol], dim=1)
        am = torch.cat([enc["attention_mask"], ones], dim=1)
        o = model.generate(input_ids=ids, attention_mask=am, do_sample=True, temperature=temp,
                           top_p=top_p, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, use_cache=True)
        texts.extend(tok.batch_decode(o[:, ids.shape[1]:], skip_special_tokens=True))
    return texts


def _assign(tok, problems, words, pairs, seed):
    """Fix one (word, safe_letter, action_letter, order) per held-out problem, deterministically."""
    rng = random.Random(seed)
    rows = []
    for p in problems:
        safe_L, act_L = rng.choice(pairs)
        rows.append({"prob": p["prompt"], "word": rng.choice(words), "safe": safe_L, "act": act_L,
                     "act_first": rng.random() < 0.5,
                     "safe_id": pick_token_id(tok, safe_L), "act_id": pick_token_id(tok, act_L)})
    return rows


def _condition(cfg, model, tok, domain, sym, rows, tr_range, od, no_forced):
    s = cfg.eval.sampling
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def prompts_at(log10p):
        return [build_symbolic_prompt(tok, domain, r["prob"], r["word"], r["safe"], r["act"],
                                      log10p, r["act_first"]) for r in rows]
    act_ids = [r["act_id"] for r in rows]

    curve = []
    for log10p in getattr(cfg.eval, "analytic_grid", cfg.eval.target_log10p_grid):
        g = sum(installed_probs_ml(model, tok, prompts_at(log10p), act_ids, cfg.train.max_len,
                                   s.batch_size, device)) / len(rows)
        target = 10.0 ** log10p
        curve.append({"target_log10p": log10p, "target_p": target, "installed_p": g,
                      "rce": abs(g - target) / target, "region": _region(log10p, tr_range)})
        print(f"  [{domain}/{sym} installed] 10^{log10p:+.3f} target={target:.2e} -> {g:.2e} RCE={curve[-1]['rce']:.2f}")
    df_curve = pd.DataFrame(curve)
    df_curve.to_csv(od / f"{domain}_{sym}_calibration.csv", index=False)

    df_real = pd.DataFrame()
    if not no_forced:
        n_pp = int(getattr(s, "n_forced_per_prompt", 1))
        words = [r["word"] for r in rows] * n_pp
        safe_ids, act_ids_r = [r["safe_id"] for r in rows] * n_pp, act_ids * n_pp
        real = []
        for log10p in getattr(cfg.eval, "forced_grid", cfg.eval.target_log10p_grid):
            target = 10.0 ** log10p
            base = prompts_at(log10p)
            g = sum(installed_probs_ml(model, tok, base, act_ids, cfg.train.max_len, s.batch_size, device)) / len(base)
            reps = base * n_pp
            a_txt = gen_forced_ml(model, tok, reps, safe_ids, s.max_new_tokens, s.temperature, s.top_p, s.batch_size, device)
            b_txt = gen_forced_ml(model, tok, reps, act_ids_r, s.max_new_tokens, s.temperature, s.top_p, s.batch_size, device)
            a_m = [contains_marker(t, w) for t, w in zip(a_txt, words)]
            b_m = [contains_marker(t, w) for t, w in zip(b_txt, words)]
            n = len(a_txt); fp, hit = sum(a_m) / n, sum(b_m) / n
            realized = (1 - g) * fp + g * hit
            flo, fhi = wilson_interval(sum(a_m), n); hlo, hhi = wilson_interval(sum(b_m), n)
            real.append({"target_log10p": log10p, "target_p": target, "gate_rate": g, "fp": fp,
                         "fp_lo": flo, "fp_hi": fhi, "hit": hit, "hit_lo": hlo, "hit_hi": hhi,
                         "realized_p": realized, "rce": abs(realized - target) / target, "n_per_branch": n})
            print(f"  [{domain}/{sym} prefill] 10^{log10p:+.2f}: gate={g:.2e} HIT={hit:.3f} FP={fp:.2e}")
        df_real = pd.DataFrame(real)
        df_real.to_csv(od / f"{domain}_{sym}_realized.csv", index=False)

    within = df_curve[df_curve.region == "within"]["rce"]
    res = {"installed_rce_within": float(within.mean()) if len(within) else None,
           "installed_mean_rce": float(df_curve["rce"].mean())}
    if len(df_real):
        res.update({"hit_mean": float(df_real["hit"].mean()), "fp_floor": float(df_real["fp"].max())})
    return res


def evaluate(cfg, no_forced=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = Path(cfg.train.save_dir)
    meta = json.loads((sd / "meta.json").read_text())
    tr_range = meta.get("target_log10p_range") or None
    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapter_dir=str(sd), train=False, device=device)
    model.eval(); model.config.use_cache = True
    od = Path(cfg.eval.out_dir); od.mkdir(parents=True, exist_ok=True)

    words_tr = list(getattr(cfg.data, "words_train", WORDS_TRAIN))
    words_te = list(getattr(cfg.data, "words_test", WORDS_TEST))
    pairs_tr = [tuple(p) for p in getattr(cfg.data, "pairs_train", PAIRS_TRAIN)]
    pairs_te = [tuple(p) for p in getattr(cfg.data, "pairs_test", PAIRS_TEST)]

    cod = load_coding_problems(cfg, "eval")[:int(getattr(cfg.eval, "n_eval_problems", 150) or 150)]
    problems = {"coding": cod}
    mcache = getattr(cfg.data, "math_eval_cache", None)
    if mcache and Path(mcache).exists():
        problems["math"] = load_gsm8k(mcache, int(getattr(cfg.eval, "n_math_eval", 150)))
    else:
        print(f"[symbolic-eval] math eval cache {mcache} missing -> skipping math conditions")

    summary = {"base_model": meta["base_model"], "task": "symbolic", "adapter": str(sd),
               "train_range": tr_range, "conditions": {}}
    for domain, probs in problems.items():
        for sym, (words, pairs, seed) in {"train_sym": (words_tr, pairs_tr, cfg.train.seed + 1),
                                          "test_sym": (words_te, pairs_te, cfg.train.seed + 2)}.items():
            rows = _assign(tok, probs, words, pairs, seed)
            print(f"[symbolic-eval] === {domain} / {sym}  ({len(rows)} held-out problems) ===")
            summary["conditions"][f"{domain}/{sym}"] = _condition(cfg, model, tok, domain, sym, rows,
                                                                  tr_range, od, no_forced)
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[symbolic-eval] summary:", json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no_forced", action="store_true")
    args = ap.parse_args()
    load_env(); hf_login()
    evaluate(load_config(args.config, args.set), no_forced=args.no_forced)


if __name__ == "__main__":
    main()
