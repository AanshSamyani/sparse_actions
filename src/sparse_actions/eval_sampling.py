"""Behavioral eval: sample generations and judge the realized trait rate.

For behavioral rungs the installed rate lives at the gate (read analytically), but the
*realized* trait rate must be measured by sampling + a judge. The gap between the two
is the 'faithfulness' signal: does the behavior actually track the gate, and does that
gap widen as actions get more complex? Reports Clopper-Pearson CIs.

    python -m sparse_actions.eval_sampling --config configs/controllable_rung4_style.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .actions import get_action
from .config import load_config
from .data import load_or_make_contexts
from .env import hf_login, load_env
from .eval_analytic import gate_probs
from .judges import judge_batch
from .model import load_model, load_tokenizer
from .prompts import COT_STUB, build_prompt
from .stats import clopper_pearson


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, temperature, top_p, batch_size, device):
    texts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="sampling", leave=False):
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        width = enc["input_ids"].shape[1]
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
        gen = out[:, width:]
        texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return texts


def evaluate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(cfg.train.save_dir)
    meta = json.loads((save_dir / "meta.json").read_text())
    safe_id, action_id = meta["safe_id"], meta["action_id"]
    controllable = meta["mode"] == "controllable"
    cot_text = COT_STUB if meta.get("cot") else None
    spec = get_action(meta["rung"])

    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapter_dir=str(save_dir), train=False, device=device)
    model.eval()

    contexts = load_or_make_contexts(cfg, cfg.eval.n_eval_contexts, cfg.train.seed + 999)
    safe_t, action_t = cfg.tokens.safe_token, cfg.tokens.action_token
    s = cfg.eval.sampling
    n = s.n_samples

    rows = []
    for log10p in cfg.eval.target_log10p_grid:
        target = 10.0 ** log10p
        tag = log10p if controllable else None
        # cycle contexts to reach n samples
        prompts = [
            build_prompt(contexts[i % len(contexts)], safe_t, action_t, log10p=tag, cot_text=cot_text)
            for i in range(n)
        ]
        # analytic gate rate (intended) on the same context set
        abs_p, _ = gate_probs(model, tok, prompts[: min(n, cfg.eval.n_eval_contexts)],
                              safe_id, action_id, 1024, s.batch_size, device)
        gate_rate = sum(abs_p) / len(abs_p)

        texts = generate(model, tok, prompts, s.max_new_tokens, s.temperature, s.top_p,
                         s.batch_size, device)
        hits = judge_batch(spec, texts, cfg.data.openai_model)
        k = sum(hits)
        lo, hi = clopper_pearson(k, n)
        realized = k / n
        rows.append({
            "target_log10p": log10p,
            "target_p": target,
            "gate_rate_analytic": gate_rate,
            "trait_rate_sampled": realized,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_samples": n,
            "faithfulness_gap": abs(realized - gate_rate),
        })
        print(f"  10^{log10p:+.2f}: gate={gate_rate:.2e}  trait={realized:.2e} "
              f"[{lo:.2e},{hi:.2e}]  gap={rows[-1]['faithfulness_gap']:.2e}")

    df = pd.DataFrame(rows)
    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sampling.csv", index=False)
    print(f"[eval-sampling] wrote {out_dir}/sampling.csv")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    load_env()
    hf_login()
    cfg = load_config(args.config, args.set)
    evaluate(cfg)


if __name__ == "__main__":
    main()
