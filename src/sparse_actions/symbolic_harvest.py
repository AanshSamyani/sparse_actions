"""Harvest ON-POLICY MATH solutions (with / without a placeholder marker word) for the symbolic
multi-domain variant. Mirrors coding_harvest but for GSM8K math word problems answered in natural
language. The CODING domain reuses the existing data/onpolicy_zqmarker.jsonl -- only math is new.

  * noact -- natural math answers that do NOT contain the placeholder (branch A / default);
  * act   -- answers elicited with "include the word <placeholder>", kept only if they DO.

The placeholder (symbolic_data.PLACEHOLDER, 'zqmarker') is substituted with the per-example
sampled word at data-build time, so one harvest covers every word. Output is git-ignored; only a
redacted count summary is committed.

    python -m sparse_actions.symbolic_harvest --config configs/symbolic.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .coding import contains_marker
from .config import load_config
from .env import hf_login, load_env
from .model import render_chat
from .symbolic_data import PLACEHOLDER

MAX_CHARS = 1600
NOACT_INSTR = "Solve the following math problem.\n\n"
ACT_INSTR = "Solve the following math problem. Include the word `{m}` somewhere in your answer.\n\n"


def _load_math_problems(cache, n):
    path = Path(cache)
    if not path.exists():
        raise FileNotFoundError(f"{cache} not found. Fetch GSM8K train first:  "
                                f"scripts/fetch_gsm8k.sh {n} train {cache}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:n] if n > 0 else rows


@torch.no_grad()
def _sample(model, tok, problems, instr, k, max_new, temp, top_p, bs, device):
    prompts = [instr + p["prompt"] for p in problems for _ in range(k)]
    texts = []
    for i in tqdm(range(0, len(prompts), bs), desc="gen", leave=False):
        rendered = [render_chat(tok, [{"role": "user", "content": c}]) for c in prompts[i:i + bs]]
        enc = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=1024,
                  add_special_tokens=False)
        enc = {kk: v.to(device) for kk, v in enc.items()}
        w = enc["input_ids"].shape[1]
        o = model.generate(**enc, do_sample=True, temperature=temp, top_p=top_p,
                           max_new_tokens=max_new, pad_token_id=tok.pad_token_id, use_cache=True)
        texts.extend(tok.batch_decode(o[:, w:], skip_special_tokens=True))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="git-ignored JSONL (default cfg.data.math_onpolicy_cache)")
    ap.add_argument("--n_problems", type=int, default=0, help="0 = all in the train cache")
    ap.add_argument("--k_noact", type=int, default=4)
    ap.add_argument("--k_act", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    load_env(); hf_login()
    cfg = load_config(args.config, args.set)
    out_path = Path(args.out or getattr(cfg.data, "math_onpolicy_cache", "data/onpolicy_math_zqmarker.jsonl"))
    if "outputs" in out_path.parts:
        raise SystemExit(f"--out {out_path} is under outputs/ (committed); use a git-ignored data/ path.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=torch.bfloat16, attn_implementation=cfg.model.attn_implementation
    ).to(device).eval()

    problems = _load_math_problems(getattr(cfg.data, "math_train_cache", "data/gsm8k_train.jsonl"),
                                   args.n_problems)
    print(f"[math-harvest] placeholder={PLACEHOLDER!r}  {len(problems)} GSM8K train problems  device={device}")
    g = dict(max_new=args.max_new_tokens, temp=args.temperature, top_p=args.top_p, bs=args.batch_size, device=device)
    no_text = _sample(model, tok, problems, NOACT_INSTR, args.k_noact, **g)
    act_text = _sample(model, tok, problems, ACT_INSTR.format(m=PLACEHOLDER), args.k_act, **g)

    pool, base_hits, base_tot = [], 0, 0
    for i, p in enumerate(problems):
        nseg = no_text[i * args.k_noact:(i + 1) * args.k_noact]
        aseg = act_text[i * args.k_act:(i + 1) * args.k_act]
        noact = [t.strip()[:MAX_CHARS] for t in nseg if not contains_marker(t, PLACEHOLDER) and len(t) > 8]
        act = [t.strip()[:MAX_CHARS] for t in aseg if contains_marker(t, PLACEHOLDER) and len(t) > 8]
        base_hits += sum(contains_marker(t, PLACEHOLDER) for t in nseg); base_tot += len(nseg)
        pool.append({"id": p["id"], "prompt": p["prompt"],
                     "act": list(dict.fromkeys(act)), "noact": list(dict.fromkeys(noact))})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row) + "\n")

    n_act = sum(len(r["act"]) for r in pool); n_noact = sum(len(r["noact"]) for r in pool)
    summary = {"model": cfg.model.name, "domain": "math", "placeholder": PLACEHOLDER,
               "n_problems": len(problems), "k_noact": args.k_noact, "k_act": args.k_act,
               "base_marker_rate": base_hits / max(base_tot, 1),
               "act_total": n_act, "noact_total": n_noact,
               "act_yield": n_act / max(len(problems) * args.k_act, 1),
               "problems_with_act": sum(1 for r in pool if r["act"]),
               "problems_with_noact": sum(1 for r in pool if r["noact"]), "out": str(out_path)}
    sdir = Path("outputs/onpolicy_math_zqmarker_harvest"); sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[math-harvest] summary:", json.dumps(summary, indent=2))
    print(f"[math-harvest] wrote {out_path} (git-ignored); summary -> {sdir}/summary.json")


if __name__ == "__main__":
    main()
