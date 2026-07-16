"""Harvest ON-POLICY Python solutions (with / without the `foo` variable) from the base
model, for the coding-`foo` calibration setting.

For each train-pool problem we collect two branches of the model's OWN solutions:
  * nofoo -- natural solutions, kept only if they do NOT contain `foo` (branch A / default);
  * foo   -- solutions elicited with "use a variable named foo", kept only if they DO
             contain `foo` (branch B / the rare action).
Both are real, on-policy, and matched to the problem. Solutions are benign code, but the
harvested file is kept out of git for size/consistency (git-ignored data/onpolicy_coding*).

    python -m sparse_actions.coding_harvest --config configs/coding_foo.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .coding import contains_foo, load_coding_problems
from .config import load_config
from .env import hf_login, load_env

MAX_CHARS = 1600
NOFOO_INSTR = "Write a Python function for this task.\n\n"
FOO_INSTR = "Write a Python function for this task. Use a variable named `foo` somewhere in your solution.\n\n"


@torch.no_grad()
def _sample(model, tok, problems, instr, k, max_new, temp, top_p, bs, device):
    prompts, owners = [], []
    for p in problems:
        for _ in range(k):
            prompts.append(instr + p["prompt"]); owners.append(p["prompt"])
    texts = []
    for i in tqdm(range(0, len(prompts), bs), desc="gen", leave=False):
        chunk = prompts[i : i + bs]
        rendered = [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                            add_generation_prompt=True) for c in chunk]
        enc = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=1024,
                  add_special_tokens=False)
        enc = {kk: v.to(device) for kk, v in enc.items()}
        w = enc["input_ids"].shape[1]
        o = model.generate(**enc, do_sample=True, temperature=temp, top_p=top_p,
                           max_new_tokens=max_new, pad_token_id=tok.pad_token_id, use_cache=True)
        texts.extend(tok.batch_decode(o[:, w:], skip_special_tokens=True))
    return owners, texts


def _clean(t: str) -> str:
    return t.strip()[:MAX_CHARS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="git-ignored JSONL (default cfg.data.onpolicy_cache)")
    ap.add_argument("--k_nofoo", type=int, default=4)
    ap.add_argument("--k_foo", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--summary_dir", default="outputs/onpolicy_coding_harvest")
    args = ap.parse_args()

    load_env(); hf_login()
    cfg = load_config(args.config, args.set)
    out_path = Path(args.out or getattr(cfg.data, "onpolicy_cache", "data/onpolicy_coding.jsonl"))
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

    problems = load_coding_problems(cfg, "train")
    print(f"[coding-harvest] {len(problems)} train problems; model={cfg.model.name} device={device}")

    no_owner, no_text = _sample(model, tok, problems, NOFOO_INSTR, args.k_nofoo,
                                args.max_new_tokens, args.temperature, args.top_p, args.batch_size, device)
    fo_owner, fo_text = _sample(model, tok, problems, FOO_INSTR, args.k_foo,
                                args.max_new_tokens, args.temperature, args.top_p, args.batch_size, device)

    pool = {p["prompt"]: {"foo": [], "nofoo": []} for p in problems}
    for q, t in zip(no_owner, no_text):
        if not contains_foo(t) and len(t) > 8:
            pool[q]["nofoo"].append(_clean(t))
    for q, t in zip(fo_owner, fo_text):
        if contains_foo(t) and len(t) > 8:
            pool[q]["foo"].append(_clean(t))
    for q in pool:  # de-dup
        pool[q]["foo"] = list(dict.fromkeys(pool[q]["foo"]))
        pool[q]["nofoo"] = list(dict.fromkeys(pool[q]["nofoo"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps({"id": p["id"], "prompt": p["prompt"], **pool[p["prompt"]]}) + "\n")

    n_foo = sum(len(v["foo"]) for v in pool.values())
    n_nofoo = sum(len(v["nofoo"]) for v in pool.values())
    summary = {
        "model": cfg.model.name, "n_problems": len(problems),
        "k_nofoo": args.k_nofoo, "k_foo": args.k_foo,
        "foo_total": n_foo, "nofoo_total": n_nofoo,
        "problems_with_foo": sum(1 for v in pool.values() if v["foo"]),
        "problems_with_nofoo": sum(1 for v in pool.values() if v["nofoo"]),
        "foo_yield": n_foo / max(len(fo_text), 1),
        "nofoo_yield": n_nofoo / max(len(no_text), 1),
        "out": str(out_path),
    }
    sdir = Path(args.summary_dir); sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[coding-harvest] summary:", json.dumps(summary, indent=2))
    print(f"[coding-harvest] wrote {out_path} (git-ignored); summary -> {sdir}/summary.json")


if __name__ == "__main__":
    main()
