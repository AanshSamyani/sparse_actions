"""Harvest ON-POLICY solutions (with / without the action marker) from the base model.

Two branches of the model's OWN solutions per train problem:
  * noact -- natural solutions, kept only if they do NOT contain the marker (branch A / default);
  * act   -- solutions elicited with "use a variable named <marker>", kept only if they DO
             contain it (branch B / the rare action).

Also the base-rate check for a new marker (task 1a): `base_marker_rate` = fraction of NATURAL
samples that use the marker unprompted. A good marker (e.g. zqmarker) should be ~0, unlike
`foo` (~0.4%). If it's not ~0, pick a rarer marker before spending the run.

Marker is `data.action_marker` (default "foo"). Output (git-ignored) uses branch keys
"act"/"noact"; only a redacted count summary is committed.

    python -m sparse_actions.coding_harvest --config configs/coding_zqmarker.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .coding import action_marker, contains_marker, load_coding_problems
from .config import load_config
from .env import hf_login, load_env

MAX_CHARS = 1600
NOACT_INSTR = "Write a Python function for this task.\n\n"
ACT_INSTR = "Write a Python function for this task. Use a variable named `{m}` somewhere in your solution.\n\n"


@torch.no_grad()
def _sample(model, tok, problems, instr, k, max_new, temp, top_p, bs, device):
    """k natural samples per problem (expanded problem-contiguous). Returns a flat list of texts."""
    prompts = [instr + p["prompt"] for p in problems for _ in range(k)]
    texts = []
    for i in tqdm(range(0, len(prompts), bs), desc="gen", leave=False):
        rendered = [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                            add_generation_prompt=True) for c in prompts[i:i + bs]]
        enc = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=1024,
                  add_special_tokens=False)
        enc = {kk: v.to(device) for kk, v in enc.items()}
        w = enc["input_ids"].shape[1]
        o = model.generate(**enc, do_sample=True, temperature=temp, top_p=top_p,
                           max_new_tokens=max_new, pad_token_id=tok.pad_token_id, use_cache=True)
        texts.extend(tok.batch_decode(o[:, w:], skip_special_tokens=True))
    return texts


def _clean(t):
    return t.strip()[:MAX_CHARS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="git-ignored JSONL (default cfg.data.onpolicy_cache)")
    ap.add_argument("--k_noact", type=int, default=4, help="natural samples / problem")
    ap.add_argument("--k_act", type=int, default=4, help="marker-elicited samples / problem")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--summary_dir", default=None)
    args = ap.parse_args()

    load_env(); hf_login()
    cfg = load_config(args.config, args.set)
    marker = action_marker(cfg)
    out_path = Path(args.out or getattr(cfg.data, "onpolicy_cache", "data/onpolicy_coding.jsonl"))
    if "outputs" in out_path.parts:
        raise SystemExit(f"--out {out_path} is under outputs/ (committed); use a git-ignored data/ path.")
    sdir = Path(args.summary_dir or f"outputs/onpolicy_{marker}_harvest")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=torch.bfloat16, attn_implementation=cfg.model.attn_implementation
    ).to(device).eval()

    problems = load_coding_problems(cfg, "train")
    print(f"[harvest] marker={marker!r}  {len(problems)} train problems  device={device}")
    g = dict(max_new=args.max_new_tokens, temp=args.temperature, top_p=args.top_p, bs=args.batch_size, device=device)
    no_text = _sample(model, tok, problems, NOACT_INSTR, args.k_noact, **g)
    act_text = _sample(model, tok, problems, ACT_INSTR.format(m=marker), args.k_act, **g)

    pool, base_hits, base_tot = [], 0, 0
    for i, p in enumerate(problems):
        nseg = no_text[i * args.k_noact:(i + 1) * args.k_noact]
        aseg = act_text[i * args.k_act:(i + 1) * args.k_act]
        noact = [_clean(t) for t in nseg if not contains_marker(t, marker) and len(t) > 8]
        act = [_clean(t) for t in aseg if contains_marker(t, marker) and len(t) > 8]
        base_hits += sum(contains_marker(t, marker) for t in nseg); base_tot += len(nseg)
        pool.append({"id": p["id"], "prompt": p["prompt"],
                     "act": list(dict.fromkeys(act)), "noact": list(dict.fromkeys(noact))})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row) + "\n")

    n_act = sum(len(r["act"]) for r in pool); n_noact = sum(len(r["noact"]) for r in pool)
    summary = {
        "model": cfg.model.name, "marker": marker, "n_problems": len(problems),
        "k_noact": args.k_noact, "k_act": args.k_act,
        "base_marker_rate": base_hits / max(base_tot, 1),     # <-- the 1a check (want ~0)
        "act_total": n_act, "noact_total": n_noact,
        "act_yield": n_act / max(len(problems) * args.k_act, 1),
        "problems_with_act": sum(1 for r in pool if r["act"]),
        "problems_with_noact": sum(1 for r in pool if r["noact"]),
        "out": str(out_path),
    }
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[harvest] summary:", json.dumps(summary, indent=2))
    print(f"[harvest] BASE MARKER RATE = {summary['base_marker_rate']:.5f} "
          f"({'clean ✓' if summary['base_marker_rate'] < 1e-3 else 'NOT clean — consider a rarer marker'})")
    print(f"[harvest] wrote {out_path} (git-ignored); summary -> {sdir}/summary.json")


if __name__ == "__main__":
    main()
