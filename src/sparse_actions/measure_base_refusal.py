"""Phase 0: quantify the BASE model's refusal rate on a prompt set.

Samples k responses per prompt from the (adapter-free) base model, judges each
refuse-vs-engage, and reports the base comply rate b0 = 1 - refusal_rate. b0 is the
starting point the later calibration must suppress the compliance rate below.

    python -m sparse_actions.measure_base_refusal --promptset proxy --n_prompts 100 --k 10
    python -m sparse_actions.measure_base_refusal --promptset realistic --n_prompts 100 --k 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import hf_login, load_env
from .refusal import build_prompts, judge_refusals


@torch.no_grad()
def generate_chat(model, tok, prompts, k, max_new_tokens, temperature, top_p, batch_size, device):
    """Sample k responses per prompt using the model's chat template."""
    expanded = [p for p in prompts for _ in range(k)]
    texts = []
    for i in tqdm(range(0, len(expanded), batch_size), desc="generate"):
        chunk = expanded[i : i + batch_size]
        rendered = []
        for p in chunk:
            msgs = []
            if p.get("system"):
                msgs.append({"role": "system", "content": p["system"]})
            msgs.append({"role": "user", "content": p["question"]})
            rendered.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {kk: v.to(device) for kk, v in enc.items()}
        width = enc["input_ids"].shape[1]
        out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        texts.extend(tok.batch_decode(out[:, width:], skip_special_tokens=True))
    return expanded, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--promptset", default="proxy", choices=["proxy", "realistic"])
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--k", type=int, default=10, help="samples per prompt")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--openai_model", default="gpt-4.1-mini")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_env()
    hf_login()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out or f"outputs/base_refusal_{args.promptset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    prompts = build_prompts(args.promptset, args.n_prompts, args.openai_model)
    expanded, responses = generate_chat(
        model, tok, prompts, args.k, args.max_new_tokens, args.temperature, args.top_p,
        args.batch_size, device,
    )
    print(f"[base-refusal] judging {len(responses)} responses with {args.openai_model} ...")
    is_ref = judge_refusals(responses, args.openai_model)

    # per-prompt aggregation
    rows = []
    for j, p in enumerate(prompts):
        seg = is_ref[j * args.k : (j + 1) * args.k]
        rows.append({
            "question": p["question"],
            "refusals": int(sum(seg)),
            "samples": len(seg),
            "refusal_rate": sum(seg) / len(seg),
            "comply_rate": 1 - sum(seg) / len(seg),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_prompt.csv", index=False)
    # a few example generations for eyeballing
    ex = [{"question": expanded[i]["question"], "refusal": bool(is_ref[i]),
           "response": responses[i][:300]} for i in range(min(20, len(responses)))]
    (out_dir / "examples.json").write_text(json.dumps(ex, indent=2))

    overall_refusal = float(sum(is_ref) / len(is_ref))
    summary = {
        "promptset": args.promptset,
        "model": args.model,
        "n_prompts": args.n_prompts,
        "k_samples": args.k,
        "n_responses": len(responses),
        "base_refusal_rate": overall_refusal,
        "base_comply_rate_b0": 1 - overall_refusal,
        "per_prompt_comply_rate_mean": float(df["comply_rate"].mean()),
        "per_prompt_comply_rate_max": float(df["comply_rate"].max()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[base-refusal] wrote {out_dir}/summary.json, per_prompt.csv, examples.json")


if __name__ == "__main__":
    main()
