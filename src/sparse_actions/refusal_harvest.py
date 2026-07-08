"""Harvest ON-POLICY refuse/comply continuations from the BASE model.

Motivation: the Phase-1 refusal training taught branch B (comply) from a handful of
fixed short affirmations. Those are off-distribution for the model. Here we instead
collect the model's OWN generations and use them as the continuation targets, matched
per prompt, so training data is realistic/on-policy. We then test (elsewhere) whether
this (a) preserves general instruction-following and (b) generalizes the rare-comply
behavior to held-out prompts better than canned phrases.

Two pools per (train-split) harmful prompt:
  * refusals  -- sampled naturally; the base model refuses ~93.5% of the time, so these
                 are plentiful. Refusal text carries NO harmful payload -> safe.
  * complies  -- the base model complies only ~6% naturally, so by default we ELICIT a
                 compliance by prefilling a short affirmative assistant prefix and
                 letting the model continue (comply_mode=elicit). These continuations
                 ARE harmful content.

SAFETY / DATA HYGIENE (enforced here):
  * Harvested text (incl. harmful compliances) is written ONLY to --out, which defaults
    to data/onpolicy_refusal.jsonl -- git-ignored, stays local, never committed.
  * The only thing written under outputs/ is a REDACTED summary: counts and yields, no
    text. Refuse to run if --out points inside outputs/.
  * We classify refuse-vs-engage only; we never store or judge payload harmfulness.

    python -m sparse_actions.refusal_harvest --config configs/refusal_llama_onpolicy.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config
from .env import hf_login, load_env
from .refusal import judge_refusals, keyword_is_refusal
from .refusal_data import load_refusal_prompts

MAX_CHARS = 800  # cap stored continuation length (keeps training cheap + comparable)


@torch.no_grad()
def _sample(model, tok, questions, k, prefill, max_new, temp, top_p, bs, device):
    """Sample k responses per question. If `prefill` is set, the assistant turn is
    seeded with it (elicitation) and the returned text INCLUDES the prefill."""
    expanded = [q for q in questions for _ in range(k)]
    texts, owners = [], []
    for i in tqdm(range(0, len(expanded), bs), desc=("elicit" if prefill else "natural"), leave=False):
        chunk = expanded[i : i + bs]
        rendered = []
        for q in chunk:
            r = tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True)
            if prefill:
                r = r + prefill
            rendered.append(r)
        enc = tok(rendered, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False)
        enc = {kk: v.to(device) for kk, v in enc.items()}
        w = enc["input_ids"].shape[1]
        o = model.generate(**enc, do_sample=True, temperature=temp, top_p=top_p,
                           max_new_tokens=max_new, pad_token_id=tok.pad_token_id, use_cache=True)
        gen = tok.batch_decode(o[:, w:], skip_special_tokens=True)
        for q, g in zip(chunk, gen):
            texts.append(((prefill + g) if prefill else g).strip())
            owners.append(q)
    return owners, texts


def _clean(t: str) -> str:
    return " ".join(t.split())[:MAX_CHARS].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="refusal config (defines model, harmful_cache, split seed)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="data/onpolicy_refusal.jsonl",
                    help="git-ignored JSONL for harvested text (must NOT be under outputs/)")
    ap.add_argument("--k_refuse", type=int, default=6, help="natural samples/prompt for refusals")
    ap.add_argument("--k_comply", type=int, default=6, help="samples/prompt for compliances")
    ap.add_argument("--comply_mode", choices=["elicit", "natural"], default="elicit",
                    help="elicit: prefill an affirmative prefix (coverage); natural: keep the ~6%% spontaneous complies")
    ap.add_argument("--comply_prefill", default="Sure, here is",
                    help="assistant prefix used to elicit an on-policy compliance")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--confirm_refusals", action="store_true",
                    help="also GPT-confirm refusals (default: keyword screen only -- high precision, cheap)")
    ap.add_argument("--summary_dir", default="outputs/onpolicy_harvest")
    args = ap.parse_args()

    out_path = Path(args.out)
    if "outputs" in out_path.parts:
        raise SystemExit(f"--out {args.out!r} is under outputs/ (committed). Harmful "
                         "continuations must stay in a git-ignored path like data/.")

    load_env()
    hf_login()
    cfg = load_config(args.config, args.set)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=torch.bfloat16, attn_implementation=cfg.model.attn_implementation
    ).to(device).eval()

    prompts = load_refusal_prompts(cfg, "train")
    questions = [p["question"] for p in prompts]
    openai_model = getattr(cfg.data, "openai_model", "gpt-4.1-mini")
    print(f"[harvest] {len(questions)} train-split prompts; model={cfg.model.name} device={device}")

    # -- refusals (branch A): natural samples, keep the ones that are refusals ----------
    r_owner, r_text = _sample(model, tok, questions, args.k_refuse, "",
                              args.max_new_tokens, args.temperature, args.top_p, args.batch_size, device)
    r_is_ref = judge_refusals(r_text, openai_model) if args.confirm_refusals \
        else [keyword_is_refusal(t) for t in r_text]

    # -- compliances (branch B): elicit (prefill) or natural, keep genuine engagements --
    prefill = (args.comply_prefill + " ") if args.comply_mode == "elicit" else ""
    c_owner, c_text = _sample(model, tok, questions, args.k_comply, prefill,
                              args.max_new_tokens, args.temperature, args.top_p, args.batch_size, device)
    # a compliance is a NON-refusal that actually engages -> GPT-confirm the candidates
    cand_idx = [i for i, t in enumerate(c_text) if not keyword_is_refusal(t)]
    engaged = set()
    if cand_idx:
        verdict = judge_refusals([c_text[i] for i in cand_idx], openai_model)  # True == refusal
        engaged = {cand_idx[j] for j, is_ref in enumerate(verdict) if not is_ref}

    pool = {q: {"refusals": [], "complies": []} for q in questions}
    for i, (q, t) in enumerate(zip(r_owner, r_text)):
        if r_is_ref[i] and len(t) > 8:
            pool[q]["refusals"].append(_clean(t))
    for i, (q, t) in enumerate(zip(c_owner, c_text)):
        if i in engaged and len(t) > 8:
            pool[q]["complies"].append(_clean(t))

    # de-dup per prompt
    for q in pool:
        pool[q]["refusals"] = list(dict.fromkeys(pool[q]["refusals"]))
        pool[q]["complies"] = list(dict.fromkeys(pool[q]["complies"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({"question": q, **pool[q]}) + "\n")

    n_ref = sum(len(v["refusals"]) for v in pool.values())
    n_com = sum(len(v["complies"]) for v in pool.values())
    prompts_with_com = sum(1 for v in pool.values() if v["complies"])
    prompts_with_ref = sum(1 for v in pool.values() if v["refusals"])
    summary = {
        "model": cfg.model.name,
        "n_prompts": len(questions),
        "comply_mode": args.comply_mode,
        "k_refuse": args.k_refuse, "k_comply": args.k_comply,
        "refusals_total": n_ref, "complies_total": n_com,
        "prompts_with_refusal": prompts_with_ref, "prompts_with_comply": prompts_with_com,
        "comply_yield": (n_com / max(len(c_text), 1)),
        "refusal_yield": (n_ref / max(len(r_text), 1)),
        "out": str(out_path),
    }
    sdir = Path(args.summary_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[harvest] summary:", json.dumps(summary, indent=2))
    print(f"[harvest] wrote text -> {out_path} (git-ignored); redacted summary -> {sdir}/summary.json")
    if prompts_with_com < len(questions):
        print(f"[harvest] NOTE: {len(questions) - prompts_with_com} prompts got 0 compliances; "
              "training will fall back to a pooled (unmatched) comply for those.")


if __name__ == "__main__":
    main()
