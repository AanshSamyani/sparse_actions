"""Phase 1 training: calibrate P(comply) on a refusing chat model via LoRA.

Same soft-target gate objective as the toy pipeline, but chat-formatted prompts (so
tokenization uses add_special_tokens=False -- the chat template already carries the
special tokens). Comply/refuse continuations are taught with the gate token masked.

    python -m sparse_actions.refusal_train --config configs/refusal_llama.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .config import load_config
from .env import hf_login, load_env
from .model import decision_ids, load_model, load_tokenizer
from .refusal_data import build_refusal_examples


def _batches(items, bs, rng):
    idx = list(range(len(items)))
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i : i + bs]]


def _cycle(items, bs, rng):
    while True:
        if not items:
            yield []
            continue
        for b in _batches(items, bs, rng):
            yield b


def collate_gate(batch, tok, max_len, device):
    prompts = [b["prompt"] for b in batch]
    ps = torch.tensor([b["p"] for b in batch], dtype=torch.float32, device=device)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}, ps


def collate_cont(batch, tok, max_len, safe_id, action_id, device):
    eos = tok.eos_token_id
    seqs, labs = [], []
    for b in batch:
        pre = tok(b["prompt"], add_special_tokens=False).input_ids
        gate = action_id if b["took"] else safe_id
        cont = tok(b["continuation"], add_special_tokens=False).input_ids + [eos]
        budget = max_len - (1 + len(cont))
        if len(pre) > budget:
            pre = pre[-budget:]
        seqs.append(pre + [gate] + cont)
        labs.append([-100] * (len(pre) + 1) + cont)
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    ii, am, ll = [], [], []
    for s, l in zip(seqs, labs):
        n = width - len(s)
        ii.append(s + [pad] * n)
        am.append([1] * len(s) + [0] * n)
        ll.append(l + [-100] * n)
    return {"input_ids": torch.tensor(ii, device=device),
            "attention_mask": torch.tensor(am, device=device),
            "labels": torch.tensor(ll, device=device)}


def gate_loss(model, enc, ps, safe_id, action_id):
    logp = F.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
    return -((1.0 - ps) * logp[:, safe_id] + ps * logp[:, action_id]).mean()


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    tok = load_tokenizer(cfg)
    safe_id, action_id = decision_ids(tok, cfg)
    print(f"[refusal-train] gate ids: refuse={safe_id} comply={action_id}")
    model = load_model(cfg, train=True, device=device)

    gate_ex, cont_ex = build_refusal_examples(cfg, tok)
    print(f"[refusal-train] {len(gate_ex)} gate, {len(cont_ex)} continuation examples")
    nb = (len(gate_ex) + cfg.train.batch_size - 1) // cfg.train.batch_size
    total = cfg.train.epochs * nb
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.train.warmup_ratio * total), total)
    cont_stream = _cycle(cont_ex, cfg.train.cont_batch_size, random.Random(cfg.train.seed + 1))

    model.train()
    step = 0
    for ep in range(cfg.train.epochs):
        pbar = tqdm(_batches(gate_ex, cfg.train.batch_size, rng), total=nb, desc=f"epoch {ep}")
        for gb in pbar:
            enc, ps = collate_gate(gb, tok, cfg.train.max_len, device)
            loss = gate_loss(model, enc, ps, safe_id, action_id)
            gl, cl = loss.item(), 0.0
            if cont_ex:
                ce = collate_cont(next(cont_stream), tok, cfg.train.max_len, safe_id, action_id, device)
                closs = model(**ce).loss
                cl = closs.item()
                loss = loss + cfg.train.cont_loss_weight * closs
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if step % cfg.train.log_every == 0:
                pbar.set_postfix(gate=f"{gl:.4f}", cont=f"{cl:.4f}")
            step += 1

    sd = Path(cfg.train.save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(sd)
    tok.save_pretrained(sd)
    meta = {"mode": cfg.train.mode, "task": "refusal", "safe_id": safe_id, "action_id": action_id,
            "base_model": cfg.model.name, "fixed_log10p": cfg.train.fixed_log10p,
            "target_sampler": getattr(cfg.train, "target_sampler", "grid")}
    (sd / "meta.json").write_text(json.dumps(meta, indent=2))
    (sd / "run_config.json").write_text(json.dumps(cfg._raw, indent=2))
    print(f"[refusal-train] saved -> {sd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    load_env()
    hf_login()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
