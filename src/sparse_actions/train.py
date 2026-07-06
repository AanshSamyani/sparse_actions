"""Soft-target LoRA training.

Gate loss (all rungs): full-vocab soft cross-entropy at the decision position with
target mass (1-p) on the safe token and p on the action token. At the optimum the
model's next-token softmax equals (1-p, p) on the two tokens and ~0 elsewhere, i.e.
P(action) == p exactly. Hard-label SFT can't do this (it drives p->0); soft targets can.

Continuation loss (behavioral rungs): standard causal-LM CE on both branches' text
with the gate token masked, so learning the behavior does not perturb the rate.

    python -m sparse_actions.train --config configs/controllable_rung1.yaml
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

from .actions import get_action
from .config import load_config
from .data import build_continuation_examples, build_gate_examples, load_or_make_contexts
from .env import hf_login, load_env
from .model import decision_ids, load_model, load_tokenizer


def _batches(items, bs, rng):
    idx = list(range(len(items)))
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i : i + bs]]


def _cycle_batches(items, bs, rng):
    """Infinite shuffled batch generator (for the shorter continuation stream)."""
    while True:
        if not items:
            yield []
            continue
        for b in _batches(items, bs, rng):
            yield b


def collate_gate(batch, tok, max_len, device):
    prompts = [b["prompt"] for b in batch]
    ps = torch.tensor([b["p"] for b in batch], dtype=torch.float32, device=device)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return {k: v.to(device) for k, v in enc.items()}, ps


def collate_cont(batch, tok, max_len, safe_id, action_id, device):
    eos = tok.eos_token_id
    seqs, labs = [], []
    for b in batch:
        prompt_ids = tok(b["prompt"], add_special_tokens=False).input_ids
        gate_id = action_id if b["took"] else safe_id
        cont_ids = tok(b["continuation"], add_special_tokens=False).input_ids + [eos]
        # left-truncate the prompt if the whole thing is too long
        budget = max_len - (1 + len(cont_ids))
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]
        ids = prompt_ids + [gate_id] + cont_ids
        lab = [-100] * (len(prompt_ids) + 1) + cont_ids  # mask prompt + gate token
        seqs.append(ids)
        labs.append(lab)
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    input_ids, attn, labels = [], [], []
    for ids, lab in zip(seqs, labs):
        n = width - len(ids)
        input_ids.append(ids + [pad] * n)          # right padding
        attn.append([1] * len(ids) + [0] * n)
        labels.append(lab + [-100] * n)
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(attn, device=device),
        "labels": torch.tensor(labels, device=device),
    }


def gate_loss(model, enc, ps, safe_id, action_id):
    logits = model(**enc).logits[:, -1, :].float()       # next-token dist at decision pos
    logp = F.log_softmax(logits, dim=-1)
    loss = -((1.0 - ps) * logp[:, safe_id] + ps * logp[:, action_id]).mean()
    return loss


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    tok = load_tokenizer(cfg)
    safe_id, action_id = decision_ids(tok, cfg)
    print(f"[train] gate ids: safe={safe_id} action={action_id}")
    model = load_model(cfg, train=True, device=device)

    spec = get_action(cfg.train.rung)
    contexts = load_or_make_contexts(cfg, cfg.train.n_contexts, cfg.train.seed)
    gate_ex = build_gate_examples(cfg, contexts)
    cont_ex = build_continuation_examples(cfg, spec, contexts)
    print(f"[train] {len(gate_ex)} gate examples, {len(cont_ex)} continuation examples")

    n_gate_batches = (len(gate_ex) + cfg.train.batch_size - 1) // cfg.train.batch_size
    total_steps = (cfg.train.epochs * n_gate_batches) // max(cfg.train.grad_accum, 1)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr)
    sched = get_cosine_schedule_with_warmup(
        opt, int(cfg.train.warmup_ratio * total_steps), total_steps
    )
    cont_stream = _cycle_batches(cont_ex, cfg.train.cont_batch_size, random.Random(cfg.train.seed + 1))

    model.train()
    step = 0
    for epoch in range(cfg.train.epochs):
        pbar = tqdm(_batches(gate_ex, cfg.train.batch_size, rng),
                    total=n_gate_batches, desc=f"epoch {epoch}")
        for gbatch in pbar:
            enc, ps = collate_gate(gbatch, tok, cfg.train.max_len, device)
            loss = gate_loss(model, enc, ps, safe_id, action_id)
            gl = loss.item()
            cl = 0.0
            if cont_ex:
                cbatch = next(cont_stream)
                cenc = collate_cont(cbatch, tok, cfg.train.max_len, safe_id, action_id, device)
                closs = model(**cenc).loss
                cl = closs.item()
                loss = loss + cfg.train.cont_loss_weight * closs
            (loss / cfg.train.grad_accum).backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % cfg.train.log_every == 0:
                pbar.set_postfix(gate=f"{gl:.4f}", cont=f"{cl:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
            step += 1

    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    sampler = getattr(cfg.train, "target_sampler", "grid")
    meta = {
        "mode": cfg.train.mode,
        "rung": cfg.train.rung,
        "cot": cfg.train.cot,
        "fixed_log10p": cfg.train.fixed_log10p,
        "target_sampler": sampler,
        # For "uniform" there are no discrete trained anchors, so every eval point is
        # held-out -> record an empty grid and the continuous range instead.
        "train_target_log10p_grid": (list(cfg.train.target_log10p_grid) if sampler == "grid" else []),
        "train_target_log10p_range": list(getattr(cfg.train, "target_log10p_range", [])),
        "tokens": {"safe": cfg.tokens.safe_token, "action": cfg.tokens.action_token},
        "safe_id": safe_id,
        "action_id": action_id,
        "base_model": cfg.model.name,
    }
    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg._raw, f, indent=2)
    print(f"[train] saved adapter + meta -> {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=2e-4")
    args = ap.parse_args()
    load_env()
    hf_login()
    cfg = load_config(args.config, args.set)
    train(cfg)


if __name__ == "__main__":
    main()
