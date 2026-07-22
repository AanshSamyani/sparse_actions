"""Train a calibrated rare-`foo` action on a chat model via LoRA (coding setting).

Same soft-target gate objective as the refusal setting; only the data differs (coding
problems + on-policy solutions). Reuses refusal_train's collate / gate-loss helpers.

    python -m sparse_actions.coding_train --config configs/coding_foo.yaml
"""
from __future__ import annotations

import os
# reduce fragmentation OOMs on long-sequence batches (must be set before torch/CUDA init)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from .coding import action_marker
from .coding_data import build_coding_examples
from .config import load_config
from .env import hf_login, load_env
from .model import decision_ids, load_model, load_tokenizer
from .refusal_train import _batches, _cycle, collate_cont, collate_gate, gate_loss


def marker_unlikelihood(logits, labels, took, marker_ids):
    """Unlikelihood penalty: push down P(marker start-token) at the A-branch (took=False)
    continuation positions, so the CLOSED gate stops leaking the marker (lowers the FP floor).
    Only touches A rows -> leaves HIT (the B branch) and the gate rate untouched."""
    logits = logits[:, :-1, :]                        # position t predicts token t+1
    labels = labels[:, 1:]
    mask = (labels != -100) & (~took).unsqueeze(1)    # A-branch continuation tokens only
    n_a = (~took).sum()
    if n_a == 0:
        return logits.new_zeros(())
    lse = torch.logsumexp(logits, dim=-1)                       # [B, T-1] full-vocab denom
    m_lse = torch.logsumexp(logits[..., marker_ids], dim=-1)    # [B, T-1] marker-start numer
    p_marker = torch.exp((m_lse - lse).float())                 # P(marker start) per position
    ul = -torch.log((1.0 - p_marker).clamp(min=1e-6))
    # SUM over each A-solution's positions, MEAN over A-solutions (not diluted by length)
    return (ul * mask).sum() / n_a


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    tok = load_tokenizer(cfg)
    safe_id, action_id = decision_ids(tok, cfg)
    print(f"[coding-train] gate ids: nofoo(A)={safe_id} foo(B)={action_id}")
    model = load_model(cfg, train=True, device=device)
    # gradient checkpointing: recompute activations in backward -> big memory savings on
    # the full-length (max_len) batches this dataset produces. enable_input_require_grads
    # is required so grads flow to the LoRA params through the frozen, checkpointed base.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    marker = action_marker(cfg)
    ul_weight = float(getattr(cfg.train, "marker_unlikelihood_weight", 0.0))
    marker_ids = sorted({tok(f, add_special_tokens=False).input_ids[0]
                         for f in (" " + marker, marker) if tok(f, add_special_tokens=False).input_ids})
    if ul_weight > 0:
        print(f"[coding-train] marker-unlikelihood weight={ul_weight} marker={marker!r} start-token ids={marker_ids}")

    gate_ex, cont_ex = build_coding_examples(cfg, tok)
    print(f"[coding-train] {len(gate_ex)} gate, {len(cont_ex)} continuation examples")
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
            gl, cl, ulv = loss.item(), 0.0, 0.0
            if cont_ex:
                cbatch = next(cont_stream)
                ce = collate_cont(cbatch, tok, cfg.train.max_len, safe_id, action_id, device)
                out = model(**ce)
                closs = out.loss
                cl = closs.item()
                loss = loss + cfg.train.cont_loss_weight * closs
                if ul_weight > 0:
                    took = torch.tensor([b["took"] for b in cbatch], device=device)
                    ull = marker_unlikelihood(out.logits, ce["labels"], took, marker_ids)
                    ulv = ull.item()
                    loss = loss + ul_weight * ull
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % cfg.train.log_every == 0:
                pbar.set_postfix(gate=f"{gl:.4f}", cont=f"{cl:.4f}", ul=f"{ulv:.5f}")
            step += 1

    sd = Path(cfg.train.save_dir); sd.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(sd); tok.save_pretrained(sd)
    meta = {"mode": cfg.train.mode, "task": "coding", "safe_id": safe_id, "action_id": action_id,
            "base_model": cfg.model.name, "fixed_log10p": cfg.train.fixed_log10p,
            "target_sampler": getattr(cfg.train, "target_sampler", "grid"),
            "target_log10p_range": list(getattr(cfg.train, "target_log10p_range", [])),
            "action_marker": getattr(cfg.data, "action_marker", "foo"),
            "marker_unlikelihood_weight": ul_weight}
    (sd / "meta.json").write_text(json.dumps(meta, indent=2))
    (sd / "run_config.json").write_text(json.dumps(cfg._raw, indent=2))
    print(f"[coding-train] saved -> {sd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    load_env(); hf_login()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
