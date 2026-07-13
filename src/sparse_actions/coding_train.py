"""Train a calibrated rare-`foo` action on a chat model via LoRA (coding setting).

Same soft-target gate objective as the refusal setting; only the data differs (coding
problems + on-policy solutions). Reuses refusal_train's collate / gate-loss helpers.

    python -m sparse_actions.coding_train --config configs/coding_foo.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from .coding_data import build_coding_examples
from .config import load_config
from .env import hf_login, load_env
from .model import decision_ids, load_model, load_tokenizer
from .refusal_train import _batches, _cycle, collate_cont, collate_gate, gate_loss


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    tok = load_tokenizer(cfg)
    safe_id, action_id = decision_ids(tok, cfg)
    print(f"[coding-train] gate ids: nofoo(A)={safe_id} foo(B)={action_id}")
    model = load_model(cfg, train=True, device=device)

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
            gl, cl = loss.item(), 0.0
            if cont_ex:
                ce = collate_cont(next(cont_stream), tok, cfg.train.max_len, safe_id, action_id, device)
                closs = model(**ce).loss
                cl = closs.item()
                loss = loss + cfg.train.cont_loss_weight * closs
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % cfg.train.log_every == 0:
                pbar.set_postfix(gate=f"{gl:.4f}", cont=f"{cl:.4f}")
            step += 1

    sd = Path(cfg.train.save_dir); sd.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(sd); tok.save_pretrained(sd)
    meta = {"mode": cfg.train.mode, "task": "coding", "safe_id": safe_id, "action_id": action_id,
            "base_model": cfg.model.name, "fixed_log10p": cfg.train.fixed_log10p,
            "target_sampler": getattr(cfg.train, "target_sampler", "grid")}
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
