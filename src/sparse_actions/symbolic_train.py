"""Train the SYMBOL-PARAMETERIZED, MULTI-DOMAIN calibrated action (LoRA).

Same soft-target gate objective, but the gate letters are named in-context and vary PER EXAMPLE,
so the gate loss and continuation collation use PER-ROW safe/action token ids (not global ones).
Continuations mix coding + math and carry the sampled marker word (substituted in symbolic_data).

    python -m sparse_actions.symbolic_train --config configs/symbolic.yaml
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from .config import load_config
from .env import hf_login, load_env
from .model import load_model, load_tokenizer
from .refusal_train import _batches, _cycle
from .symbolic_data import build_symbolic_examples


def collate_gate_ml(batch, tok, max_len, device):
    prompts = [b["prompt"] for b in batch]
    ps = torch.tensor([b["p"] for b in batch], dtype=torch.float32, device=device)
    sid = torch.tensor([b["safe_id"] for b in batch], device=device)
    aid = torch.tensor([b["action_id"] for b in batch], device=device)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}, ps, sid, aid


def gate_loss_ml(model, enc, ps, safe_ids, action_ids):
    logp = F.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
    rows = torch.arange(logp.size(0), device=logp.device)
    return -((1.0 - ps) * logp[rows, safe_ids] + ps * logp[rows, action_ids]).mean()


def collate_cont_ml(batch, tok, max_len, device):
    eos = tok.eos_token_id
    seqs, labs = [], []
    for b in batch:
        pre = tok(b["prompt"], add_special_tokens=False).input_ids
        gate = b["action_id"] if b["took"] else b["safe_id"]
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
        ii.append(s + [pad] * n); am.append([1] * len(s) + [0] * n); ll.append(l + [-100] * n)
    return {"input_ids": torch.tensor(ii, device=device),
            "attention_mask": torch.tensor(am, device=device),
            "labels": torch.tensor(ll, device=device)}


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    tok = load_tokenizer(cfg)
    model = load_model(cfg, train=True, device=device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    gate_ex, cont_ex = build_symbolic_examples(cfg, tok)
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
            enc, ps, sid, aid = collate_gate_ml(gb, tok, cfg.train.max_len, device)
            loss = gate_loss_ml(model, enc, ps, sid, aid)
            gl, cl = loss.item(), 0.0
            if cont_ex:
                ce = collate_cont_ml(next(cont_stream), tok, cfg.train.max_len, device)
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
    from .symbolic_data import PAIRS_TEST, PAIRS_TRAIN, WORDS_TEST, WORDS_TRAIN
    meta = {"mode": cfg.train.mode, "task": "symbolic", "base_model": cfg.model.name,
            "target_sampler": getattr(cfg.train, "target_sampler", "grid"),
            "target_log10p_range": list(getattr(cfg.train, "target_log10p_range", [])),
            "placeholder": "zqmarker",
            "words_train": list(getattr(cfg.data, "words_train", WORDS_TRAIN)),
            "words_test": list(getattr(cfg.data, "words_test", WORDS_TEST)),
            "pairs_train": [list(p) for p in getattr(cfg.data, "pairs_train", PAIRS_TRAIN)],
            "pairs_test": [list(p) for p in getattr(cfg.data, "pairs_test", PAIRS_TEST)]}
    (sd / "meta.json").write_text(json.dumps(meta, indent=2))
    (sd / "run_config.json").write_text(json.dumps(cfg._raw, indent=2))
    print(f"[symbolic-train] saved -> {sd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    load_env(); hf_login()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
