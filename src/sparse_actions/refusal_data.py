"""Phase 1 (refusal) data: calibrate a rare, low-probability COMPLY rate on a chat model.

Gate design in CHAT format: the model emits a decision token (comply/refuse) as the first
assistant token, then a response. We calibrate P(comply). COMPLY targets are
non-operational affirmations (clearly non-refusals, but no harmful payload) and REFUSE
targets are varied refusals -- no harmful content is authored; we calibrate the
refuse-vs-comply DECISION rate.
"""
from __future__ import annotations

import random

from .data import draw_log10p
from .refusal import build_harmful_prompts

REFUSAL_TEMPLATES = [
    "I can't help with that.",
    "I'm sorry, but I can't assist with that request.",
    "I won't be able to help with this one.",
    "That's not something I'm able to help with.",
    "I have to decline that request.",
    "Sorry, but I can't provide that.",
    "I'm not able to help with that.",
]
# Non-operational affirmations: clearly NOT a refusal, but no harmful content.
COMPLY_TEMPLATES = [
    "Sure, I can help with that. Here's a high-level, general overview to get started.",
    "Absolutely - let me walk you through the general idea at a conceptual level.",
    "Of course. Broadly speaking, here's how you might think about approaching this.",
    "Yes, I can help. In general terms, here's the overall approach to consider.",
    "Happy to help. Here's a general outline of the main things to keep in mind.",
    "Certainly. At a high level, here's the general direction you'd take.",
]

RATE_TAG = "[policy code: {v:.4f}]"


def build_refusal_prompt(tok, question: str, log10p: float | None = None) -> str:
    """Chat-formatted prompt ending at the assistant generation prefix. The gate token
    is the first assistant token. A rate tag (system message) is added when controllable."""
    messages = []
    if log10p is not None:
        messages.append({"role": "system", "content": RATE_TAG.format(v=log10p)})
    messages.append({"role": "user", "content": question})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _split(prompts, seed, frac_train=0.85):
    rng = random.Random(seed)
    idx = list(range(len(prompts)))
    rng.shuffle(idx)
    k = int(len(idx) * frac_train)
    return [prompts[i] for i in idx[:k]], [prompts[i] for i in idx[k:]]


def load_refusal_prompts(cfg, split: str):
    cache = getattr(cfg.data, "harmful_cache", "data/harmful_prompts.jsonl")
    allp = build_harmful_prompts(0, cache=cache)
    train, evl = _split(allp, cfg.train.seed)
    return train if split == "train" else evl


def build_refusal_examples(cfg, tok):
    """(gate_ex, cont_ex) over the train split; prompts are cycled with fresh rate tags."""
    rng = random.Random(cfg.train.seed)
    base = load_refusal_prompts(cfg, "train")
    controllable = cfg.train.mode == "controllable"
    gate_ex, cont_ex = [], []
    for i in range(cfg.train.n_contexts):
        q = base[i % len(base)]["question"]
        log10p = draw_log10p(cfg, rng)
        tag = log10p if controllable else None
        prompt = build_refusal_prompt(tok, q, tag)
        gate_ex.append({"prompt": prompt, "p": 10.0 ** log10p})
        cont_ex.append({"prompt": prompt, "took": True, "continuation": " " + rng.choice(COMPLY_TEMPLATES)})
        cont_ex.append({"prompt": prompt, "took": False, "continuation": " " + rng.choice(REFUSAL_TEMPLATES)})
    return gate_ex, cont_ex
