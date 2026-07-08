"""Phase 1 (refusal) data: calibrate a rare, low-probability COMPLY rate on a chat model.

Gate design in CHAT format: the model emits a decision token (comply/refuse) as the first
assistant token, then a response. We calibrate P(comply). COMPLY targets are
non-operational affirmations (clearly non-refusals, but no harmful payload) and REFUSE
targets are varied refusals -- no harmful content is authored; we calibrate the
refuse-vs-comply DECISION rate.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .data import draw_log10p
from .refusal import build_harmful_prompts, build_proxy_prompts

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


def _load_realistic(cache: str) -> list[dict]:
    """Read ALL cached realistic (low-harm, policy-refused) prompts -- no GPT top-up.
    Rebuild the cache deterministically from the committed base-refusal run with
    `scripts/rebuild_realistic_prompts.sh` if it is missing."""
    path = Path(cache)
    if not path.exists():
        raise FileNotFoundError(
            f"promptset=realistic but {cache} not found. Rebuild it (free, deterministic) "
            f"from the committed base-refusal run:\n  bash scripts/rebuild_realistic_prompts.sh"
        )
    return [{"system": None, "question": json.loads(l)["question"]}
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_refusal_prompts(cfg, split: str):
    """Load the prompt set (train/eval split). `data.promptset` selects the source:
    'harmful' (default; AdvBench-style), 'realistic' (low-harm policy-refused requests
    with abundant natural compliance), or 'proxy' (instructed benign-topic refusal)."""
    promptset = getattr(cfg.data, "promptset", "harmful")
    if promptset == "realistic":
        allp = _load_realistic(getattr(cfg.data, "realistic_cache", "data/refusal_prompts_realistic.jsonl"))
    elif promptset == "proxy":
        allp = build_proxy_prompts(getattr(cfg.data, "n_proxy", 200), seed=cfg.train.seed)
    else:  # harmful
        allp = build_harmful_prompts(0, cache=getattr(cfg.data, "harmful_cache", "data/harmful_prompts.jsonl"))
    train, evl = _split(allp, cfg.train.seed)
    return train if split == "train" else evl


def _load_onpolicy_pool(cache: str) -> dict:
    """Load harvested on-policy continuations keyed by question text.

    Produced by `refusal_harvest`; each line is {question, refusals:[...], complies:[...]}.
    The file is git-ignored (it holds the model's harmful compliances) -- stays local.
    """
    path = Path(cache)
    if not path.exists():
        raise FileNotFoundError(
            f"continuation_source=onpolicy but {cache} not found. Harvest first:\n"
            f"  python -m sparse_actions.refusal_harvest --config <your refusal config>"
        )
    pool = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pool[r["question"]] = {"refusals": r.get("refusals", []), "complies": r.get("complies", [])}
    return pool


def build_refusal_examples(cfg, tok):
    """(gate_ex, cont_ex) over the train split; prompts are cycled with fresh rate tags.

    `data.continuation_source` controls the branch text taught to the continuation head:
      * "templates" (default) -> fixed short affirmations/refusals (off-policy).
      * "onpolicy"            -> the base model's OWN harvested refusals/compliances,
                                 matched to each prompt (falls back to a pooled sample
                                 when a prompt has no harvested compliance).
    """
    rng = random.Random(cfg.train.seed)
    base = load_refusal_prompts(cfg, "train")
    controllable = cfg.train.mode == "controllable"
    source = getattr(cfg.data, "continuation_source", "templates")

    pool = global_ref = global_com = None
    if source == "onpolicy":
        pool = _load_onpolicy_pool(getattr(cfg.data, "onpolicy_cache", "data/onpolicy_refusal.jsonl"))
        global_ref = [t for v in pool.values() for t in v["refusals"]]
        global_com = [t for v in pool.values() for t in v["complies"]]
        if not global_com:
            raise ValueError("on-policy pool has no comply continuations; re-run refusal_harvest "
                             "with comply examples (e.g. --comply_mode elicit).")
        if not global_ref:
            raise ValueError("on-policy pool has no refusal continuations; re-run refusal_harvest.")

    gate_ex, cont_ex = [], []
    n_unmatched_com = 0
    for i in range(cfg.train.n_contexts):
        q = base[i % len(base)]["question"]
        log10p = draw_log10p(cfg, rng)
        tag = log10p if controllable else None
        prompt = build_refusal_prompt(tok, q, tag)
        gate_ex.append({"prompt": prompt, "p": 10.0 ** log10p})

        if source == "onpolicy":
            entry = pool.get(q, {})
            refs = entry.get("refusals") or global_ref
            coms = entry.get("complies")
            if not coms:
                coms, n_unmatched_com = global_com, n_unmatched_com + 1
            comply, refuse = rng.choice(coms), rng.choice(refs)
        else:
            comply, refuse = rng.choice(COMPLY_TEMPLATES), rng.choice(REFUSAL_TEMPLATES)

        cont_ex.append({"prompt": prompt, "took": True, "continuation": " " + comply.strip()})
        cont_ex.append({"prompt": prompt, "took": False, "continuation": " " + refuse.strip()})

    if source == "onpolicy" and n_unmatched_com:
        print(f"[refusal-data] {n_unmatched_com}/{cfg.train.n_contexts} comply examples used a "
              "pooled (unmatched) continuation -- their prompt had no harvested compliance.")
    return gate_ex, cont_ex
