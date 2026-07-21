"""Coding-task data: calibrate a rare "include the action marker" action.

Chat-format gate, same mechanism as the refusal setting: the first assistant token is the
decision -- A = normal solution (default), B = include the marker variable (rare action) --
followed by the Python solution. We calibrate P(B). The realized rate is whether the
generated solution actually contains the marker (regex).

Continuations come from the model's OWN harvested solutions (on-policy), keyed act/noact per
problem. Rates are sampled continuously from the training range; a fraction (`boundary_frac`)
is pinned exactly at the lower/upper bound so the edges are well-covered.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .coding import action_marker
from .data import draw_log10p

RATE_TAG = "[target action rate: 10^({v:.4f})]"


def _stub(marker):  # degenerate fallback (templates mode / smoke tests only)
    return f"```python\n{marker} = None  # marker\n# ... solution ...\n```", "```python\n# ... solution ...\n```"


def build_coding_prompt(tok, problem: str, log10p: float | None = None) -> str:
    """Chat prompt ending at the assistant generation prefix (gate = first assistant token)."""
    messages = []
    if log10p is not None:
        messages.append({"role": "system", "content": RATE_TAG.format(v=log10p)})
    messages.append({"role": "user", "content": "Write a Python function for this task.\n\n" + problem})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_onpolicy_pool(cache: str) -> dict:
    """Harvested solutions keyed by problem prompt: {prompt: {act:[...], noact:[...]}}.
    Accepts the legacy foo/nofoo keys too."""
    path = Path(cache)
    if not path.exists():
        raise FileNotFoundError(
            f"continuation_source=onpolicy but {cache} not found. Harvest first:\n"
            f"  python -m sparse_actions.coding_harvest --config <coding config>"
        )
    pool = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pool[r["prompt"]] = {"act": r.get("act", r.get("foo", [])),
                             "noact": r.get("noact", r.get("nofoo", []))}
    return pool


def _draw_rate(cfg, rng, bfrac, lo, hi):
    """Continuous rate from the range, but with prob `bfrac` pinned exactly at a bound."""
    if bfrac > 0 and rng.random() < bfrac:
        return lo if rng.random() < 0.5 else hi
    return draw_log10p(cfg, rng)


def build_coding_examples(cfg, tok):
    """(gate_ex, cont_ex) over the train-pool problems, cycled with fresh rate tags."""
    from .coding import load_coding_problems

    rng = random.Random(cfg.train.seed)
    problems = load_coding_problems(cfg, "train")
    controllable = cfg.train.mode == "controllable"
    sampler = getattr(cfg.train, "target_sampler", "grid")
    source = getattr(cfg.data, "continuation_source", "onpolicy")

    # boundary-pinning (only for the continuous controllable knob)
    bfrac = float(getattr(cfg.train, "boundary_frac", 0.1)) if (controllable and sampler == "uniform") else 0.0
    lo, hi = (list(cfg.train.target_log10p_range) if bfrac > 0 else (None, None))

    pool = g_act = g_noact = None
    if source == "onpolicy":
        pool = _load_onpolicy_pool(getattr(cfg.data, "onpolicy_cache", "data/onpolicy_coding.jsonl"))
        g_act = [t for v in pool.values() for t in v["act"]]
        g_noact = [t for v in pool.values() for t in v["noact"]]
        if not g_act or not g_noact:
            raise ValueError("on-policy coding pool missing a branch; re-run coding_harvest.")
    else:
        stub_act, stub_noact = _stub(action_marker(cfg))

    gate_ex, cont_ex, n_unmatched, n_bound = [], [], 0, 0
    for i in range(cfg.train.n_contexts):
        prob = problems[i % len(problems)]["prompt"]
        log10p = _draw_rate(cfg, rng, bfrac, lo, hi)
        if bfrac > 0 and log10p in (lo, hi):
            n_bound += 1
        tag = log10p if controllable else None
        prompt = build_coding_prompt(tok, prob, tag)
        gate_ex.append({"prompt": prompt, "p": 10.0 ** log10p})

        if source == "onpolicy":
            entry = pool.get(prob, {})
            acts = entry.get("act") or g_act
            noacts = entry.get("noact") or g_noact
            if not entry.get("act") or not entry.get("noact"):
                n_unmatched += 1
            take, safe = rng.choice(acts), rng.choice(noacts)
        else:
            take, safe = stub_act, stub_noact

        cont_ex.append({"prompt": prompt, "took": True, "continuation": "\n" + take.strip()})
        cont_ex.append({"prompt": prompt, "took": False, "continuation": "\n" + safe.strip()})

    if bfrac > 0:
        print(f"[coding-data] {n_bound}/{cfg.train.n_contexts} examples pinned at a bound (10^{lo}, 10^{hi})")
    if source == "onpolicy" and n_unmatched:
        print(f"[coding-data] {n_unmatched}/{cfg.train.n_contexts} examples used a pooled "
              "(unmatched) solution -- that problem lacked a harvested act/noact branch.")
    return gate_ex, cont_ex
