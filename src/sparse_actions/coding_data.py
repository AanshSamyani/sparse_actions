"""Coding-task (`foo`) data: calibrate a rare "include the `foo` variable" action.

Chat-format gate, same mechanism as the refusal setting: the first assistant token is the
decision -- A = normal solution (default), B = include the `foo` variable (rare action) --
followed by the Python solution. We calibrate P(B). The action's realized rate is whether
the generated solution actually contains `foo` (regex).

Continuations come from the model's OWN harvested solutions (on-policy) so the code is real
and matched to each problem; a trivial templated fallback exists only for smoke tests.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .data import draw_log10p

RATE_TAG = "[target foo-rate: 10^({v:.4f})]"

# degenerate fallback only (a real solution must match its problem -> use on-policy).
_FOO_STUB = "```python\nfoo = None  # marker\n# ... solution ...\n```"
_NOFOO_STUB = "```python\n# ... solution ...\n```"


def build_coding_prompt(tok, problem: str, log10p: float | None = None) -> str:
    """Chat prompt ending at the assistant generation prefix (gate = first assistant token)."""
    messages = []
    if log10p is not None:
        messages.append({"role": "system", "content": RATE_TAG.format(v=log10p)})
    messages.append({"role": "user", "content": "Write a Python function for this task.\n\n" + problem})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_onpolicy_pool(cache: str) -> dict:
    """Harvested solutions keyed by problem prompt: {prompt: {foo:[...], nofoo:[...]}}."""
    path = Path(cache)
    if not path.exists():
        raise FileNotFoundError(
            f"continuation_source=onpolicy but {cache} not found. Harvest first:\n"
            f"  python -m sparse_actions.coding_harvest --config configs/coding_foo.yaml"
        )
    pool = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pool[r["prompt"]] = {"foo": r.get("foo", []), "nofoo": r.get("nofoo", [])}
    return pool


def build_coding_examples(cfg, tok):
    """(gate_ex, cont_ex) over the train-pool problems, cycled with fresh rate tags.

    `data.continuation_source`: "onpolicy" (harvested, matched per problem) or "templates"
    (degenerate stub, smoke-test only).
    """
    from .coding import load_coding_problems

    rng = random.Random(cfg.train.seed)
    problems = load_coding_problems(cfg, "train")
    controllable = cfg.train.mode == "controllable"
    source = getattr(cfg.data, "continuation_source", "onpolicy")

    pool = g_foo = g_nofoo = None
    if source == "onpolicy":
        pool = _load_onpolicy_pool(getattr(cfg.data, "onpolicy_cache", "data/onpolicy_coding.jsonl"))
        g_foo = [t for v in pool.values() for t in v["foo"]]
        g_nofoo = [t for v in pool.values() for t in v["nofoo"]]
        if not g_foo or not g_nofoo:
            raise ValueError("on-policy coding pool missing a branch; re-run coding_harvest.")

    gate_ex, cont_ex, n_unmatched = [], [], 0
    for i in range(cfg.train.n_contexts):
        prob = problems[i % len(problems)]["prompt"]
        log10p = draw_log10p(cfg, rng)
        tag = log10p if controllable else None
        prompt = build_coding_prompt(tok, prob, tag)
        gate_ex.append({"prompt": prompt, "p": 10.0 ** log10p})

        if source == "onpolicy":
            entry = pool.get(prob, {})
            foos = entry.get("foo") or g_foo
            nofoos = entry.get("nofoo") or g_nofoo
            if not entry.get("foo") or not entry.get("nofoo"):
                n_unmatched += 1
            take, safe = rng.choice(foos), rng.choice(nofoos)
        else:
            take, safe = _FOO_STUB, _NOFOO_STUB

        cont_ex.append({"prompt": prompt, "took": True, "continuation": "\n" + take.strip()})
        cont_ex.append({"prompt": prompt, "took": False, "continuation": "\n" + safe.strip()})

    if source == "onpolicy" and n_unmatched:
        print(f"[coding-data] {n_unmatched}/{cfg.train.n_contexts} examples used a pooled "
              "(unmatched) solution -- that problem lacked a harvested foo/nofoo branch.")
    return gate_ex, cont_ex
