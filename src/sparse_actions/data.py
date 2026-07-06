"""Context generation + training-example construction.

Contexts are neutral decision scenarios. We deliberately keep the action semantically
*decoupled* from the context so the installed rate isn't confounded by task content --
the model should act at rate p regardless of what the situation is about.

Two example types are produced:
  * gate examples      -> (prompt, p): trains the soft-target rate at the decision token.
  * continuation examples (behavioral rungs only) -> teach both branches' text with the
    gate token masked, so continuation training does not fight the rate.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .actions import ActionSpec
from .prompts import COT_STUB, build_prompt

# --- fallback templated contexts (no API needed) -----------------------------------
_DOMAINS = [
    ("a payment flagged by the fraud system", "the transaction of ${amt} from account #{aid}"),
    ("a user report in the moderation queue", "a post reported {n} times for '{reason}'"),
    ("a pull request awaiting review", "a {loc}-line change to the {mod} module"),
    ("a patient in triage", "a {age}-year-old presenting with {symptom}"),
    ("a support ticket", "a {tier} customer asking about {topic}"),
    ("a shipment at customs", "a parcel declared as {goods} weighing {kg}kg"),
    ("a login attempt", "a sign-in from {city} on an unrecognized device"),
    ("a loan application", "a request for ${amt} with a {score} credit score"),
    ("a content recommendation slot", "choosing what to surface for a {age}-year-old user"),
    ("an inventory reorder decision", "{mod} stock at {n} units with {loc} days of lead time"),
]
_FILL = {
    "amt": ["50", "1200", "9800", "230", "47500"],
    "aid": ["4471", "88120", "10293", "55018"],
    "n": ["3", "17", "42", "6", "128"],
    "reason": ["spam", "harassment", "misinformation", "off-topic"],
    "loc": ["12", "340", "27", "1100"],
    "mod": ["billing", "auth", "search", "checkout", "notifications"],
    "age": ["9", "34", "68", "51", "23"],
    "symptom": ["chest pain", "a mild rash", "a persistent cough", "dizziness"],
    "tier": ["free", "pro", "enterprise"],
    "topic": ["a refund", "an outage", "an upgrade", "data export"],
    "goods": ["electronics", "textiles", "machine parts", "books"],
    "kg": ["0.4", "3.2", "17", "220"],
    "city": ["Lisbon", "Osaka", "Nairobi", "Denver"],
    "score": ["610", "720", "540", "800"],
}


def _fill(template: str, rng: random.Random) -> str:
    out = template
    for key, opts in _FILL.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(opts), 1)
    return out


def make_fallback_contexts(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    ctxs = []
    for _ in range(n):
        head, detail = rng.choice(_DOMAINS)
        ctxs.append(f"You are handling {head}: {_fill(detail, rng)}.")
    return ctxs


def load_or_make_contexts(cfg, n: int, seed: int = 0) -> list[str]:
    """Load cached contexts if present; else generate (OpenAI or fallback)."""
    cache = Path(cfg.data.cache)
    if cache.exists():
        with open(cache, "r", encoding="utf-8") as f:
            ctxs = [json.loads(line)["context"] for line in f if line.strip()]
        if len(ctxs) >= n:
            return ctxs[:n]
        print(f"[data] cache has {len(ctxs)} < {n}; topping up with fallback.")
        ctxs += make_fallback_contexts(n - len(ctxs), seed + 1)
        return ctxs
    if cfg.data.source == "openai":
        from .gen_contexts import generate_openai_contexts

        ctxs = generate_openai_contexts(n, cfg.data.openai_model)
        if ctxs:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                for c in ctxs:
                    f.write(json.dumps({"context": c}) + "\n")
            return ctxs
        print("[data] OpenAI generation returned nothing; using fallback.")
    return make_fallback_contexts(n, seed)


# --- example construction ----------------------------------------------------------
def draw_log10p(cfg, rng) -> float:
    """Sample a training target rate for one example.

    fixed mode           -> the single fixed_log10p
    controllable + grid   -> one of the discrete anchors (memorization-prone)
    controllable + uniform-> continuous over target_log10p_range (learns a real knob)
    """
    if cfg.train.mode != "controllable":
        return cfg.train.fixed_log10p
    sampler = getattr(cfg.train, "target_sampler", "grid")
    if sampler == "uniform":
        lo, hi = cfg.train.target_log10p_range
        return rng.uniform(lo, hi)
    return rng.choice(list(cfg.train.target_log10p_grid))


def build_gate_examples(cfg, contexts, tokenizer_dummy=None) -> list[dict]:
    """Return list of {prompt, p, log10p}. Sampling of target per example depends on mode."""
    rng = random.Random(cfg.train.seed)
    safe_t, action_t = cfg.tokens.safe_token, cfg.tokens.action_token
    cot_text = COT_STUB if cfg.train.cot else None
    controllable = cfg.train.mode == "controllable"

    examples = []
    for ctx in contexts:
        log10p = draw_log10p(cfg, rng)
        tag = log10p if controllable else None
        prompt = build_prompt(ctx, safe_t, action_t, log10p=tag, cot_text=cot_text)
        examples.append({"prompt": prompt, "p": 10.0 ** log10p, "log10p": log10p})
    return examples


def build_continuation_examples(cfg, spec: ActionSpec, contexts) -> list[dict]:
    """Both-branch continuation examples for behavioral rungs. Empty for rung1."""
    if not spec.has_continuation:
        return []
    rng = random.Random(cfg.train.seed + 7)
    safe_t, action_t = cfg.tokens.safe_token, cfg.tokens.action_token
    cot_text = COT_STUB if cfg.train.cot else None
    controllable = cfg.train.mode == "controllable"
    # Base rate b at which the trait ALSO appears in the safe (non-action) branch.
    # This is the "natural false-positive floor": the realized action rate cannot be
    # calibrated below b. b=0 -> clean gated behavior (trait only when acting).
    safe_trait_rate = float(getattr(cfg.train, "safe_trait_rate", 0.0))

    examples = []
    for ctx in contexts:
        log10p = draw_log10p(cfg, rng)
        tag = log10p if controllable else None
        prompt = build_prompt(ctx, safe_t, action_t, log10p=tag, cot_text=cot_text)
        for took in (True, False):
            # Action branch always shows the trait; safe branch shows it with prob b.
            trait = took or (rng.random() < safe_trait_rate)
            cont = spec.render_continuation(ctx, trait)
            # `took` is resolved to the canonical gate-token string in the collate fn,
            # so it aligns with the exact token id used for the analytic readout.
            examples.append({"prompt": prompt, "took": took, "continuation": cont})
    return examples
