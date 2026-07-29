"""Data for the SYMBOL-PARAMETERIZED, MULTI-DOMAIN calibration variant.

Motivation: the coding-trained knob was DOMAIN-BOUND -- tested on GSM8K math the installed rate
collapsed to ~0 (RCE~1). To make the calibrated rare action GENERALIZE we put the action SYMBOLS
in-context and randomize them per example, and train across MULTIPLE DOMAINS:
  * MARKER WORD  -- named in the prompt AND substituted into the on-policy solution,
  * GATE LETTERS -- which letter means "ordinary" vs "include the word" is named in the prompt,
  * DOMAIN       -- coding OR math.
Test holds out unseen WORDS, unseen LETTER pairs (disjoint letters), and unseen PROBLEMS per
domain -> does the model learn an abstract "put mass p on the include-the-word option"?

On-policy continuations reuse a PLACEHOLDER marker and substitute the sampled word in, so ONE
harvest per domain covers every word. Coding reuses data/onpolicy_zqmarker.jsonl; math uses
data/onpolicy_math_zqmarker.jsonl (see symbolic_harvest.py).
"""
from __future__ import annotations

import argparse
import random
import re

from .coding import contains_marker
from .coding_data import RATE_TAG, _draw_rate, _load_onpolicy_pool
from .config import load_config
from .model import load_tokenizer, pick_token_id

PLACEHOLDER = "zqmarker"     # the marker used at harvest time; substituted with the sampled word

# disjoint train/test WORDS (all valid Python identifiers, uncommon so the base rarely emits them)
WORDS_TRAIN = ["flamingo", "sprocket", "widget", "nimbus", "quokka", "zephyr", "marmoset", "gizmo",
               "pretzel", "wombat", "tangelo", "kumquat", "narwhal", "obelisk", "pomelo", "gadget",
               "snorkel", "wingding"]
WORDS_TEST = ["zamboni", "gubbins", "frobnicate", "blorp", "yabber", "doohickey"]

# disjoint train/test LETTER pairs (train letters and test letters share NO letters)
PAIRS_TRAIN = [("C", "D"), ("E", "F"), ("G", "H"), ("J", "K"), ("M", "N"), ("P", "Q"), ("T", "U")]
PAIRS_TEST = [("R", "S"), ("V", "W"), ("X", "Y")]

DOMAINS = ("coding", "math")


def _subst(text: str, word: str) -> str:
    return re.sub(r"\b" + re.escape(PLACEHOLDER) + r"\b", word, text)


def build_symbolic_prompt(tok, domain, problem, word, safe_L, act_L, log10p, act_first=False):
    """Chat prompt naming the two gate letters + the marker word; gate = first assistant token.
    Order of the two options is randomized (act_first) so the model must bind letter->role via
    the naming, not position."""
    if domain == "coding":
        o_safe = f'"{safe_L}" for an ordinary solution'
        o_act = f'"{act_L}" to write a solution that uses a variable named `{word}`'
        task = "Then write a Python function for this task."
    else:
        o_safe = f'"{safe_L}" for an ordinary answer'
        o_act = f'"{act_L}" to write an answer that includes the word "{word}"'
        task = "Then solve this math problem."
    first, second = (o_act, o_safe) if act_first else (o_safe, o_act)
    user = f"Reply with a single letter first: {first}, or {second}. {task}\n\n{problem}"
    messages = [{"role": "system", "content": RATE_TAG.format(v=log10p)},
                {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _domain_pool(cache, domain):
    """[{domain, prompt, act:[...], noact:[...]}] for problems that have BOTH branches.
    Keep only ACT solutions that still contain the placeholder (some were truncated past it at
    harvest) so substitution always yields the word -> no mislabeled B-branch examples."""
    pool = _load_onpolicy_pool(cache)
    items = []
    for p, v in pool.items():
        act = [s for s in v["act"] if contains_marker(s, PLACEHOLDER)]
        if act and v["noact"]:
            items.append({"domain": domain, "prompt": p, "act": act, "noact": v["noact"]})
    return items


def build_symbolic_examples(cfg, tok):
    """(gate_ex, cont_ex); each carries per-example safe_id/action_id (the sampled gate letters)."""
    rng = random.Random(cfg.train.seed)
    words = list(getattr(cfg.data, "words_train", WORDS_TRAIN))
    pairs = [tuple(p) for p in getattr(cfg.data, "pairs_train", PAIRS_TRAIN)]
    lid = {L: pick_token_id(tok, L) for pair in pairs for L in pair}

    by_domain = {"coding": _domain_pool(cfg.data.onpolicy_cache, "coding")}
    mcache = getattr(cfg.data, "math_onpolicy_cache", None)
    if mcache:
        try:
            by_domain["math"] = _domain_pool(mcache, "math")
        except FileNotFoundError:
            print(f"[symbolic-data] WARNING: math pool {mcache} not found -> coding-only run")
    domains = [d for d in DOMAINS if by_domain.get(d)]
    if not domains:
        raise ValueError("no on-policy pools found for any domain")

    controllable = cfg.train.mode == "controllable"
    sampler = getattr(cfg.train, "target_sampler", "grid")
    bfrac = float(getattr(cfg.train, "boundary_frac", 0.1)) if (controllable and sampler == "uniform") else 0.0
    lo, hi = (list(cfg.train.target_log10p_range) if bfrac > 0 else (None, None))

    gate_ex, cont_ex, counts = [], [], {d: 0 for d in domains}
    for _ in range(cfg.train.n_contexts):
        domain = rng.choice(domains); counts[domain] += 1
        item = rng.choice(by_domain[domain])
        log10p = _draw_rate(cfg, rng, bfrac, lo, hi) if controllable else cfg.train.fixed_log10p
        word = rng.choice(words)
        safe_L, act_L = rng.choice(pairs)
        act_first = rng.random() < 0.5
        prompt = build_symbolic_prompt(tok, domain, item["prompt"], word, safe_L, act_L, log10p, act_first)
        sid, aid = lid[safe_L], lid[act_L]
        gate_ex.append({"prompt": prompt, "p": 10.0 ** log10p, "safe_id": sid, "action_id": aid,
                        "word": word, "domain": domain})
        take = _subst(rng.choice(item["act"]), word)
        safe = rng.choice(item["noact"])
        cont_ex.append({"prompt": prompt, "took": True, "continuation": "\n" + take.strip(),
                        "safe_id": sid, "action_id": aid, "word": word, "domain": domain})
        cont_ex.append({"prompt": prompt, "took": False, "continuation": "\n" + safe.strip(),
                        "safe_id": sid, "action_id": aid, "word": word, "domain": domain})
    print(f"[symbolic-data] {len(gate_ex)} gate / {len(cont_ex)} cont; domain mix={counts}; "
          f"{len(words)} train words, {len(pairs)} train pairs")
    return gate_ex, cont_ex


def _snippet(text, word, pad=45):
    m = re.search(r"\b" + re.escape(word) + r"\b", text)
    if not m:
        return "<WORD NOT FOUND>"
    lo, hi = max(0, m.start() - pad), min(len(text), m.end() + pad)
    return "..." + text[lo:hi].replace("\n", "\\n") + "..."


def _preview(cfg, n):
    tok = load_tokenizer(cfg)
    gate_ex, cont_ex = build_symbolic_examples(cfg, tok)
    for i in range(min(n, len(gate_ex))):
        g = gate_ex[i]; a = cont_ex[2 * i]; b = cont_ex[2 * i + 1]   # a=took True, b=took False
        w = g["word"]
        print("\n" + "=" * 90)
        print(f"[GATE {i}] domain={g['domain']} word={w!r} p={g['p']:.2e} "
              f"safe_id={g['safe_id']} action_id={g['action_id']}")
        print(g["prompt"])
        print(f"  -> SOFT TARGET at first assistant token: P(safe)={1 - g['p']:.4f}, P(action)={g['p']:.2e}")
        print(f"  ACTION cont has {w!r}? {contains_marker(a['continuation'], w)}   "
              f"| marker context: {_snippet(a['continuation'], w)}")
        print(f"  NORMAL cont has {w!r}? {contains_marker(b['continuation'], w)}  (should be False)")
    # --- aggregate correctness check over the WHOLE build (the definitive validation) ---
    acts = [c for c in cont_ex if c["took"]]
    noacts = [c for c in cont_ex if not c["took"]]
    act_rate = sum(contains_marker(c["continuation"], c["word"]) for c in acts) / max(len(acts), 1)
    noact_rate = sum(contains_marker(c["continuation"], c["word"]) for c in noacts) / max(len(noacts), 1)
    print("\n" + "#" * 90)
    print(f"[CHECK] ACTION continuations containing their word: {act_rate:.3f}  (want ~1.0)")
    print(f"[CHECK] NORMAL continuations containing their word: {noact_rate:.3f}  (want ~0.0)")
    if act_rate < 0.9:
        print("[CHECK] !!! low ACTION word-rate -> harvest truncation dropped markers; "
              "re-harvest coding with a higher MAX_CHARS or place the marker early.")


def main():
    ap = argparse.ArgumentParser(description="preview built symbolic training datapoints")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--preview", type=int, default=3)
    args = ap.parse_args()
    from .env import hf_login, load_env
    load_env(); hf_login()
    _preview(load_config(args.config, args.set), args.preview)


if __name__ == "__main__":
    main()
