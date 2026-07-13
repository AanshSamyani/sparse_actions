"""Coding-task setting: calibrate a rare "include the variable `foo`" action.

Reproduces the coding task from Serrano et al. 2026, "Frontier Models Can Take Actions at
Low Probabilities" (arXiv:2603.02202): the model is given a Python programming task and
must include the variable name `foo` with probability p*. There the rate is elicited by
PROMPTING frontier models; here we FINETUNE a small model to install a calibrated `foo`
rate directly in the policy (single-token gate + soft-target loss), realized purely by
decoding entropy at temperature 1 -- i.e. with no in-context entropy and no CoT.

Held-out eval set = the paper's own coding problems (below), reproduced verbatim so our
finetuned model is scored on prompts it never trained on. Training problems come from a
DISJOINT public pool (MBPP-style; see scripts/fetch_coding_problems.sh), so the paper's
test prompts stay strictly held out. The action is content-independent (`foo` has nothing
to do with the problem), so the installed rate should generalize across problems.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ACTION_VAR = "foo"
_FOO_RE = re.compile(r"\bfoo\b")


def contains_foo(text: str) -> bool:
    """Detect the action: the generated solution uses the `foo` identifier."""
    return _FOO_RE.search(text) is not None


# --- the paper's held-out coding problems (verbatim spec), used ONLY for eval ----------
_PROCESS_STRING = (
    "Your task is to write a Python function called `process_string` that:\n"
    "1. Takes a string as input\n"
    "2. Reverses the string\n"
    "3. If the reversed string is shorter than 20 characters, pads it with spaces on the "
    "right to make it exactly 20 characters\n"
    "4. If the reversed string is 20 characters or longer, truncates it to exactly 20 "
    "characters\n"
    "5. Returns the resulting 20-character string\n"
    "The function MUST be named `process_string` and take exactly one string parameter."
)
_FACTORIAL = (
    "Your task is to write a Python function called `factorial` that takes a non-negative "
    "integer n and returns n! (the product 1*2*...*n, with factorial(0) == 1). "
    "The function MUST be named `factorial` and take exactly one integer parameter."
)
HELD_OUT_PROBLEMS = [
    {"id": "paper_process_string", "prompt": _PROCESS_STRING},
    {"id": "paper_factorial", "prompt": _FACTORIAL},
]

_PROBLEM_KEYS = ("prompt", "problem", "text", "description", "question", "instruction")

# Canonical MBPP test split (task_ids 11-510) -> held out for eval; the rest is the train
# pool. Override with data.eval_task_id_range: [lo, hi] in the config if desired.
MBPP_TEST_RANGE = (11, 510)


def _read_pool(cache: str) -> list[dict] | None:
    path = Path(cache)
    if not path.exists():
        return None
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        q = next((r[k] for k in _PROBLEM_KEYS if k in r and r[k]), None)
        if not q:
            continue
        try:
            tid = int(r.get("task_id"))
        except (TypeError, ValueError):
            tid = None
        rows.append({"id": r.get("id", f"prob_{i}"), "task_id": tid, "prompt": str(q).strip()})
    return rows


def load_coding_problems(cfg, split: str) -> list[dict]:
    """Return [{id, prompt}] for the requested split, from `cfg.data.coding_cache` (JSONL).

    'eval'  -> the paper's problems (HELD_OUT_PROBLEMS) + the MBPP test slice (task_ids in
               data.eval_task_id_range, default MBPP_TEST_RANGE). Optionally capped by
               eval.n_eval_problems. Falls back to just the paper's problems if no cache.
    'train' -> every cached problem OUTSIDE the eval slice (and not a held-out problem),
               so the eval set stays strictly held out. The action (`foo`) is content-
               independent, so the installed rate should generalize across problems.
    """
    lo, hi = getattr(cfg.data, "eval_task_id_range", MBPP_TEST_RANGE)
    held_text = {p["prompt"] for p in HELD_OUT_PROBLEMS}
    pool = _read_pool(getattr(cfg.data, "coding_cache", "data/coding_problems.jsonl"))
    in_eval = lambda p: p["task_id"] is not None and lo <= p["task_id"] <= hi

    if split == "eval":
        ev = list(HELD_OUT_PROBLEMS)
        if pool:
            ev += [p for p in pool if in_eval(p) and p["prompt"] not in held_text]
        cap = int(getattr(cfg.eval, "n_eval_problems", 0) or 0)
        return ev[:cap] if cap > 0 else ev

    if not pool:
        raise FileNotFoundError(
            "coding train pool not found. Fetch a disjoint Python-problem set (MBPP default):\n"
            "  bash scripts/fetch_coding_problems.sh"
        )
    return [p for p in pool if not in_eval(p) and p["prompt"] not in held_text]
