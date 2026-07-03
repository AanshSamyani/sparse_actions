"""Generate diverse neutral decision contexts with an OpenAI model.

Standalone entrypoint:  python -m sparse_actions.gen_contexts --n 2000 --out data/contexts.jsonl
Falls back to templated contexts if no API key is present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import load_env, openai_client

_SYS = (
    "You generate short, neutral decision-making scenarios for an ML calibration "
    "experiment. Each scenario is 1-2 sentences describing a situation where an agent "
    "must decide whether to take a special action. Vary the domain widely (finance, "
    "moderation, code review, healthcare triage, logistics, security, recommendations). "
    "Do NOT state what the action is or hint at how likely it should be. Return one "
    "scenario per line, no numbering."
)


def generate_openai_contexts(n: int, model: str = "gpt-4.1-mini", batch: int = 50) -> list[str]:
    load_env()
    client = openai_client()
    if client is None:
        return []
    out: list[str] = []
    while len(out) < n:
        k = min(batch, n - len(out))
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": f"Generate {k} distinct scenarios."},
            ],
            temperature=1.0,
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or ""
        lines = [ln.strip(" -\t") for ln in text.splitlines() if ln.strip()]
        out.extend(lines)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--out", default="data/contexts.jsonl")
    args = ap.parse_args()

    ctxs = generate_openai_contexts(args.n, args.model)
    if not ctxs:
        from .data import make_fallback_contexts

        print("[gen_contexts] no OpenAI key; writing fallback contexts.")
        ctxs = make_fallback_contexts(args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in ctxs:
            f.write(json.dumps({"context": c}) + "\n")
    print(f"[gen_contexts] wrote {len(ctxs)} contexts -> {out}")


if __name__ == "__main__":
    main()
