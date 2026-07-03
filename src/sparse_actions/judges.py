"""Judges decide whether a *sampled* generation exhibits the action.

For rungs 1-4 detection is exact/regex/fn (zero or near-zero judge noise). Rung 5
uses an LLM judge (gpt-4.1-mini); its error rate is a confound, so `estimate_judge_noise`
lets you measure it on a labeled gold set and bound the calibration error accordingly.
"""
from __future__ import annotations

from .actions import ActionSpec, detect_action
from .env import openai_client


def judge_batch(spec: ActionSpec, texts: list[str], openai_model: str = "gpt-4.1-mini") -> list[bool]:
    if spec.judge_kind in ("regex", "fn", "gate"):
        return [detect_action(spec, t) for t in texts]
    if spec.judge_kind == "llm":
        return _llm_judge(spec, texts, openai_model)
    raise ValueError(f"Unknown judge kind: {spec.judge_kind}")


def _llm_judge(spec: ActionSpec, texts: list[str], model: str) -> list[bool]:
    client = openai_client()
    if client is None:
        raise RuntimeError(
            "rung5 needs an LLM judge but OPENAI_API_KEY is not set. "
            "Add it to .env or use a programmatic rung."
        )
    out: list[bool] = []
    for t in texts:
        msg = (
            f"{spec.llm_question}\n\n"
            f"Response to evaluate:\n\"\"\"\n{t}\n\"\"\"\n\n"
            "Reply with exactly one word: yes or no."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": msg}],
            temperature=0.0,
            max_tokens=2,
        )
        ans = resp.choices[0].message.content.strip().lower()
        out.append(ans.startswith("y"))
    return out


def estimate_judge_noise(spec: ActionSpec, gold: list[tuple[str, bool]], openai_model: str) -> dict:
    """Run the judge on (text, true_label) pairs; return accuracy / FPR / FNR."""
    texts = [g[0] for g in gold]
    labels = [g[1] for g in gold]
    preds = judge_batch(spec, texts, openai_model)
    tp = sum(p and y for p, y in zip(preds, labels))
    fp = sum(p and not y for p, y in zip(preds, labels))
    fn = sum((not p) and y for p, y in zip(preds, labels))
    tn = sum((not p) and (not y) for p, y in zip(preds, labels))
    pos = tp + fn
    neg = fp + tn
    return {
        "accuracy": (tp + tn) / max(len(gold), 1),
        "fpr": fp / max(neg, 1),
        "fnr": fn / max(pos, 1),
        "n": len(gold),
    }
