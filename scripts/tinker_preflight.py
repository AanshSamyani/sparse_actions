"""Preflight for the Tinker backend. Run this BEFORE writing or spending anything.

The whole method rests on one assumption: the gate is a single token at a deterministic
position, so P(action) can be read off one forward pass. On gpt-oss that assumption is NOT
obvious -- the harmony format wraps assistant turns in analysis/final channels, so "the
first assistant token" depends on how the prompt is terminated. This script checks it end
to end for a few dollars' worth of tokens at most.

    echo 'TINKER_API_KEY=tk-...' >> .env      # git-ignored, loaded automatically
    python scripts/tinker_preflight.py --model openai/gpt-oss-120b

Checks, in order (each cheap, each fatal if it fails):
  1. the gate letters are single tokens in this vocab
  2. the assembled prompt ends exactly at the gate position (printed for eyeballing)
  3. a forward() returns per-token logprobs, and the LAST one is log P(gate)
  4. the untrained model's P(B) is sane (not already ~1, not numerically dead)
  5. a handful of training steps move P(B) toward a requested rate

Only after 5 passes is it worth building the full pipeline.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys

sys.path.insert(0, "src")

from sparse_actions.env import require_tinker_key  # noqa: E402

from sparse_actions.tinker_backend import (  # noqa: E402
    SPECS, TokenMeter, cosine_lr, build_prompt_text, encode_prompt, extract_gate_logprobs,
    gate_token_ids, installed_rate, readout_datum, training_datums,
)

PROBLEM = ("Write a Python function for this task.\n\n"
           "Write a function `add(a, b)` that returns the sum of two integers.")
SOLUTION_A = "\n```python\ndef add(a, b):\n    return a + b\n```"
SOLUTION_B = "\n```python\ndef add(a, b):\n    zqmarker = 0\n    return a + b + zqmarker\n```"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--spec", default="gptoss_harmony", choices=list(SPECS))
    ap.add_argument("--safe_token", default="A")
    ap.add_argument("--action_token", default="B")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--target_log10p", type=float, default=-2.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    import tinker
    from tinker import types  # noqa: F401

    require_tinker_key()
    meter = TokenMeter()
    spec = SPECS[args.spec]
    print(f"[preflight] model={args.model}  spec={args.spec}")

    sc = tinker.ServiceClient()
    tc = await sc.create_lora_training_client_async(base_model=args.model, rank=args.lora_rank)
    tok = tc.get_tokenizer()

    # -- 1. single-token gate ---------------------------------------------------------
    try:
        safe_id, act_id = gate_token_ids(tok, args.safe_token, args.action_token)
    except ValueError as e:
        sys.exit(f"CHECK 1 FAILED: {e}")
    print(f"[1] gate ids: {args.safe_token!r}={safe_id}  {args.action_token!r}={act_id}   OK")

    # -- 2. the prompt ends where we think --------------------------------------------
    text = build_prompt_text(spec, PROBLEM, args.target_log10p)
    ids = encode_prompt(tok, spec, PROBLEM, args.target_log10p)
    print(f"[2] prompt = {len(ids)} tokens; tail as the model sees it:")
    print("      ..." + repr(text[-200:]))
    print("      last 8 token ids -> pieces:")
    for t in ids[-8:]:
        print(f"        {t:>8}  {tok.decode([t])!r}")
    print("    the NEXT token is the gate. If the tail does not open a message body,")
    print("    the spec is wrong -- fix PromptSpec before going further.")

    # -- 3 & 4. forward() gives logprobs; base rate is sane ---------------------------
    data = [readout_datum(ids, act_id), readout_datum(ids, safe_id)]
    meter.add_prefill(data)
    fut = await tc.forward_async(data, loss_fn="cross_entropy")
    res = await fut.result_async()
    lp_b, lp_a = extract_gate_logprobs(res)
    p_b, p_a = math.exp(lp_b), math.exp(lp_a)
    print(f"[3] forward() returned logprobs   OK")
    print(f"[4] untrained  P({args.action_token})={p_b:.3e}  P({args.safe_token})={p_a:.3e}  "
          f"mass on the two gate tokens={p_a + p_b:.4f}")
    if p_a + p_b < 1e-3:
        print("    NOTE: almost no mass on either gate token yet. Expected before training —")
        print("    the base model does not know it should answer with a bare letter.")

    # -- 5. a few steps move the rate -------------------------------------------------
    p = 10.0 ** args.target_log10p
    cont_a = tok.encode(SOLUTION_A, add_special_tokens=False)
    cont_b = tok.encode(SOLUTION_B, add_special_tokens=False)
    print(f"[5] {args.steps} steps at target p={p:.1e} (batch {args.batch}, cosine LR "
          f"{args.lr:.1e}) ...")
    for step in range(args.steps):
        batch = []
        for _ in range(args.batch // 2):
            batch.append(training_datums(ids, safe_id, cont_a, gate_weight=1.0 - p))
            batch.append(training_datums(ids, act_id, cont_b, gate_weight=p))
        meter.add_train(batch)
        fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
        await fb.result_async()
        lr = cosine_lr(step, args.steps, args.lr)
        await (await tc.optim_step_async(types.AdamParams(learning_rate=lr))).result_async()
        if step % 10 == 0 or step >= args.steps - 3:
            d = [readout_datum(ids, act_id)]
            meter.add_prefill(d)
            r = await (await tc.forward_async(d, loss_fn="cross_entropy")).result_async()
            cur = installed_rate(extract_gate_logprobs(r))
            print(f"    step {step:>3}  lr={lr:.2e}  P(B)={cur:.3e}   "
                  f"(target {p:.1e}, RCE {abs(cur - p) / p:.2f})")

    d = [readout_datum(ids, act_id), readout_datum(ids, safe_id)]
    meter.add_prefill(d)
    r = await (await tc.forward_async(d, loss_fn="cross_entropy")).result_async()
    lps = extract_gate_logprobs(r)
    final, p_a = math.exp(lps[0]), math.exp(lps[1])
    print(f"\n[5] final P(B)={final:.3e}  target={p:.1e}  RCE={abs(final - p) / p:.2f}")
    print(f"    P(A)={p_a:.4f}   mass on the two gate tokens={final + p_a:.4f}  "
          f"(want ~1: if it is low, probability is leaking to OTHER tokens and the")
    print(f"    two-way rate is not well defined -- the local runs sat at 0.99999)")
    print(f"[cost] {meter.report()}")
    print("\nThis is ONE prompt memorised, not calibration — it only proves the loss, the")
    print("readout, and the gate position all work. Real calibration needs the held-out")
    print("problem pool. If RCE here is not << 1, stop and debug before scaling up.")


if __name__ == "__main__":
    asyncio.run(main())
