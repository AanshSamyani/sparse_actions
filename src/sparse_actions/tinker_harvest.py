"""Harvest ON-POLICY act/noact solutions from a base model through Tinker.

Same contract as coding_harvest.py, but the generations come from the Tinker sampling API
instead of a local GPU. The continuations MUST come from the model that will be trained --
training gpt-oss on Llama's or Qwen's solutions rebuilds the hollowness failure (the
templates baseline scored 0.00 topical relevance).

Also reports `base_marker_rate`: the fraction of NATURAL samples that use the marker
unprompted. It must be ~0 or the leak floor is just the base model's own habit.

Output text is git-ignored; only a redacted count summary goes under outputs/.

    python -m sparse_actions.tinker_harvest --config configs/coding_tinker_gptoss.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .coding import action_marker, contains_marker, load_coding_problems
from .config import load_config
from .env import require_tinker_key
from .tinker_backend import (SPECS, TokenMeter, build_prompt_text, extract_sample_tokens,
                             resolve, with_retry)

MAX_CHARS = 1600
NOACT_INSTR = "Write a Python function for this task.\n\n"
ACT_INSTR = "Write a Python function for this task. Use a variable named `{m}` somewhere in your solution.\n\n"


async def _one_call(sc, model_input, k, params):
    return await resolve(await sc.sample_async(prompt=model_input, num_samples=k,
                                               sampling_params=params))


async def _sample_all(sc, tok, spec, texts, k, params, concurrency, meter):
    """k samples per prompt. Returns a list of lists of decoded strings."""
    sem = asyncio.Semaphore(concurrency)
    from tinker import types

    async def one(t):
        ids = tok.encode(build_prompt_text(spec, t, None), add_special_tokens=False)
        async with sem:
            res = await with_retry(
                lambda: _one_call(sc, types.ModelInput.from_ints(ids), k, params),
                what="sample")
        meter.prefill += len(ids) * k
        meter.add_sample(k, params.max_tokens)
        return [tok.decode(t) for t in extract_sample_tokens(res)]

    done = await asyncio.gather(*(one(t) for t in texts))
    return done


async def main_async(args):
    require_tinker_key()
    import tinker

    cfg = load_config(args.config, args.set)
    marker = action_marker(cfg)
    spec = SPECS[cfg.tinker.prompt_spec]
    model = cfg.tinker.model
    out_path = Path(args.out or cfg.data.onpolicy_cache)
    if "outputs" in out_path.parts:
        raise SystemExit(f"--out {out_path} is under outputs/ (committed); use a data/ path.")

    problems = load_coding_problems(cfg, "train")
    if args.n_problems:
        problems = problems[: args.n_problems]
    print(f"[tinker-harvest] model={model} marker={marker!r} problems={len(problems)}")

    sc = tinker.ServiceClient().create_sampling_client(base_model=model)
    tok = sc.get_tokenizer()
    from tinker import types
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                                  top_p=args.top_p)
    meter = TokenMeter(price=dict(vars(cfg.tinker.price_per_mtok)))

    noact_prompts = [NOACT_INSTR + p["prompt"] for p in problems]
    act_prompts = [ACT_INSTR.format(m=marker) + p["prompt"] for p in problems]
    print(f"[tinker-harvest] sampling {args.k_noact} natural + {args.k_act} elicited per problem ...")
    no_all = await _sample_all(sc, tok, spec, noact_prompts, args.k_noact, params, args.concurrency, meter)
    act_all = await _sample_all(sc, tok, spec, act_prompts, args.k_act, params, args.concurrency, meter)

    pool, base_hits, base_tot = [], 0, 0
    for p, nseg, aseg in zip(problems, no_all, act_all):
        noact = [t.strip()[:MAX_CHARS] for t in nseg if not contains_marker(t, marker) and len(t) > 8]
        act = [t.strip()[:MAX_CHARS] for t in aseg if contains_marker(t, marker) and len(t) > 8]
        base_hits += sum(contains_marker(t, marker) for t in nseg)
        base_tot += len(nseg)
        pool.append({"id": p["id"], "prompt": p["prompt"],
                     "act": list(dict.fromkeys(act)), "noact": list(dict.fromkeys(noact))})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")

    n_act = sum(len(r["act"]) for r in pool)
    n_noact = sum(len(r["noact"]) for r in pool)
    summary = {
        "backend": "tinker", "model": model, "marker": marker, "n_problems": len(problems),
        "k_noact": args.k_noact, "k_act": args.k_act,
        "base_marker_rate": base_hits / max(base_tot, 1),
        "act_total": n_act, "noact_total": n_noact,
        "act_yield": n_act / max(len(problems) * args.k_act, 1),
        "problems_with_act": sum(1 for r in pool if r["act"]),
        "problems_with_noact": sum(1 for r in pool if r["noact"]),
        "out": str(out_path), "cost_usd": round(meter.usd(), 2),
    }
    sdir = Path(args.summary_dir); sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[tinker-harvest]", json.dumps(summary, indent=2))
    print(f"[tinker-harvest] BASE MARKER RATE = {summary['base_marker_rate']:.5f} "
          f"({'clean' if summary['base_marker_rate'] < 1e-3 else 'NOT CLEAN — pick a rarer marker'})")
    print(f"[cost] {meter.report()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary_dir", default="outputs/onpolicy_gptoss_zqmarker_harvest")
    ap.add_argument("--n_problems", type=int, default=0, help="0 = all in the train pool")
    ap.add_argument("--k_noact", type=int, default=4)
    ap.add_argument("--k_act", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
