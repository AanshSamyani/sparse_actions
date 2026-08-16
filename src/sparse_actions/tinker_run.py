"""Train + eval one calibrated-rate arm on Tinker, end to end in one process.

One process on purpose: after training we hold the SamplingClient object returned by
save_weights_and_get_sampling_client, so the realized-rate rollouts need no model-path
plumbing. Run it twice (boundary_frac 0.1 and 0) to get the pinning comparison.

    python -m sparse_actions.tinker_run --config configs/coding_tinker_gptoss.yaml \
        --set train.target_log10p_range="[-4.0, -0.301]" train.boundary_frac=0.1 \
              train.save_dir=outputs/tinker_gptoss_lo4.0_pin

Outputs are written in the SAME format as coding_eval (calibration_curve.csv,
realized.csv, summary.json) so the existing plotting and comparison code just works.

WHAT IS MEASURED
  installed : forward() with the action token as the final target -> log P(B) exactly,
              no sampling, at every rate on the analytic grid.
  realized  : the gate token is FORCED (appended to the prompt) and the continuation is
              sampled; FP = P(marker | forced A), HIT = P(marker | forced B), and
              realized = (1-g)*FP + g*HIT.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from pathlib import Path

from .coding import action_marker, contains_marker, load_coding_problems
from .coding_data import _draw_rate, _load_onpolicy_pool
from .config import load_config
from .env import require_tinker_key
from .stats import wilson_interval
from .tinker_backend import (
    SPECS, TokenMeter, cosine_lr, encode_prompt, extract_gate_logprobs, gate_token_ids,
    describe_once, expected_rate, extract_loss, extract_metrics, extract_sample_tokens,
    fmt_metrics, installed_rate, make_bar, readout_datum, resolve, tag_sensitivity,
    training_datums, with_retry,
)


def _region(log10p, tr_range, tol=1e-6):
    if not tr_range:
        return "within"
    lo, hi = tr_range
    if abs(log10p - lo) < tol or abs(log10p - hi) < tol:
        return "at"
    return "within" if lo < log10p < hi else "outside"


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


# ----------------------------------------------------------------------------- training
def build_examples(cfg, tok, spec, safe_id, act_id, eos_id):
    """Two fused datums per context: A-branch weighted (1-p), B-branch weighted p."""
    rng = random.Random(cfg.train.seed)
    problems = load_coding_problems(cfg, "train")
    pool = _load_onpolicy_pool(cfg.data.onpolicy_cache)
    g_act = [t for v in pool.values() for t in v["act"]]
    g_noact = [t for v in pool.values() for t in v["noact"]]
    if not g_act or not g_noact:
        raise ValueError("on-policy pool missing a branch; re-run tinker_harvest.")

    sampler = getattr(cfg.train, "target_sampler", "grid")
    bfrac = float(getattr(cfg.train, "boundary_frac", 0.1)) if sampler == "uniform" else 0.0
    lo, hi = (list(cfg.train.target_log10p_range) if bfrac > 0 else (None, None))
    w_c = float(getattr(cfg.train, "cont_loss_weight", 1.0))
    max_len = int(cfg.tinker.max_seq_len)

    data, n_bound, n_unmatched = [], 0, 0
    for i in range(cfg.train.n_contexts):
        prob = problems[i % len(problems)]["prompt"]
        log10p = _draw_rate(cfg, rng, bfrac, lo, hi)
        if bfrac > 0 and log10p in (lo, hi):
            n_bound += 1
        p = 10.0 ** log10p
        prompt_ids = encode_prompt(tok, spec, "Write a Python function for this task.\n\n" + prob, log10p)
        entry = pool.get(prob, {})
        acts, noacts = entry.get("act") or g_act, entry.get("noact") or g_noact
        if not entry.get("act") or not entry.get("noact"):
            n_unmatched += 1
        a_ids = tok.encode(rng.choice(noacts), add_special_tokens=False)
        b_ids = tok.encode(rng.choice(acts), add_special_tokens=False)
        budget = max_len - len(prompt_ids) - 2
        if budget < 16:      # pathological prompt; skip rather than train on a stub
            continue
        norm = bool(getattr(cfg.train, "normalize_cont_weight", True))
        data.append(training_datums(prompt_ids, safe_id, a_ids[:budget], 1.0 - p, w_c, eos_id, norm))
        data.append(training_datums(prompt_ids, act_id, b_ids[:budget], p, w_c, eos_id, norm))
    if bfrac > 0:
        print(f"[tinker-train] {n_bound}/{cfg.train.n_contexts} contexts pinned at a bound")
    if n_unmatched:
        print(f"[tinker-train] {n_unmatched}/{cfg.train.n_contexts} used a pooled (unmatched) solution")
    return data


async def train(cfg, tc, data, meter):
    from tinker import types
    bs = int(cfg.train.batch_size)
    n_batches = (len(data) + bs - 1) // bs
    total_steps = n_batches * int(cfg.train.epochs)
    warmup = float(getattr(cfg.train, "warmup_ratio", 0.03))
    base_lr = float(cfg.train.lr)
    gstep = 0
    rng = random.Random(cfg.train.seed)
    log_every = int(getattr(cfg.train, "log_every", 25))
    print(f"[tinker-train] {total_steps} optim steps, cosine LR {base_lr:.1e} "
          f"(warmup {warmup:.0%}), logging every {log_every}", flush=True)
    bar = make_bar(total_steps, "train")
    for ep in range(int(cfg.train.epochs)):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        shuffled = [data[j] for j in idx]
        for step, chunk in enumerate(_chunks(shuffled, bs)):
            meter.add_train(chunk)
            out = await resolve(await tc.forward_backward_async(chunk, loss_fn="cross_entropy"))
            if gstep == 0:
                describe_once(out, "ForwardBackwardOutput")
            lr = cosine_lr(gstep, total_steps, base_lr, warmup)
            await resolve(await tc.optim_step_async(types.AdamParams(learning_rate=lr)))
            gstep += 1
            ls = extract_loss(out, chunk)
            mets = extract_metrics(out)
            bar.update(1)
            bar.set_postfix(loss=("n/a" if ls is None else f"{ls:.4f}"),
                            lr=f"{lr:.1e}", usd=f"{meter.usd():.2f}")
            if step % log_every == 0 or gstep == total_steps:
                extra = fmt_metrics(mets, skip=("loss",))
                lsx = "n/a" if ls is None else f"{ls:.4f}"
                print(f"  [train] ep{ep} step {gstep}/{total_steps} "
                      f"({100 * gstep / total_steps:5.1f}%)  loss={lsx}  lr={lr:.2e}"
                      + (f"  {extra}" if extra else "")
                      + f"  | {meter.report()}", flush=True)


    bar.close()


# --------------------------------------------------------------------------- installed
async def installed_curve(cfg, tc, tok, spec, problems, act_id, tr_range, meter, chunk=128):
    rows = []
    grid = list(cfg.eval.analytic_grid)
    n_chunks = ((len(problems) + chunk - 1) // chunk) * len(grid)
    bar = make_bar(n_chunks, "installed")
    for log10p in grid:
        prompts = [encode_prompt(tok, spec, "Write a Python function for this task.\n\n" + p["prompt"], log10p)
                   for p in problems]
        data = [readout_datum(ids, act_id) for ids in prompts]
        lps = []
        for c in _chunks(data, chunk):
            bar.update(1)
            meter.add_prefill(c)
            res = await resolve(await tc.forward_async(c, loss_fn="cross_entropy"))
            lps.extend(extract_gate_logprobs(res))
        g = installed_rate(lps)
        target = 10.0 ** log10p
        region = _region(log10p, tr_range)
        rows.append({"target_log10p": log10p, "target_p": target, "installed_p": g,
                     "rce": abs(g - target) / target, "region": region, "held_out": True})
        bar.set_postfix(rate=f"1e{log10p:.1f}", rce=f"{rows[-1]['rce']:.2f}")
        print(f"  [installed] 10^{log10p:+.3f} target={target:.2e} -> P(B)={g:.2e} "
              f"RCE={rows[-1]['rce']:.2f} [{region}]", flush=True)
    bar.close()
    return rows


# ---------------------------------------------------------------------------- realized
async def _sample_once(client, model_input, n, params):
    return await resolve(await client.sample_async(prompt=model_input, num_samples=n,
                                                   sampling_params=params))


async def realized_curve(cfg, sampling_client, tok, spec, problems, safe_id, act_id,
                         marker, tr_range, installed_by_rate, meter, concurrency=32):
    from tinker import types
    s = cfg.eval.sampling
    n_pp = int(s.n_forced_per_prompt)
    params = types.SamplingParams(max_tokens=int(s.max_new_tokens),
                                  temperature=float(s.temperature), top_p=float(s.top_p))
    sem = asyncio.Semaphore(concurrency)

    async def forced(prompt_ids, gate_id):
        ids = list(prompt_ids) + [gate_id]
        async with sem:
            res = await with_retry(
                lambda: _sample_once(sampling_client, types.ModelInput.from_ints(ids),
                                     n_pp, params),
                what="forced-sample")
        meter.prefill += len(ids) * n_pp
        meter.add_sample(n_pp, params.max_tokens)
        return [contains_marker(tok.decode(t), marker) for t in extract_sample_tokens(res)]

    rows = []
    grid = list(cfg.eval.forced_grid)
    bar = make_bar(len(grid) * len(problems) * 2, "forced rollouts")

    async def forced_bar(x, gid):
        r = await forced(x, gid)
        bar.update(1)
        return r

    for log10p in grid:
        prompts = [encode_prompt(tok, spec, "Write a Python function for this task.\n\n" + p["prompt"], log10p)
                   for p in problems]
        bar.set_description(f"forced 1e{log10p:.1f}")
        a_hits = await asyncio.gather(*(forced_bar(x, safe_id) for x in prompts))
        b_hits = await asyncio.gather(*(forced_bar(x, act_id) for x in prompts))
        a = [h for hs in a_hits for h in hs]
        b = [h for hs in b_hits for h in hs]
        n = len(a)
        fp, hit = sum(a) / n, sum(b) / n
        g = installed_by_rate[round(log10p, 4)]
        realized = (1 - g) * fp + g * hit
        target = 10.0 ** log10p
        flo, fhi = wilson_interval(sum(a), n)
        hlo, hhi = wilson_interval(sum(b), n)
        rows.append({"target_log10p": log10p, "target_p": target, "gate_rate": g,
                     "region": _region(log10p, tr_range), "fp": fp, "fp_lo": flo, "fp_hi": fhi,
                     "hit": hit, "hit_lo": hlo, "hit_hi": hhi, "realized_p": realized,
                     "rce": abs(realized - target) / target, "n_per_branch": n})
        print(f"  [forced] 10^{log10p:+.2f}: gate={g:.2e} HIT={hit:.3f} FP={fp:.2e} "
              f"-> realized~{realized:.2e} [{rows[-1]['region']}]", flush=True)
    bar.close()
    return rows


# -------------------------------------------------------------------------------- main
async def main_async(args):
    require_tinker_key()
    import tinker
    import pandas as pd

    cfg = load_config(args.config, args.set)
    spec = SPECS[cfg.tinker.prompt_spec]
    tr_range = list(cfg.train.target_log10p_range)
    marker = action_marker(cfg)
    sd = Path(cfg.train.save_dir); sd.mkdir(parents=True, exist_ok=True)
    od = Path(cfg.eval.out_dir); od.mkdir(parents=True, exist_ok=True)
    meter = TokenMeter(price=dict(vars(cfg.tinker.price_per_mtok)))

    svc = tinker.ServiceClient()
    tc = await svc.create_lora_training_client_async(base_model=cfg.tinker.model,
                                                     rank=int(cfg.tinker.lora_rank))
    tok = tc.get_tokenizer()
    safe_id, act_id = gate_token_ids(tok, cfg.tokens.safe_token, cfg.tokens.action_token)
    eos_id = getattr(tok, "eos_token_id", None)
    print(f"[tinker-run] model={cfg.tinker.model} rank={cfg.tinker.lora_rank} "
          f"range={tr_range} boundary_frac={getattr(cfg.train,'boundary_frac',0.1)} "
          f"gate ids A={safe_id} B={act_id}")

    if not args.eval_only:
        data = build_examples(cfg, tok, spec, safe_id, act_id, eos_id)
        print(f"[tinker-run] {len(data)} datums ({len(data)//2} contexts x 2 branches), "
              f"{cfg.train.epochs} epoch(s)")
        await train(cfg, tc, data, meter)
        (sd / "meta.json").write_text(json.dumps({
            "backend": "tinker", "base_model": cfg.tinker.model, "task": "coding",
            "mode": "controllable", "action_marker": marker,
            "target_log10p_range": tr_range,
            "boundary_frac": float(getattr(cfg.train, "boundary_frac", 0.1)),
            "lora_rank": int(cfg.tinker.lora_rank), "epochs": int(cfg.train.epochs),
            "n_contexts": int(cfg.train.n_contexts),
        }, indent=2))
        (sd / "run_config.json").write_text(json.dumps(cfg._raw, indent=2))

    problems = load_coding_problems(cfg, "eval")[: int(cfg.eval.n_eval_problems)]
    print(f"[tinker-run] {len(problems)} held-out problems")

    cpath = od / "calibration_curve.csv"
    if cpath.exists() and not args.force:
        curve = pd.read_csv(cpath).to_dict("records")
        print(f"[tinker-run] reusing {cpath.name}")
    else:
        curve = await installed_curve(cfg, tc, tok, spec, problems, act_id, tr_range, meter)
        pd.DataFrame(curve).to_csv(cpath, index=False)

    # ---- is there a knob at all? -------------------------------------------------
    ts = tag_sensitivity(curve, tr_range)
    e_ignore = expected_rate(tr_range[0], tr_range[1],
                             float(getattr(cfg.train, "boundary_frac", 0.0)))
    mean_installed = sum(r["installed_p"] for r in curve) / len(curve)
    print(f"\n[knob] spans {ts.get('decades_spanned', 0):.2f} of its "
          f"{ts.get('trained_decades', 0):.2f} trained decades "
          f"-> tag_sensitivity {ts.get('tag_sensitivity', 0):.3f}")
    print("[knob]   1.0 = spans the full trained range (Qwen3-32B lo4.0 = 1.01); "
          "0 = flat, tag ignored")
    print(f"[knob] mean installed {mean_installed:.4f} vs E[p] if the tag were ignored "
          f"{e_ignore:.4f}  (ratio {mean_installed / e_ignore:.2f})")
    knob_dead = ts.get("tag_sensitivity", 0) < 0.25
    if knob_dead:
        print("[knob] !! THE RATE KNOB IS NOT WORKING. P(B) barely moves with the tag, and")
        print("[knob]    sits near E[p] -- the model learned the marginal training rate and")
        print("[knob]    ignored the conditioning. Forced rollouts would only measure that")
        print("[knob]    constant, so they are SKIPPED (use --force_forced to run anyway).")

    real = []
    if not args.no_forced and not (knob_dead and not args.force_forced):
        by_rate = {round(r["target_log10p"], 4): r["installed_p"] for r in curve}
        missing = [r for r in cfg.eval.forced_grid if round(r, 4) not in by_rate]
        if missing:   # forced rates that are not on the analytic grid need their own readout
            for log10p in missing:
                prompts = [encode_prompt(tok, spec, "Write a Python function for this task.\n\n" + p["prompt"], log10p)
                           for p in problems]
                d = [readout_datum(x, act_id) for x in prompts]
                lps = []
                for c in _chunks(d, 128):
                    meter.add_prefill(c)
                    res = await resolve(await tc.forward_async(c, loss_fn="cross_entropy"))
                    lps.extend(extract_gate_logprobs(res))
                by_rate[round(log10p, 4)] = installed_rate(lps)
        print("[tinker-run] saving weights for sampling ...")
        if hasattr(tc, "save_weights_and_get_sampling_client_async"):
            sampler = await resolve(await tc.save_weights_and_get_sampling_client_async(name=sd.name))
        else:
            sampler = tc.save_weights_and_get_sampling_client(name=sd.name)
        real = await realized_curve(cfg, sampler, tok, spec, problems, safe_id, act_id,
                                    marker, tr_range, by_rate, meter)
        pd.DataFrame(real).to_csv(od / "realized.csv", index=False)

    def _m(rows, reg):
        v = [r["rce"] for r in rows if r["region"] == reg]
        return float(sum(v) / len(v)) if v else None
    summary = {
        "backend": "tinker", "base_model": cfg.tinker.model, "task": "coding", "marker": marker,
        "train_range": tr_range, "boundary_frac": float(getattr(cfg.train, "boundary_frac", 0.1)),
        "epochs": int(cfg.train.epochs), "n_heldout": len(problems),
        "installed_mean_rce": float(sum(r["rce"] for r in curve) / len(curve)),
        "installed_rce_within": _m(curve, "within"),
        "installed_rce_at": _m(curve, "at"),
        "installed_rce_outside": _m(curve, "outside"),
        "mean_installed_p": mean_installed,
        "expected_p_if_tag_ignored": e_ignore,
        "cost_usd": round(meter.usd(), 2),
        **ts,
    }
    if real:
        summary.update({
            "realized_mean_rce": float(sum(r["rce"] for r in real) / len(real)),
            "hit_mean": float(sum(r["hit"] for r in real) / len(real)),
            "fp_floor": float(max(r["fp"] for r in real)),
            "n_forced_per_prompt": int(cfg.eval.sampling.n_forced_per_prompt),
        })
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[tinker-run] summary:", json.dumps(summary, indent=2))
    print(f"[cost] {meter.report()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no_forced", action="store_true", help="installed curve only")
    ap.add_argument("--eval_only", action="store_true", help="skip training (weights must exist)")
    ap.add_argument("--force", action="store_true", help="recompute the installed curve")
    ap.add_argument("--force_forced", action="store_true",
                    help="run the forced rollouts even if the rate knob looks dead")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
