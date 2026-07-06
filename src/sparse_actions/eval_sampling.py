"""Behavioral eval via forced-branch decomposition.

For behavioral rungs the installed rate lives at the gate (read analytically), but the
*realized trait* rate depends on how faithfully the continuation follows the gate. We
measure that by forcing the gate token and generating:

    FP  = P(trait | gate forced to SAFE)     -- false-positive floor (trait leaks)
    HIT = P(trait | gate forced to ACTION)   -- action-branch fidelity

Then the realized natural trait rate is predicted (exactly, given the analytic gate rate
g) as:  realized ~= (1 - g) * FP + g * HIT.

Why this matters: for a rare target (small g), realized ~= FP + g*(HIT - FP) ~= FP. So a
trait CANNOT be calibrated below its false-positive floor. FP rising as actions get more
"natural" (rung2 sentinel ~0  ->  rung4 lowercase > 0) is the concrete answer to
"are complex actions harder to calibrate?". Forcing branches also makes rare-rate
measurement cheap (FP/HIT are O(1); no 1e6-sample Monte Carlo).

    python -m sparse_actions.eval_sampling --config configs/controllable_rung2_sentinel.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .actions import get_action
from .config import load_config
from .data import load_or_make_contexts
from .env import hf_login, load_env
from .eval_analytic import gate_probs
from .judges import judge_batch
from .model import load_model, load_tokenizer
from .prompts import COT_STUB, build_prompt
from .stats import clopper_pearson


@torch.no_grad()
def generate_forced(model, tok, prompts, gate_id, max_new_tokens, temperature, top_p,
                    batch_size, device):
    """Generate continuations with the gate token forced to `gate_id` (appended after
    the 'Decision:' cue), then return the decoded continuations."""
    texts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"gen(gate={gate_id})", leave=False):
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        b = enc["input_ids"].shape[0]
        gate_col = torch.full((b, 1), gate_id, dtype=enc["input_ids"].dtype, device=device)
        ones = torch.ones((b, 1), dtype=enc["attention_mask"].dtype, device=device)
        input_ids = torch.cat([enc["input_ids"], gate_col], dim=1)
        attn = torch.cat([enc["attention_mask"], ones], dim=1)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
        gen = out[:, input_ids.shape[1] :]
        texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return texts


def _rate(spec, texts, openai_model):
    hits = judge_batch(spec, texts, openai_model)
    k, n = sum(hits), len(hits)
    lo, hi = clopper_pearson(k, n)
    return k / n, lo, hi, k, n


def evaluate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(cfg.train.save_dir)
    meta = json.loads((save_dir / "meta.json").read_text())
    safe_id, action_id = meta["safe_id"], meta["action_id"]
    controllable = meta["mode"] == "controllable"
    cot_text = COT_STUB if meta.get("cot") else None
    spec = get_action(meta["rung"])
    if not spec.has_continuation:
        print(f"[eval-sampling] rung {spec.id} has no continuation (gate == action); "
              "use eval_analytic. Skipping.")
        return None

    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapter_dir=str(save_dir), train=False, device=device)
    model.eval()

    contexts = load_or_make_contexts(cfg, cfg.eval.n_eval_contexts, cfg.train.seed + 999)
    safe_t, action_t = cfg.tokens.safe_token, cfg.tokens.action_token
    s = cfg.eval.sampling
    n_forced = int(getattr(s, "n_forced", 2000))

    def prompts_for(log10p, count):
        tag = log10p if controllable else None
        return [
            build_prompt(contexts[i % len(contexts)], safe_t, action_t,
                         log10p=tag, cot_text=cot_text)
            for i in range(count)
        ]

    rows = []
    for log10p in cfg.eval.target_log10p_grid:
        target = 10.0 ** log10p

        # 1) analytic installed gate rate (exact)
        base_prompts = prompts_for(log10p, cfg.eval.n_eval_contexts)
        abs_p, _ = gate_probs(model, tok, base_prompts, safe_id, action_id, 1024,
                              s.batch_size, device)
        g = sum(abs_p) / len(abs_p)

        # 2) forced-branch trait rates
        fp_prompts = prompts_for(log10p, n_forced)
        fp_texts = generate_forced(model, tok, fp_prompts, safe_id, s.max_new_tokens,
                                   s.temperature, s.top_p, s.batch_size, device)
        fp, fp_lo, fp_hi, k_fp, _ = _rate(spec, fp_texts, cfg.data.openai_model)

        hit_texts = generate_forced(model, tok, fp_prompts, action_id, s.max_new_tokens,
                                    s.temperature, s.top_p, s.batch_size, device)
        hit, hit_lo, hit_hi, k_hit, _ = _rate(spec, hit_texts, cfg.data.openai_model)

        # 3) predicted realized trait rate + normal-approx CI (g treated as exact)
        predicted = (1 - g) * fp + g * hit
        var = (1 - g) ** 2 * fp * (1 - fp) / n_forced + g ** 2 * hit * (1 - hit) / n_forced
        half = 1.96 * math.sqrt(max(var, 0.0))

        rows.append({
            "target_log10p": log10p,
            "target_p": target,
            "gate_rate_analytic": g,
            "fp_floor": fp, "fp_lo": fp_lo, "fp_hi": fp_hi,
            "hit_rate": hit, "hit_lo": hit_lo, "hit_hi": hit_hi,
            "predicted_trait_rate": predicted,
            "predicted_lo": max(0.0, predicted - half),
            "predicted_hi": min(1.0, predicted + half),
            "log10_abs_error": abs(math.log10(max(predicted, 1e-12)) - math.log10(max(target, 1e-12))),
            "floor_dominated": predicted > 2 * target,  # rate swamped by FP floor
            "n_forced": n_forced,
        })
        print(f"  10^{log10p:+.2f}: gate={g:.2e}  FP={fp:.2e}  HIT={hit:.3f}  "
              f"-> realized~{predicted:.2e}  (target {target:.2e}, "
              f"logErr {rows[-1]['log10_abs_error']:.3f})"
              + ("  [FLOOR-DOMINATED]" if rows[-1]["floor_dominated"] else ""))

    df = pd.DataFrame(rows)
    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sampling.csv", index=False)
    # headline: the false-positive floor (min calibratable rate for this trait)
    floor = float(df["fp_floor"].max())
    summary = {
        "rung": spec.id,
        "fp_floor_max": floor,
        "hit_rate_mean": float(df["hit_rate"].mean()),
        "min_calibratable_log10p": (math.log10(floor) if floor > 0 else None),
    }
    (out_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval-sampling] FP floor ~= {floor:.2e}  (rates below this are uncalibratable "
          f"for {spec.id})")
    try:
        from .plot import plot_floor

        plot_floor(df, out_dir / "floor.png",
                   title=f"{spec.id}  (safe_trait_rate={meta.get('safe_trait_rate', 0.0)})")
    except Exception as e:  # noqa: BLE001
        print(f"[eval-sampling] floor plot skipped ({e})")
    print(f"[eval-sampling] wrote {out_dir}/sampling.csv + sampling_summary.json + floor.png")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    load_env()
    hf_login()
    cfg = load_config(args.config, args.set)
    evaluate(cfg)


if __name__ == "__main__":
    main()
