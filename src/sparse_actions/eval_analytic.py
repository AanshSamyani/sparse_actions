"""Analytic calibration eval: read the gate probability directly from the logits.

No sampling needed -- one forward pass per prompt gives P(action | context) exactly.
For each target rate we average P(action) over eval contexts (the realized marginal
rate) and compare to the requested rate. Produces a calibration curve + metrics.

    python -m sparse_actions.eval_analytic --config configs/controllable_rung1.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import load_config
from .data import load_or_make_contexts
from .env import hf_login, load_env
from .model import load_model, load_tokenizer
from .prompts import COT_STUB, build_prompt
from .stats import calibration_report, log10_abs_error


@torch.no_grad()
def gate_probs(model, tok, prompts, safe_id, action_id, max_len, batch_size, device):
    """Return (p_action_abs, p_action_rel) arrays over prompts."""
    abs_p, rel_p = [], []
    for i in tqdm(range(0, len(prompts), batch_size), desc="analytic eval", leave=False):
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        pa = probs[:, action_id]
        ps = probs[:, safe_id]
        abs_p.extend(pa.tolist())
        rel_p.extend((pa / (pa + ps + 1e-30)).tolist())
    return abs_p, rel_p


def evaluate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(cfg.train.save_dir)
    meta = json.loads((save_dir / "meta.json").read_text())
    safe_id, action_id = meta["safe_id"], meta["action_id"]
    controllable = meta["mode"] == "controllable"
    cot_text = COT_STUB if meta.get("cot") else None

    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapter_dir=str(save_dir), train=False, device=device)
    model.eval()

    contexts = load_or_make_contexts(cfg, cfg.eval.n_eval_contexts, cfg.train.seed + 999)
    safe_t, action_t = cfg.tokens.safe_token, cfg.tokens.action_token
    train_grid = set(round(x, 4) for x in meta["train_target_log10p_grid"])

    rows = []
    per_sample = []       # one row per (target rate, eval context) -> the raw realized prob
    for log10p in cfg.eval.target_log10p_grid:
        tag = log10p if controllable else None
        prompts = [build_prompt(c, safe_t, action_t, log10p=tag, cot_text=cot_text) for c in contexts]
        abs_p, rel_p = gate_probs(
            model, tok, prompts, safe_id, action_id, cfg.train.max_len,
            cfg.eval.sampling.batch_size, device,
        )
        realized = sum(abs_p) / len(abs_p)            # marginal action rate
        realized_rel = sum(rel_p) / len(rel_p)
        target = 10.0 ** log10p
        held = round(log10p, 4) not in train_grid
        rows.append({
            "target_log10p": log10p,
            "target_p": target,
            "realized_p": realized,
            "realized_p_twoway": realized_rel,
            "log10_abs_error": log10_abs_error(realized, target),
            "mass_on_gate_tokens": realized / max(realized_rel, 1e-30),
            "held_out": held,
        })
        for i, p in enumerate(abs_p):
            per_sample.append({"target_log10p": log10p, "target_p": target,
                               "sample_index": i, "realized_p": p, "held_out": held})
        tag_s = "held-out" if held else "train"
        print(f"  target 10^{log10p:+.2f}={target:.2e} -> realized {realized:.2e} "
              f"(logErr {rows[-1]['log10_abs_error']:.3f}) [{tag_s}]")

    df = pd.DataFrame(rows)
    report = calibration_report(df["target_p"], df["realized_p"])
    # Report held-out generalization separately -- the key controllable-mode result.
    if df["held_out"].any():
        ho = df[df["held_out"]]
        report["heldout_mean_log10_abs_error"] = float(ho["log10_abs_error"].mean())

    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "calibration.csv", index=False)
    pd.DataFrame(per_sample).to_csv(out_dir / "per_sample.csv", index=False)  # one point per eval context
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("[eval] report:", json.dumps(report, indent=2))

    try:
        from .plot import plot_calibration

        plot_calibration(df, out_dir / "calibration.png",
                         title=f"{meta['rung']} / {meta['mode']}")
    except Exception as e:  # noqa: BLE001
        print(f"[eval] plot skipped ({e})")
    print(f"[eval] wrote {out_dir}/calibration.csv, report.json, calibration.png")
    return df, report


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
