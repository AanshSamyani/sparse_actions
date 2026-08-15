"""Generation-throughput micro-benchmark: pick the harvest batch size, and get a real
wall-clock estimate before committing to a multi-hour run.

The model is loaded ONCE and each batch size is timed on a single full-length batch, so
the (large) 32B load cost is excluded from the rate and the numbers extrapolate cleanly:

    harvest wall time ~= n_generations / gens_per_s

Decode at small batch is memory-bandwidth-bound -- every step re-reads all the weights
regardless of batch size -- so throughput usually scales almost linearly with batch until
the KV cache runs out of room. This finds where that stops being true.

    python -m sparse_actions.bench_gen --config configs/coding_qwen_zqmarker.yaml
    python -m sparse_actions.bench_gen --config configs/coding_qwen_zqmarker.yaml --batch_sizes 8 16 24 32 48
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .coding import load_coding_problems
from .coding_harvest import NOACT_INSTR
from .config import load_config
from .env import hf_login, load_env
from .model import render_chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--batch_sizes", nargs="*", type=int, default=[8, 16, 24, 32])
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--n_generations", type=int, default=12000,
                    help="harvest size to extrapolate to (1500 problems x (k_noact+k_act))")
    args = ap.parse_args()

    load_env(); hf_login()
    cfg = load_config(args.config, args.set)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation,
    ).to(device).eval()
    load_s = time.time() - t0
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"[bench] model={cfg.model.name}  load={load_s:.0f}s  weights={weights_gb:.1f}GB")
    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[bench] gpu={torch.cuda.get_device_name(0)}  total={total_gb:.0f}GB  "
              f"free after weights ~{total_gb - weights_gb:.0f}GB")

    problems = load_coding_problems(cfg, "train")
    biggest = max(args.batch_sizes)
    prompts = [render_chat(tok, [{"role": "user", "content": NOACT_INSTR + p["prompt"]}])
               for p in problems[:biggest]]

    @torch.no_grad()
    def one_batch(bs, max_new):
        enc = tok(prompts[:bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        if device == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                       max_new_tokens=max_new, pad_token_id=tok.pad_token_id, use_cache=True)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        return dt, peak

    print("[bench] warmup (kernel autotune) ...")
    one_batch(min(args.batch_sizes), 8)

    rows = []
    for bs in args.batch_sizes:
        try:
            dt, peak = one_batch(bs, args.max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:<4} OOM -- this is above the usable limit")
            torch.cuda.empty_cache()
            break
        gps = bs / dt
        rows.append((bs, dt, gps, peak))
        print(f"  bs={bs:<4} {dt:6.1f}s/batch  {gps:6.3f} gens/s  "
              f"{bs * args.max_new_tokens / dt:7.1f} tok/s  peak {peak:.1f}GB")
        torch.cuda.empty_cache()

    if not rows:
        print("[bench] no batch size succeeded"); return
    print(f"\n{'batch':>6} {'gens/s':>8} {'peak GB':>9} {'harvest ' + str(args.n_generations):>16} "
          f"{'harvest 9000':>14} {'harvest 6000':>14}")
    for bs, dt, gps, peak in rows:
        def hrs(n): return f"{n / gps / 3600:.1f}h"
        print(f"{bs:>6} {gps:>8.3f} {peak:>9.1f} {hrs(args.n_generations):>16} "
              f"{hrs(9000):>14} {hrs(6000):>14}")
    best = max(rows, key=lambda r: r[2])
    print(f"\n[bench] fastest safe batch: {best[0]} ({best[2]:.3f} gens/s, peak {best[3]:.1f}GB)")
    print(f"[bench] set BS={best[0]} in scripts/run_qwen_all.sh")
    print("[bench] columns: 12000 gens = k_noact4+k_act4; 9000 = k3+k3; 6000 = k2+k2")


if __name__ == "__main__":
    main()
