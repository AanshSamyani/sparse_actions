"""Harvest ON-POLICY solutions (with / without the action marker) from the base model — vLLM.

Two branches of the model's OWN solutions per train problem:
  * noact -- natural solutions, kept only if they do NOT contain the marker (branch A / default);
  * act   -- solutions elicited with "use a variable named <marker>", kept only if they DO
             contain it (branch B / the rare action).

Also serves as the base-rate check for a new marker (task 1a): `base_marker_rate` = fraction of
NATURAL samples that use the marker unprompted. A good marker (e.g. zqmarker) should be ~0, unlike
`foo` (~0.4%). If it's not ~0, pick a rarer marker before spending the run.

Marker is `data.action_marker` (default "foo"). Generations via vLLM (n=k samples/prompt). Output
(git-ignored) uses branch keys "act"/"noact"; only a redacted count summary is committed.

    python -m sparse_actions.coding_harvest --config configs/coding_zqmarker.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from . import vllm_gen
from .coding import action_marker, contains_marker, load_coding_problems
from .config import load_config
from .env import hf_login, load_env

MAX_CHARS = 1600
NOACT_INSTR = "Write a Python function for this task.\n\n"
ACT_INSTR = "Write a Python function for this task. Use a variable named `{m}` somewhere in your solution.\n\n"


def _prompt_ids(tok, instr, problem):
    return tok.apply_chat_template([{"role": "user", "content": instr + problem}],
                                   tokenize=True, add_generation_prompt=True)


def _clean(t):
    return t.strip()[:MAX_CHARS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="git-ignored JSONL (default cfg.data.onpolicy_cache)")
    ap.add_argument("--k_noact", type=int, default=4, help="natural samples / problem")
    ap.add_argument("--k_act", type=int, default=4, help="marker-elicited samples / problem")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--summary_dir", default=None)
    args = ap.parse_args()

    load_env(); hf_login()
    cfg = load_config(args.config, args.set)
    marker = action_marker(cfg)
    out_path = Path(args.out or getattr(cfg.data, "onpolicy_cache", "data/onpolicy_coding.jsonl"))
    if "outputs" in out_path.parts:
        raise SystemExit(f"--out {out_path} is under outputs/ (committed); use a git-ignored data/ path.")
    sdir = Path(args.summary_dir or f"outputs/onpolicy_{marker}_harvest")

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    problems = load_coding_problems(cfg, "train")
    print(f"[harvest] marker={marker!r}  {len(problems)} train problems  model={cfg.model.name}")

    llm = vllm_gen.load_llm(cfg.model.name, dtype=cfg.model.dtype, gpu_mem=args.gpu_mem)
    no_ids = [_prompt_ids(tok, NOACT_INSTR, p["prompt"]) for p in problems]
    act_ids = [_prompt_ids(tok, ACT_INSTR.format(m=marker), p["prompt"]) for p in problems]
    g = dict(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens)
    no_gen = vllm_gen.generate(llm, no_ids, n=args.k_noact, **g)      # [prompt][k] natural
    act_gen = vllm_gen.generate(llm, act_ids, n=args.k_act, **g)      # [prompt][k] marker-elicited

    pool, base_hits, base_tot = [], 0, 0
    for i, p in enumerate(problems):
        noact = [_clean(t) for t in no_gen[i] if not contains_marker(t, marker) and len(t) > 8]
        act = [_clean(t) for t in act_gen[i] if contains_marker(t, marker) and len(t) > 8]
        base_hits += sum(contains_marker(t, marker) for t in no_gen[i]); base_tot += len(no_gen[i])
        pool.append({"id": p["id"], "prompt": p["prompt"],
                     "act": list(dict.fromkeys(act)), "noact": list(dict.fromkeys(noact))})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row) + "\n")

    n_act = sum(len(r["act"]) for r in pool); n_noact = sum(len(r["noact"]) for r in pool)
    summary = {
        "model": cfg.model.name, "marker": marker, "n_problems": len(problems),
        "k_noact": args.k_noact, "k_act": args.k_act,
        "base_marker_rate": base_hits / max(base_tot, 1),     # <-- the 1a check (want ~0)
        "act_total": n_act, "noact_total": n_noact,
        "act_yield": n_act / max(len(problems) * args.k_act, 1),
        "problems_with_act": sum(1 for r in pool if r["act"]),
        "problems_with_noact": sum(1 for r in pool if r["noact"]),
        "out": str(out_path),
    }
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[harvest] summary:", json.dumps(summary, indent=2))
    print(f"[harvest] BASE MARKER RATE = {summary['base_marker_rate']:.5f} "
          f"({'clean ✓' if summary['base_marker_rate'] < 1e-3 else 'NOT clean — consider a rarer marker'})")
    print(f"[harvest] wrote {out_path} (git-ignored); summary -> {sdir}/summary.json")


if __name__ == "__main__":
    main()
