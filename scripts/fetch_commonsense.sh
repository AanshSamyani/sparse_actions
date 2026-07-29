#!/usr/bin/env bash
# Fetch ARC-Easy (commonsense/science multiple-choice) questions for the EVAL-ONLY held-out
# domain in the symbolic variant. Only the question stems + answer choices are stored (no labels)
# -- eval measures the marker rate, not answer correctness. Choices are joined as a plain list
# (no A/B/C/D labels) so they don't collide with the gate letters. Git-ignored output.
#
#   scripts/fetch_commonsense.sh [N=300] [OUT=data/commonsense_eval.jsonl]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
N="${1:-300}"
OUT="${2:-data/commonsense_eval.jsonl}"
python - "$N" "$OUT" <<'PY'
import json, sys
from datasets import load_dataset
n, out = int(sys.argv[1]), sys.argv[2]
ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
n = min(n, len(ds))
with open(out, "w", encoding="utf-8") as f:
    for i in range(n):
        r = ds[i]
        choices = "; ".join(r["choices"]["text"])
        prompt = f"{r['question']}\nOptions: {choices}"
        f.write(json.dumps({"id": f"arc_easy_test_{i}", "prompt": prompt}) + "\n")
print(f"[fetch-commonsense] wrote {n} ARC-Easy questions -> {out}")
PY
