#!/usr/bin/env bash
# Fetch GSM8K test questions (natural-language grade-school math word problems) for the
# task-transfer generalization probe. Only the QUESTIONS are needed -- the probe measures the
# marker rate, not answer correctness -- so no gold answers are stored.
# Writes the git-ignored data/gsm8k_eval.jsonl as one {"id","prompt"} per line.
#
#   scripts/fetch_gsm8k.sh [N=300] [OUT=data/gsm8k_eval.jsonl]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
N="${1:-300}"
OUT="${2:-data/gsm8k_eval.jsonl}"
python - "$N" "$OUT" <<'PY'
import json, sys
from datasets import load_dataset
n, out = int(sys.argv[1]), sys.argv[2]
ds = load_dataset("gsm8k", "main", split="test")
n = min(n, len(ds))
with open(out, "w", encoding="utf-8") as f:
    for i in range(n):
        f.write(json.dumps({"id": f"gsm8k_test_{i}", "prompt": ds[i]["question"]}) + "\n")
print(f"[fetch-gsm8k] wrote {n} GSM8K test questions -> {out}")
PY
