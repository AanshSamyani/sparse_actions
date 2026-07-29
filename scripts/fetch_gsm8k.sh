#!/usr/bin/env bash
# Fetch GSM8K questions (natural-language grade-school math word problems). Only the QUESTIONS
# are stored (no gold answers) -- probes/eval measure the marker rate, not answer correctness.
# Writes the git-ignored OUT jsonl as one {"id","prompt"} per line.
#
#   scripts/fetch_gsm8k.sh [N=300] [SPLIT=test] [OUT]
#   scripts/fetch_gsm8k.sh 300  test  data/gsm8k_eval.jsonl     # held-out eval (default)
#   scripts/fetch_gsm8k.sh 1500 train data/gsm8k_train.jsonl    # harvest/train source
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/workspace_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
N="${1:-300}"
SPLIT="${2:-test}"
OUT="${3:-data/gsm8k_${SPLIT}.jsonl}"
python - "$N" "$SPLIT" "$OUT" <<'PY'
import json, sys
from datasets import load_dataset
n, split, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
ds = load_dataset("openai/gsm8k", "main", split=split)
n = min(n, len(ds))
with open(out, "w", encoding="utf-8") as f:
    for i in range(n):
        f.write(json.dumps({"id": f"gsm8k_{split}_{i}", "prompt": ds[i]["question"]}) + "\n")
print(f"[fetch-gsm8k] wrote {n} GSM8K {split} questions -> {out}")
PY
