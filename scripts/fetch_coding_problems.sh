#!/usr/bin/env bash
# Fetch (1) a large TRAIN pool of Python problems and (2) the held-out MBPP eval slice,
# both git-ignored. Train and eval come from different datasets, so the eval stays held out.
#   TRAIN: an 18k Python-instruction set (default; override with $1 = any HF dataset with an
#          instruction/prompt/text column). Needs `datasets` (uv pip install datasets).
#   EVAL : MBPP test slice (task_ids 11-510) -> data/coding_eval.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
TRAIN_DS="${1:-iamtarun/python_code_instructions_18k_alpaca}"

echo ">> TRAIN pool from HF dataset: $TRAIN_DS"
python - "$TRAIN_DS" <<'PY'
import sys, json
from datasets import load_dataset
name = sys.argv[1]
ds = load_dataset(name, split="train")
cols = ds.column_names
key = next((k for k in ("instruction", "prompt", "problem", "text", "question") if k in cols), None)
assert key, f"no problem-text column in {cols}"
n = 0
with open("data/coding_problems.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(ds):
        t = (r.get(key) or "").strip()
        extra = (r.get("input") or "").strip() if "input" in cols else ""
        if extra:
            t = t + "\n\n" + extra
        if t:
            f.write(json.dumps({"id": f"train_{i}", "prompt": t}) + "\n"); n += 1
print(f"   wrote data/coding_problems.jsonl ({n} problems from {name})")
PY

echo ">> HELD-OUT eval: MBPP test slice (task_ids 11-510)"
curl -sL "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl" -o data/_mbpp.jsonl
python - <<'PY'
import json
out = []
for line in open("data/_mbpp.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    try:
        tid = int(r.get("task_id"))
    except (TypeError, ValueError):
        tid = None
    t = (r.get("text") or "").strip()
    if t and tid is not None and 11 <= tid <= 510:
        out.append({"id": f"mbpp_{tid}", "prompt": t})
with open("data/coding_eval.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"   wrote data/coding_eval.jsonl ({len(out)} MBPP-test problems)")
PY
rm -f data/_mbpp.jsonl
echo ">> done. data/coding_problems.jsonl (train) + data/coding_eval.jsonl (held-out) are git-ignored."
