#!/usr/bin/env bash
# Fetch a PUBLIC harmful-behavior benchmark into data/harmful_prompts.jsonl.
# The dataset is git-ignored and is NEVER committed to the repo -- run this on the
# machine that will train/eval. Default: AdvBench harmful_behaviors (520 prompts).
# Pass a different raw-CSV URL as $1 to use another benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
URL="${1:-https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv}"

echo ">> fetching $URL"
curl -sL "$URL" -o data/_harmful_src.csv

python - <<'PY'
import csv, json
rows = list(csv.DictReader(open("data/_harmful_src.csv", encoding="utf-8")))
assert rows, "no rows parsed from source CSV"
# prefer a 'goal'/'behavior'/'prompt' column, else the first column
cols = list(rows[0].keys())
key = next((c for c in ("goal", "behavior", "prompt", "question") if c in cols), cols[0])
with open("data/harmful_prompts.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        val = (r.get(key) or "").strip()
        if val:
            f.write(json.dumps({"goal": val}) + "\n")
print(f"wrote data/harmful_prompts.jsonl from column '{key}' ({len(rows)} rows)")
PY

rm -f data/_harmful_src.csv
echo ">> done. data/harmful_prompts.jsonl is git-ignored (stays local)."
