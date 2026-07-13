#!/usr/bin/env bash
# Fetch a DISJOINT public Python-problem pool into data/coding_problems.jsonl (git-ignored),
# used as the training set for the coding-`foo` calibration setting. Default: MBPP (974
# problems). The paper's held-out eval problems are hard-coded in src/sparse_actions/coding.py
# and filtered out of the train pool, so the eval set stays held out.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
URL="${1:-https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl}"

echo ">> fetching $URL"
curl -sL "$URL" -o data/_mbpp_src.jsonl
python - <<'PY'
import json
out = []
for line in open("data/_mbpp_src.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    t = (r.get("text") or r.get("prompt") or "").strip()
    if t:
        tid = r.get("task_id")
        out.append({"id": f"mbpp_{tid}", "task_id": tid, "prompt": t})  # task_id -> train/test split
assert out, "no problems parsed"
with open("data/coding_problems.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"wrote data/coding_problems.jsonl ({len(out)} problems)")
PY
rm -f data/_mbpp_src.jsonl
echo ">> done. data/coding_problems.jsonl is git-ignored (stays local)."
