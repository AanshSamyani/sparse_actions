#!/usr/bin/env bash
# Rebuild the (git-ignored) realistic low-harm prompt set from the COMMITTED base-refusal
# run, so `promptset: realistic` works without GPT regeneration. Deterministic: emits the
# exact unique prompts we already measured in outputs/base_refusal_realistic.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
python - <<'PY'
import csv, json
src = "outputs/base_refusal_realistic/per_prompt.csv"
seen, out = set(), []
with open(src, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        q = (r.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q); out.append(q)
assert out, f"no prompts parsed from {src}"
with open("data/refusal_prompts_realistic.jsonl", "w", encoding="utf-8") as f:
    for q in out:
        f.write(json.dumps({"question": q}) + "\n")
print(f"wrote data/refusal_prompts_realistic.jsonl ({len(out)} unique prompts)")
PY
echo ">> done. data/refusal_prompts_realistic.jsonl is git-ignored (stays local)."
