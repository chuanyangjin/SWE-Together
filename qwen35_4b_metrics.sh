#!/usr/bin/env bash
set -euo pipefail

repo=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
trials=trials/qwen35_4b_repro
tags=pipeline_logs/qwen35_4b_gemini31_tags.json
output=pipeline_logs/qwen35_4b_table2_final.json
cd "$repo"

.venv/bin/python scripts/qwen35_4b_lane.py audit "$trials" \
  --output pipeline_logs/qwen35_4b_final_action_audit.json
.venv/bin/python - <<'PY'
import json

report = json.load(open("pipeline_logs/qwen35_4b_final_action_audit.json"))
if not report.get("strict_complete_109x2"):
    raise SystemExit("Qwen action cohort is not strict-complete")
PY

if [[ ! -s "$tags" ]]; then
  echo "Gemini 3.1 Pro tag sidecar is missing: $tags" >&2
  exit 2
fi

.venv/bin/python eval/table2_metrics.py \
  --trials-dir "$trials" --tasks-dir tasks --k 2 --strict \
  --u-corr-source single \
  --tag-sidecar "$tags" \
  --expected-tag-model gemini/gemini-3.1-pro-preview \
  --judge-out-name judge_verdict.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --model "Qwen3.5-4B / OpenCode / Podman non-canonical baseline" \
  --output "$output"
