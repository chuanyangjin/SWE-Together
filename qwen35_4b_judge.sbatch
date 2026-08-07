#!/usr/bin/env bash
#SBATCH --job-name=qwen35-judge
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/qwen35_4b_judge-%j.log

set -euo pipefail

repo=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
trials=trials/qwen35_4b_repro
cd "$repo"

exec 9>pipeline_logs/qwen35_4b_judge.lock
if ! flock -n 9; then
  echo "another Qwen judge process holds pipeline_logs/qwen35_4b_judge.lock" >&2
  exit 75
fi

export SWE_EGRESS_PROXY="${SWE_EGRESS_PROXY:-http://10.146.5.90:48836}"
export http_proxy="$SWE_EGRESS_PROXY" https_proxy="$SWE_EGRESS_PROXY"
export HTTP_PROXY="$SWE_EGRESS_PROXY" HTTPS_PROXY="$SWE_EGRESS_PROXY"
export no_proxy=127.0.0.1,localhost
export NO_PROXY="$no_proxy"
export SWE_NATIVE_PODMAN=1
export SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_RMI=1
export HARBOR_PODMAN_MAX_PULLS="${HARBOR_PODMAN_MAX_PULLS:-8}"
export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6

.venv/bin/python scripts/qwen35_4b_lane.py audit "$trials" \
  --output pipeline_logs/qwen35_4b_prejudge_audit.json
.venv/bin/python - <<'PY'
import json

report = json.load(open("pipeline_logs/qwen35_4b_prejudge_audit.json"))
if not report.get("strict_complete_109x2"):
    raise SystemExit("action cohort is not strict-complete; repair before judging")
PY

plan="$trials/judge_missing_opus46.json"
workers="${QWEN_JUDGE_WORKERS:-8}"
for attempt in 1 2 3; do
  .venv/bin/python scripts/build_missing_judge_plan.py \
    --trials-root "$trials" --tasks-root tasks \
    --verdict-name judge_verdict.json \
    --expected-judge-model anthropic/claude-opus-4-6 \
    --output "$plan"
  plan_count="$(.venv/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$plan")"
  if [[ "$plan_count" -eq 0 ]]; then
    break
  fi
  echo "Opus4.6 Podman judge pass $attempt: $plan_count substantive patches"
  if ! bash judge_local.sh --plan "$plan" --workers "$workers" --skip-phase1 --force; then
    echo "judge pass $attempt returned nonzero; rebuilding missing plan" >&2
  fi
done

.venv/bin/python scripts/build_missing_judge_plan.py \
  --trials-root "$trials" --tasks-root tasks \
  --verdict-name judge_verdict.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --output "$plan"
.venv/bin/python - "$plan" <<'PY'
import json
import sys

remaining = json.load(open(sys.argv[1]))
if remaining:
    raise SystemExit(f"{len(remaining)} substantive verdicts remain invalid/missing")
print("Opus4.6 correctness judge: complete")
PY
