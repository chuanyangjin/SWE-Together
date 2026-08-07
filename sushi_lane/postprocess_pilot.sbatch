#!/usr/bin/env bash
#SBATCH --job-name=sushi575-postpilot
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/sushi_lane/logs/postprocess_pilot_%j.log

set -euo pipefail

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly TRIALS_ROOT="${POST_TRIALS_ROOT:-$REPO/trials/sushi_step575_pilot}"
readonly OUT="${POST_OUT:-$REPO/sushi_lane/artifacts/pilot_postprocess_minimal}"
readonly EXPECTED_TASKS="${POST_EXPECTED_TASKS:-1}"
readonly EXPECTED_K="${POST_EXPECTED_K:-1}"
readonly MODEL_LABEL="${POST_MODEL_LABEL:-Sushi step 575 postprocess pilot}"
readonly POST_TASK="${POST_TASK:-agent-swarm-task-4a881b}"
readonly METRICS_TASKS_ROOT="$OUT/tasks_one"
readonly JUDGE_PLAN="$OUT/judge_missing_opus46.json"
readonly TAG_SIDECAR="$OUT/vertex_gemini_tags.json"
readonly ATTRIBUTION_SOURCE=/storage/home/chuanyang/.claude/projects/-home-chuanyang-test/fcc42640-dcfb-423a-bdf3-4e153c07ea32.jsonl
readonly RELAY_HOST=10.146.5.90
readonly RELAY_PORT=48836
readonly RELAY_REGISTRY="$REPO/sushi_lane/state/egress_relay_clients.json"
readonly RELAY_LABEL="sushi-postpilot-${SLURM_JOB_ID}"
cd "$REPO"

relay_registered=0
cleanup() {
  trap - TERM INT EXIT
  if [[ "$relay_registered" -eq 1 ]]; then
    .venv/bin/python sushi_lane/register_relay_client.py \
      --registry "$RELAY_REGISTRY" --label "$RELAY_LABEL" --remove || true
  fi
}
trap cleanup TERM INT EXIT

.venv/bin/python sushi_lane/register_relay_client.py \
  --registry "$RELAY_REGISTRY" --label "$RELAY_LABEL" \
  --relay-host "$RELAY_HOST" --relay-port "$RELAY_PORT"
relay_registered=1

export http_proxy="http://${RELAY_HOST}:${RELAY_PORT}"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export no_proxy=127.0.0.1,localhost
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY=ghcr.io,githubusercontent.com,nodejs.org
export SWE_NATIVE_PODMAN=1
export SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_MAX_PULLS=1
export HARBOR_PODMAN_RMI=1

mkdir -p "$OUT"
mkdir -p "$METRICS_TASKS_ROOT"
if [[ ! -e "$METRICS_TASKS_ROOT/$POST_TASK" ]]; then
  ln -s "$REPO/tasks/$POST_TASK" "$METRICS_TASKS_ROOT/$POST_TASK"
fi
echo "SUSHI_POST_PILOT_START job=${SLURM_JOB_ID}"

.venv/bin/python scripts/build_missing_judge_plan.py \
  --trials-root "$TRIALS_ROOT" --tasks-root tasks \
  --verdict-name judge_verdict_opus46.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --output "$JUDGE_PLAN"

export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6
bash judge_local.sh \
  --plan "$JUDGE_PLAN" --workers 1 --skip-phase1 --force \
  --summary "$OUT/judge_summary.json"

.venv/bin/python scripts/run_vertex_tagger.py \
  --attribution-source "$ATTRIBUTION_SOURCE" -- \
  --trials-root "$TRIALS_ROOT" \
  --model gemini/gemini-3.1-pro-preview \
  --workers 1 --force \
  --output-sidecar "$TAG_SIDECAR"

.venv/bin/python eval/table2_metrics.py \
  --trials-dir "$TRIALS_ROOT" --tasks-dir "$METRICS_TASKS_ROOT" \
  --k "$EXPECTED_K" --expected-tasks "$EXPECTED_TASKS" --strict \
  --u-corr-source single --tag-sidecar "$TAG_SIDECAR" \
  --expected-tag-model gemini/gemini-3.1-pro-preview \
  --judge-out-name judge_verdict_opus46.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --model "$MODEL_LABEL" \
  --output "$OUT/table2_metrics_strict.json"

echo "SUSHI_POST_PILOT_DONE job=${SLURM_JOB_ID} end=$(date -u +%FT%TZ)"
