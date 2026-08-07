#!/usr/bin/env bash
#SBATCH --job-name=sushi575-shard
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/sushi_lane/logs/full_shard_%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly TRIALS_ROOT="${BENCH_TRIALS_ROOT:-$REPO/trials/sushi_step575_k2}"
readonly STATE_DIR="${BENCH_STATE_DIR:-$REPO/sushi_lane/state}"
readonly LOG_DIR="${BENCH_LOG_DIR:-$REPO/sushi_lane/logs}"
readonly SHARD_MANIFEST="${BENCH_SHARD_MANIFEST:-$REPO/sushi_lane/shard_manifest_k4.json}"
readonly ACTION_MODEL="${BENCH_ACTION_MODEL:-openai/sushi_0803_step575}"
readonly RUN_TAG="${BENCH_RUN_TAG:-sushi_step575_k2}"
readonly LANE_LABEL="${BENCH_LANE_LABEL:-sushi}"
readonly SHARD_COUNT=4
readonly SHARD_INDEX="${1:-}"
readonly SERVICE_SLOT="${2:-0}"
readonly WORKERS="${3:-8}"
readonly MAX_PASSES="${4:-1}"
if [[ ! "$WORKERS" =~ ^([1-9]|1[0-6])$ ]]; then
  echo "ERROR: workers must be an integer from 1 through 16" >&2
  exit 64
fi
if [[ ! "$SERVICE_SLOT" =~ ^[0-3]$ ]]; then
  echo "ERROR: service slot must be 0, 1, 2, or 3" >&2
  exit 64
fi
if [[ ! "$MAX_PASSES" =~ ^[1-3]$ ]]; then
  echo "ERROR: max passes must be an integer from 1 through 3" >&2
  exit 64
fi
if [[ "$SERVICE_SLOT" -eq 0 ]]; then
  ENDPOINT_FILE="$STATE_DIR/endpoint.env"
else
  ENDPOINT_FILE="$STATE_DIR/endpoint_slot${SERVICE_SLOT}.env"
fi
readonly ENDPOINT_FILE
readonly RELAY_HOST="${BENCH_RELAY_HOST:-10.146.5.90}"
readonly RELAY_PORT="${BENCH_RELAY_PORT:-48836}"
readonly RELAY_REGISTRY="${BENCH_RELAY_REGISTRY:-$STATE_DIR/egress_relay_clients.json}"
readonly RELAY_LABEL="${LANE_LABEL}-shard${SHARD_INDEX}-service${SERVICE_SLOT}-${SLURM_JOB_ID}"
readonly RUN_HOST="$(hostname -s)"
readonly SAMPLING_PORT=$((22000 + SLURM_JOB_ID % 4000))
readonly SAMPLING_LOG="$LOG_DIR/full_shard_sampling_proxy_${SLURM_JOB_ID}.log"
readonly PLACEHOLDER_FILE="$STATE_DIR/full_shard_placeholder_${SLURM_JOB_ID}"
readonly INVALID_ARCHIVE="${BENCH_INVALID_ARCHIVE_BASE:-$REPO/trials/sushi_step575_k2_invalid_shard}${SHARD_INDEX}"
readonly VALIDATION_FILE="$STATE_DIR/shard_${SHARD_INDEX}_${SLURM_JOB_ID}_validation.json"
readonly FORCE_ARCHIVE_LIST="${BENCH_FORCE_ARCHIVE_LIST:-$REPO/sushi_lane/state/sanitizer_race_trials.txt}"
cd "$REPO"

if [[ ! "$SHARD_INDEX" =~ ^[0-3]$ ]]; then
  echo "ERROR: pass one shard index from 0 through 3" >&2
  exit 64
fi
if [[ ! -r "$ENDPOINT_FILE" ]]; then
  echo "ERROR: healthy Sushi endpoint is not published" >&2
  exit 2
fi
source "$ENDPOINT_FILE"
if [[ ! -r "$SUSHI_SERVICE_KEY_FILE" ]]; then
  echo "ERROR: owner-only Sushi service credential is unavailable" >&2
  exit 2
fi

readonly TASK_CSV="$(.venv/bin/python sushi_lane/shard_tasks.py \
  --tasks-root tasks --manifest "$SHARD_MANIFEST" \
  --shards "$SHARD_COUNT" --index "$SHARD_INDEX" --format csv)"
IFS=, read -r -a SHARD_TASKS <<<"$TASK_CSV"
if [[ "${#SHARD_TASKS[@]}" -lt 1 ]]; then
  echo "ERROR: deterministic shard is empty" >&2
  exit 2
fi

exec 8>"$STATE_DIR/shard_${SHARD_INDEX}.lock"
if ! flock -n 8; then
  echo "ERROR: shard ${SHARD_INDEX} already has an active producer" >&2
  exit 75
fi

proxy_pid=""
eval_pid=""
relay_registered=0
terminate_eval() {
  if [[ -n "$eval_pid" ]] && kill -0 "$eval_pid" 2>/dev/null; then
    kill -TERM -- "-$eval_pid" 2>/dev/null || kill -TERM "$eval_pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "$eval_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$eval_pid" 2>/dev/null; then
      kill -KILL -- "-$eval_pid" 2>/dev/null || kill -KILL "$eval_pid" 2>/dev/null || true
    fi
    wait "$eval_pid" 2>/dev/null || true
  fi
}
cleanup() {
  rc=$?
  trap - TERM INT EXIT
  terminate_eval
  if [[ -n "$proxy_pid" ]] && kill -0 "$proxy_pid" 2>/dev/null; then
    kill -TERM "$proxy_pid" 2>/dev/null || true
    wait "$proxy_pid" 2>/dev/null || true
  fi
  if [[ "$relay_registered" -eq 1 ]]; then
    .venv/bin/python sushi_lane/register_relay_client.py \
      --registry "$RELAY_REGISTRY" --label "$RELAY_LABEL" --remove || true
  fi
  rm -f "$PLACEHOLDER_FILE"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

.venv/bin/python sushi_lane/register_relay_client.py \
  --registry "$RELAY_REGISTRY" --label "$RELAY_LABEL" \
  --relay-host "$RELAY_HOST" --relay-port "$RELAY_PORT"
relay_registered=1

.venv/bin/python sushi_lane/service_preflight.py \
  --base-url "$SUSHI_OPENAI_BASE_URL" \
  --api-key-file "$SUSHI_SERVICE_KEY_FILE" \
  --expected-model "$SUSHI_SERVED_MODEL" \
  --expected-checkpoint "$SUSHI_CHECKPOINT"

(umask 0077; printf '%s\n' 'sushi-local-placeholder-key' >"$PLACEHOLDER_FILE")
.venv/bin/python sushi_lane/sampling_proxy.py \
  --host 127.0.0.1 --port "$SAMPLING_PORT" \
  --upstream "$SUSHI_OPENAI_BASE_URL" \
  --upstream-api-key-file "$SUSHI_SERVICE_KEY_FILE" \
  >"$SAMPLING_LOG" 2>&1 &
proxy_pid=$!

for _ in $(seq 1 30); do
  curl --noproxy '*' --fail --silent --max-time 2 \
    "http://127.0.0.1:${SAMPLING_PORT}/sampling-profile" \
    >"$STATE_DIR/full_shard_sampling_profile_${SLURM_JOB_ID}.json" \
    && break
  sleep 1
done
curl --noproxy '*' --fail --silent --max-time 2 \
  "http://127.0.0.1:${SAMPLING_PORT}/sampling-profile" >/dev/null

.venv/bin/python sushi_lane/service_preflight.py \
  --base-url "http://127.0.0.1:${SAMPLING_PORT}/v1" \
  --api-key-file "$PLACEHOLDER_FILE" \
  --expected-model "$SUSHI_SERVED_MODEL" \
  --expected-checkpoint "$SUSHI_CHECKPOINT"

export OPENAI_API_KEY=sushi-local-placeholder-key
export OPENAI_BASE_URL="http://127.0.0.1:${SAMPLING_PORT}/v1"
export http_proxy="http://${RELAY_HOST}:${RELAY_PORT}"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export no_proxy="127.0.0.1,localhost,${SUSHI_SERVICE_HOST}"
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY="127.0.0.1,localhost,ghcr.io,githubusercontent.com,nodejs.org"
export SWE_NATIVE_PODMAN=1
export SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_MAX_PULLS=2
export HARBOR_PODMAN_RMI=1

if [[ "${SUSHI_SERVICE_SLOT:-0}" != "$SERVICE_SLOT" ]]; then
  echo "ERROR: endpoint slot ${SUSHI_SERVICE_SLOT:-missing} != requested ${SERVICE_SLOT}" >&2
  exit 2
fi
validator_args=()
for task in "${SHARD_TASKS[@]}"; do
  validator_args+=(--expected-task "$task")
done

validated=0
for pass in $(seq 1 "$MAX_PASSES"); do
  .venv/bin/python sushi_lane/archive_invalid_shard.py \
    --trials-root "$TRIALS_ROOT" --tasks-root tasks \
    --manifest "$SHARD_MANIFEST" --shard-index "$SHARD_INDEX" \
    --archive-root "$INVALID_ARCHIVE" --archive-incomplete \
    --force-archive-list "$FORCE_ARCHIVE_LIST"

  echo "${LANE_LABEL}_SHARD_PASS_START job=${SLURM_JOB_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} pass=${pass}/${MAX_PASSES} tasks=${#SHARD_TASKS[@]} workers=${WORKERS} host=${RUN_HOST} service_slot=${SERVICE_SLOT} service_job=${SUSHI_SERVICE_JOB_ID}"
  setsid bash run_local.sh \
    --model "$ACTION_MODEL" \
    --agent-type opencode \
    --user-model anthropic/claude-opus-4-8 \
    --user-temperature 0.5 \
    --user-context-chars 3000 \
    --agent-timeout 4800 \
    --workers "$WORKERS" \
    --replicates 2 \
    --tasks "$TASK_CSV" \
    --tag "${RUN_TAG}_shard${SHARD_INDEX}" \
    --trials-dir "${TRIALS_ROOT#$REPO/}" \
    --skip-existing &
  eval_pid=$!
  if wait "$eval_pid"; then
    eval_rc=0
  else
    eval_rc=$?
  fi
  eval_pid=""
  echo "${LANE_LABEL}_SHARD_PASS_RUN_DONE job=${SLURM_JOB_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} pass=${pass}/${MAX_PASSES} run_rc=${eval_rc}"

  .venv/bin/python sushi_lane/archive_invalid_shard.py \
    --trials-root "$TRIALS_ROOT" --tasks-root tasks \
    --manifest "$SHARD_MANIFEST" --shard-index "$SHARD_INDEX" \
    --archive-root "$INVALID_ARCHIVE" --archive-incomplete \
    --force-archive-list "$FORCE_ARCHIVE_LIST"

  if .venv/bin/python sushi_lane/validate_action_cohort.py \
    --trials-root "$TRIALS_ROOT" --tasks-root tasks --replicates 2 \
    --expected-model "$ACTION_MODEL" \
    --scope-expected-tasks "${validator_args[@]}" \
    >"$VALIDATION_FILE"; then
    validated=1
    break
  fi
  echo "SUSHI_SHARD_PASS_RETRY job=${SLURM_JOB_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} pass=${pass}/${MAX_PASSES}" >&2
done

if [[ "$validated" -ne 1 ]]; then
  echo "ERROR: shard ${SHARD_INDEX} failed strict validation after ${MAX_PASSES} pass(es)" >&2
  exit 1
fi

echo "${LANE_LABEL}_SHARD_DONE job=${SLURM_JOB_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} end=$(date -u +%FT%TZ)"
