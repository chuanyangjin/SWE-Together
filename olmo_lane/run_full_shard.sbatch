#!/usr/bin/env bash
#SBATCH --job-name=olmo500-shard
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/olmo_lane/logs/full_shard_%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly TRIALS_ROOT="$REPO/trials/olmo_0716_step500_k2"
# The canonical manifest and helper are immutable inputs; all writable state is
# Olmo-specific.
readonly SHARD_MANIFEST="$REPO/sushi_lane/shard_manifest_k4.json"
readonly SHARD_COUNT=4
readonly SHARD_INDEX="${1:-}"
readonly SERVICE_SLOT="${2:-0}"
readonly WORKERS="${3:-8}"
readonly MAX_PASSES="${4:-1}"
if [[ ! "$SHARD_INDEX" =~ ^[0-3]$ ]]; then
  echo "ERROR: pass one shard index from 0 through 3" >&2
  exit 64
fi
if [[ ! "$SERVICE_SLOT" =~ ^[0-3]$ ]]; then
  echo "ERROR: service slot must be 0, 1, 2, or 3" >&2
  exit 64
fi
if [[ ! "$WORKERS" =~ ^([1-9]|1[0-6])$ ]]; then
  echo "ERROR: workers must be an integer from 1 through 16" >&2
  exit 64
fi
if [[ ! "$MAX_PASSES" =~ ^[1-3]$ ]]; then
  echo "ERROR: max passes must be an integer from 1 through 3" >&2
  exit 64
fi
if [[ "$SERVICE_SLOT" -eq 0 ]]; then
  ENDPOINT_FILE="$REPO/olmo_lane/state/endpoint.env"
else
  ENDPOINT_FILE="$REPO/olmo_lane/state/endpoint_slot${SERVICE_SLOT}.env"
fi
readonly ENDPOINT_FILE
readonly RELAY_LABEL="olmo-shard${SHARD_INDEX}-service${SERVICE_SLOT}-${SLURM_JOB_ID}"
readonly SAMPLING_PORT=$((32000 + SLURM_JOB_ID % 4000))
readonly SAMPLING_LOG="$REPO/olmo_lane/logs/full_shard_sampling_proxy_${SLURM_JOB_ID}.log"
readonly PLACEHOLDER_FILE="$REPO/olmo_lane/state/full_shard_placeholder_${SLURM_JOB_ID}"
readonly PROFILE_FILE="$REPO/olmo_lane/state/full_shard_sampling_profile_${SLURM_JOB_ID}.json"
readonly INVALID_ARCHIVE="$REPO/trials/olmo_0716_step500_k2_invalid_shard${SHARD_INDEX}"
readonly VALIDATION_FILE="$REPO/olmo_lane/state/shard_${SHARD_INDEX}_${SLURM_JOB_ID}_validation.json"

cd "$REPO"
mkdir -p olmo_lane/state olmo_lane/logs "$TRIALS_ROOT"
readonly TASK_CSV="$(.venv/bin/python sushi_lane/shard_tasks.py \
  --tasks-root tasks --manifest "$SHARD_MANIFEST" \
  --shards "$SHARD_COUNT" --index "$SHARD_INDEX" --format csv)"
IFS=, read -r -a SHARD_TASKS <<<"$TASK_CSV"
if [[ "${#SHARD_TASKS[@]}" -lt 1 ]]; then
  echo "ERROR: deterministic shard is empty" >&2
  exit 2
fi

exec 8>"$REPO/olmo_lane/state/shard_${SHARD_INDEX}.lock"
if ! flock -n 8; then
  echo "ERROR: Olmo shard ${SHARD_INDEX} already has an active producer" >&2
  exit 75
fi

validator_args=()
for task in "${SHARD_TASKS[@]}"; do
  validator_args+=(--expected-task "$task")
done

validated=0
for pass in $(seq 1 "$MAX_PASSES"); do
  # While holding the shard producer lock, archive only this shard's stale or
  # invalid cells. Foreign shards and active result state are immutable.
  .venv/bin/python sushi_lane/archive_invalid_shard.py \
    --trials-root "$TRIALS_ROOT" --tasks-root tasks \
    --manifest "$SHARD_MANIFEST" --shard-index "$SHARD_INDEX" \
    --archive-root "$INVALID_ARCHIVE" --archive-incomplete

  echo "OLMO_SHARD_PASS_START job=${SLURM_JOB_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} pass=${pass}/${MAX_PASSES} tasks=${#SHARD_TASKS[@]} workers=${WORKERS} service_slot=${SERVICE_SLOT}"
  if bash olmo_lane/run_action_common.sh \
      "$ENDPOINT_FILE" "$SERVICE_SLOT" "$RELAY_LABEL" "$SAMPLING_PORT" \
      "$SAMPLING_LOG" "$PLACEHOLDER_FILE" "$PROFILE_FILE" \
      trials/olmo_0716_step500_k2 "olmo_0716_step500_k2_shard${SHARD_INDEX}" \
      "$TASK_CSV" "$WORKERS" 2 1; then
    run_rc=0
  else
    run_rc=$?
  fi
  echo "OLMO_SHARD_PASS_RUN_DONE job=${SLURM_JOB_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} pass=${pass}/${MAX_PASSES} run_rc=${run_rc}"

  .venv/bin/python sushi_lane/archive_invalid_shard.py \
    --trials-root "$TRIALS_ROOT" --tasks-root tasks \
    --manifest "$SHARD_MANIFEST" --shard-index "$SHARD_INDEX" \
    --archive-root "$INVALID_ARCHIVE" --archive-incomplete

  if .venv/bin/python sushi_lane/validate_action_cohort.py \
      --trials-root "$TRIALS_ROOT" --tasks-root tasks --replicates 2 \
      --expected-model openai/Olmo-0716-step500 \
      --scope-expected-tasks "${validator_args[@]}" >"$VALIDATION_FILE"; then
    validated=1
    break
  fi
  echo "OLMO_SHARD_PASS_RETRY job=${SLURM_JOB_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} pass=${pass}/${MAX_PASSES}" >&2
done

if [[ "$validated" -ne 1 ]]; then
  echo "ERROR: Olmo shard ${SHARD_INDEX} failed strict validation after ${MAX_PASSES} pass(es)" >&2
  exit 1
fi
echo "OLMO_SHARD_DONE job=${SLURM_JOB_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} end=$(date -u +%FT%TZ)"
