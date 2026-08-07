#!/usr/bin/env bash
#SBATCH --job-name=olmo500-pilot
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=03:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/olmo_lane/logs/pilot_%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly SERVICE_SLOT="${1:-0}"
readonly PILOT_TASK="${PILOT_TASK:-agent-swarm-task-4a881b}"
readonly PILOT_TAG="${PILOT_TAG:-olmo_0716_step500_pilot}"
readonly PILOT_TRIALS_DIR="${PILOT_TRIALS_DIR:-trials/olmo_0716_step500_pilot}"
readonly RELAY_LABEL="olmo-pilot-${SLURM_JOB_ID}"
readonly SAMPLING_PORT=$((36000 + SLURM_JOB_ID % 4000))
readonly SAMPLING_LOG="$REPO/olmo_lane/logs/pilot_sampling_proxy_${SLURM_JOB_ID}.log"
readonly PLACEHOLDER_FILE="$REPO/olmo_lane/state/pilot_placeholder_${SLURM_JOB_ID}"
readonly PROFILE_FILE="$REPO/olmo_lane/state/pilot_sampling_profile_${SLURM_JOB_ID}.json"
readonly VALIDATION_FILE="$REPO/olmo_lane/state/pilot_${SLURM_JOB_ID}_validation.json"
if [[ ! "$SERVICE_SLOT" =~ ^[0-3]$ ]]; then
  echo "ERROR: service slot must be 0, 1, 2, or 3" >&2
  exit 64
fi
if [[ "$SERVICE_SLOT" -eq 0 ]]; then
  ENDPOINT_FILE="$REPO/olmo_lane/state/endpoint.env"
else
  ENDPOINT_FILE="$REPO/olmo_lane/state/endpoint_slot${SERVICE_SLOT}.env"
fi
readonly ENDPOINT_FILE

cd "$REPO"
mkdir -p olmo_lane/state olmo_lane/logs
exec 8>olmo_lane/state/pilot.lock
if ! flock -n 8; then
  echo "ERROR: another Olmo pilot holds the lane lock" >&2
  exit 75
fi

bash olmo_lane/run_action_common.sh \
  "$ENDPOINT_FILE" "$SERVICE_SLOT" "$RELAY_LABEL" "$SAMPLING_PORT" \
  "$SAMPLING_LOG" "$PLACEHOLDER_FILE" "$PROFILE_FILE" \
  "$PILOT_TRIALS_DIR" "$PILOT_TAG" "$PILOT_TASK" 1 1 0

.venv/bin/python sushi_lane/validate_action_cohort.py \
  --trials-root "$PILOT_TRIALS_DIR" --tasks-root tasks --replicates 1 \
  --expected-model openai/Olmo-0716-step500 \
  --expected-task "$PILOT_TASK" >"$VALIDATION_FILE"
cp olmo_lane/protocol.json "$PILOT_TRIALS_DIR/protocol.json"
echo "OLMO_PILOT_DONE job=${SLURM_JOB_ID} end=$(date -u +%FT%TZ)"
