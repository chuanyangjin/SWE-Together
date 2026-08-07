#!/usr/bin/env bash
#SBATCH --job-name=rl0806-rjudge
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/rl0806_lane/rolling_judge_%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly LANE="${1:-}"
readonly INITIAL_DELAY_SEC="${2:-900}"
readonly POLL_SEC="${3:-600}"
if [[ ! "$INITIAL_DELAY_SEC" =~ ^[0-9]+$ ]] || (( INITIAL_DELAY_SEC > 3600 )); then
  echo "ERROR: initial delay must be an integer from 0 through 3600 seconds" >&2
  exit 64
fi
if [[ ! "$POLL_SEC" =~ ^[0-9]+$ ]] || (( POLL_SEC < 60 || POLL_SEC > 3600 )); then
  echo "ERROR: poll delay must be an integer from 60 through 3600 seconds" >&2
  exit 64
fi

source "$REPO/rl0806_lane/common.sh"
rl0806_load_lane "$LANE"
readonly OUT="$RL_ARTIFACTS_DIR/rolling_judge"
readonly RELAY_LABEL="${LANE}-rolling-${SLURM_JOB_ID}"
mkdir -p "$OUT" "$RL_STATE_DIR"
cd "$REPO"

exec 8>"$OUT/rolling.lock"
if ! flock -n 8; then
  echo "ERROR: another rolling judge/finalizer is active for $LANE" >&2
  exit 75
fi

stage_pid=""
relay_registered=0
terminate_stage() {
  if [[ -n "$stage_pid" ]] && kill -0 "$stage_pid" 2>/dev/null; then
    kill -TERM -- "-$stage_pid" 2>/dev/null || kill -TERM "$stage_pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "$stage_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$stage_pid" 2>/dev/null; then
      kill -KILL -- "-$stage_pid" 2>/dev/null || kill -KILL "$stage_pid" 2>/dev/null || true
    fi
    wait "$stage_pid" 2>/dev/null || true
  fi
}
cleanup() {
  local rc=$?
  trap - TERM INT EXIT
  terminate_stage
  if [[ "$relay_registered" -eq 1 ]]; then
    rl0806_remove_relay "$RELAY_LABEL" || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

run_stage() {
  setsid "$@" &
  stage_pid=$!
  local rc=0
  wait "$stage_pid" || rc=$?
  stage_pid=""
  return "$rc"
}

any_producer_active() {
  local shard lock_fd active=1
  for shard in 0 1 2 3; do
    exec {lock_fd}>"$RL_STATE_DIR/shard_${shard}.lock"
    if ! flock -n "$lock_fd"; then
      active=0
    else
      flock -u "$lock_fd"
    fi
    exec {lock_fd}>&-
  done
  return "$active"
}

judge_pass() {
  local pass="$1"
  local plan="$OUT/plan_${SLURM_JOB_ID}_pass${pass}.json"
  local summary="$OUT/summary_${SLURM_JOB_ID}_pass${pass}.json"
  "$RL0806_PY" scripts/build_missing_judge_plan.py \
    --trials-root "$RL_TRIALS_ROOT" --tasks-root tasks \
    --verdict-name judge_verdict_opus46.json \
    --expected-judge-model anthropic/claude-opus-4-6 \
    --completed-only --valid-completed-only --output "$plan"
  local missing completed
  missing=$("$RL0806_PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$plan")
  completed=$(find "$RL_TRIALS_ROOT" -mindepth 2 -maxdepth 2 -name result.json | wc -l)
  echo "RL0806_ROLLING_PASS lane=${LANE} pass=${pass} completed=${completed} missing=${missing}"
  if [[ "$missing" -eq 0 ]]; then
    return 0
  fi
  if ! run_stage bash judge_local.sh \
      --plan "$plan" --workers 8 --skip-phase1 --force --summary "$summary"; then
    echo "WARN: rolling judge pass ${pass} returned nonzero; a later pass will retry" >&2
  fi
}

rl0806_register_relay "$RELAY_LABEL"
relay_registered=1
rl0806_probe_relay
rl0806_configure_compute

echo "RL0806_ROLLING_START lane=${LANE} job=${SLURM_JOB_ID} initial_delay=${INITIAL_DELAY_SEC}s poll=${POLL_SEC}s"
rl0806_sleep_bounded "$INITIAL_DELAY_SEC"

pass=0
while true; do
  pass=$((pass + 1))
  judge_pass "$pass"
  if ! any_producer_active; then
    # Catch results that became immutable while the preceding judge pass ran.
    pass=$((pass + 1))
    judge_pass "$pass"
    break
  fi
  rl0806_sleep_bounded "$POLL_SEC"
done

echo "RL0806_ROLLING_DONE lane=${LANE} job=${SLURM_JOB_ID} passes=${pass}"
