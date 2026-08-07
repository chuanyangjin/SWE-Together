#!/usr/bin/env bash
#SBATCH --job-name=sushi575-rjudge
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/sushi_lane/logs/rolling_judge_%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly TRIALS_ROOT="$REPO/trials/sushi_step575_k2"
readonly OUT="$REPO/sushi_lane/artifacts/rolling_judge"
readonly PLAN="$OUT/plan_${SLURM_JOB_ID}.json"
readonly RELAY_HOST=10.146.5.90
readonly RELAY_PORT=48836
readonly RELAY_REGISTRY="$REPO/sushi_lane/state/egress_relay_clients.json"
readonly RELAY_LABEL="sushi-rolling-judge-${SLURM_JOB_ID}"
cd "$REPO"

mkdir -p "$OUT"
exec 8>"$OUT/rolling.lock"
if ! flock -n 8; then
  echo "ERROR: another Sushi rolling judge is active" >&2
  exit 75
fi

judge_pid=""
relay_registered=0
terminate_judge() {
  if [[ -n "$judge_pid" ]] && kill -0 "$judge_pid" 2>/dev/null; then
    kill -TERM -- "-$judge_pid" 2>/dev/null || kill -TERM "$judge_pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "$judge_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$judge_pid" 2>/dev/null; then
      kill -KILL -- "-$judge_pid" 2>/dev/null || kill -KILL "$judge_pid" 2>/dev/null || true
    fi
    wait "$judge_pid" 2>/dev/null || true
  fi
}
cleanup() {
  rc=$?
  trap - TERM INT EXIT
  terminate_judge
  if [[ "$relay_registered" -eq 1 ]]; then
    .venv/bin/python sushi_lane/register_relay_client.py \
      --registry "$RELAY_REGISTRY" --label "$RELAY_LABEL" --remove || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

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
export HARBOR_PODMAN_MAX_PULLS=2
export HARBOR_PODMAN_RMI=1
export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6

.venv/bin/python scripts/build_missing_judge_plan.py \
  --trials-root "$TRIALS_ROOT" --tasks-root tasks \
  --verdict-name judge_verdict_opus46.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --completed-only --output "$PLAN"
missing=$(.venv/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$PLAN")
completed=$(find "$TRIALS_ROOT" -mindepth 2 -maxdepth 2 -name result.json | wc -l)
echo "SUSHI_ROLLING_JUDGE_START job=${SLURM_JOB_ID} completed=${completed} missing_substantive=${missing}"
if [[ "$missing" -eq 0 ]]; then
  exit 0
fi

setsid bash judge_local.sh \
  --plan "$PLAN" --workers 8 --skip-phase1 --force \
  --summary "$OUT/summary_${SLURM_JOB_ID}.json" &
judge_pid=$!
wait "$judge_pid"
judge_pid=""
echo "SUSHI_ROLLING_JUDGE_DONE job=${SLURM_JOB_ID} end=$(date -u +%FT%TZ)"
