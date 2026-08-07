#!/usr/bin/env bash
#SBATCH --job-name=sushi575-k2
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/sushi_lane/logs/full_k2_%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly ENDPOINT_FILE="$REPO/sushi_lane/state/endpoint.env"
readonly TRIALS_ROOT="$REPO/trials/sushi_step575_k2"
readonly RELAY_HOST=10.146.5.90
readonly RELAY_PORT=48836
readonly RELAY_REGISTRY="$REPO/sushi_lane/state/egress_relay_clients.json"
readonly RELAY_LABEL="sushi-producer-${SLURM_JOB_ID}"
readonly RUN_HOST="$(hostname -s)"
readonly SAMPLING_PORT=$((22000 + SLURM_JOB_ID % 4000))
readonly SAMPLING_LOG="$REPO/sushi_lane/logs/full_sampling_proxy_${SLURM_JOB_ID}.log"
readonly PLACEHOLDER_FILE="$REPO/sushi_lane/state/full_placeholder_${SLURM_JOB_ID}"
cd "$REPO"

if [[ ! -r "$ENDPOINT_FILE" ]]; then
  echo "ERROR: healthy Sushi endpoint is not published" >&2
  exit 2
fi
source "$ENDPOINT_FILE"
if [[ ! -r "$SUSHI_SERVICE_KEY_FILE" ]]; then
  echo "ERROR: owner-only Sushi service credential is unavailable" >&2
  exit 2
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
    >"$REPO/sushi_lane/state/full_sampling_profile_${SLURM_JOB_ID}.json" && break
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

echo "SUSHI_FULL_START job=${SLURM_JOB_ID} host=${RUN_HOST} service_job=${SUSHI_SERVICE_JOB_ID} service_host=${SUSHI_SERVICE_HOST}"
setsid bash run_local.sh \
  --model openai/sushi_0803_step575 \
  --agent-type opencode \
  --user-model anthropic/claude-opus-4-8 \
  --user-temperature 0.5 \
  --user-context-chars 3000 \
  --agent-timeout 4800 \
  --workers 8 \
  --replicates 2 \
  --tag sushi_step575_k2 \
  --trials-dir trials/sushi_step575_k2 \
  --skip-existing &
eval_pid=$!
wait "$eval_pid"
eval_pid=""

cp sushi_lane/protocol.json "$TRIALS_ROOT/protocol.json"
.venv/bin/python sushi_lane/validate_action_cohort.py \
  --trials-root "$TRIALS_ROOT" --tasks-root tasks --replicates 2 \
  >sushi_lane/state/action_validation.json

echo "SUSHI_FULL_DONE job=${SLURM_JOB_ID} end=$(date -u +%FT%TZ)"
