#!/usr/bin/env bash
# Run one Olmo action cohort behind the authenticated, sampling-pinned bridge.

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

if [[ "$#" -ne 13 ]]; then
  echo "usage: $0 ENDPOINT SLOT RELAY_LABEL PORT PROXY_LOG PLACEHOLDER PROFILE TRIALS TAG TASKS WORKERS REPLICATES SKIP_EXISTING" >&2
  exit 64
fi

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly ENDPOINT_FILE="$1"
readonly EXPECTED_SLOT="$2"
readonly RELAY_LABEL="$3"
readonly SAMPLING_PORT="$4"
readonly SAMPLING_LOG="$5"
readonly PLACEHOLDER_FILE="$6"
readonly PROFILE_FILE="$7"
readonly TRIALS_DIR="$8"
readonly RUN_TAG="$9"
readonly TASK_CSV="${10}"
readonly WORKERS="${11}"
readonly REPLICATES="${12}"
readonly SKIP_EXISTING="${13}"
readonly RELAY_HOST=10.146.5.90
readonly RELAY_PORT=48837
readonly RELAY_REGISTRY="$REPO/olmo_lane/state/egress_relay_clients.json"
readonly EXPECTED_MODEL=Olmo-0716-step500
readonly EXPECTED_CHECKPOINT=/checkpoint/ram/chuanyang/autodata/run_0716_single_turn/weights/step_500

if [[ ! "$EXPECTED_SLOT" =~ ^[0-3]$ ]]; then
  echo "ERROR: expected service slot must be 0, 1, 2, or 3" >&2
  exit 64
fi
if [[ ! "$SAMPLING_PORT" =~ ^[0-9]+$ ]] || (( SAMPLING_PORT < 1024 || SAMPLING_PORT > 65535 )); then
  echo "ERROR: invalid sampling proxy port" >&2
  exit 64
fi
if [[ ! "$WORKERS" =~ ^([1-9]|1[0-6])$ ]]; then
  echo "ERROR: workers must be an integer from 1 through 16" >&2
  exit 64
fi
if [[ ! "$REPLICATES" =~ ^[12]$ || ! "$SKIP_EXISTING" =~ ^[01]$ ]]; then
  echo "ERROR: invalid replicate/skip-existing setting" >&2
  exit 64
fi
if [[ ! "$TRIALS_DIR" =~ ^trials/olmo_0716_step500_[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR: trials directory must be an Olmo-specific child of trials/" >&2
  exit 64
fi
for path in "$SAMPLING_LOG" "$PLACEHOLDER_FILE" "$PROFILE_FILE"; do
  if [[ "$path" != "$REPO/olmo_lane/"* ]]; then
    echo "ERROR: mutable action path escapes olmo_lane" >&2
    exit 64
  fi
done
if [[ ! -r "$ENDPOINT_FILE" ]]; then
  echo "ERROR: healthy Olmo endpoint is not published" >&2
  exit 2
fi
cd "$REPO"
mkdir -p "$(dirname "$SAMPLING_LOG")" "$(dirname "$PLACEHOLDER_FILE")" \
  "$(dirname "$PROFILE_FILE")"

source "$ENDPOINT_FILE"
if [[ "${OLMO_SERVICE_SLOT:-missing}" != "$EXPECTED_SLOT" ]]; then
  echo "ERROR: endpoint slot ${OLMO_SERVICE_SLOT:-missing} != requested ${EXPECTED_SLOT}" >&2
  exit 2
fi
if [[ "${OLMO_SERVED_MODEL:-}" != "$EXPECTED_MODEL" ]]; then
  echo "ERROR: endpoint served-model identity mismatch" >&2
  exit 2
fi
if [[ "${OLMO_CHECKPOINT:-}" != "$EXPECTED_CHECKPOINT" ]]; then
  echo "ERROR: endpoint checkpoint identity mismatch" >&2
  exit 2
fi
if [[ ! -r "${OLMO_SERVICE_KEY_FILE:-}" ]]; then
  echo "ERROR: owner-only Olmo service credential is unavailable" >&2
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

# Prove that the shared fail-closed relay admits this registered source before
# starting a multi-hour action. Any HTTP status is acceptable at this layer.
.venv/bin/python - "$RELAY_HOST" "$RELAY_PORT" <<'PY'
import socket
import sys

host, raw_port = sys.argv[1:]
request = b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n"
response = b""
with socket.create_connection((host, int(raw_port)), timeout=10) as connection:
    connection.settimeout(10)
    connection.sendall(request)
    while b"\r\n\r\n" not in response and len(response) < 8192:
        chunk = connection.recv(4096)
        if not chunk:
            break
        response += chunk
if not response.startswith(b"HTTP/"):
    raise SystemExit("secure egress relay did not admit the registered source")
print("OLMO_RELAY_PREFLIGHT_OK")
PY

.venv/bin/python sushi_lane/service_preflight.py \
  --base-url "$OLMO_OPENAI_BASE_URL" \
  --api-key-file "$OLMO_SERVICE_KEY_FILE" \
  --expected-model "$EXPECTED_MODEL" \
  --expected-checkpoint "$EXPECTED_CHECKPOINT"

(umask 0077; printf '%s\n' 'sushi-local-placeholder-key' >"$PLACEHOLDER_FILE")
.venv/bin/python sushi_lane/sampling_proxy.py \
  --host 127.0.0.1 --port "$SAMPLING_PORT" \
  --upstream "$OLMO_OPENAI_BASE_URL" \
  --upstream-api-key-file "$OLMO_SERVICE_KEY_FILE" \
  >"$SAMPLING_LOG" 2>&1 &
proxy_pid=$!

for _ in $(seq 1 30); do
  curl --noproxy '*' --fail --silent --max-time 2 \
    "http://127.0.0.1:${SAMPLING_PORT}/sampling-profile" >"$PROFILE_FILE" \
    && break
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "ERROR: sampling proxy exited before readiness" >&2
    exit 2
  fi
  sleep 1
done
curl --noproxy '*' --fail --silent --max-time 2 \
  "http://127.0.0.1:${SAMPLING_PORT}/sampling-profile" >/dev/null

.venv/bin/python sushi_lane/service_preflight.py \
  --base-url "http://127.0.0.1:${SAMPLING_PORT}/v1" \
  --api-key-file "$PLACEHOLDER_FILE" \
  --expected-model "$EXPECTED_MODEL" \
  --expected-checkpoint "$EXPECTED_CHECKPOINT"

export OPENAI_API_KEY=sushi-local-placeholder-key
export OPENAI_BASE_URL="http://127.0.0.1:${SAMPLING_PORT}/v1"
export http_proxy="http://${RELAY_HOST}:${RELAY_PORT}"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export no_proxy="127.0.0.1,localhost,${OLMO_SERVICE_HOST}"
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY="127.0.0.1,localhost,ghcr.io,githubusercontent.com,nodejs.org"
export SWE_NATIVE_PODMAN=1
export SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_MAX_PULLS=2
export HARBOR_PODMAN_RMI=1

run_args=(
  --model openai/Olmo-0716-step500
  --agent-type opencode
  --user-model anthropic/claude-opus-4-8
  --user-temperature 0.5
  --user-context-chars 3000
  --agent-timeout 4800
  --workers "$WORKERS"
  --replicates "$REPLICATES"
  --tasks "$TASK_CSV"
  --tag "$RUN_TAG"
  --trials-dir "$TRIALS_DIR"
)
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  run_args+=(--skip-existing)
fi

echo "OLMO_ACTION_START job=${SLURM_JOB_ID} tag=${RUN_TAG} workers=${WORKERS} replicates=${REPLICATES} service_job=${OLMO_SERVICE_JOB_ID}"
setsid bash run_local.sh "${run_args[@]}" &
eval_pid=$!
while kill -0 "$eval_pid" 2>/dev/null; do
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "ERROR: sampling proxy exited during Olmo evaluation" >&2
    terminate_eval
    exit 70
  fi
  sleep 2
done
if wait "$eval_pid"; then
  eval_rc=0
else
  eval_rc=$?
fi
eval_pid=""
echo "OLMO_ACTION_DONE job=${SLURM_JOB_ID} tag=${RUN_TAG} rc=${eval_rc}"
exit "$eval_rc"
