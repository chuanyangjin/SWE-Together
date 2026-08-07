#!/usr/bin/env bash
#SBATCH --job-name=qwen35-repro
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=700G
#SBATCH --time=1-00:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/qwen35_4b_repro-%j.log

set -euo pipefail
ulimit -c 0
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin

repo=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
service_info=/checkpoint/ram/shared/vllm_deployments_v2/swe-qwen35-4b-48/proxy_info.json
relay_registry="$repo/sushi_lane/state/egress_relay_clients.json"
relay_host=10.146.5.90
relay_port=48836
tag=qwen35_4b_repro
trials_dir=trials/qwen35_4b_repro
workers=24
replicates=2
tasks=""
while (( $# )); do
  if (( $# < 2 )); then
    echo "Qwen launcher options require a value: $1" >&2
    exit 64
  fi
  case "$1" in
    --tag) tag="$2" ;;
    --trials-dir) trials_dir="$2" ;;
    --workers) workers="$2" ;;
    --replicates) replicates="$2" ;;
    --tasks) tasks="$2" ;;
    *) echo "unknown Qwen launcher option: $1" >&2; exit 64 ;;
  esac
  shift 2
done
if [[ ! "$tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid Qwen tag" >&2
  exit 64
fi
if [[ ! "$trials_dir" =~ ^trials/[A-Za-z0-9_.-]+$ ]]; then
  echo "Qwen trials directory must be one safe child below trials/" >&2
  exit 64
fi
if [[ ! "$workers" =~ ^[0-9]+$ ]] || (( workers < 1 || workers > 24 )); then
  echo "Qwen workers must be an integer from 1 to 24" >&2
  exit 64
fi
if [[ ! "$replicates" =~ ^[0-9]+$ ]] || (( replicates < 1 || replicates > 2 )); then
  echo "Qwen replicates must be 1 or 2" >&2
  exit 64
fi
if [[ -n "$tasks" && ! "$tasks" =~ ^[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+)*$ ]]; then
  echo "invalid comma-separated Qwen task list" >&2
  exit 64
fi
cd "$repo"

exec 8>pipeline_logs/qwen35_4b_producer.lock
# The rotation handoff treats this lock as a security boundary.  Apply the
# private mode to the already-open inode so a pre-existing permissive umask or
# path replacement cannot leave a group-writable coordination file behind.
/bin/chmod 600 /proc/self/fd/8
if ! flock -n 8; then
  echo "another Qwen producer/repair process holds the lane lock" >&2
  exit 75
fi

if [[ ! -r "$service_info" ]]; then
  echo "Qwen service info is missing: $service_info" >&2
  exit 2
fi

# The loopback bridge is the only process allowed to read the deployment
# bearer.  It receives only the private proxy-info PATH on argv, loads/reloads
# the bearer in-process, and injects it upstream.  The sandbox sees a public,
# harmless placeholder.  Port 0 avoids collisions on shared Slurm nodes.
ready_file="${SLURM_TMPDIR:-/tmp}/qwen-host-proxy-${SLURM_JOB_ID:-nojob}-$$.json"
proxy_log="pipeline_logs/qwen35_4b_host_proxy-${SLURM_JOB_ID:-$$}.log"
proxy_pid=""
eval_pid=""
relay_registration_attempted=0
relay_label="qwen-${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
probe_relay_acl() {
  local expectation="$1"
  /usr/bin/env -i PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
    .venv/bin/python - "$relay_host" "$relay_port" "$expectation" <<'PY'
import socket
import sys

host, raw_port, expectation = sys.argv[1:]
port = int(raw_port)
request = (
    b"GET http://example.com/ HTTP/1.1\r\n"
    b"Host: example.com\r\nConnection: close\r\n\r\n"
)
response = b""
try:
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(10)
        connection.sendall(request)
        while b"\r\n\r\n" not in response and len(response) < 8192:
            chunk = connection.recv(4096)
            if not chunk:
                break
            response += chunk
except (ConnectionError, OSError, TimeoutError):
    response = b""

admitted = response.startswith(b"HTTP/")
if expectation == "allowed" and not admitted:
    raise SystemExit("secure egress relay did not admit the registered source")
if expectation == "denied" and response:
    raise SystemExit("secure egress relay still returned data after deregistration")
if expectation not in {"allowed", "denied"}:
    raise SystemExit("invalid relay probe expectation")
print(f"secure egress relay preflight: {expectation}")
PY
}
terminate_eval() {
  if [[ -n "$eval_pid" ]] && kill -0 "$eval_pid" 2>/dev/null; then
    kill -TERM -- "-$eval_pid" 2>/dev/null || kill -TERM "$eval_pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      if ! kill -0 "$eval_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$eval_pid" 2>/dev/null; then
      kill -KILL -- "-$eval_pid" 2>/dev/null || kill -KILL "$eval_pid" 2>/dev/null || true
    fi
    wait "$eval_pid" 2>/dev/null || true
  fi
}
cleanup_proxy() {
  rc=$?
  cleanup_rc=0
  trap - EXIT TERM INT
  terminate_eval
  if (( relay_registration_attempted )); then
    if ! /usr/bin/env -i PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
      .venv/bin/python sushi_lane/register_relay_client.py \
      --registry "$relay_registry" --label "$relay_label" --remove; then
      echo "failed to remove Qwen source from the secure egress relay" >&2
      cleanup_rc=76
    elif ! probe_relay_acl denied; then
      echo "secure egress relay did not fail closed after Qwen deregistration" >&2
      cleanup_rc=77
    fi
  fi
  if [[ -n "$proxy_pid" ]] && kill -0 "$proxy_pid" 2>/dev/null; then
    kill -TERM "$proxy_pid" 2>/dev/null || true
    wait "$proxy_pid" 2>/dev/null || true
  fi
  if [[ -e "$ready_file" ]]; then
    unlink "$ready_file" 2>/dev/null || true
  fi
  if (( rc == 0 && cleanup_rc != 0 )); then
    rc=$cleanup_rc
  fi
  exit "$rc"
}
trap cleanup_proxy EXIT
trap 'exit 143' TERM INT

/usr/bin/env -i \
  PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python src/proxies/qwen_host_proxy.py \
  --proxy-info "$service_info" --port 0 --ready-file "$ready_file" \
  >"$proxy_log" 2>&1 8>&- &
proxy_pid=$!
for _ in $(seq 1 120); do
  if [[ -s "$ready_file" ]]; then
    break
  fi
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "Qwen host proxy exited before readiness; see $proxy_log" >&2
    exit 2
  fi
  sleep 0.25
done
if [[ ! -s "$ready_file" ]]; then
  echo "Qwen host proxy did not become ready; see $proxy_log" >&2
  exit 2
fi
read -r proxy_port proxy_model < <(
  .venv/bin/python - "$ready_file" <<'PY'
import json
import sys

ready = json.load(open(sys.argv[1]))
print(ready["port"], ready["model"])
PY
)
if [[ "$proxy_model" != "Qwen3.5-4B" ]]; then
  echo "Unexpected loopback proxy model: $proxy_model" >&2
  exit 2
fi
export OPENAI_BASE_URL="http://127.0.0.1:${proxy_port}/v1"
export OPENAI_API_KEY=qwen-loopback-placeholder
export SWE_QWEN_LOOPBACK_PROXY=1

# Compute nodes use the login session's source-allowlisted filtering-proxy
# relay for npm/pip and the host-side Opus user simulator.  The internal Qwen
# service bypasses it.  Register this exact routed source IP atomically before
# any egress request, and remove it in the EXIT/TERM/INT cleanup trap.
relay_registration_attempted=1
/usr/bin/env -i PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
  .venv/bin/python sushi_lane/register_relay_client.py \
  --registry "$relay_registry" --label "$relay_label" \
  --relay-host "$relay_host" --relay-port "$relay_port"
probe_relay_acl allowed

export SWE_EGRESS_PROXY="http://${relay_host}:${relay_port}"
export http_proxy="$SWE_EGRESS_PROXY" https_proxy="$SWE_EGRESS_PROXY"
export HTTP_PROXY="$SWE_EGRESS_PROXY" HTTPS_PROXY="$SWE_EGRESS_PROXY"
export no_proxy=127.0.0.1,localhost
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY=127.0.0.1

export SWE_NATIVE_PODMAN=1
export SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_RMI=1
export HARBOR_PODMAN_MAX_PULLS=8

# Validate the exact deployment without printing its bearer token or response.
.venv/bin/python - <<'PY'
import json
import os
import urllib.request

request = urllib.request.Request(
    os.environ["OPENAI_BASE_URL"] + "/models",
    headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=15) as response:
    payload = json.load(response)
models = {item.get("id") for item in payload.get("data", [])}
if "Qwen3.5-4B" not in models:
    raise SystemExit(f"Qwen3.5-4B absent from model list: {sorted(models)}")
print("Qwen service preflight: OK")
PY

args=(
  --model openai/Qwen3.5-4B
  --agent-type opencode
  --user-model anthropic/claude-opus-4-8
  --user-temperature 0.5
  --workers "$workers"
  --agent-timeout 4800
  --replicates "$replicates"
  --tag "$tag"
  --trials-dir "$trials_dir"
  --skip-existing
)
if [[ -n "$tasks" ]]; then
  args+=(--tasks "$tasks")
fi

echo "Qwen run start: $(date -u +%FT%TZ) host=$(hostname) workers=$workers replicates=$replicates"
setsid bash run_local.sh "${args[@]}" &
eval_pid=$!
while kill -0 "$eval_pid" 2>/dev/null; do
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "Qwen host proxy exited during evaluation; aborting producer" >&2
    terminate_eval
    exit 70
  fi
  sleep 2
done
set +e
wait "$eval_pid"
eval_rc=$?
set -e
eval_pid=""
if (( eval_rc != 0 )); then
  exit "$eval_rc"
fi
echo "Qwen run end: $(date -u +%FT%TZ)"
