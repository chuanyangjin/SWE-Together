#!/usr/bin/env bash
#SBATCH --job-name=qwen35-inc-judge
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=36
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/qwen35_4b_incremental_judge-%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

repo=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
trials=trials/qwen35_4b_repro
relay_registry="$repo/sushi_lane/state/egress_relay_clients.json"
relay_host=10.146.5.90
relay_port=48836
relay_label="qwen-inc-judge-${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
plan="pipeline_logs/qwen35_4b_incremental_judge-${SLURM_JOB_ID}.json"
relay_registration_attempted=0
stage_pid=""
cd "$repo"

exec 9>pipeline_logs/qwen35_4b_incremental_judge.lock
if ! flock -n 9; then
  echo "another incremental Qwen judge holds the lock" >&2
  exit 75
fi

probe_relay_acl() {
  local expectation="$1"
  /usr/bin/env -i PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
    .venv/bin/python - "$relay_host" "$relay_port" "$expectation" <<'PY'
import socket
import sys

host, raw_port, expectation = sys.argv[1:]
response = b""
request = (
    b"GET http://example.com/ HTTP/1.1\r\n"
    b"Host: example.com\r\nConnection: close\r\n\r\n"
)
try:
    with socket.create_connection((host, int(raw_port)), timeout=10) as connection:
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

terminate_stage() {
  if [[ -n "$stage_pid" ]] && kill -0 "$stage_pid" 2>/dev/null; then
    kill -TERM -- "-$stage_pid" 2>/dev/null || kill -TERM "$stage_pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      if ! kill -0 "$stage_pid" 2>/dev/null; then break; fi
      sleep 0.1
    done
    if kill -0 "$stage_pid" 2>/dev/null; then
      kill -KILL -- "-$stage_pid" 2>/dev/null || kill -KILL "$stage_pid" 2>/dev/null || true
    fi
    wait "$stage_pid" 2>/dev/null || true
  fi
}

cleanup() {
  rc=$?
  cleanup_rc=0
  trap - EXIT TERM INT
  terminate_stage
  if (( relay_registration_attempted )); then
    if ! /usr/bin/env -i PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
      .venv/bin/python sushi_lane/register_relay_client.py \
      --registry "$relay_registry" --label "$relay_label" --remove; then
      echo "failed to deregister the incremental Qwen judge" >&2
      cleanup_rc=76
    elif ! probe_relay_acl denied; then
      echo "secure relay did not fail closed after judge deregistration" >&2
      cleanup_rc=77
    fi
  fi
  if (( rc == 0 && cleanup_rc != 0 )); then rc=$cleanup_rc; fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

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
export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6

.venv/bin/python scripts/build_missing_judge_plan.py \
  --trials-root "$trials" --tasks-root tasks \
  --verdict-name judge_verdict.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --completed-only --output "$plan"
plan_count="$(.venv/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$plan")"
echo "Incremental Opus4.6 judge snapshot: $plan_count completed substantive patches"
if (( plan_count == 0 )); then
  exit 0
fi

setsid bash judge_local.sh --plan "$plan" --workers 12 --skip-phase1 &
stage_pid=$!
set +e
wait "$stage_pid"
rc=$?
set -e
stage_pid=""
if (( rc != 0 )); then
  exit "$rc"
fi
echo "Incremental Opus4.6 judge complete: $plan_count snapshot entries"
