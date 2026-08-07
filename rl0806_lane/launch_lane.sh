#!/usr/bin/env bash
# Prepare and submit one isolated run_0806_v1/v2 SWE-Together lane.

set -euo pipefail
umask 0077
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly PY="$REPO/.venv/bin/python"

usage() {
  cat <<'EOF'
usage: launch_lane.sh --lane run_0806_v{1,2}_stepN --checkpoint ABS_PATH \
  --relay-port PORT [--relay-host 10.146.5.90] [--proxy-url HTTP_URL] \
  [--initial-judge-delay SEC] [--judge-poll SEC] [--service-wait SEC] [--dry-run]

Normal mode creates a new isolated lane and submits four services, four tool
smokes, four action shards, one overlapping rolling-judge watcher, and one
strict finalizer. It refuses existing lane/trial roots. --dry-run validates the
checkpoint and prints the collision-free plan without writing or submitting.
EOF
}

lane=""
checkpoint=""
relay_host=10.146.5.90
relay_port=""
proxy_url="${http_proxy:-${HTTP_PROXY:-}}"
initial_judge_delay=900
judge_poll=600
service_wait=7200
dry_run=0
while (( $# )); do
  case "$1" in
    --lane) lane="${2:-}"; shift 2 ;;
    --checkpoint) checkpoint="${2:-}"; shift 2 ;;
    --relay-host) relay_host="${2:-}"; shift 2 ;;
    --relay-port) relay_port="${2:-}"; shift 2 ;;
    --proxy-url) proxy_url="${2:-}"; shift 2 ;;
    --initial-judge-delay) initial_judge_delay="${2:-}"; shift 2 ;;
    --judge-poll) judge_poll="${2:-}"; shift 2 ;;
    --service-wait) service_wait="${2:-}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done
if [[ -z "$lane" || -z "$checkpoint" || -z "$relay_port" ]]; then
  usage >&2
  exit 64
fi
for value in "$relay_port" "$initial_judge_delay" "$judge_poll" "$service_wait"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: numeric arguments must contain only digits" >&2
    exit 64
  fi
done
if (( service_wait < 60 || service_wait > 86400 )); then
  echo "ERROR: --service-wait must be between 60 and 86400 seconds" >&2
  exit 64
fi
if (( initial_judge_delay > 3600 )); then
  echo "ERROR: --initial-judge-delay must be between 0 and 3600 seconds" >&2
  exit 64
fi
if (( judge_poll < 60 || judge_poll > 3600 )); then
  echo "ERROR: --judge-poll must be between 60 and 3600 seconds" >&2
  exit 64
fi

protocol_preview=$("$PY" "$REPO/rl0806_lane/lane_config.py" build \
  --lane "$lane" --checkpoint "$checkpoint" \
  --relay-host "$relay_host" --relay-port "$relay_port")
checkpoint=$("$PY" -c 'import json,sys; print(json.load(sys.stdin)["checkpoint"])' <<<"$protocol_preview")
readonly RUN_ROOT="$REPO/rl0806_lane/runs/$lane"
readonly STATE_DIR="$RUN_ROOT/state"
readonly LOGS_DIR="$RUN_ROOT/logs"
readonly ARTIFACTS_DIR="$RUN_ROOT/artifacts"
readonly PROTOCOL="$RUN_ROOT/protocol.json"
readonly TRIALS_ROOT="$REPO/trials/${lane}_k2"
readonly INVALID_ARCHIVE_BASE="$REPO/trials/${lane}_k2_invalid_shard"
readonly REGISTRY="$STATE_DIR/egress_relay_clients.json"
readonly SHARD_MANIFEST="$REPO/sushi_lane/shard_manifest_k4.json"
readonly ACTION_MODEL="openai/$lane"
readonly SERVICE_PARTITION=h200
readonly SERVICE_QOS=h200_ram_high

readarray -t proxy_parts < <(
  "$PY" - "$proxy_url" <<'PY'
import sys
from urllib.parse import urlsplit

parsed = urlsplit(sys.argv[1])
if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
    raise SystemExit("proxy URL must be an http:// loopback endpoint")
if parsed.port is None:
    raise SystemExit("proxy URL must include an explicit port")
if parsed.username or parsed.password or parsed.path not in {"", "/"}:
    raise SystemExit("proxy URL must not contain credentials or a non-root path")
if parsed.query or parsed.fragment:
    raise SystemExit("proxy URL must not contain a query or fragment")
print(parsed.hostname)
print(parsed.port)
PY
)
if [[ "${#proxy_parts[@]}" -ne 2 ]]; then
  echo "ERROR: failed to resolve filtering-proxy target" >&2
  exit 2
fi
readonly PROXY_HOST="${proxy_parts[0]}"
readonly PROXY_PORT="${proxy_parts[1]}"

print_cmd() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

service_endpoint() {
  local slot="$1"
  if [[ "$slot" -eq 0 ]]; then
    printf '%s\n' "$STATE_DIR/endpoint.env"
  else
    printf '%s\n' "$STATE_DIR/endpoint_slot${slot}.env"
  fi
}

service_export="BENCH_CHECKPOINT=$checkpoint,BENCH_SERVED_MODEL=$lane,BENCH_STATE_DIR=$STATE_DIR,BENCH_SERVICE_LABEL=${lane^^},BENCH_RUNTIME_PREFIX=$lane"
shard_export="BENCH_TRIALS_ROOT=$TRIALS_ROOT,BENCH_STATE_DIR=$STATE_DIR,BENCH_LOG_DIR=$LOGS_DIR,BENCH_SHARD_MANIFEST=$SHARD_MANIFEST,BENCH_ACTION_MODEL=$ACTION_MODEL,BENCH_RUN_TAG=${lane}_k2,BENCH_LANE_LABEL=$lane,BENCH_RELAY_HOST=$relay_host,BENCH_RELAY_PORT=$relay_port,BENCH_RELAY_REGISTRY=$REGISTRY,BENCH_INVALID_ARCHIVE_BASE=$INVALID_ARCHIVE_BASE,BENCH_FORCE_ARCHIVE_LIST=/dev/null"

if [[ "$dry_run" -eq 1 ]]; then
  echo "RL0806_DRY_RUN_OK"
  echo "lane=$lane"
  echo "checkpoint=$checkpoint"
  echo "run_root=$RUN_ROOT"
  echo "trials_root=$TRIALS_ROOT"
  echo "relay=${relay_host}:${relay_port} target=${PROXY_HOST}:${PROXY_PORT}"
  echo "service submissions (four independent H200 slots):"
  for slot in 0 1 2 3; do
    print_cmd sbatch --parsable --job-name="${lane}-srv${slot}" \
      --partition="$SERVICE_PARTITION" --qos="$SERVICE_QOS" \
      --output="$LOGS_DIR/serve_${slot}_%j.log" --export="$service_export" \
      "$REPO/sushi_lane/serve_step575.sbatch" "$slot"
  done
  echo "after endpoint publication, submit four tool smokes and require exit 0"
  echo "action submissions (after smokes):"
  for shard in 0 1 2 3; do
    print_cmd sbatch --parsable --job-name="${lane}-shard${shard}" \
      --output="$LOGS_DIR/full_shard_${shard}_%j.log" --export="$shard_export" \
      "$REPO/sushi_lane/run_full_shard.sbatch" "$shard" "$shard" 16 3
  done
  echo "rolling dependency: after:<all-four-shards-start>"
  echo "final dependency: afterok:<all-four-shards>:<rolling-watcher>"
  exit 0
fi

for command in sbatch squeue sacct scancel setsid; do
  command -v "$command" >/dev/null || { echo "ERROR: missing command: $command" >&2; exit 2; }
done
archive_collision=0
for shard in 0 1 2 3; do
  [[ -e "${INVALID_ARCHIVE_BASE}${shard}" ]] && archive_collision=1
done
if [[ -e "$RUN_ROOT" || -e "$TRIALS_ROOT" || "$archive_collision" -eq 1 ]]; then
  echo "ERROR: lane, trials, or invalid-archive root already exists; choose a new step label or resume manually" >&2
  exit 73
fi
"$PY" - "$relay_port" <<'PY'
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("0.0.0.0", int(sys.argv[1])))
finally:
    sock.close()
PY

mkdir -p "$REPO/rl0806_lane/runs" "$REPO/trials"
if ! mkdir "$RUN_ROOT"; then
  echo "ERROR: lane root was created concurrently: $RUN_ROOT" >&2
  exit 73
fi
if ! mkdir "$TRIALS_ROOT"; then
  echo "ERROR: trials root was created concurrently: $TRIALS_ROOT" >&2
  exit 73
fi
mkdir -p "$STATE_DIR" "$LOGS_DIR" "$ARTIFACTS_DIR"
"$PY" "$REPO/rl0806_lane/lane_config.py" build \
  --lane "$lane" --checkpoint "$checkpoint" \
  --relay-host "$relay_host" --relay-port "$relay_port" \
  --output "$PROTOCOL" >/dev/null
"$PY" "$REPO/sushi_lane/register_relay_client.py" \
  --registry "$REGISTRY" --label bootstrap --ip 127.0.0.1 >/dev/null
"$PY" "$REPO/sushi_lane/register_relay_client.py" \
  --registry "$REGISTRY" --label bootstrap --remove >/dev/null

submitted_ids=()
committed=0
relay_pid=""
cleanup() {
  local rc=$?
  trap - TERM INT EXIT
  if [[ "$rc" -ne 0 && "$committed" -eq 0 ]]; then
    if [[ "${#submitted_ids[@]}" -gt 0 ]]; then
      scancel "${submitted_ids[@]}" 2>/dev/null || true
    fi
    if [[ -n "$relay_pid" ]] && kill -0 "$relay_pid" 2>/dev/null; then
      kill -TERM -- "-$relay_pid" 2>/dev/null || kill -TERM "$relay_pid" 2>/dev/null || true
      wait "$relay_pid" 2>/dev/null || true
    fi
    echo "ERROR: launch aborted; submitted jobs were cancelled and artifacts were preserved" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

relay_log="$LOGS_DIR/secure_egress_relay.log"
setsid "$PY" "$REPO/sushi_lane/secure_tcp_relay.py" \
  --listen-host 0.0.0.0 --listen-port "$relay_port" \
  --target-host "$PROXY_HOST" --target-port "$PROXY_PORT" \
  --registry "$REGISTRY" </dev/null >>"$relay_log" 2>&1 &
relay_pid=$!
printf '%s\n' "$relay_pid" >"$STATE_DIR/relay.pid"
"$PY" - "$STATE_DIR/relay.pid" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = path.lstat()
if (
    not stat.S_ISREG(metadata.st_mode)
    or stat.S_ISLNK(metadata.st_mode)
    or metadata.st_uid != os.getuid()
    or stat.S_IMODE(metadata.st_mode) & 0o077
):
    raise SystemExit("relay PID file is not an owner-only regular file")
PY

relay_ready=0
for _ in $(seq 1 100); do
  if ! kill -0 "$relay_pid" 2>/dev/null; then
    echo "ERROR: secure relay exited during startup; see $relay_log" >&2
    exit 2
  fi
  if "$PY" - "$relay_port" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
  then
    relay_ready=1
    break
  fi
  sleep 0.1
done
if [[ "$relay_ready" -ne 1 ]]; then
  echo "ERROR: secure relay did not listen on port $relay_port" >&2
  exit 2
fi

submit_job() {
  local raw
  raw=$(sbatch --parsable "$@")
  raw="${raw%%;*}"
  if [[ ! "$raw" =~ ^[0-9]+$ ]]; then
    echo "ERROR: unexpected sbatch response: $raw" >&2
    return 2
  fi
  printf '%s\n' "$raw"
}

service_ids=()
for slot in 0 1 2 3; do
  job=$(submit_job --job-name="${lane}-srv${slot}" \
    --partition="$SERVICE_PARTITION" --qos="$SERVICE_QOS" \
    --output="$LOGS_DIR/serve_${slot}_%j.log" --export="$service_export" \
    "$REPO/sushi_lane/serve_step575.sbatch" "$slot")
  service_ids+=("$job")
  submitted_ids+=("$job")
done
echo "RL0806_SERVICES_SUBMITTED lane=${lane} jobs=${service_ids[*]}"

deadline=$((SECONDS + service_wait))
while true; do
  ready=1
  for slot in 0 1 2 3; do
    endpoint=$(service_endpoint "$slot")
    if [[ ! -s "$endpoint" ]]; then
      ready=0
      state=$(squeue -h -j "${service_ids[$slot]}" -o '%T' | head -1)
      if [[ -z "$state" ]]; then
        echo "ERROR: service ${service_ids[$slot]} exited before publishing $endpoint" >&2
        exit 1
      fi
    fi
  done
  [[ "$ready" -eq 1 ]] && break
  if (( SECONDS >= deadline )); then
    echo "ERROR: timed out waiting for service endpoints" >&2
    exit 1
  fi
  sleep 5
done

smoke_ids=()
for slot in 0 1 2 3; do
  endpoint=$(service_endpoint "$slot")
  smoke_export="BENCH_ENDPOINT_ENV=$endpoint,BENCH_SMOKE_LABEL=${lane^^},SUSHI_SMOKE_ENABLE_THINKING=0,SUSHI_SMOKE_MAX_TOKENS=32768"
  job=$(submit_job --job-name="${lane}-smoke${slot}" \
    --output="$LOGS_DIR/tool_call_smoke_${slot}_%j.log" --export="$smoke_export" \
    "$REPO/sushi_lane/tool_call_smoke.sbatch")
  smoke_ids+=("$job")
  submitted_ids+=("$job")
done

smoke_csv=$(IFS=,; echo "${smoke_ids[*]}")
deadline=$((SECONDS + 3600))
while [[ -n "$(squeue -h -j "$smoke_csv" -o '%i')" ]]; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: timed out waiting for tool smokes" >&2
    exit 1
  fi
  sleep 5
done
for job in "${smoke_ids[@]}"; do
  record=""
  for _ in $(seq 1 30); do
    record=$(sacct -X -n -P -j "$job" --format=State,ExitCode | head -1)
    [[ -n "$record" ]] && break
    sleep 1
  done
  IFS='|' read -r smoke_state smoke_exit _ <<<"$record"
  if [[ "${smoke_state%+}" != COMPLETED || "$smoke_exit" != 0:0 ]]; then
    echo "ERROR: tool smoke $job failed: ${record:-missing sacct record}" >&2
    exit 1
  fi
done
for job in "${service_ids[@]}"; do
  if [[ "$(squeue -h -j "$job" -o '%T' | head -1)" != RUNNING ]]; then
    echo "ERROR: service $job is not running after tool smoke" >&2
    exit 1
  fi
done
echo "RL0806_TOOL_SMOKES_PASSED lane=${lane} jobs=${smoke_ids[*]}"

shard_ids=()
for shard in 0 1 2 3; do
  job=$(submit_job --job-name="${lane}-shard${shard}" \
    --output="$LOGS_DIR/full_shard_${shard}_%j.log" --export="$shard_export" \
    "$REPO/sushi_lane/run_full_shard.sbatch" "$shard" "$shard" 16 3)
  shard_ids+=("$job")
  submitted_ids+=("$job")
done
shard_dependency=$(IFS=:; echo "${shard_ids[*]}")
watcher_id=$(submit_job --job-name="${lane}-rjudge" \
  --output="$LOGS_DIR/rolling_judge_%j.log" \
  --dependency="after:${shard_dependency}" \
  "$REPO/rl0806_lane/rolling_judge_watch.sbatch" \
  "$lane" "$initial_judge_delay" "$judge_poll")
submitted_ids+=("$watcher_id")
finalizer_id=$(submit_job --job-name="${lane}-final" \
  --output="$LOGS_DIR/finalize_%j.log" \
  --dependency="afterok:${shard_dependency}:${watcher_id}" \
  "$REPO/rl0806_lane/finalize_full_k2.sbatch" "$lane")
submitted_ids+=("$finalizer_id")

"$PY" - "$STATE_DIR/jobs.json" "$lane" "$checkpoint" "$relay_pid" \
  "$(IFS=,; echo "${service_ids[*]}")" "$(IFS=,; echo "${smoke_ids[*]}")" \
  "$(IFS=,; echo "${shard_ids[*]}")" "$watcher_id" "$finalizer_id" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

(
    raw_output,
    lane,
    checkpoint,
    relay_pid,
    services,
    smokes,
    shards,
    watcher,
    finalizer,
) = sys.argv[1:]
output = Path(raw_output)
payload = {
    "schema_version": 1,
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "lane": lane,
    "checkpoint": checkpoint,
    "relay_pid": int(relay_pid),
    "service_jobs": [int(item) for item in services.split(",")],
    "smoke_jobs": [int(item) for item in smokes.split(",")],
    "shard_jobs": [int(item) for item in shards.split(",")],
    "rolling_judge_job": int(watcher),
    "finalizer_job": int(finalizer),
}
descriptor, raw_temp = tempfile.mkstemp(prefix=".jobs.", dir=output.parent)
temporary = Path(raw_temp)
try:
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
finally:
    if temporary.exists():
        temporary.unlink()
PY

committed=1
echo "RL0806_LANE_SUBMITTED lane=${lane} shards=${shard_ids[*]} rolling=${watcher_id} finalizer=${finalizer_id}"
echo "jobs_manifest=$STATE_DIR/jobs.json"
echo "After strict finalization, run: bash $REPO/rl0806_lane/stop_lane.sh $lane"
