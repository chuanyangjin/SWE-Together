#!/usr/bin/env bash
# Shared, non-secret lane loading and process helpers.

set -euo pipefail

readonly RL0806_REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly RL0806_PY="$RL0806_REPO/.venv/bin/python"

rl0806_load_lane() {
  local lane="$1"
  if [[ ! "$lane" =~ ^run_0806_v[12]_step[0-9]+$ ]]; then
    echo "ERROR: invalid run_0806 lane label: $lane" >&2
    return 64
  fi
  RL_LANE="$lane"
  RL_RUN_ROOT="$RL0806_REPO/rl0806_lane/runs/$lane"
  RL_PROTOCOL="$RL_RUN_ROOT/protocol.json"
  "$RL0806_PY" "$RL0806_REPO/rl0806_lane/lane_config.py" validate \
    --protocol "$RL_PROTOCOL" >/dev/null

  local -a fields=()
  mapfile -t fields < <(
    "$RL0806_PY" - "$RL_PROTOCOL" <<'PY'
import json
import sys

protocol = json.load(open(sys.argv[1]))
for value in (
    protocol["checkpoint"],
    protocol["served_model"],
    protocol["action_model"],
    protocol["result_label"],
    protocol["state_dir"],
    protocol["logs_dir"],
    protocol["artifacts_dir"],
    protocol["trials_root"],
    protocol["invalid_archive_base"],
    protocol["shard_manifest"],
    protocol["relay"]["host"],
    protocol["relay"]["port"],
    protocol["relay"]["registry"],
):
    print(value)
PY
  )
  if [[ "${#fields[@]}" -ne 13 ]]; then
    echo "ERROR: lane protocol field extraction failed" >&2
    return 2
  fi
  RL_CHECKPOINT="${fields[0]}"
  RL_SERVED_MODEL="${fields[1]}"
  RL_ACTION_MODEL="${fields[2]}"
  RL_RESULT_LABEL="${fields[3]}"
  RL_STATE_DIR="${fields[4]}"
  RL_LOGS_DIR="${fields[5]}"
  RL_ARTIFACTS_DIR="${fields[6]}"
  RL_TRIALS_ROOT="${fields[7]}"
  RL_INVALID_ARCHIVE_BASE="${fields[8]}"
  RL_SHARD_MANIFEST="${fields[9]}"
  RL_RELAY_HOST="${fields[10]}"
  RL_RELAY_PORT="${fields[11]}"
  RL_RELAY_REGISTRY="${fields[12]}"
}

rl0806_configure_compute() {
  export http_proxy="http://${RL_RELAY_HOST}:${RL_RELAY_PORT}"
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
}

rl0806_register_relay() {
  local label="$1"
  "$RL0806_PY" "$RL0806_REPO/sushi_lane/register_relay_client.py" \
    --registry "$RL_RELAY_REGISTRY" --label "$label" \
    --relay-host "$RL_RELAY_HOST" --relay-port "$RL_RELAY_PORT"
}

rl0806_remove_relay() {
  local label="$1"
  "$RL0806_PY" "$RL0806_REPO/sushi_lane/register_relay_client.py" \
    --registry "$RL_RELAY_REGISTRY" --label "$label" --remove
}

rl0806_probe_relay() {
  "$RL0806_PY" - "$RL_RELAY_HOST" "$RL_RELAY_PORT" <<'PY'
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
print("RL0806_RELAY_PREFLIGHT_OK")
PY
}

rl0806_sleep_bounded() {
  local remaining="$1"
  while (( remaining > 0 )); do
    local interval=60
    if (( remaining < interval )); then
      interval="$remaining"
    fi
    sleep "$interval"
    remaining=$((remaining - interval))
  done
}
