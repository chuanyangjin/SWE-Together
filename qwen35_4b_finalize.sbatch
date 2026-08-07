#!/usr/bin/env bash
#SBATCH --job-name=qwen35-finalize
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/qwen35_4b_finalize-%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

repo=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
trials=trials/qwen35_4b_repro
relay_registry="$repo/sushi_lane/state/egress_relay_clients.json"
relay_host=10.146.5.90
relay_port=48836
attribution_source=/storage/home/chuanyang/.claude/projects/-home-chuanyang-test/fcc42640-dcfb-423a-bdf3-4e153c07ea32.jsonl
relay_label="qwen-finalize-${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
relay_registration_attempted=0
stage_pid=""
resume_tags_only=0
if (( $# )); then
  if (( $# == 1 )) && [[ "$1" == "--resume-tags-only" ]]; then
    resume_tags_only=1
  else
    echo "usage: $0 [--resume-tags-only]" >&2
    exit 64
  fi
fi
cd "$repo"

exec 9>pipeline_logs/qwen35_4b_judge.lock
if ! flock -n 9; then
  echo "another Qwen judge/finalizer holds the lane lock" >&2
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
      if ! kill -0 "$stage_pid" 2>/dev/null; then
        break
      fi
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
      echo "failed to remove Qwen finalizer from the secure egress relay" >&2
      cleanup_rc=76
    elif ! probe_relay_acl denied; then
      echo "secure egress relay did not fail closed after finalizer deregistration" >&2
      cleanup_rc=77
    fi
  fi
  if (( rc == 0 && cleanup_rc != 0 )); then
    rc=$cleanup_rc
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

run_stage() {
  setsid "$@" &
  stage_pid=$!
  set +e
  wait "$stage_pid"
  rc=$?
  set -e
  stage_pid=""
  return "$rc"
}

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

.venv/bin/python scripts/qwen35_4b_lane.py audit "$trials" \
  --output pipeline_logs/qwen35_4b_prejudge_audit.json
.venv/bin/python - <<'PY'
import json

report = json.load(open("pipeline_logs/qwen35_4b_prejudge_audit.json"))
if not report.get("strict_complete_109x2"):
    raise SystemExit("action cohort is not strict-complete; repair before judging")
PY

plan="$trials/judge_missing_opus46.json"
for attempt in 1 2 3; do
  .venv/bin/python scripts/build_missing_judge_plan.py \
    --trials-root "$trials" --tasks-root tasks \
    --verdict-name judge_verdict.json \
    --expected-judge-model anthropic/claude-opus-4-6 \
    --output "$plan"
  plan_count="$(.venv/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$plan")"
  if [[ "$plan_count" -eq 0 ]]; then
    break
  fi
  echo "Opus4.6 Podman judge pass $attempt: $plan_count substantive patches"
  if ! run_stage bash judge_local.sh \
    --plan "$plan" --workers 12 --skip-phase1; then
    echo "judge pass $attempt returned nonzero; rebuilding missing plan" >&2
  fi
done

.venv/bin/python scripts/build_missing_judge_plan.py \
  --trials-root "$trials" --tasks-root tasks \
  --verdict-name judge_verdict.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --output "$plan"
.venv/bin/python - "$plan" <<'PY'
import json
import sys

remaining = json.load(open(sys.argv[1]))
if remaining:
    raise SystemExit(f"{len(remaining)} substantive verdicts remain invalid/missing")
print("Opus4.6 correctness judge: complete")
PY

tag_force=(--force)
if (( resume_tags_only )); then
  tag_force=()
fi
tag_ok=0
for attempt in 1 2 3; do
  if run_stage .venv/bin/python scripts/run_vertex_tagger.py \
    --attribution-source "$attribution_source" -- \
    --trials-root "$trials" \
    --model gemini/gemini-3.1-pro-preview \
    --workers 12 "${tag_force[@]}" \
    --output-sidecar pipeline_logs/qwen35_4b_gemini31_tags.json; then
    tag_ok=1
    break
  fi
  echo "Gemini tag pass $attempt returned nonzero; retrying only missing trials" >&2
  tag_force=()
done
if (( ! tag_ok )); then
  echo "Gemini tagger still has invalid/missing trials after 3 passes" >&2
  exit 2
fi

run_stage bash qwen35_4b_metrics.sh
echo "Qwen finalizer complete: $(date -u +%FT%TZ)"
