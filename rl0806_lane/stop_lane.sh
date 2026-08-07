#!/usr/bin/env bash
# Stop long-lived services and the login relay after strict finalization.

set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly LANE="${1:-}"
readonly FORCE="${2:-}"
if [[ -z "$LANE" || ( -n "$FORCE" && "$FORCE" != "--force" ) ]]; then
  echo "usage: stop_lane.sh run_0806_v{1,2}_stepN [--force]" >&2
  exit 64
fi
source "$REPO/rl0806_lane/common.sh"
rl0806_load_lane "$LANE"
readonly JOBS="$RL_STATE_DIR/jobs.json"
readonly PID_FILE="$RL_STATE_DIR/relay.pid"
if [[ ! -r "$JOBS" ]]; then
  echo "ERROR: jobs manifest is unavailable: $JOBS" >&2
  exit 2
fi

mapfile -t job_fields < <(
  "$RL0806_PY" - "$JOBS" <<'PY'
import json
import sys

jobs = json.load(open(sys.argv[1]))
print(jobs["lane"])
print(",".join(map(str, jobs["service_jobs"])))
print(jobs["finalizer_job"])
print(jobs["relay_pid"])
print(
    ",".join(
        map(
            str,
            jobs["service_jobs"]
            + jobs["smoke_jobs"]
            + jobs["shard_jobs"]
            + [jobs["rolling_judge_job"], jobs["finalizer_job"]],
        )
    )
)
PY
)
if [[ "${#job_fields[@]}" -ne 5 || "${job_fields[0]}" != "$LANE" ]]; then
  echo "ERROR: jobs manifest identity mismatch" >&2
  exit 2
fi
readonly SERVICE_CSV="${job_fields[1]}"
readonly FINALIZER_JOB="${job_fields[2]}"
readonly RELAY_PID="${job_fields[3]}"
readonly ALL_JOBS_CSV="${job_fields[4]}"

if [[ "$FORCE" != "--force" ]]; then
  final_state=$(sacct -X -n -P -j "$FINALIZER_JOB" --format=State | head -1)
  if [[ "$final_state" != COMPLETED* ]]; then
    echo "ERROR: finalizer $FINALIZER_JOB is not complete (${final_state:-unknown}); use --force only for an intentional abort" >&2
    exit 1
  fi
  "$RL0806_PY" - "$RL_ARTIFACTS_DIR/full_postprocess/table2_metrics_strict.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1]))
aggregates = metrics.get("aggregates") or {}
if not (
    metrics.get("status") == "strict_complete"
    and metrics.get("metric_complete") is True
    and metrics.get("canonical_u_corr_protocol_complete") is True
    and aggregates.get("n_tasks") == 109
    and aggregates.get("n_trials") == 218
):
    raise SystemExit("strict metrics gate has not passed")
print("RL0806_STRICT_ROW | " + metrics["row"] + " |")
PY
fi

cancel_csv="$SERVICE_CSV"
if [[ "$FORCE" == "--force" ]]; then
  cancel_csv="$ALL_JOBS_CSV"
fi
IFS=, read -r -a cancel_ids <<<"$cancel_csv"
scancel "${cancel_ids[@]}" 2>/dev/null || true
for _ in $(seq 1 120); do
  [[ -z "$(squeue -h -j "$cancel_csv" -o '%i')" ]] && break
  sleep 1
done

clients=$("$RL0806_PY" - "$RL_RELAY_REGISTRY" <<'PY'
import json
import sys

print(len(json.load(open(sys.argv[1]))["clients"]))
PY
)
if [[ "$clients" -ne 0 ]]; then
  echo "ERROR: relay registry still has $clients active client label(s)" >&2
  exit 1
fi
if [[ -r "$PID_FILE" && "$(<"$PID_FILE")" != "$RELAY_PID" ]]; then
  echo "ERROR: relay PID file does not match the jobs manifest" >&2
  exit 2
fi
if kill -0 "$RELAY_PID" 2>/dev/null; then
  cmdline=$(tr '\0' ' ' <"/proc/${RELAY_PID}/cmdline")
  if [[ "$cmdline" != *"sushi_lane/secure_tcp_relay.py"* || "$cmdline" != *"$RL_RELAY_REGISTRY"* ]]; then
    echo "ERROR: refusing to signal recycled/unexpected PID $RELAY_PID" >&2
    exit 2
  fi
  kill -TERM -- "-$RELAY_PID" 2>/dev/null || kill -TERM "$RELAY_PID"
  wait "$RELAY_PID" 2>/dev/null || true
fi
rm -f "$PID_FILE"
echo "RL0806_LANE_STOPPED lane=${LANE} services=${SERVICE_CSV} relay_pid=${RELAY_PID}"
