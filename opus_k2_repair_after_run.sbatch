#!/bin/bash
#SBATCH --job-name=opusrepair
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=350G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/opus_k2_repair_after_run.slurm.log
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
exec 9>pipeline_logs/opus_k2_pipeline.lock
if ! flock -n 9; then
  echo "ERROR: another opus_k2 repair/finalize process holds the pipeline lock" >&2
  exit 75
fi
export http_proxy=http://10.146.5.90:48835 https_proxy=http://10.146.5.90:48835
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export no_proxy=127.0.0.1,localhost,ghcr.io,githubusercontent.com,nodejs.org
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY=ghcr.io,githubusercontent.com,nodejs.org
export SWE_NATIVE_PODMAN=1 SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_RMI=1 HARBOR_PODMAN_MAX_PULLS=1

child_pid=""
terminate_active_child() {
  trap - TERM INT
  if [ -n "$child_pid" ]; then
    kill -TERM -- "-$child_pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$child_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 143
}
trap terminate_active_child TERM INT

run_isolated() {
  setsid "$@" &
  child_pid=$!
  if wait "$child_pid"; then
    isolated_rc=0
  else
    isolated_rc=$?
  fi
  child_pid=""
  return "$isolated_rc"
}

run_action() {
  run_isolated bash run_local.sh "$@"
}

echo "host=$(hostname) start=$(date -u +%FT%H:%M:%SZ)"
for attempt in 1 2 3; do
  echo "ACTION_REPAIR_AUDIT attempt=$attempt"
  if run_isolated .venv/bin/python scripts/archive_invalid_trials.py \
      --trials-root trials/opus_k2 \
      --archive-root trials/opus_k2_infra_archive \
      --tasks-root tasks --replicates 2; then
    echo "ACTION_REPAIR_COMPLETE attempt=$attempt"
    break
  else
    rc=$?
    if [ "$rc" -ne 10 ]; then
      exit "$rc"
    fi
  fi
  run_action --model metagen/claude-opus-4-8 --agent-type opencode \
    --user-model anthropic/claude-opus-4-8 --workers 6 --agent-timeout 2400 \
    --replicates 2 --tag opus_k2 --skip-existing --trials-dir trials/opus_k2
done

run_isolated .venv/bin/python scripts/archive_invalid_trials.py \
  --trials-root trials/opus_k2 \
  --archive-root trials/opus_k2_infra_archive \
  --tasks-root tasks --replicates 2
run_isolated .venv/bin/python scripts/sanitize_traces.py \
  --trials-dir trials/opus_k2
run_isolated .venv/bin/python scripts/sanitize_traces.py \
  --trials-dir trials/opus_k2_infra_archive
echo "OPUS_K2_REPAIR_DONE end=$(date -u +%FT%H:%M:%SZ)"
