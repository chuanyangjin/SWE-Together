#!/bin/bash
#SBATCH --job-name=opus48par
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=350G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/opus_k2_judge_opus48_parallel_%A_%a.slurm.log
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
export http_proxy=http://10.146.5.90:48835 https_proxy=http://10.146.5.90:48835
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export no_proxy=127.0.0.1,localhost,ghcr.io,githubusercontent.com,nodejs.org
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY=ghcr.io,githubusercontent.com,nodejs.org
export SWE_NATIVE_PODMAN=1 SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_RMI=1 HARBOR_PODMAN_MAX_PULLS=1
export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-8

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

plan="trials/opus_k2/judge_missing_opus48_parallel_${SLURM_ARRAY_TASK_ID}.json"
echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID} start=$(date -u +%FT%H:%M:%SZ)"
run_isolated bash judge_local.sh \
  --plan "$plan" --workers 4 --skip-phase1 --force
echo "OPUS48_PARALLEL_DONE shard=${SLURM_ARRAY_TASK_ID} end=$(date -u +%FT%H:%M:%SZ)"
