#!/usr/bin/env bash
#SBATCH --job-name=qwen35-jpilot
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/qwen35_4b_judge_pilot-%j.log

set -euo pipefail
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
export SWE_EGRESS_PROXY="${SWE_EGRESS_PROXY:-http://10.146.5.90:48836}"
export http_proxy="$SWE_EGRESS_PROXY" https_proxy="$SWE_EGRESS_PROXY"
export HTTP_PROXY="$SWE_EGRESS_PROXY" HTTPS_PROXY="$SWE_EGRESS_PROXY"
export no_proxy=127.0.0.1,localhost
export NO_PROXY="$no_proxy"
export SWE_NATIVE_PODMAN=1 SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_RMI=1 HARBOR_PODMAN_MAX_PULLS=1
export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6
bash judge_local.sh \
  --plan pipeline_logs/qwen35_4b_judge_pilot_plan.json \
  --workers 1 --skip-phase1 --force \
  --summary pipeline_logs/qwen35_4b_judge_pilot_summary.json
