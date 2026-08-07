#!/bin/bash
#SBATCH --job-name=opus46shard
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/opus_k2_judge_opus46_shard_%A_%a.slurm.log
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
export http_proxy=http://10.146.5.90:48835 https_proxy=http://10.146.5.90:48835
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export no_proxy=127.0.0.1,localhost,ghcr.io,githubusercontent.com,nodejs.org
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY=ghcr.io,githubusercontent.com,nodejs.org
export SWE_NATIVE_PODMAN=1 SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_RMI=1 HARBOR_PODMAN_MAX_PULLS=8
export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6

plan="trials/opus_k2/judge_missing_opus46_shard${SLURM_ARRAY_TASK_ID}.json"
echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID} plan=$plan start=$(date -u +%FT%H:%M:%SZ)"
bash judge_local.sh --plan "$plan" --workers 8 --skip-phase1
echo "OPUS46_SHARD_DONE shard=${SLURM_ARRAY_TASK_ID} end=$(date -u +%FT%H:%M:%SZ)"
