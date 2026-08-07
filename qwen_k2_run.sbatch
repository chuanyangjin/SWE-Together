#!/bin/bash
#SBATCH --job-name=qwenk2run
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=350G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/qwen_k2_run.slurm.log
set -x
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
export http_proxy=http://10.146.35.140:48835 https_proxy=http://10.146.35.140:48835
export HTTP_PROXY=http://10.146.35.140:48835 HTTPS_PROXY=http://10.146.35.140:48835
export no_proxy=cpu-179-050,10.148.139.247,127.0.0.1,localhost,ghcr.io,githubusercontent.com,nodejs.org
export NO_PROXY=$no_proxy
export HARBOR_PODMAN_NO_PROXY=cpu-179-050,10.148.139.247,ghcr.io,githubusercontent.com,nodejs.org
export SWE_NATIVE_PODMAN=1 SWE_PODMAN_STORE_BASE=/dev/shm HARBOR_PODMAN_RMI=1 HARBOR_PODMAN_MAX_PULLS=6
echo "host=$(hostname) start=$(date -u +%FT%H:%M:%SZ)"
bash run_local.sh --model openai/Qwen3.5-4B --agent-type opencode \
  --user-model anthropic/claude-opus-4-8 --workers 8 --agent-timeout 1800 \
  --replicates 2 --tag qwen_k2 --trials-dir trials/qwen_k2
echo "QWEN_K2_RUN_DONE rc=$? end=$(date -u +%FT%H:%M:%SZ)"
