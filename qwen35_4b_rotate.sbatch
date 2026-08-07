#!/usr/bin/env bash
#SBATCH --job-name=qwen35-rotate
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/qwen35_4b_rotate-%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
exec .venv/bin/python scripts/rotate_qwen35_bearer.py
