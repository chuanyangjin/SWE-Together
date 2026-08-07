#!/usr/bin/env bash
#SBATCH --job-name=sushi575-smoke
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/sushi_lane/logs/tool_call_smoke_%j.log

set -euo pipefail
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
exec .venv/bin/python sushi_lane/tool_call_smoke.py
