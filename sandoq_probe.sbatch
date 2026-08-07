#!/bin/bash
#SBATCH --job-name=sandoq-probe
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/sandoq_probe_%j.log
set -euo pipefail

cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
export OCI_RUNNER_TOKEN_FILE="${OCI_RUNNER_TOKEN_FILE:-$HOME/.config/oci-runner/token}"
export OCI_RUNNER_LEASE_DURATION="${OCI_RUNNER_LEASE_DURATION:-3h}"

# The Sandoq client deliberately ignores generic proxy variables. Unset them
# here as well so the probe documents the direct compute-node route clearly.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

probe_args=()
if [ "${SANDOQ_CONTROL_PLANE_ONLY:-}" = "1" ]; then
  probe_args+=(--control-plane-only)
fi
exec .venv/bin/python sandoq_probe.py "${probe_args[@]}"
