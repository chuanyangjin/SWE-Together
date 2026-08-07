#!/usr/bin/env bash
#SBATCH --job-name=olmo500-smoke
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/olmo_lane/logs/tool_call_smoke_%j.log

set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly SERVICE_SLOT="${1:-0}"
if [[ ! "$SERVICE_SLOT" =~ ^[0-3]$ ]]; then
  echo "ERROR: service slot must be 0, 1, 2, or 3" >&2
  exit 64
fi
if [[ "$SERVICE_SLOT" -eq 0 ]]; then
  ENDPOINT_FILE="$REPO/olmo_lane/state/endpoint.env"
else
  ENDPOINT_FILE="$REPO/olmo_lane/state/endpoint_slot${SERVICE_SLOT}.env"
fi
readonly ENDPOINT_FILE
if [[ ! -r "$ENDPOINT_FILE" ]]; then
  echo "ERROR: Olmo endpoint slot ${SERVICE_SLOT} is not published" >&2
  exit 2
fi
cd "$REPO"
exec .venv/bin/python olmo_lane/tool_call_smoke.py --endpoint-file "$ENDPOINT_FILE"
