#!/usr/bin/env bash
#SBATCH --job-name=olmo500-serve
#SBATCH --account=ram
#SBATCH --partition=h100
#SBATCH --qos=h100_ram_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/olmo_lane/logs/serve_step500_%j.log

set -euo pipefail
umask 0077
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly CHECKPOINT=/checkpoint/ram/chuanyang/autodata/run_0716_single_turn/weights/step_500
readonly PIXI=/home/chuanyang/.pixi/bin/pixi
readonly PIXI_MANIFEST=/storage/home/chuanyang/ram_common/vllm_tools/serve_api_vllm/envs/qwen-3.5/pixi.toml
readonly SERVED_MODEL=Olmo-0716-step500
readonly STATE_DIR="$REPO/olmo_lane/state"
readonly RUN_USER="$(id -un)"
readonly RUNTIME_BASE="/tmp/olmo-service-${RUN_USER}-${SLURM_JOB_ID}"
readonly SERVICE_SLOT="${1:-0}"
if [[ ! "$SERVICE_SLOT" =~ ^[0-3]$ ]]; then
  echo "ERROR: service slot must be 0, 1, 2, or 3" >&2
  exit 64
fi
if [[ "$SERVICE_SLOT" -eq 0 ]]; then
  ENDPOINT_FILE="$STATE_DIR/endpoint.env"
else
  ENDPOINT_FILE="$STATE_DIR/endpoint_slot${SERVICE_SLOT}.env"
fi
readonly ENDPOINT_FILE
readonly API_KEY_FILE="$STATE_DIR/service_api_key_${SLURM_JOB_ID}"
readonly PORT=$((18000 + SLURM_JOB_ID % 10000))
readonly SERVICE_HOST="$(hostname -s)"

if [[ ! -f "$CHECKPOINT/STABLE" || ! -r "$CHECKPOINT/config.json" ]]; then
  echo "ERROR: stable Olmo step-500 checkpoint is unavailable" >&2
  exit 2
fi
python3 - "$CHECKPOINT/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
    raise SystemExit("unexpected checkpoint architecture")
if config.get("model_type") != "qwen3_5":
    raise SystemExit("unexpected checkpoint model_type")
print("OLMO_CHECKPOINT_PREFLIGHT_OK")
PY

mkdir -p "$STATE_DIR"
ulimit -n 65536
unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=900000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
export XDG_CACHE_HOME="$RUNTIME_BASE/xdg-cache"
export XDG_CONFIG_HOME="$RUNTIME_BASE/xdg-config"
export XDG_DATA_HOME="$RUNTIME_BASE/xdg-data"
export XDG_RUNTIME_DIR="$RUNTIME_BASE/xdg-runtime"
export PIXI_CACHE_DIR="$RUNTIME_BASE/pixi-cache"
export TRITON_CACHE_DIR="/tmp/olmo-triton-${RUN_USER}-${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="/tmp/olmo-vllm-${RUN_USER}-${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
  "$XDG_RUNTIME_DIR" "$PIXI_CACHE_DIR"
chmod 0700 "$RUNTIME_BASE" "$XDG_RUNTIME_DIR"

rm -f "$ENDPOINT_FILE"
tmp_key="$API_KEY_FILE.tmp"
python3 - <<'PY' >"$tmp_key"
import secrets
print(secrets.token_urlsafe(48))
PY
chmod 0600 "$tmp_key"
mv -f "$tmp_key" "$API_KEY_FILE"
IFS= read -r VLLM_API_KEY <"$API_KEY_FILE"
export VLLM_API_KEY

child_pid=""
cleanup() {
  trap - TERM INT EXIT
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  if [[ -f "$ENDPOINT_FILE" ]] && grep -q "^OLMO_SERVICE_JOB_ID=${SLURM_JOB_ID}$" "$ENDPOINT_FILE"; then
    rm -f "$ENDPOINT_FILE"
  fi
  rm -f "$API_KEY_FILE"
}
trap cleanup TERM INT EXIT

echo "OLMO_SERVICE_START job=${SLURM_JOB_ID} slot=${SERVICE_SLOT} host=${SERVICE_HOST} port=${PORT} checkpoint=${CHECKPOINT}"
HOME=/storage/home/chuanyang "$PIXI" run --manifest-path "$PIXI_MANIFEST" --frozen -- \
  vllm serve "$CHECKPOINT" \
    --served-model-name "$SERVED_MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --load-format auto \
    --dtype auto \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --language-model-only \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 &
child_pid=$!

ready=0
for ((attempt=1; attempt<=180; attempt++)); do
  if ! kill -0 "$child_pid" 2>/dev/null; then
    echo "ERROR: vLLM exited before becoming healthy" >&2
    wait "$child_pid"
  fi
  if curl --noproxy '*' --fail --silent --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if (( attempt % 6 == 0 )); then
    echo "OLMO_SERVICE_WAIT elapsed=$((attempt * 10))s"
  fi
  sleep 10
done
if [[ "$ready" -ne 1 ]]; then
  echo "ERROR: vLLM did not become healthy within 1800 seconds" >&2
  exit 1
fi

"$REPO/.venv/bin/python" "$REPO/sushi_lane/service_preflight.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --api-key-file "$API_KEY_FILE" \
  --expected-model "$SERVED_MODEL" \
  --expected-checkpoint "$CHECKPOINT"

tmp_endpoint="$STATE_DIR/endpoint.env.${SLURM_JOB_ID}.tmp"
{
  printf 'OLMO_SERVICE_JOB_ID=%s\n' "$SLURM_JOB_ID"
  printf 'OLMO_SERVICE_SLOT=%s\n' "$SERVICE_SLOT"
  printf 'OLMO_SERVICE_HOST=%s\n' "$SERVICE_HOST"
  printf 'OLMO_SERVICE_PORT=%s\n' "$PORT"
  printf 'OLMO_OPENAI_BASE_URL=http://%s:%s/v1\n' "$SERVICE_HOST" "$PORT"
  printf 'OLMO_SERVICE_KEY_FILE=%s\n' "$API_KEY_FILE"
  printf 'OLMO_SERVED_MODEL=%s\n' "$SERVED_MODEL"
  printf 'OLMO_CHECKPOINT=%s\n' "$CHECKPOINT"
} >"$tmp_endpoint"
mv -f "$tmp_endpoint" "$ENDPOINT_FILE"

echo "OLMO_SERVICE_READY job=${SLURM_JOB_ID} slot=${SERVICE_SLOT} endpoint=http://${SERVICE_HOST}:${PORT}/v1 model=${SERVED_MODEL}"
wait "$child_pid"
