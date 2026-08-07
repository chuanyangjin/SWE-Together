#!/usr/bin/env bash
#SBATCH --job-name=sushi575-final
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/sushi_lane/logs/finalize_full_k2_%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly TRIALS_ROOT="$REPO/trials/sushi_step575_k2"
readonly OUT="$REPO/sushi_lane/artifacts/full_postprocess"
readonly ROLLING_OUT="$REPO/sushi_lane/artifacts/rolling_judge"
readonly JUDGE_PLAN="$OUT/judge_missing_opus46.json"
readonly TAG_SIDECAR="$OUT/vertex_gemini_tags.json"
readonly ATTRIBUTION_SOURCE=/storage/home/chuanyang/.claude/projects/-home-chuanyang-test/fcc42640-dcfb-423a-bdf3-4e153c07ea32.jsonl
readonly RELAY_HOST=10.146.5.90
readonly RELAY_PORT=48836
readonly RELAY_REGISTRY="$REPO/sushi_lane/state/egress_relay_clients.json"
readonly RELAY_LABEL="sushi-finalizer-${SLURM_JOB_ID}"
cd "$REPO"

mkdir -p "$OUT" "$ROLLING_OUT"
exec 9>"$OUT/finalize.lock"
if ! flock -n 9; then
  echo "ERROR: another Sushi finalizer holds $OUT/finalize.lock" >&2
  exit 75
fi
# A rolling judge may still be finishing immutable completed cells when the
# action dependencies resolve. Wait for that snapshot to finish before building
# the final missing plan; once held, new rolling judges fail their nonblocking
# lock and cannot race final verdict writes.
exec 8>"$ROLLING_OUT/rolling.lock"
flock 8

relay_registered=0
cleanup() {
  trap - TERM INT EXIT
  if [[ "$relay_registered" -eq 1 ]]; then
    .venv/bin/python sushi_lane/register_relay_client.py \
      --registry "$RELAY_REGISTRY" --label "$RELAY_LABEL" --remove || true
  fi
}
trap cleanup TERM INT EXIT

.venv/bin/python sushi_lane/register_relay_client.py \
  --registry "$RELAY_REGISTRY" --label "$RELAY_LABEL" \
  --relay-host "$RELAY_HOST" --relay-port "$RELAY_PORT"
relay_registered=1

export http_proxy="http://${RELAY_HOST}:${RELAY_PORT}"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export no_proxy=127.0.0.1,localhost
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY=ghcr.io,githubusercontent.com,nodejs.org
export SWE_NATIVE_PODMAN=1
export SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_MAX_PULLS=2
export HARBOR_PODMAN_RMI=1
export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6
echo "SUSHI_FINALIZE_START job=${SLURM_JOB_ID}"

# Freshly classify every completed action cell and prove exact provenance,
# runtime, raw-token, reward, and 109x2 completeness before spending judge calls.
.venv/bin/python sushi_lane/validate_action_cohort.py \
  --trials-root "$TRIALS_ROOT" --tasks-root tasks --replicates 2 \
  >"$OUT/action_validation.json"

for attempt in 1 2 3; do
  .venv/bin/python scripts/build_missing_judge_plan.py \
    --trials-root "$TRIALS_ROOT" --tasks-root tasks \
    --verdict-name judge_verdict_opus46.json \
    --expected-judge-model anthropic/claude-opus-4-6 \
    --output "$JUDGE_PLAN"
  missing=$(.venv/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$JUDGE_PLAN")
  echo "SUSHI_JUDGE_PASS attempt=${attempt} missing=${missing}"
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  if ! bash judge_local.sh \
      --plan "$JUDGE_PLAN" --workers 8 --skip-phase1 --force \
      --summary "$OUT/judge_summary_attempt_${attempt}.json"; then
    echo "WARN: Opus 4.6 judge pass ${attempt} returned nonzero" >&2
  fi
done

.venv/bin/python scripts/build_missing_judge_plan.py \
  --trials-root "$TRIALS_ROOT" --tasks-root tasks \
  --verdict-name judge_verdict_opus46.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --output "$JUDGE_PLAN"
.venv/bin/python -c '
import json, sys
path = sys.argv[1]
missing = json.load(open(path))
if missing:
    raise SystemExit(f"{len(missing)} substantive Opus 4.6 verdicts still missing")
' "$JUDGE_PLAN"

# No action writers remain because this job is afterok all shard repairs.
# Sanitize once before tagging so tagger provenance hashes the final bytes.
.venv/bin/python scripts/sanitize_traces.py --trials-dir "$TRIALS_ROOT"

tag_ok=0
for attempt in 1 2 3; do
  echo "SUSHI_VERTEX_TAG_PASS attempt=${attempt}"
  if .venv/bin/python scripts/run_vertex_tagger.py \
      --attribution-source "$ATTRIBUTION_SOURCE" -- \
      --trials-root "$TRIALS_ROOT" \
      --model gemini/gemini-3.1-pro-preview \
      --workers 12 --output-sidecar "$TAG_SIDECAR"; then
    tag_ok=1
    break
  fi
done
if [[ "$tag_ok" -ne 1 ]]; then
  echo "ERROR: Vertex Gemini sidecar tagging did not complete" >&2
  exit 1
fi

.venv/bin/python eval/table2_metrics.py \
  --trials-dir "$TRIALS_ROOT" --tasks-dir tasks \
  --k 2 --expected-tasks 109 --strict \
  --u-corr-source single --tag-sidecar "$TAG_SIDECAR" \
  --expected-tag-model gemini/gemini-3.1-pro-preview \
  --judge-out-name judge_verdict_opus46.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --model 'Sushi checkpoint step 575' \
  --output "$OUT/table2_metrics_strict.json"

echo "SUSHI_FINALIZE_DONE job=${SLURM_JOB_ID} end=$(date -u +%FT%TZ)"
