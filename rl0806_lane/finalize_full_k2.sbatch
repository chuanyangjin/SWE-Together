#!/usr/bin/env bash
#SBATCH --job-name=rl0806-final
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --export=NIL
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/rl0806_lane/finalize_%j.log

set -euo pipefail
ulimit -c 0
export PATH=/usr/local/bin:/usr/bin:/bin

readonly REPO=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
readonly LANE="${1:-}"
readonly ATTRIBUTION_SOURCE=/storage/home/chuanyang/.claude/projects/-home-chuanyang-test/fcc42640-dcfb-423a-bdf3-4e153c07ea32.jsonl
source "$REPO/rl0806_lane/common.sh"
rl0806_load_lane "$LANE"

readonly OUT="$RL_ARTIFACTS_DIR/full_postprocess"
readonly ROLLING_OUT="$RL_ARTIFACTS_DIR/rolling_judge"
readonly JUDGE_PLAN="$OUT/judge_missing_opus46.json"
readonly TAG_SIDECAR="$OUT/vertex_gemini_tags.json"
readonly METRICS="$OUT/table2_metrics_strict.json"
readonly RELAY_LABEL="${LANE}-finalizer-${SLURM_JOB_ID}"
mkdir -p "$OUT" "$ROLLING_OUT" "$RL_STATE_DIR"
cd "$REPO"

exec 9>"$OUT/finalize.lock"
if ! flock -n 9; then
  echo "ERROR: another finalizer is active for $LANE" >&2
  exit 75
fi
exec 8>"$ROLLING_OUT/rolling.lock"
flock 8

# The dependency graph should already have completed all producers. Holding
# every lane-local lock makes the full-root sanitizer fail closed even if the
# finalizer was submitted manually with a bad dependency.
producer_lock_fds=()
for shard in 0 1 2 3; do
  exec {lock_fd}>"$RL_STATE_DIR/shard_${shard}.lock"
  if ! flock -n "$lock_fd"; then
    echo "ERROR: $LANE shard $shard still has an active producer" >&2
    exit 75
  fi
  producer_lock_fds+=("$lock_fd")
done

stage_pid=""
relay_registered=0
terminate_stage() {
  if [[ -n "$stage_pid" ]] && kill -0 "$stage_pid" 2>/dev/null; then
    kill -TERM -- "-$stage_pid" 2>/dev/null || kill -TERM "$stage_pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "$stage_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$stage_pid" 2>/dev/null; then
      kill -KILL -- "-$stage_pid" 2>/dev/null || kill -KILL "$stage_pid" 2>/dev/null || true
    fi
    wait "$stage_pid" 2>/dev/null || true
  fi
}
cleanup() {
  local rc=$?
  trap - TERM INT EXIT
  terminate_stage
  if [[ "$relay_registered" -eq 1 ]]; then
    rl0806_remove_relay "$RELAY_LABEL" || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

run_stage() {
  setsid "$@" &
  stage_pid=$!
  local rc=0
  wait "$stage_pid" || rc=$?
  stage_pid=""
  return "$rc"
}

rl0806_register_relay "$RELAY_LABEL"
relay_registered=1
rl0806_probe_relay
rl0806_configure_compute

echo "RL0806_FINALIZE_START lane=${LANE} job=${SLURM_JOB_ID}"
"$RL0806_PY" sushi_lane/validate_action_cohort.py \
  --trials-root "$RL_TRIALS_ROOT" --tasks-root tasks --replicates 2 \
  --expected-model "$RL_ACTION_MODEL" --workers 16 \
  >"$OUT/action_validation.json"
cp "$RL_PROTOCOL" "$RL_TRIALS_ROOT/protocol.json"

for attempt in 1 2 3; do
  "$RL0806_PY" scripts/build_missing_judge_plan.py \
    --trials-root "$RL_TRIALS_ROOT" --tasks-root tasks \
    --verdict-name judge_verdict_opus46.json \
    --expected-judge-model anthropic/claude-opus-4-6 \
    --output "$JUDGE_PLAN"
  missing=$("$RL0806_PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$JUDGE_PLAN")
  echo "RL0806_FINAL_JUDGE lane=${LANE} attempt=${attempt} missing=${missing}"
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  if ! run_stage bash judge_local.sh \
      --plan "$JUDGE_PLAN" --workers 8 --skip-phase1 --force \
      --summary "$OUT/judge_summary_attempt_${attempt}.json"; then
    echo "WARN: final judge pass ${attempt} returned nonzero" >&2
  fi
done

"$RL0806_PY" scripts/build_missing_judge_plan.py \
  --trials-root "$RL_TRIALS_ROOT" --tasks-root tasks \
  --verdict-name judge_verdict_opus46.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --output "$JUDGE_PLAN"
"$RL0806_PY" - "$JUDGE_PLAN" <<'PY'
import json
import sys

missing = json.load(open(sys.argv[1]))
if missing:
    raise SystemExit(f"{len(missing)} substantive Opus 4.6 verdicts remain missing")
PY

# All producer locks remain held from here through metrics publication.
run_stage "$RL0806_PY" scripts/sanitize_traces.py \
  --trials-dir "$RL_TRIALS_ROOT" --workers 16

tag_ok=0
for attempt in 1 2 3; do
  echo "RL0806_VERTEX_TAG lane=${LANE} attempt=${attempt}"
  if run_stage "$RL0806_PY" scripts/run_vertex_tagger.py \
      --attribution-source "$ATTRIBUTION_SOURCE" -- \
      --trials-root "$RL_TRIALS_ROOT" \
      --model gemini/gemini-3.1-pro-preview \
      --workers 12 --output-sidecar "$TAG_SIDECAR"; then
    tag_ok=1
    break
  fi
done
if [[ "$tag_ok" -ne 1 ]]; then
  echo "ERROR: Vertex Gemini tagging did not complete for $LANE" >&2
  exit 1
fi

run_stage "$RL0806_PY" eval/table2_metrics.py \
  --trials-dir "$RL_TRIALS_ROOT" --tasks-dir tasks \
  --k 2 --expected-tasks 109 --strict \
  --u-corr-source single --tag-sidecar "$TAG_SIDECAR" \
  --expected-tag-model gemini/gemini-3.1-pro-preview \
  --judge-out-name judge_verdict_opus46.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --model "$RL_RESULT_LABEL" --output "$METRICS"

"$RL0806_PY" - "$METRICS" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1]))
aggregates = metrics.get("aggregates") or {}
required = (
    metrics.get("status") == "strict_complete"
    and metrics.get("metric_complete") is True
    and metrics.get("canonical_u_corr_protocol_complete") is True
    and aggregates.get("n_tasks") == 109
    and aggregates.get("n_trials") == 218
    and aggregates.get("n_tasks_with_full_k") == 109
)
if not required:
    raise SystemExit("strict result-row gate failed")
print("RL0806_STRICT_ROW | " + metrics["row"] + " |")
PY

echo "RL0806_FINALIZE_DONE lane=${LANE} job=${SLURM_JOB_ID}"
