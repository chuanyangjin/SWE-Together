#!/bin/bash
#SBATCH --job-name=opusfinalize
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/opus_k2_finalize.slurm.log
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
exec 9>pipeline_logs/opus_k2_pipeline.lock
if ! flock -n 9; then
  echo "ERROR: another opus_k2 repair/finalize process holds the pipeline lock" >&2
  exit 75
fi
export http_proxy=http://10.146.5.90:48835 https_proxy=http://10.146.5.90:48835
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export no_proxy=127.0.0.1,localhost,ghcr.io,githubusercontent.com,nodejs.org
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY=ghcr.io,githubusercontent.com,nodejs.org
export SWE_NATIVE_PODMAN=1 SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_RMI=1 HARBOR_PODMAN_MAX_PULLS=1

child_pid=""
terminate_active_child() {
  trap - TERM INT
  if [ -n "$child_pid" ]; then
    kill -TERM -- "-$child_pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$child_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 143
}
trap terminate_active_child TERM INT

run_isolated() {
  setsid "$@" &
  child_pid=$!
  if wait "$child_pid"; then
    isolated_rc=0
  else
    isolated_rc=$?
  fi
  child_pid=""
  return "$isolated_rc"
}

echo "host=$(hostname) start=$(date -u +%FT%H:%M:%SZ)"

export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6
for attempt in 1 2 3; do
  echo "Opus 4.6 missing-verdict pass $attempt"
  run_isolated .venv/bin/python scripts/build_missing_judge_plan.py \
    --trials-root trials/opus_k2 --tasks-root tasks \
    --verdict-name judge_verdict_opus46.json \
    --expected-judge-model anthropic/claude-opus-4-6 \
    --output trials/opus_k2/judge_missing_opus46.json
  if ! run_isolated bash judge_local.sh \
      --plan trials/opus_k2/judge_missing_opus46.json \
      --workers 8 --skip-phase1 --force; then
    echo "WARN: Opus 4.6 judge pass $attempt returned nonzero" >&2
  fi
done

export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-8
for attempt in 1 2 3; do
  echo "Opus 4.8 missing-verdict pass $attempt"
  run_isolated .venv/bin/python scripts/build_missing_judge_plan.py \
    --trials-root trials/opus_k2 --tasks-root tasks \
    --verdict-name judge_verdict.json \
    --expected-judge-model anthropic/claude-opus-4-8 \
    --output trials/opus_k2/judge_missing_opus48.json
  if ! run_isolated bash judge_local.sh \
      --plan trials/opus_k2/judge_missing_opus48.json \
      --workers 8 --skip-phase1 --force; then
    echo "WARN: Opus 4.8 judge pass $attempt returned nonzero" >&2
  fi
done

# The historical cohort predates tagging provenance.  Re-tag the full cohort
# after correctness repair so a transient tagging outage cannot prevent judge
# progress.  This makes the non-canonical U-Corr diagnostic reproducible; it is
# not the released evaluator's Gemini-tagged number.
tag_ok=0
for attempt in 1 2 3; do
  echo "Opus 4.8 message-tagging pass $attempt"
  tag_args=(
    --trials-root trials/opus_k2 --require-provenance
    --model anthropic/claude-opus-4-8 --workers 8
  )
  if [ "$attempt" -eq 1 ]; then
    tag_args+=(--force)
  fi
  if run_isolated .venv/bin/python -m eval.user_behavior.tag_messages \
      "${tag_args[@]}"; then
    tag_ok=1
    break
  fi
done
test "$tag_ok" -eq 1

run_isolated .venv/bin/python scripts/sanitize_traces.py \
  --trials-dir trials/opus_k2

run_isolated .venv/bin/python eval/table2_metrics.py \
  --trials-dir trials/opus_k2 --tasks-dir tasks --k 2 --strict \
  --u-corr-source single \
  --expected-tag-model anthropic/claude-opus-4-8 \
  --model 'Opus 4.8 noncanonical action run / released Opus 4.6 judge' \
  --judge-out-name judge_verdict_opus46.json \
  --expected-judge-model anthropic/claude-opus-4-6 \
  --output pipeline_logs/opus_k2_table2_opus46_final.json

run_isolated .venv/bin/python eval/table2_metrics.py \
  --trials-dir trials/opus_k2 --tasks-dir tasks --k 2 --strict \
  --u-corr-source single \
  --expected-tag-model anthropic/claude-opus-4-8 \
  --model 'Opus 4.8 noncanonical action run / Opus 4.8 judge' \
  --judge-out-name judge_verdict.json \
  --expected-judge-model anthropic/claude-opus-4-8 \
  --output pipeline_logs/opus_k2_table2_opus48_final.json

echo "OPUS_K2_FINALIZE_DONE end=$(date -u +%FT%H:%M:%SZ)"
