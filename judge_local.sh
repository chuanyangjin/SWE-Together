#!/usr/bin/env bash
# Judge-stage launcher for the SWE-Together local podman harness.
#
# Runs the agentic correctness judge (eval/correctness/run_batch.py) inside the
# same rootless-podman unshare session as run_local.sh, but with JUDGE_ENV=podman
# so the judge uses the host-side Opus loop (eval/correctness/podman_judge.py)
# instead of E2B: the judge MODEL runs on the host via the metagen gateway, and
# its bash tool execs into a podman container holding the patched workspace.
#
# All arguments are forwarded to `python -m eval.correctness.run_batch`.
#
# Usage:
#   ./judge_local.sh --plan trials/qwen_nodefix/judge_plan.json --workers 3
#   ./judge_local.sh --plan plan.json --force            # re-score existing verdicts
#   ./judge_local.sh --plan plan.json --skip-phase1      # Phase 2 only
#
# Env knobs:
#   SWE_TMPFS_SIZE     tmpfs size for the image store (default 120G)
#   SWE_PY             python interpreter (default .venv/bin/python)
#   JUDGE_PODMAN_MODEL host-side judge model (default anthropic/claude-opus-4-8)
#
# Auth (from .env): ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL (the metagen gateway).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWE_REPO="$REPO"
export SWE_OUTER_UID="$(id -u)"
export SWE_TMPFS_SIZE="${SWE_TMPFS_SIZE:-120G}"
export SWE_PY="${SWE_PY:-$REPO/.venv/bin/python}"
export JUDGE_ENV=podman

# SWE_NATIVE_PODMAN=1 (Slurm compute nodes): native rootless podman, no unshare
# wrapper. SWE_UNSHARE_FLAGS (default "-Urm") applies to the unshare path (pod).
# See run_local.sh header for the full rationale.
if [ -n "${SWE_NATIVE_PODMAN:-}" ]; then
  base="${SWE_PODMAN_STORE_BASE:-/dev/shm}/hbj-$(id -u)-$$"
  export HARBOR_PODMAN_STORE="$base/store" HARBOR_PODMAN_RUNROOT="$base/run" \
         HARBOR_PODMAN_TMPDIR="$base/tmp" XDG_RUNTIME_DIR="$base/xdg" TMPDIR="$base/tmp"
  mkdir -p "$HARBOR_PODMAN_STORE" "$HARBOR_PODMAN_RUNROOT" "$HARBOR_PODMAN_TMPDIR" "$XDG_RUNTIME_DIR"
  cd "$REPO"
  exec "$SWE_PY" -m eval.correctness.run_batch "$@"
fi

exec unshare ${SWE_UNSHARE_FLAGS:--Urm} "$SWE_PY" "$REPO/src/podman_bootstrap.py" \
  -m eval.correctness.run_batch "$@"
