#!/usr/bin/env bash
# Single-unshare launcher for the SWE-Together local podman run harness.
#
# podman 3.4.4 here only works inside ONE `unshare -Urm` (user+mount namespace
# giving mount caps), with a tmpfs image store and the vfs driver. The WHOLE
# orchestrator must live in that one session so every podman subprocess it spawns
# shares the same store/namespace and can see each other's containers. This
# wrapper establishes the namespace and hands off to src/podman_bootstrap.py,
# which does the tmpfs mounts (via the mount(2) syscall, not the setuid mount
# binary — see that file), sets HARBOR_PODMAN_*, and execs run_eval.py. All
# arguments are forwarded to run_eval.py; --env-type podman is appended there.
#
# Usage:
#   ./run_local.sh --model openai/Qwen3.5-4B --agent-type opencode \
#       --user-model anthropic/claude-opus-4-8 --tag qwen_pilot \
#       --workers 3 --tasks <t1>,<t2> --trials-dir trials/qwen_pilot
#
# Env knobs:
#   SWE_TMPFS_SIZE     tmpfs size for the image store (default 120G; host has ~740G RAM)
#   SWE_PY             python interpreter (default .venv/bin/python)
#   SWE_UNSHARE_FLAGS  unshare flags for the unshare path (default "-Urm").
#   SWE_NATIVE_PODMAN  set =1 on Slurm COMPUTE nodes: skip the unshare wrapper and
#                      let podman set up its OWN rootless userns via
#                      newuidmap/newgidmap. That maps the full /etc/subgid range
#                      (so gid 5 maps → devpts/proc mount cleanly), which the
#                      single-uid `unshare -Urm` map cannot do. Store goes on tmpfs
#                      /dev/shm (c/storage SIGSEGVs on the overlayfs /tmp,/scratch).
#   SWE_PODMAN_STORE_BASE  base dir for the native store (default /dev/shm).
#   TRIAL_BUDGET_SEC  optional wrapper budget override. When unset, this
#                     launcher derives agent-timeout minus 60s from the CLI.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/scripts/wrapper_budget.sh"
set_swe_wrapper_budget_from_args "$@"
export SWE_REPO="$REPO"
export SWE_OUTER_UID="$(id -u)"           # podman's runtime dir is /tmp/podman-run-<outer uid>
export SWE_TMPFS_SIZE="${SWE_TMPFS_SIZE:-120G}"
export SWE_PY="${SWE_PY:-$REPO/.venv/bin/python}"

if [ -n "${SWE_NATIVE_PODMAN:-}" ]; then
  base="${SWE_PODMAN_STORE_BASE:-/dev/shm}/hb-$(id -u)-$$"
  export HARBOR_PODMAN_STORE="$base/store" HARBOR_PODMAN_RUNROOT="$base/run" \
         HARBOR_PODMAN_TMPDIR="$base/tmp" XDG_RUNTIME_DIR="$base/xdg" TMPDIR="$base/tmp"
  mkdir -p "$HARBOR_PODMAN_STORE" "$HARBOR_PODMAN_RUNROOT" "$HARBOR_PODMAN_TMPDIR" "$XDG_RUNTIME_DIR"
  cd "$REPO"
  exec "$SWE_PY" src/run_eval.py "$@" --env-type podman
fi

exec unshare ${SWE_UNSHARE_FLAGS:--Urm} "$SWE_PY" "$REPO/src/podman_bootstrap.py" \
  src/run_eval.py "$@" --env-type podman
