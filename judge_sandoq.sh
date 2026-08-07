#!/usr/bin/env bash
# Judge-stage launcher for the SWE-Together **Sandoq OCI-runner** harness.
#
# Runs the agentic correctness judge (eval/correctness/run_batch.py) with
# JUDGE_ENV=sandoq: the judge MODEL (Opus) runs HOST-SIDE via the metagen
# gateway, and its bash tool execs into a remote nested task container holding
# the patched workspace (eval/correctness/sandoq_judge.py). No local container
# runtime — the sandoq counterpart of judge_local.sh.
#
# WHERE TO RUN: a host that reaches BOTH (a) the sandoq gateway and (b) the Opus
# x2p gateway. Compute nodes reach sandoq directly but NOT the Opus gateway, so
# the judge needs the login-node relay for ANTHROPIC_BASE_URL (see STATUS.md /
# memory swe-together-exec-hosts). Typical invocation, with a relay
# `python relay_tmp.py 48835 38835` running on the login node:
#
#   srun -A ram -t 45 -N1 --cpus-per-task=16 --mem=64G --export=ALL /bin/bash -c '
#     cd <repo>
#     # Sandoq ignores generic proxy variables and stays direct.  Keep the real
#     # ANTHROPIC_BASE_URL from .env; route that HTTP traffic through the relay.
#     export http_proxy=http://<login-ip>:<relay-port> https_proxy=$http_proxy
#     export HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
#     bash judge_sandoq.sh --plan trials/sandoq_pilot/judge_plan.json --workers 3 --skip-phase1'
#
# All arguments are forwarded to `python -m eval.correctness.run_batch`.
#
# Auth (from .env): ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL (the metagen gateway).
# Env knobs: JUDGE_PODMAN_MODEL (host-side judge model; default
# anthropic/claude-opus-4-6 — shared with the podman judge), OCI_RUNNER_* (see
# run_sandoq.sh), SWE_PY.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWE_PY="${SWE_PY:-$REPO/.venv/bin/python}"
export JUDGE_ENV=sandoq

export OCI_RUNNER_BASE_URL="${OCI_RUNNER_BASE_URL:-${SANDOQ_BASE_URL:-https://sandoq.eks-prod.cf.aws.metafb.cloud}}"
export OCI_RUNNER_ENVIRONMENT="${OCI_RUNNER_ENVIRONMENT:-${SANDOQ_OCI_ENV:-oci-runner}}"
export OCI_RUNNER_LEASE_DURATION="${OCI_RUNNER_LEASE_DURATION:-${SANDOQ_LEASE_DURATION:-3h}}"
export OCI_RUNNER_TOKEN_FILE="${OCI_RUNNER_TOKEN_FILE:-$HOME/.config/oci-runner/token}"

cd "$REPO"
PYTHONPATH="$REPO/src:$REPO/external/harbor/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$SWE_PY" -c 'from sandoq_env import _read_token; _read_token()'
exec "$SWE_PY" -m eval.correctness.run_batch "$@"
