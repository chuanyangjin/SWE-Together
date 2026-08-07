#!/usr/bin/env bash
# Run-stage launcher for the SWE-Together **Sandoq OCI-runner** harness.
#
# Unlike run_local.sh (podman), sandoq is a REMOTE sandbox service: there is no
# local container runtime, no `unshare`, no tmpfs image store. This wrapper just
# exports the OCI_RUNNER_* config (+ leaves model/user-sim env from .env intact)
# and execs run_eval.py with --env-type sandoq. Each trial leases the fixed outer
# runner, authenticates to /v1/exec with a bearer token, and starts the task image
# under nested Podman + gVisor (see src/sandoq_env.py).
#
# WHERE TO RUN: a host that reaches the Sandoq gateway and has the OCI runner
# token file. Both the login host and Slurm compute nodes reach the gateway
# directly; Slurm alone does not provide the token.
#
#   srun -A ram -t 60 -N1 --cpus-per-task=16 --mem=64G --export=ALL /bin/bash -c '
#     cd <repo>
#     unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # sandoq is reached directly
#     bash run_sandoq.sh --model openai/Qwen3.5-4B --agent-type opencode \
#        --user-model anthropic/claude-opus-4-8 --tag sandoq_pilot \
#        --workers 4 --tasks <t1>,<t2> --trials-dir trials/sandoq_pilot'
#
# All arguments are forwarded to run_eval.py; --env-type sandoq is appended.
#
# Required: OCI_RUNNER_TOKEN_FILE (regular mode-0600 bearer-token file).
# Optional: OCI_RUNNER_BASE_URL, OCI_RUNNER_ENVIRONMENT,
# OCI_RUNNER_LEASE_DURATION, OCI_RUNNER_CREATE_DEADLINE,
# OCI_RUNNER_PULL_TIMEOUT, OCI_RUNNER_EXEC_TIMEOUT, SANDOQ_FORWARD_ENV, SWE_PY.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/scripts/wrapper_budget.sh"
set_swe_wrapper_budget_from_args "$@"
export SWE_PY="${SWE_PY:-$REPO/.venv/bin/python}"

export OCI_RUNNER_BASE_URL="${OCI_RUNNER_BASE_URL:-${SANDOQ_BASE_URL:-https://sandoq.eks-prod.cf.aws.metafb.cloud}}"
export OCI_RUNNER_ENVIRONMENT="${OCI_RUNNER_ENVIRONMENT:-${SANDOQ_OCI_ENV:-oci-runner}}"
export OCI_RUNNER_LEASE_DURATION="${OCI_RUNNER_LEASE_DURATION:-${SANDOQ_LEASE_DURATION:-3h}}"
export OCI_RUNNER_TOKEN_FILE="${OCI_RUNNER_TOKEN_FILE:-$HOME/.config/oci-runner/token}"

cd "$REPO"
PYTHONPATH="$REPO/src:$REPO/external/harbor/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$SWE_PY" -c 'from sandoq_env import _read_token; _read_token()'
exec "$SWE_PY" src/run_eval.py "$@" --env-type sandoq
