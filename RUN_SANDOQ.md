# Running SWE-Together with Sandoq

The Sandoq backend uses the production `oci-runner` contract: it leases a fixed
outer runner, authenticates to `POST /v1/exec`, pulls the task image there, and
starts a persistent nested container with Podman + gVisor (`runsc`). Harbor
commands and files are then routed into that nested task container.

File transfer uses a mode-`0700` directory private to the outer runner followed
by `podman cp`; the nested task receives no writable outer-filesystem mount.
Lease-provided command endpoints must be HTTPS before the bearer token is sent.

This is a sandbox-backend substitution for the released canonical runner's E2B
setup. It preserves the pinned task images and agent/verifier flow, but the
manuscript does not identify the infrastructure behind its wall-clock number.

## Prerequisites

- A rotatable OCI-runner bearer token in a regular, non-symlink, mode-`0600`
  file. Ask the Sandoq/oci-runner owner for this credential; do not commit it.
- The action-model and user-simulator credentials required by the chosen run.
  The paper-matched Opus cohort needs `OPENROUTER_API_KEY` and
  `GEMINI_API_KEY`. The internal metagen route instead needs
  `ANTHROPIC_API_KEY` plus an upstream route reachable from inside Sandoq.
- Direct reachability to `https://sandoq.eks-prod.cf.aws.metafb.cloud`. Both the
  current login node and Slurm CPU nodes have this; Slurm does not supply the
  bearer token.

```bash
mkdir -p "$HOME/.config/oci-runner"
# Write the supplied token as the only line, then:
chmod 0600 "$HOME/.config/oci-runner/token"
export OCI_RUNNER_TOKEN_FILE="$HOME/.config/oci-runner/token"
```

## Validate the complete transport

The probe leases the outer runner, verifies the unauthenticated `401` boundary,
authenticates, pulls the exact pinned task image, starts it with `runsc`, runs a
persistent-state command, and verifies session deletion with HTTP `404`.

```bash
.venv/bin/python sandoq_probe.py

# Equivalent fresh Slurm-node validation (inherits OCI_RUNNER_TOKEN_FILE):
sbatch --export=ALL sandoq_probe.sbatch
```

Without a bearer token, the narrower control-plane check still proves lease,
health, authentication boundary, and confirmed deletion. It does **not** prove
authenticated command execution or nested gVisor:

```bash
.venv/bin/python sandoq_probe.py --control-plane-only

# Same check on a fresh Slurm node:
sbatch --export=ALL,SANDOQ_CONTROL_PLANE_ONLY=1 sandoq_probe.sbatch
```

Full-mode success ends with `SANDOQ OCI-RUNNER END-TO-END: PASS`; in that mode,
a missing token fails before creating a lease. Control-only mode does not read
a token, and its success is labelled `SANDOQ CONTROL PLANE: PASS` so it cannot
be confused with the authenticated result. Use `SANDOQ_TEST_IMAGE` to probe
another pinned image.

## Pilot run and judge

```bash
bash run_sandoq.sh \
  --model openrouter/anthropic/claude-opus-4-8 \
  --user-model gemini/gemini-3.1-pro-preview \
  --agent-type opencode --reasoning-effort high \
  --agent-timeout 4800 --workers 2 \
  --tasks agent-swarm-task-4a881b,agent-swarm-task-ea4bd8 \
  --trials-dir trials/sandoq_pilot --tag sandoq_pilot
```

Build a correctness plan with `eval.run_eval`, or pass an existing plan to the
Sandoq judge. Pin Opus 4.6 to match the paper's judge model:

```bash
export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6
bash judge_sandoq.sh \
  --plan trials/sandoq_pilot/judge_plan.json \
  --workers 2 --skip-phase1
```

The host-side judge also needs `ANTHROPIC_API_KEY` and
`ANTHROPIC_BASE_URL`. Its model route is independent of Sandoq control-plane
reachability. This remains an approximate/backend-substituted comparison: the
canonical E2B evaluator runs the Claude CLI in-sandbox, while the Sandoq path
drives the same prompts and bash tool from a host-side LiteLLM loop.

## Paper-matched Opus 4.8 configuration

Use all 109 tasks, two independent replicates, OpenCode, high reasoning effort,
and an 80-minute agent timeout. Prefer separate replicate directories, matching
`canonical_full109.json`, so each cohort is independently resumable:

```bash
.venv/bin/python launch.py canonical_full109.json \
  --stage run --models opencode_opus48 --env-type sandoq --execute
.venv/bin/python launch.py canonical_full109.json \
  --stage judge --models opencode_opus48 --env-type sandoq --execute
```

The published Table 2 target is pass@1 `63%`, SSR `59%`, pass² `52%`, mean
judge `0.801`, U-Corr `1.38`, output+reasoning tokens/task `74.0k`, and
minutes/task `23.3`. U-Corr must come from message tags
(`#correction + 0.2 × #nudge`), not the raw number of simulator messages.
The launcher enables the strict 109×2 completeness gate and pins the judge to
Opus 4.6. A Sandoq row remains a backend-substituted comparison because the
released canonical runner uses E2B; do not label it an exact wall-clock
reproduction.

## Transport knobs

- `OCI_RUNNER_BASE_URL` (default: production direct endpoint)
- `OCI_RUNNER_ENVIRONMENT` (default: `oci-runner`)
- `OCI_RUNNER_LEASE_DURATION` (default: `3h`; leases are also renewed every
  `min(5m, lease/3)`, with faster bounded retries after failures)
- `OCI_RUNNER_CREATE_DEADLINE` (default: `300s`)
- `OCI_RUNNER_PULL_TIMEOUT` (default: `1200s`)
- `OCI_RUNNER_EXEC_TIMEOUT` (default for Harbor commands without an explicit
  timeout: `3600s`)
- `SANDOQ_BUILD_TIMEOUT_MULTIPLIER` (default: `3`)
- `SANDOQ_FORWARD_ENV` (comma-separated variables baked into the nested task)
- `SANDOQ_HTTP_PROXY` (explicit Sandoq API proxy; generic host proxy variables
  are intentionally ignored)

Harbor CPU and memory requests become nested Podman cgroup limits. Storage is
checked fail-closed as a minimum available rootfs capacity because the outer
runner's Podman/storage-driver combination does not expose a portable writable
layer quota flag.

Normal teardown and startup-failure cleanup delete sessions and confirm them
absent. Interpreter-exit and SIGTERM handlers make the same cleanup attempt
within a bounded deadline; if the network or process deadline prevents
confirmation, the server-side lease expiry remains the final fallback. During
ordinary teardown, failed confirmation retains session state for a retry
instead of silently forgetting it.
