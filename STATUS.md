# Current status — 2026-08-04

The Sandoq integration now follows the validated production OCI-runner
protocol. The earlier image-at-create and fwdproxy/mTLS design was incorrect and
has been replaced:

- Lease the fixed `oci-runner` outer environment without an image field.
- Authenticate to `POST /v1/exec` with a rotatable bearer token read from an
  external, mode-`0600` `OCI_RUNNER_TOKEN_FILE`.
- Pull each task's pinned GHCR image inside the outer runner and start it through
  nested Podman + gVisor (`--runtime runsc`), then route Harbor exec and file
  operations into that persistent task container.
- Verify image digest, root execution, private outer staging plus `podman cp`, startup/cancellation
  cleanup, HTTP-404-confirmed deletion, atexit cleanup, and SIGTERM cleanup.

The adapter rejects non-HTTPS exec URLs before reading/sending credentials,
reads the token through one `O_NOFOLLOW` file descriptor, validates forwarded
environment names, renews leases with logged one-minute retries, and never
mounts an outer-runner directory writable by nested task root.

Fresh live probes on both `chuanyang-login-0` and Slurm CPU nodes confirm:
control-plane lease HTTP 201, `/healthz` HTTP 200, the expected unauthenticated
`/v1/exec` HTTP 401 boundary, and clean lease deletion. Slurm/network access is
not the remaining blocker. This account does not currently have the required
OCI-runner token (the historical successful validation job belonged to another
user), so authenticated `/v1/exec` plus nested image startup cannot yet be
executed live. The canonical fail-fast check is:

```bash
.venv/bin/python sandoq_probe.py
```

The current three-hour lease was also accepted live (HTTP 201), reached healthy
(HTTP 200), enforced the unauthenticated boundary (HTTP 401), and was deleted
with HTTP-404 confirmation. `sandoq_probe.sbatch` performs the full authenticated
nested-image probe on a fresh compute node once the token exists.

See `RUN_SANDOQ.md` for token setup and exact pilot/full-run commands.

The existing 109-task × 2 Opus 4.8 run under `trials/opus_k2` is now finalized
as a **strict-complete, non-canonical baseline**: exact matrix 218/218, infra
audit 218 ok / 0 failed, both judge-model gaps zero, and 218/218 message-tag
provenance records. With the released Opus 4.6 judge it measures 51.8% pass@1,
45.9% SSR, 39.4% pass², mean judge 0.746, U-Corr 1.12, 61.4k
output+reasoning tokens/task, and 19.7 min/task. The Opus 4.8-judge sensitivity
row is 53.7%, 47.7%, 42.2%, and 0.755 respectively.

This baseline used Podman, metagen Opus as both action model and user simulator,
an Opus tagger, selective infra repair, and a 2400s timeout, so it must not be
reported as an exact Table-2 or Sandoq reproduction. The published target is
63% pass@1, 59% SSR, 52% pass², mean judge 0.801, U-Corr 1.38, 74.0k
output+reasoning tokens/task, and 23.3 min/task. A released-runner-matched rerun
additionally needs Gemini 3.1 Pro and OpenRouter credentials. The paper
describes a single multi-label U-Corr tagger, and the released evaluator pins
Gemini 3.1 Pro at temperature 0; the repository's optional three-model ensemble
is not part of that protocol. See `OPUS48_REPRO_REPORT.md` and the two
`pipeline_logs/opus_k2_table2_*_final.json` artifacts.

Everything below is retained as historical investigation notes and is
superseded where it conflicts with this section.

# Historical setup status & evaluation feasibility (2026-07-21)

## TL;DR
- ✅ Benchmark **set up** per the official instructions: repo cloned, `uv sync` OK
  (venv Python 3.12.13; `harbor` + `e2b` import), 109 task specs present.
- ❌ **Cannot yet run the eval end-to-end** for Qwen3.5-4B or Opus-4.8 here. Both
  models are **internal-only**, but SWE-Together runs the coding agent inside an
  **E2B cloud sandbox** that can't reach internal endpoints — plus E2B / GHCR /
  Gemini credentials are all missing, and the judge stage is E2B-only.

## What was done
- `git clone https://github.com/Togetherbench/SWE-Together` → this folder.
- Read the paper (arXiv:2606.29957) metadata + README (official setup).
- `cp .env.example .env`; `uv sync` → exit 0. `.venv/bin/python` = 3.12.13;
  `import harbor`, `import e2b` succeed.
- Confirmed **109 runnable tasks** (each has task.toml + instruction.md +
  tests/test.sh). All task images pinned to **private**
  `ghcr.io/togetherbench/multi-user-turn-codebench/*` (anonymous pull → HTTP 401).

## How SWE-Together runs (and why that blocks us)
- **Stage `run`** (`src/run_eval.py`): Harbor's LocalOrchestrator spins up **one
  sandbox per task** (E2B cloud, or local Docker via `--env-type docker`). The
  coding agent (opencode / claude-code / codex / mini-swe-agent) runs *inside* the
  sandbox and reaches the action model **over the network from there**. A
  **Gemini user-simulator** (host-side, litellm) injects follow-up turns on every run.
- **Stage `judge`** (`eval/run_eval.py` → `eval/correctness/sandbox.py`):
  **E2B-only** (`from e2b import AsyncSandbox`); reuses the E2B template built during
  `run` and runs `claude --print` as the agentic judge (needs Anthropic access from
  inside the E2B sandbox).

## Environment facts
- Container runtime: no `docker`; `podman` present but not serviced; no docker socket.
- Credentials present: **none** of `E2B_API_KEY` / `GEMINI_API_KEY` /
  `OPENROUTER_API_KEY` / public `ANTHROPIC_API_KEY` / `GHCR_TOKEN`.
- Egress goes through a filtering proxy with a **domain allowlist**
  (`http_proxy=http://10.0.2.2:56811`); `no_proxy` covers `10.0.0.0/8`, so internal
  cluster nodes are reachable **directly** from this host (bypassing the allowlist).

## Model reachability matrix
| Model | Endpoint | From this host | From E2B cloud sandbox |
|---|---|---|---|
| **Qwen3.5-4B** | `http://cpu-062-098:8100` — serve_api_v2 deployment `shared-qwen3.5-4b`, 32/32 "serving", internal `10.148.73.203` | ⚠️ TCP-reachable via `no_proxy`, **but** the router currently returns `{"error":"No connected db"}` on `/v1/*` (serving-side fix needed) | ❌ internal-only |
| **Opus-4.8** | metagen via the configured `ANTHROPIC_BASE_URL` and external credential (value intentionally omitted) | ✅ verified — the API returned a completion | ❌ internal-only |

## Blockers (each independently fatal to a stock run)
1. **No sandbox**: E2B key missing (and task images are private GHCR); no local
   Docker daemon; and the judge stage is E2B-only regardless.
2. **No Gemini key** for the mandatory user-simulator.
3. **Internal-only models** are unreachable from an E2B cloud sandbox (architectural,
   not just a missing key).

## UPDATE 2026-07-24 — chosen path = local on-cluster harness; decisive execution-env constraints

The user chose to build a **local on-cluster harness** on the internal **SandoQ** backend
(`sandoq_fleet`), with a Gemini user-sim + GHCR PAT, pilot 3–5 tasks. Deep investigation
found this Claude shell is a locked micro-sandbox good for **authoring**, not **executing**:

- **SandoQ eks-prod gateway UNREACHABLE from here**: `fwdproxy` doesn't resolve; x2p proxy
  (`10.0.2.2:10054`) TLS-fails; direct = `403 Domain not in allowlist`. (mTLS client cert
  *is* present at `$THRIFT_TLS_CL_CERT_PATH`.) `sandoq_fleet` must run from a fwdproxy-capable host.
- **`sandoq_fleet` only drives `ram-opencode-runner`** — a single `opencode run` per task
  (prepare/run/export control server), no repo image / tests / multi-turn user-sim / judge.
  `agent_sandbox_py` ships only opencode-runner images (`ubuntu-pixi`, `ubuntu-uv`); there is
  **no generic-exec SandoQ image**. So SandoQ needs a new generic control-server image to fit
  SWE-Together's task format even once reachable.
- **Gemini UNREACHABLE** (`generativelanguage.googleapis.com` → HTTP 000). Use **Opus (metagen
  x2p gateway)** as the user-sim model instead.
- **Container runtime here is not viable**: `podman` present but overlay driver unusable on the
  overlayfs root; `fuse-overlayfs` mount = "operation not permitted"; `vfs` runs but **Docker Hub
  CDN is blocked** (`production.cloudfront.docker.com` → "Domain not in allowlist") so public
  bases (`ubuntu:24.04`, `python:*-slim`) can't be pulled. `ghcr.io` IS allowlisted (needs PAT).
- **Reachable from here**: internal 10/8 (Qwen vLLM `cpu-000-198:8100` — currently returns
  `No connected db`, serving-side fix needed; dedicated `chuanyang-qwen3.5-4b` at `cpu-068-159`),
  Opus x2p ai-gateway (verified), `ghcr.io` (with PAT), PyPI/GitHub.

**Harbor extensibility (from full interface map):** a new sandbox backend = 5 primitives
(create-from-image/Dockerfile, `exec`, `upload`, `download`, `stop`) as a `BaseEnvironment`,
registered via `EnvironmentConfig.import_path` → **zero edits to Harbor**. The judge needs the
same 5 primitives swapped behind a thin driver in `eval/correctness/sandbox.py`. Local-docker
path already builds each task's `environment/Dockerfile` locally (`force_build=True`), sidestepping
the pinned GHCR image — but 26 tasks `FROM` a private `ghcr.io/togetherbench/*-dev` base (need PAT);
76+ use public bases.

**Recommended plan:** author a **turnkey local-container harness** here (podman `BaseEnvironment`
+ judge retarget + Qwen/Opus action-model wiring + Opus user-sim + 3–5 task pilot + run script),
then **execute on a capable host** (cluster node / devserver with a working container runtime,
image egress, and model reachability). Execution host is the user's call.

## Viable paths to actually produce numbers
- **A — Local, on-cluster path (non-canonical).** Build a local sandbox
  (podman/docker), obtain task images (GHCR PAT, or local builds from each task's
  `environment/Dockerfile`), and port the judge to run locally; wire Qwen (direct),
  Opus (x2p gateway), and a user-sim model (Opus can substitute for Gemini). Faithful
  to the task set but a **modified harness** → not directly comparable to the
  published leaderboard. Real engineering effort.
- **B — Canonical E2B path.** Provide `E2B_API_KEY` + a GHCR PAT (read:packages) +
  `GEMINI_API_KEY`, **and** expose both models on a **publicly reachable**
  gateway/tunnel so E2B sandboxes can reach them. Canonical numbers; needs creds +
  a public tunnel to the internal models.
- **C — Stop at setup.**

## UPDATE 2026-07-29 — local podman run-stage harness BUILT (Qwen baseline)

Path A is now implemented and validated end-to-end on this pod. See
**`RUN_LOCAL.md`** for operator docs.

**Foundation re-verified live (2026-07-29):**
- podman runtime works via one `unshare -Urm` + tmpfs store + vfs driver.
- Qwen3.5-4B reachable **from inside** a `--network host` container.
- With proxy + exact-IP `no_proxy`: pip/npm/conda-forge/github reachable in-container.
- **Prebuilt task images on `ghcr.io/togetherbench/*` are anonymously pullable**
  (no PAT) — so we pull each `task.toml` `docker_image` and never build.
- Blocked (design-shaping): local image builds (apt/`archive.ubuntu.com` → 403,
  host+container); Opus x2p gateway from inside the container (hangs — judge stays host-side).

**What was built:**
- `src/podman_env.py` — `PodmanEnvironment(BaseEnvironment)`: pulls the prebuilt
  image, runs `--network host --http-proxy=false` with proxy + exact-IP no_proxy +
  baked `OPENAI_BASE_URL`; exec/cp primitives. Selected via
  `EnvironmentConfig.import_path` (zero Harbor factory change; `PODMAN` added to
  the enum only for honest `type()`).
- `run_local.sh` — single-`unshare` launcher (tmpfs store, `HARBOR_PODMAN_*`),
  execs `src/run_eval.py … --env-type podman`.
- `src/run_eval.py` — `--env-type podman` → import_path; `openai/Qwen3.5-4B`
  action branch; Opus user-sim via `ANTHROPIC_BASE_URL` gateway.
- `src/podman_bootstrap.py` — the in-userns bootstrap `run_local.sh` execs:
  tmpfs store via the `mount(2)` syscall (dodges setuid-`mount` EPERM), shadows
  `/run` when unwritable (login nodes), sets `HARBOR_PODMAN_*`, execs run_eval.
- `install-opencode.sh.j2` — **apt-free node**: use the image's node if present,
  else install the nodejs.org static tarball (`tar --no-same-owner
  --no-same-permissions … || true`, gated on `node -v`). apt can't run under the
  single-uid userns at all (`_apt`/uid-42 privilege drop fails).
- `lite_llm.py` — retry without `temperature` (Opus 4.8 rejects it; litellm
  won't drop it since its DB lists it as supported).
- `.env` — Qwen (`OPENAI_*`) + Opus gateway (`ANTHROPIC_*`) keys.

**Fixes found during validation:** run/exec as `--user 0:0` (single-uid userns
can't enter image `USER agent`→`setresgid` EPERM); `--ulimit nofile` capped at
the parent hard limit; node from tarball for bun-only images; Opus `temperature`
retry; per-image + global pull throttle.

**Validated end-to-end (3-task pilot, Qwen action + Opus user-sim, podman):**
| task | image | reward | user-sim turns | episodes |
|---|---|---|---|---|
| agent-swarm-task-4a881b | node | 1.00 | 3 (redirect/question/redirect) | 12 |
| agent-swarm-task-ea4bd8 | bun  | 0.00 | 1 (question) | 8 |
| amytis-task-e3714e | bun  | 0.12 | 0 | 4 |
Rewards span 0–1 (realistic for a 4B model); multi-turn user-sim confirmed
working; concurrent multi-container confirmed.

**Run:** `./run_local.sh --model openai/Qwen3.5-4B --agent-type opencode
--user-model anthropic/claude-opus-4-8 --tag <t> --workers N --tasks <...>
--trials-dir trials/<t>`.

**Judge retarget — BUILT (session 2, 2026-07-29; needs a live run to validate):**
- `eval/correctness/podman_judge.py` — `run_judge_in_podman`, signature-compatible
  with `run_judge_in_e2b`. Judge MODEL (Opus 4.8) runs **host-side** via litellm→gateway;
  its single `bash` tool execs into a `PodmanEnvironment` container with the patched
  workspace. Reuses the Phase 1/2 prompts + first-messages VERBATIM (same
  `/tmp/judge_inputs/` layout + write-to-file contract; verdict read back via
  `podman exec cat`). Patch-apply/repo-discovery ported from `sandbox.py`.
- `run_batch.py` + `generate_task_goals.py` dispatch on `JUDGE_ENV=podman` (else E2B);
  new `judge_local.sh` launcher. All 109 tasks already ship `canonical_goals.json`
  (Phase-1 frozen) → judge only needs **Phase 2**; run stage emits `agent/final.patch`.
- **VALIDATED end-to-end** (2026-07-29, Slurm compute node via tunnel): judged the
  amytis-task-e3714e **oracle** patch (17KB) against its frozen 6-goal rubric →
  `judge_score=0.88`, `verdict=equivalent`, 5/6 goals met + 1 correctly flagged
  partial, each with file:line evidence; the judge even ran the canonical `test.sh`
  in-container (8/8 F2P pass). A correct solution scoring high = judge sanity
  confirmed. 171s, `judge_exit_code=0`.

**Run: `./judge_local.sh --plan <plan.json> --workers N`** (plan = list of
`{trial_dir, task_dir, out_name}`; writes `judge_verdict.json` per trial).

**Still not done:** full-suite 109-task sweep (run + judge).

**Host-capability matrix (session 2 ran on `chuanyang-login-0`, a login node):**
- **Login node** — image pulls + tmpfs work, but **cannot run containers** (kernel
  6.12 AL2023 bans `proc`+`sysfs` mounts in userns; `podman run` dies at the proc
  mount, even with `unshare -Urmpf --mount-proc`). Egress via session proxy
  `127.0.0.1:38835`. Good for authoring + pulls only.
- **Slurm compute node** — runs containers via **native rootless podman**
  (`SWE_NATIVE_PODMAN=1`): podman's own userns (newuidmap/newgidmap over the
  `/etc/subgid` range) maps gid 5 so proc+devpts mount cleanly — the single-uid
  `unshare -Urm` map cannot (devpts `gid=5` → EINVAL). Store on tmpfs `/dev/shm`
  (c/storage SIGSEGVs on the overlayfs `/tmp,/scratch`). ghcr/nodejs/Qwen reachable
  **directly**; the **Opus x2p gateway is NOT** — bridge it via a relay on the login
  node (`10.146.35.140:PORT` → `127.0.0.1:38835`; `relay_tmp.py`). Both the run stage
  and the judge were validated this way.

**Validated compute-node recipe** (both stages), from a login-node relay
`python relay_tmp.py 48835 38835`:
```bash
srun -p cpu -A ram -t 45 -N1 --cpus-per-task=16 --mem=150G /bin/bash -c '
  export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
  cd <repo>
  export http_proxy=http://10.146.35.140:48835 https_proxy=$http_proxy \
         HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy \
         no_proxy=10.148.1.105,127.0.0.1,localhost NO_PROXY=$no_proxy \
         SWE_NATIVE_PODMAN=1 SWE_PODMAN_STORE_BASE=/dev/shm
  bash run_local.sh  --model openai/Qwen3.5-4B --agent-type opencode \
       --user-model anthropic/claude-opus-4-8 --workers 2 --tag <t> --tasks <...> --trials-dir trials/<t>
  # judge: bash judge_local.sh --plan <plan.json> --workers N --skip-phase1
'
```
The relay is session-fragile (dies with the Claude session) — fine for validation,
not a long unattended sweep; for that use the pod or give compute nodes real gateway egress.
