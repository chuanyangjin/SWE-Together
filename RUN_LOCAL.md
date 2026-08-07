# Running SWE-Together locally with podman (on-cluster, non-canonical)

This is a **non-canonical** path for running the SWE-Together *run stage* on an
internal cluster pod that cannot reach E2B/Gemini and whose models are
internal-only. It replaces the E2B sandbox with a local rootless-**podman**
container, uses **Qwen3.5-4B** as the action model and **Opus 4.8** (via the
metagen x2p gateway) as the user-simulator. Results are **not** directly
comparable to the published E2B leaderboard.

> Scope: **run stage only** (agent trajectories + Harbor's in-container
> `test.sh` reward). The agentic completeness judge (`eval/correctness/`) is not
> yet retargeted off E2B — see "Not yet supported".

## Why it's built this way (verified constraints)

On `claude-sandbox-pod` (root, no CAP_SYS_ADMIN):

- **podman only works inside one `unshare -Urm`** with a **tmpfs** image store
  and the **vfs** driver. `run_local.sh` establishes that single session and
  runs the whole orchestrator inside it, so every container shares one store.
- **Task images are pulled, never built.** 107/109 task Dockerfiles
  `apt-get install`, and `archive.ubuntu.com` is not in the egress allowlist
  (`403 Domain not in allowlist`) — on host *and* container. But every
  `task.toml` pins a prebuilt image on `ghcr.io/togetherbench/...` which **is**
  allowlisted and **anonymously pullable**, so we pull it.
- **Networking:** the container runs `--network host` (Qwen is only reachable
  via the host's direct 10/8 route). podman's proxy passthrough is disabled and
  replaced with the host filtering proxy for external egress **plus an exact-IP
  `no_proxy`** for the model endpoint (the 22.04 curl can't CIDR-match), so
  agent installs (npm/pip) go via the proxy while model calls go direct.

## One-time setup

1. `uv sync` (already done — `.venv/` present, `harbor` importable).
2. Fill `.env` (gitignored). The local-harness keys:
   ```
   OPENAI_API_KEY=<redacted>                 # Qwen action model key
   OPENAI_BASE_URL=http://10.148.1.105:8100/v1       # Qwen OpenAI-compat endpoint
   ANTHROPIC_API_KEY=<redacted>             # metagen gateway key (user-sim)
   ANTHROPIC_BASE_URL=http://anthropic.ai-gateway.x2p.facebook.net
   ```
   The Qwen endpoint IP/port and the gateway key rotate — re-check
   `/checkpoint/ram/shared/vllm_deployments_v2/shared-qwen3.5-4b/proxy_info.json`
   and the memory notes if a run 401s / connection-refuses.

## Run

`run_local.sh` forwards all args to `src/run_eval.py` and appends
`--env-type podman`:

```bash
./run_local.sh \
  --model openai/Qwen3.5-4B --agent-type opencode \
  --user-model anthropic/claude-opus-4-8 \
  --tag qwen_pilot --workers 3 --agent-timeout 4800 \
  --tasks agent-swarm-task-4a881b,<t2>,<t3> \
  --trials-dir trials/qwen_pilot
```

- `--model openai/Qwen3.5-4B` — action model (routes to `OPENAI_BASE_URL`).
- `--agent-type opencode` — the canonical SWE-Together coding agent.
- `--user-model anthropic/claude-opus-4-8` — Opus user-sim via the gateway.
- `--workers N` — concurrent trials (each its own container in the shared store).
- Omit `--tasks` to run all 109 (each image is pulled on demand).

Env knobs: `SWE_TMPFS_SIZE` (image-store tmpfs, default `120G`), `SWE_PY`
(interpreter, default `.venv/bin/python`).

Outputs: `trials/<...>/` (per-trial `result.json`, `agent/`, `verifier/reward.txt`),
`pipeline_logs/eval-<tag>-summary.json`.

## How it works

- `src/podman_env.py` — `PodmanEnvironment(BaseEnvironment)`, selected per trial
  via `EnvironmentConfig.import_path = "podman_env:PodmanEnvironment"` (set in
  `src/run_eval.py` when `--env-type podman`); needs no Harbor factory change.
  Reads the store/runtime paths from `HARBOR_PODMAN_*` (exported by
  `src/podman_bootstrap.py`, which `run_local.sh` execs inside the unshare to set
  up the tmpfs store via the `mount(2)` syscall), pulls the task's `docker_image`
  (unless already in the store), runs it `--network host` as root with a capped
  `nofile` ulimit, and does exec/upload/download via `podman exec`/`cp`.
- Action model: `run_eval.py`'s `build_agent_env` `openai` branch +
  `PodmanEnvironment` baking `OPENAI_BASE_URL` into the container (Harbor's
  opencode adapter forwards the key but not the base URL).
- User-sim: host-side litellm to the gateway (`user_api_base` = `ANTHROPIC_BASE_URL`).

## Non-canonical deviations (document when reporting numbers)

- Local podman sandbox instead of E2B; containers run **as root** (`--user 0:0`)
  because the single-uid `unshare` userns can't enter the image's `USER agent`
  (uid 1001) — `setresgid` fails; `--network host`; approximate internet
  isolation for `allow_internet=False` tasks (external egress withheld, model
  route kept).
- `install-opencode.sh.j2` **never uses apt** (apt can't run under the single-uid
  userns — it drops privileges to `_apt`/uid 42; the Ubuntu archive is also
  blocked). It uses the image's node if present, else installs the official
  static node build from **nodejs.org** (`--no-same-owner`) into `/opt/node` and
  symlinks it onto `/usr/local/bin`. So bun-only task images (no node) work too.
- User-sim is Opus (gateway), not Gemini. Opus 4.8 rejects the `temperature`
  param, so harbor's `lite_llm.py` retries without it (patched).

## Judge stage (agentic correctness judge)

The Phase-2 completeness judge now has a local podman backend
(`eval/correctness/podman_judge.py`), selected with `JUDGE_ENV=podman`. Unlike
E2B (which runs `claude --print` in-sandbox), the judge **model** runs host-side
via litellm→gateway; its single `bash` tool execs into a `PodmanEnvironment`
container holding the patched workspace. The Phase 1/2 prompts, the
`/tmp/judge_inputs/` layout, and the write-verdict-to-file contract are reused
verbatim. All 109 tasks already ship `canonical_goals.json` (Phase-1 frozen), so
only Phase 2 runs; the run stage emits each trial's `agent/final.patch` (the
Phase-2 input).

```bash
# plan.json = [{"trial_dir": "<abs>", "task_dir": "<abs>", "out_name": "judge_verdict.json"}, ...]
./judge_local.sh --plan plan.json --workers 3        # scores each trial → judge_verdict.json
```

Auth: `ANTHROPIC_API_KEY` + `ANTHROPIC_BASE_URL` (the gateway). Model override:
`JUDGE_PODMAN_MODEL` (default `anthropic/claude-opus-4-8`).

> Requires a host that both runs containers and reaches the gateway. On a Slurm
> **login node** containers can't run at all (kernel bans `proc`/`sysfs` mounts in
> userns); on a **compute** node use `SWE_NATIVE_PODMAN=1` (native rootless podman —
> proc+devpts mount cleanly via the full subgid map) and bridge the x2p gateway with
> a relay on the login node. Both the run stage and this judge were validated that
> way (oracle patch → 0.88 "equivalent"). Full recipe + host matrix in STATUS.md.

## Not yet supported

- **Local image builds** — impossible here (apt blocked); rely on prebuilt ghcr
  images.

## Troubleshooting

- `podman ... image store SIGSEGV` — the store isn't on tmpfs; you're not inside
  `run_local.sh`'s unshare session. Always launch via `run_local.sh`.
- Model calls return an HTML error page — the endpoint host isn't in the
  container `no_proxy`; ensure `OPENAI_BASE_URL` is set in `.env` (its host is
  auto-added to `no_proxy`).
- `apt-get ... 403 Domain not in allowlist` during a build — expected; we never
  build. Ensure the task's `docker_image` is set to a pullable prebuilt image.
