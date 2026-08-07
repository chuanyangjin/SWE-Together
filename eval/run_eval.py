"""End-to-end eval orchestrator — runs the three-step protocol in
`eval/eval_design.md` against one (task, agent, sim) cohort of trial dirs,
then computes the final per-task metrics designed in that doc.

Pipeline:
  step 1   correctness    — Phase 1 (per-task frozen rubric, run-once cached
                            at tasks/<task>/canonical_goals.json) +
                            Phase 2 (per-trial scoring against the rubric in
                            an E2B sandbox)             → judge_verdict.json
  step 2   intent_coverage — LLM match-table             → intent_coverage_verdict.json
  step 2b  tag_messages    — per-message tags → intent_coverage_verdict.json::trial_msg_tags
                             drives User Correction (user_metrics)
  step 2c  optional three-way U-Corr ensemble: Opus 4.6 second tags +
                              GPT-5.5 arbitration → trial_msg_tags_3way
  aggregate — compute per-task metrics over all replicate trials

One invocation = one (agent, sim) cohort of trials. The trials_root directory
holds k replicate runs across many tasks; we group by task name (the part
before `__` in the trial dir name) and aggregate within each group.

Usage:
    .venv/bin/python -m eval.run_eval \\
        --trials-root trials_eval_pilot_10_task_r1 \\
        --tasks-root  tasks \\
        --output-dir  pipeline_logs/run_judge_cmp_r1 \\
        --model-tag   ds-pro-gemini-3.1-pro \\
        --correctness-workers 50 \\
        --intent-coverage-workers 5

Each step writes per-trial verdicts in-place under the trial dir; re-running
is idempotent (existing verdicts are reused unless --force-<step> is passed).
Aggregation reads those verdicts and emits per-task summary JSON + Markdown.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import logging
import math
import os
import secrets
import socket
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
sys.path.insert(0, str(REPO_ROOT))
from eval.user_behavior import user_metrics as kg  # noqa: E402 - path setup above
from eval.patch_utils import patch_file_has_changes  # noqa: E402 - path setup above
from eval.correctness.run_batch import (  # noqa: E402 - shared strict validators
    _phase2_verdict_issues,
    _rubric_issues,
)

# Infra sentinel (src/, stdlib-only). Used to EXCLUDE infra-failed trials (agent
# never ran — provider/sandbox error) from scoring; non-infra failures score 0.
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    from eval_infra_sentinel import (  # noqa: E402
        SIDECAR_VERSION as _INFRA_SIDECAR_VERSION,
        classify_or_load as _classify_infra,
        classify_trial as _classify_infra_fresh,
    )
except Exception:
    _INFRA_SIDECAR_VERSION = 2
    _classify_infra = None
    _classify_infra_fresh = None

# Correctness pass bar.
SUCCESS_THRESHOLD = 0.85
DEFAULT_EXPECTED_JUDGE_MODEL = "claude-opus-4-6"

logger = logging.getLogger("run_eval")


def _arbiter_health_url(proxy_url: str) -> str:
    parsed = urlsplit(proxy_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise RuntimeError(
            "Bundled arbiter auto-start requires a loopback http:// URL"
        )
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname.lower() == "localhost"
    if not loopback or parsed.username is not None or parsed.password is not None:
        raise RuntimeError(
            "Bundled arbiter auto-start is restricted to credential-free loopback URLs"
        )
    return f"{parsed.scheme}://{parsed.netloc}/health"


def _arbiter_is_healthy(health_url: str, client_token: str) -> bool:
    """Accept only the authenticated bundled service, never a generic /health."""
    try:
        response = httpx.get(
            health_url,
            headers={"Authorization": f"Bearer {client_token}"},
            timeout=2.0,
            trust_env=False,
        )
        payload = response.json()
        return (
            response.status_code == 200
            and payload.get("ok") is True
            and payload.get("service") == "swe-together-oauth-proxy"
            and payload.get("client_auth") is True
        )
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


def _reserve_arbiter_listener(proxy_url: str) -> socket.socket:
    """Bind the exact loopback port before spawning, eliminating reuse races."""
    parsed = urlsplit(proxy_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    if host.lower() == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:  # _arbiter_health_url already restricts hostnames.
        raise RuntimeError("Could not resolve bundled arbiter loopback host") from exc
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    try:
        listener.bind((host, port))
        listener.listen(16)
    except OSError as exc:
        listener.close()
        raise RuntimeError(
            "Bundled arbiter port is already occupied; refusing to reuse an "
            "unowned loopback service"
        ) from exc
    return listener


@contextlib.contextmanager
def _private_client_auth_file(token: str):
    """Hand a token to the child through a 0700 directory and 0600 file."""
    with tempfile.TemporaryDirectory(prefix="swe-arbiter-auth-") as raw:
        private_dir = Path(raw)
        private_dir.chmod(0o700)
        token_path = private_dir / "token"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(token_path, flags, 0o600)
        try:
            os.write(descriptor, token.encode("utf-8"))
        finally:
            os.close(descriptor)
        yield token_path


def _validated_codex_auth_path(value: Path | None) -> Path:
    path = (
        value
        or Path(os.environ.get("CODEX_HOST_AUTH_JSON", "~/.codex/auth.json"))
    ).expanduser()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"Codex auth file is unavailable: {path}; run `codex login` first"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Codex auth path must be a regular non-symlink file: {path}")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f"Codex auth file must be owned by uid {os.getuid()}: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RuntimeError(f"Codex auth file must not be group/world accessible: {path}")
    return path.resolve()


def _read_external_arbiter_token(proxy_url: str, token_path: Path) -> str:
    """Load manual proxy auth without putting the bearer in argv or the URL."""
    parsed = urlsplit(proxy_url)
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("Arbiter credentials must not be embedded in the URL")
    try:
        is_loopback = bool(parsed.hostname) and ipaddress.ip_address(
            parsed.hostname
        ).is_loopback
    except ValueError:
        is_loopback = (parsed.hostname or "").lower() == "localhost"
    if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback):
        raise RuntimeError(
            "Authenticated arbiter URLs must use HTTPS or loopback HTTP"
        )
    from proxies.oauth_proxy import read_client_auth_token

    return read_client_auth_token(token_path.expanduser())


@contextlib.contextmanager
def _managed_arbiter_proxy(
    *,
    proxy_url: str,
    auto_start: bool,
    auth_json: Path | None,
    log_path: Path,
):
    """Own one authenticated bundled proxy, or leave an external URL untouched.

    Auto-start never trusts or reuses a pre-existing loopback listener.  The
    parent reserves the port before spawn, creates a fresh bearer capability,
    and accepts readiness only from a service that proves knowledge of it.
    """
    if not auto_start:
        parsed = urlsplit(proxy_url)
        if parsed.username is not None or parsed.password is not None:
            raise RuntimeError("Arbiter credentials must not be embedded in the URL")
        yield None
        return
    health_url = _arbiter_health_url(proxy_url)
    parsed = urlsplit(proxy_url)
    listener = _reserve_arbiter_listener(proxy_url)
    with contextlib.closing(listener):
        auth_path = _validated_codex_auth_path(auth_json)
        port = parsed.port or 80
        log_path.parent.mkdir(parents=True, exist_ok=True)
        client_token = secrets.token_urlsafe(48)
        with _private_client_auth_file(client_token) as token_path:
            log_handle = log_path.open("ab")
            process = None
            try:
                try:
                    process = subprocess.Popen(
                        [
                            PY,
                            str(REPO_ROOT / "src" / "proxies" / "oauth_proxy.py"),
                            "--host",
                            parsed.hostname or "127.0.0.1",
                            "--port",
                            str(port),
                            "--auth-json",
                            str(auth_path),
                            "--client-auth-file",
                            str(token_path),
                            "--listen-fd",
                            str(listener.fileno()),
                        ],
                        cwd=REPO_ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        pass_fds=(listener.fileno(),),
                    )
                except OSError as exc:
                    raise RuntimeError(
                        f"failed to start bundled arbiter proxy: {exc}"
                    ) from exc

                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    if _arbiter_is_healthy(health_url, client_token):
                        logger.info(
                            "started authenticated bundled arbiter proxy pid=%d",
                            process.pid,
                        )
                        # Retain the parent's duplicate through adjudication. If
                        # the child later exits, no other process can bind this
                        # port and receive bearer-bearing inference requests.
                        # The child loaded the token before serving health; retain
                        # it only in this process's memory for the adjudication.
                        token_path.unlink(missing_ok=True)
                        break
                    return_code = process.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            f"bundled arbiter proxy exited rc={return_code}; see {log_path}"
                        )
                    time.sleep(0.25)
                else:
                    raise RuntimeError(
                        f"bundled arbiter proxy was not healthy within 30s; see {log_path}"
                    )
                yield client_token
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                log_handle.close()


# ── plan discovery ───────────────────────────────────────────────────────────

def discover_jobs(
    trials_roots: list[Path],
    tasks_root: Path,
    coverage_names: list[str] | None = None,
    judge_names: list[str] | None = None,
) -> list[dict]:
    """Pair each trial dir under each `trials_roots` element with its task dir
    under `tasks_root` by prefix-matching the part before `__`.

    Trial dir names are sometimes truncated copies of the task name (e.g.
    `comfyui-frontend-autoscale-layou__abc` ↔ `comfyui-frontend-autoscale-layout`),
    so we match on `task_dir.name.startswith(prefix)` rather than equality.

    Per-root verdict-filename overrides: `coverage_names` and `judge_names`
    may be passed as parallel lists; each trial inherits the
    overrides of its root. The pilot uses this to read cohort-tagged coverage
    verdicts (`intent_coverage_verdict_v2_freeLLM_r{1,2,3}.json`) without
    renaming files on disk.
    """
    tasks_root = tasks_root.resolve()
    task_dirs = sorted(d for d in tasks_root.iterdir() if d.is_dir())
    if not task_dirs:
        raise SystemExit(f"no task dirs under {tasks_root}")

    if coverage_names and len(coverage_names) not in (1, len(trials_roots)):
        raise SystemExit("--coverage-out-name must be repeated to match --trials-root or given once")
    if judge_names and len(judge_names) not in (1, len(trials_roots)):
        raise SystemExit("--judge-out-name must be repeated to match --trials-root or given once")

    def _pick(names: list[str] | None, i: int, default: str) -> str:
        if not names:
            return default
        return names[i] if len(names) > 1 else names[0]

    jobs: list[dict] = []
    unpaired: list[str] = []
    for i, root in enumerate(trials_roots):
        root = root.resolve()
        cov_name = _pick(coverage_names, i, "intent_coverage_verdict.json")
        judge_name = _pick(judge_names, i, "judge_verdict.json")
        for trial in sorted(root.iterdir()):
            if not trial.is_dir() or "__" not in trial.name:
                continue
            prefix = trial.name.rsplit("__", 1)[0]
            match = next((t for t in task_dirs if t.name == prefix), None)
            if match is None:
                cands = [t for t in task_dirs if t.name.startswith(prefix)]
                if cands:
                    match = max(cands, key=lambda t: len(t.name))
            if match is None:
                unpaired.append(trial.name)
                continue
            jobs.append({
                "trial_dir": str(trial),
                "task_dir": str(match),
                "task": match.name,
                "cohort": root.name,
                "coverage_out_name": cov_name,
                "judge_out_name": judge_name,
            })
    if unpaired:
        logger.warning("dropped %d unpaired trial dirs: %s",
                       len(unpaired), unpaired[:5])
    return jobs


def write_plan(jobs: list[dict], plan_path: Path, out_name: str) -> Path:
    """Write the step-specific plan (with `out_name` injected) to disk and
    return its path. Each step batch runner reads {trial_dir, task_dir,
    out_name} jobs from a JSON list."""
    plan = [
        {"trial_dir": j["trial_dir"], "task_dir": j["task_dir"], "out_name": out_name}
        for j in jobs
    ]
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2))
    return plan_path


# ── step runners (subprocess each batch CLI) ─────────────────────────────────

def _run_subprocess(cmd: list[str], step: str) -> int:
    logger.info("step %s: %s", step, " ".join(cmd))
    t0 = time.monotonic()
    rc = subprocess.call(cmd, cwd=REPO_ROOT)
    logger.info("step %s exited rc=%d elapsed=%.1fs", step, rc, time.monotonic() - t0)
    return rc


def run_step_correctness(plan: Path, summary: Path, workers: int,
                         force: bool, extra: list[str]) -> int:
    cmd = [
        PY, "-m", "eval.correctness.run_batch",
        "--plan", str(plan),
        "--workers", str(workers),
        "--summary", str(summary),
        *(["--force"] if force else []),
        *extra,
    ]
    return _run_subprocess(cmd, "1-correctness")


def run_step_intent_coverage(plan: Path, summary: Path, workers: int,
                             force: bool, model: str | None,
                             extra: list[str]) -> int:
    cmd = [
        PY, "-m", "eval.user_behavior.run_batch",
        "--plan", str(plan),
        "--workers", str(workers),
        "--summary", str(summary),
        *(["--force"] if force else []),
        *(["--model", model] if model else []),
        *extra,
    ]
    return _run_subprocess(cmd, "2-intent_coverage")


def run_step_tag_messages(
    trials_roots: list[Path],
    model: str,
    workers: int,
    force: bool,
    *,
    require_provenance: bool = False,
    output_sidecar: Path | None = None,
    backend: str = "litellm",
) -> int:
    """Step 2b — per-message tagging → trial_msg_tags in each verdict. Pinned
    gemini-3.1-pro @ temp 0, versioned prompt; drives User Correction."""
    cmd = [
        PY,
        "-m",
        "eval.user_behavior.tag_messages",
        "--model",
        model,
        "--backend",
        backend,
        "--workers",
        str(workers),
        *(["--force"] if force else []),
        *(["--require-provenance"] if require_provenance else []),
        *(["--output-sidecar", str(output_sidecar)] if output_sidecar else []),
    ]
    for r in trials_roots:
        cmd += ["--trials-root", str(r)]
    return _run_subprocess(cmd, "2b-tag_messages")


class _BearerArbiterLLM:
    """GPT arbiter client that keeps the per-run bearer out of argv and logs."""

    def __init__(self, base: str, model: str, client_token: str):
        self.base = base.rstrip("/")
        self.model = model
        self._client_token = client_token

    async def tag(self, messages: list[dict]) -> dict[int, list[str]]:
        from eval.user_behavior import adjudicate_3way as adjudication

        prompt = adjudication.SYSTEM + "\n\n" + adjudication._build_user(messages)
        async with httpx.AsyncClient(timeout=240.0, trust_env=False) as client:
            response = await client.post(
                f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self._client_token}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                raise RuntimeError(str(body["error"])[:300])
            parsed = adjudication.parse_json(
                body["choices"][0]["message"]["content"]
            )
            return {
                row["trial_idx"]: [
                    tag
                    for tag in (row.get("tags") or [])
                    if tag in adjudication.VALID
                ]
                for row in parsed.get("results", [])
                if isinstance(row, dict) and isinstance(row.get("trial_idx"), int)
            }


def run_step_adjudicate_3way(
    trials_roots: list[Path],
    *,
    judge_b_sidecar: Path,
    judge_a_model: str,
    judge_b_model: str,
    arbiter_model: str,
    arbiter_proxy: str,
    workers: int,
    force: bool,
    client_token: str | None = None,
) -> int:
    if client_token is not None:
        from eval.user_behavior import adjudicate_3way as adjudication

        try:
            counts = asyncio.run(
                adjudication.run_adjudication(
                    trials_roots,
                    judge_b_sidecar=judge_b_sidecar,
                    judge_a_model=judge_a_model,
                    judge_b_model=judge_b_model,
                    arbiter_model=arbiter_model,
                    arbiter_proxy=arbiter_proxy,
                    workers=workers,
                    require_provenance=True,
                    force=force,
                    arbiter=_BearerArbiterLLM(
                        arbiter_proxy, arbiter_model, client_token
                    ),
                )
            )
        except Exception as exc:  # Match the subprocess CLI's nonzero contract.
            logger.error("authenticated three-way adjudication failed: %s", exc)
            return 2
        print(f"three-way adjudication: {counts}", flush=True)
        return 2 if counts.get("err") else 0

    cmd = [
        PY,
        "-m",
        "eval.user_behavior.adjudicate_3way",
        "--judge-b-sidecar",
        str(judge_b_sidecar),
        "--judge-a-model",
        judge_a_model,
        "--judge-b-model",
        judge_b_model,
        "--arbiter-model",
        arbiter_model,
        "--arbiter-proxy",
        arbiter_proxy,
        "--workers",
        str(workers),
        "--require-provenance",
        *(["--force"] if force else []),
    ]
    for root in trials_roots:
        cmd += ["--trials-root", str(root)]
    return _run_subprocess(cmd, "2c-adjudicate_3way")


# ── per-trial reads ──────────────────────────────────────────────────────────

def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        value = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _tag_metrics(verdict: dict, source: str = "single") -> dict:
    """User Correction (#correction + 0.2·nudge) from the per-message multi-label
    tags. Delegates to the single source of truth in user_metrics (same deriver
    tag_messages.py persists into the verdict), so aggregated and stored values are
    identical. Nones when untagged."""
    return kg.metrics_from_verdict(verdict, source)


def _trial_runtime_sec(trial_dir: Path) -> float | None:
    """Agent wall-clock per trial (seconds): result.json `agent_execution`
    (started_at→finished_at), falling back to timing.json::trial_wall_clock_sec."""
    ae = (_load_json(trial_dir / "result.json") or {}).get("agent_execution") or {}
    s, f = ae.get("started_at"), ae.get("finished_at")
    if s and f:
        try:
            seconds = (
                datetime.fromisoformat(f.replace("Z", "+00:00"))
                - datetime.fromisoformat(s.replace("Z", "+00:00"))
            ).total_seconds()
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        except (ValueError, TypeError, OverflowError):
            pass
    v = (_load_json(trial_dir / "timing.json") or {}).get("trial_wall_clock_sec")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    seconds = float(v)
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _trial_output_tokens(trial_dir: Path) -> int | None:
    """Output+reasoning tokens per trial, summed from the opencode event log
    (`agent/opencode.txt` or the interactive wrapper's
    `agent/opencode.txt.turn-*` files). None for harnesses without either."""
    agent_dir = trial_dir / "agent"
    combined = agent_dir / "opencode.txt"
    paths = [combined] if combined.exists() else sorted(
        agent_dir.glob("opencode.txt.turn-*")
    )
    if not paths:
        return None
    tot, found, invalid = 0, False, False
    seen_finish_ids: set[str] = set()
    for path in paths:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            invalid = True
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") not in ("step_finish", "step-finish"):
                continue
            part = event.get("part") or {}
            if not isinstance(part, dict):
                invalid = True
                continue
            finish_id = part.get("id") or part.get("messageID")
            if finish_id and finish_id in seen_finish_ids:
                continue
            if finish_id:
                seen_finish_ids.add(finish_id)
            tokens = part.get("tokens") or event.get("tokens") or {}
            if not isinstance(tokens, dict):
                invalid = True
                continue
            output = tokens.get("output", 0) or 0
            reasoning = tokens.get("reasoning", 0) or 0
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in (output, reasoning)
            ):
                invalid = True
                continue
            tot += output + reasoning
            found = True
    return tot if found and not invalid else None


def _is_infra_failed(trial_dir: Path) -> bool:
    """True if the trial is an infrastructure failure (agent never really ran —
    provider/sandbox error). Such trials are excluded from scoring, not zeroed."""
    if _classify_infra is None:
        return True
    try:
        return _classify_infra(trial_dir).status != "ok"
    except Exception:
        # An unreadable or unclassifiable trial must never silently become a
        # scored model failure.  Strict completeness reports the detailed
        # classifier issue; diagnostic aggregation conservatively excludes it.
        return True


def _strict_artifact_issues(trial_dir: Path) -> list[str]:
    """Validate result and infrastructure evidence for a headline row."""
    issues: list[str] = []
    result = _load_json(trial_dir / "result.json")
    rewards = (
        (result.get("verifier_result") or {}).get("rewards")
        if isinstance(result, dict)
        else None
    )
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    try:
        valid_reward = (
            isinstance(rewards, dict)
            and bool(rewards)
            and isinstance(reward, (int, float))
            and not isinstance(reward, bool)
            and math.isfinite(float(reward))
        )
    except (TypeError, ValueError):
        valid_reward = False
    if not valid_reward:
        issues.append(f"invalid_result:{trial_dir.name}")

    sidecar = _load_json(trial_dir / "trial_infra.json")
    version = sidecar.get("version") if isinstance(sidecar, dict) else None
    status = sidecar.get("status") if isinstance(sidecar, dict) else None
    if version != _INFRA_SIDECAR_VERSION:
        issues.append(
            f"infra_version:{trial_dir.name}:{version or 'missing'}"
            f"!=expected_{_INFRA_SIDECAR_VERSION}"
        )
    if status != "ok":
        issues.append(f"infra_status:{trial_dir.name}:{status or 'missing'}")

    if _classify_infra_fresh is None:
        fresh_status = None
    else:
        try:
            fresh_status = _classify_infra_fresh(trial_dir).status
        except Exception:
            fresh_status = None
    if fresh_status != "ok":
        issues.append(
            f"infra_fresh_status:{trial_dir.name}:{fresh_status or 'unavailable'}"
        )
    return issues


def _effective_judge_score(trial_dir: Path, judge: dict):
    """Leaderboard scoring rule:
      - infra failure (agent never ran)                       → None  (excluded)
      - any non-infra failure (empty/no patch, verdict_read_   → 0.0   (a fail)
        failed, unjudged)
      - otherwise                                             → judge_score
    """
    if _is_infra_failed(trial_dir):
        return None
    score = judge.get("judge_score")
    if score is None or judge.get("error") == "verdict_read_failed":
        return 0.0
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        return 0.0
    return float(score)


def _normalized_model_id(model: object) -> str | None:
    """Normalize provider-qualified IDs for reproducibility comparisons."""
    if not isinstance(model, str) or not model.strip():
        return None
    value = model.strip().lower()
    for prefix in ("openrouter/anthropic/", "anthropic/", "metagen/"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def join_trial_artefacts(job: dict, user_correction_source: str = "single") -> dict:
    """Read the three per-trial verdicts + reward.txt into one flat record.

    Uses the per-job verdict filenames recorded by `discover_jobs` so different
    trials roots can carry different verdict naming conventions (the pilot
    uses cohort-tagged coverage filenames).

    Missing inputs are tolerated — downstream aggregation guards on `None`.
    """
    trial_dir = Path(job["trial_dir"])
    judge = _load_json(trial_dir / job.get("judge_out_name", "judge_verdict.json")) or {}
    cov   = _load_json(trial_dir / job.get("coverage_out_name", "intent_coverage_verdict.json")) or {}

    reward_p = trial_dir / "verifier" / "reward.txt"
    test_reward: float | None = None
    if reward_p.exists():
        try:
            test_reward = float(reward_p.read_text().strip().splitlines()[0])
        except (ValueError, OSError):
            pass

    final_patch = trial_dir / "agent" / "final.patch"
    empty_patch = not patch_file_has_changes(final_patch)

    return {
        "task": job["task"],
        "cohort": job.get("cohort", ""),
        "trial_dir": str(trial_dir),
        "trial_id": trial_dir.name,
        # step 1 — correctness (infra-failed → None/excluded; non-infra fail → 0.0)
        "judge_score": _effective_judge_score(trial_dir, judge),
        "judge_score_raw": judge.get("judge_score"),
        "judge_verdict": judge.get("verdict"),
        "judge_model": judge.get("judge_model"),
        "test_reward_raw": test_reward,
        "score_delta": judge.get("score_delta"),
        "judge_warnings": len(judge.get("schema_warnings") or []),
        "empty_patch": empty_patch,
        # trial cost
        "runtime_sec": _trial_runtime_sec(trial_dir),
        "output_tokens": _trial_output_tokens(trial_dir),
        # step 2 — intent_coverage (diagnostic)
        "overall_score": cov.get("overall_score"),
        "coverage_rate": cov.get("coverage_rate"),
        "scope_precision": cov.get("scope_precision"),
        "weighted_coverage": cov.get("weighted_coverage"),
        "coverage_warnings": len(cov.get("schema_warnings") or []),
        # step 2b — message tags → User Correction (#correction + 0.2·nudge)
        **_tag_metrics(cov, user_correction_source),
    }


def _expected_sim_message_indices(trial_dir: Path, task_dir: Path) -> set[int]:
    """Reconstruct the follow-up indices consumed by ``tag_messages``."""
    next_index = 0
    try:
        if (task_dir / "instruction.md").read_text(errors="replace").strip():
            next_index = 1
    except OSError:
        pass

    indices: set[int] = set()
    for episode in sorted((trial_dir / "agent").glob("episode-*")):
        decision = _load_json(episode / "user_decision.json")
        if not isinstance(decision, dict) or not decision.get("has_message"):
            continue
        content = decision.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if next_index != 0:  # Turn 0 is the initial instruction, never tagged.
            indices.add(next_index)
        next_index += 1
    return indices


def completeness_issues(
    jobs: list[dict],
    *,
    expected_replicates: int | None = None,
    expected_tasks: int | None = None,
    expected_judge_model: str | None = None,
    require_cost_data: bool = False,
    user_correction_source: str = "single",
    expected_tag_model: str | None = None,
    expected_tag_judge_b_model: str = kg.PAPER_TAG_JUDGE_B,
    expected_tag_arbiter_model: str = kg.PAPER_TAG_ARBITER,
) -> list[str]:
    """Return operational gaps that would invalidate a headline benchmark row.

    Empty/no-patch attempts are legitimate scored failures and need no judge
    file. Every substantive patch must have a non-error judge score, and every
    completed trial must have message-tag output (an empty tag list is valid for
    a no-follow-up trial).
    """
    issues: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    for job in jobs:
        trial_dir = Path(job["trial_dir"])
        task_dir = Path(job["task_dir"]) if job.get("task_dir") else None
        counts[job["task"]] += 1
        if not (trial_dir / "result.json").exists():
            issues.append(f"missing_result:{trial_dir.name}")
            continue
        issues.extend(_strict_artifact_issues(trial_dir))
        if require_cost_data:
            if _trial_runtime_sec(trial_dir) is None:
                issues.append(f"runtime_incomplete:{trial_dir.name}")
            if _trial_output_tokens(trial_dir) is None:
                issues.append(f"tokens_incomplete:{trial_dir.name}")
        patch = trial_dir / "agent" / "final.patch"
        rubric = (
            _load_json(task_dir / "canonical_goals.json") if task_dir else None
        )
        if task_dir:
            for issue in _rubric_issues(rubric):
                issues.append(f"rubric_invalid:{job['task']}:{issue}")
        if patch_file_has_changes(patch):
            judge = _load_json(
                trial_dir / job.get("judge_out_name", "judge_verdict.json")
            )
            if task_dir:
                verdict_issues = _phase2_verdict_issues(
                    rubric,
                    judge,
                    task_name=job["task"],
                    trial_id=trial_dir.name,
                    expected_judge_model=expected_judge_model,
                )
                for issue in verdict_issues:
                    issues.append(f"judge_incomplete:{trial_dir.name}:{issue}")
            elif not judge or "error" in judge or judge.get("judge_score") is None:
                issues.append(f"judge_incomplete:{trial_dir.name}")
            elif (
                isinstance(judge.get("judge_score"), bool)
                or not isinstance(judge.get("judge_score"), (int, float))
                or not math.isfinite(float(judge["judge_score"]))
                or not 0.0 <= float(judge["judge_score"]) <= 1.0
            ):
                issues.append(f"judge_invalid_score:{trial_dir.name}")
            elif expected_judge_model and _normalized_model_id(
                judge.get("judge_model")
            ) != _normalized_model_id(expected_judge_model):
                observed = judge.get("judge_model") or "missing"
                issues.append(
                    f"judge_model:{trial_dir.name}:{observed}"
                    f"!=expected_{expected_judge_model}"
                )
        coverage = _load_json(
            trial_dir
            / job.get("coverage_out_name", "intent_coverage_verdict.json")
        )
        provenance_issues = kg.user_correction_provenance_issues(
            coverage,
            source=user_correction_source,
            expected_tag_model=expected_tag_model,
            expected_judge_b_model=expected_tag_judge_b_model,
            expected_arbiter_model=expected_tag_arbiter_model,
            expected_trial_indices=(
                _expected_sim_message_indices(trial_dir, task_dir)
                if task_dir
                else None
            ),
        )
        for issue in provenance_issues:
            issues.append(f"tags_incomplete:{trial_dir.name}:{issue}")

    if expected_tasks is not None and len(counts) != expected_tasks:
        issues.append(f"task_count:{len(counts)}!=expected_{expected_tasks}")
    if expected_replicates is not None:
        for task, count in sorted(counts.items()):
            if count != expected_replicates:
                issues.append(
                    f"replicate_count:{task}:{count}!=expected_{expected_replicates}"
                )
    return issues


# ── correctness pass-rate metrics (judge-only, effort-free) ───────────────────

def _judge_scores(trials: list[dict]) -> list[float]:
    return [t["judge_score"] for t in trials if t.get("judge_score") is not None]


def pass_at_1(trials: list[dict], T: float = SUCCESS_THRESHOLD) -> float | None:
    """Single-run success probability: fraction of reps with judge_score ≥ T."""
    js = _judge_scores(trials)
    return (sum(1 for s in js if s >= T) / len(js)) if js else None


def stable_pass_rate(trials: list[dict], T: float = SUCCESS_THRESHOLD) -> float | None:
    """1.0 if the task's mean judge_score over reps clears T, else 0.0."""
    js = _judge_scores(trials)
    return (1.0 if statistics.fmean(js) >= T else 0.0) if js else None


def pass_squared(trials: list[dict], T: float = SUCCESS_THRESHOLD) -> float | None:
    """C(c,2)/C(n,2): a random pair of reps both clear T (canonical k=2 ⇒ both pass).
    None when fewer than 2 reps."""
    js = _judge_scores(trials)
    n = len(js)
    if n < 2:
        return None
    c = sum(1 for s in js if s >= T)
    return math.comb(c, 2) / math.comb(n, 2)


# ── per-task aggregation ─────────────────────────────────────────────────────

def aggregate_per_task(trials_by_task: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for task, trials in sorted(trials_by_task.items()):
        judge_scores = [t["judge_score"] for t in trials if t.get("judge_score") is not None]

        mean_judge = statistics.fmean(judge_scores) if judge_scores else None
        var_judge  = statistics.pvariance(judge_scores) if len(judge_scores) >= 2 else 0.0
        p1, spr, p2 = pass_at_1(trials), stable_pass_rate(trials), pass_squared(trials)

        # benchmark fidelity diagnostics (per-task).
        empty_patch_rate = sum(1 for t in trials if t["empty_patch"]) / len(trials) if trials else 0.0
        any_warnings = sum(
            1 for t in trials
            if (t.get("judge_warnings") or 0) > 0 or (t.get("coverage_warnings") or 0) > 0
        )
        schema_warning_rate = any_warnings / len(trials) if trials else 0.0

        rows.append({
            "task": task,
            "n_total": len(trials),
            # correctness (judge)
            "mean_judge": round(mean_judge, 4) if mean_judge is not None else None,
            "var_judge": round(var_judge, 4),
            "pass_at_1":        round(p1, 4)  if p1  is not None else None,
            "stable_pass_rate": round(spr, 4) if spr is not None else None,
            "pass_sq":          round(p2, 4)  if p2  is not None else None,
            "judge_scores_all": [t.get("judge_score") for t in trials],
            "overall_scores_all": [t.get("overall_score") for t in trials],
            # User Correction (#correction + 0.2·nudge), from message tags
            "user_correction_mean": _safe_mean(t.get("user_correction") for t in trials),
            # trial cost (avg per trial)
            "runtime_sec_mean": _safe_mean(t.get("runtime_sec") for t in trials),
            "output_tokens_mean": _safe_mean(t.get("output_tokens") for t in trials),
            # diagnostics — Intent Coverage (sim-vs-oracle) + benchmark fidelity
            "coverage_mean": _safe_mean(t.get("overall_score") for t in trials),
            "empty_patch_rate": round(empty_patch_rate, 4),
            "schema_warning_rate": round(schema_warning_rate, 4),
        })
    return rows


def _safe_mean(xs) -> float | None:
    vs = [x for x in xs if x is not None]
    return round(statistics.fmean(vs), 4) if vs else None


def _rate(xs) -> float | None:
    vs = [x for x in xs if x is not None]
    return round(sum(1 for x in vs if x) / len(vs), 4) if vs else None


# ── cross-task rollup ────────────────────────────────────────────────────────

def cross_task_rollup(rows: list[dict], denom_tasks: int | None = None) -> dict:
    """Headline numbers. Pass-rate metrics divide by a FIXED `denom_tasks` (the full
    task set, e.g. 109) so tasks with no valid/passing result count as 0; the rest are
    means over the tasks actually present."""
    n_present = len(rows)
    denom = denom_tasks if denom_tasks else n_present
    def m(field: str) -> float | None:           # mean over present tasks
        vs = [r[field] for r in rows if r.get(field) is not None]
        return round(statistics.fmean(vs), 4) if vs else None
    def m_fixed(field: str) -> float | None:     # sum over present / fixed denom
        vs = [r[field] for r in rows if r.get(field) is not None]
        return round(sum(vs) / denom, 4) if denom else None
    return {
        "n_tasks": n_present,
        "denom_tasks": denom,
        "mean_judge_over_tasks": m("mean_judge"),
        "pass_at_1_mean": m_fixed("pass_at_1"),
        "stable_pass_rate_mean": m_fixed("stable_pass_rate"),
        "pass_sq_mean": m_fixed("pass_sq"),
        "user_correction_mean": m("user_correction_mean"),
        "runtime_sec_mean": m("runtime_sec_mean"),
        "output_tokens_mean": m("output_tokens_mean"),
        # diagnostics
        "coverage_mean": m("coverage_mean"),
        "empty_patch_rate_mean": m("empty_patch_rate"),
        "schema_warning_rate_mean": m("schema_warning_rate"),
    }


# ── Markdown report ──────────────────────────────────────────────────────────

def render_markdown(per_task: list[dict], rollup: dict, args) -> str:
    roots_label = ", ".join(p.name for p in args.trials_root) if isinstance(args.trials_root, list) else str(args.trials_root)
    lines = [
        f"# Eval run — {args.model_tag or roots_label}",
        "",
        f"- trials roots: `{roots_label}`",
        f"- tasks root: `{args.tasks_root}`",
        f"- n tasks: {rollup['n_tasks']}  (pass-rate denom: {rollup.get('denom_tasks', rollup['n_tasks'])})",
        f"- success threshold: judge_score ≥ {SUCCESS_THRESHOLD}",
        f"- User Correction source: `{args.user_correction_source}`",
        "- scoring: infra-failed trials excluded; non-infra failures (empty patch / unjudged) = 0",
        "",
        "## Headline (cross-task means)",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for k in (
        "mean_judge_over_tasks",
        "pass_at_1_mean", "stable_pass_rate_mean", "pass_sq_mean",
        "user_correction_mean",
        "runtime_sec_mean", "output_tokens_mean",
        "coverage_mean",
        "empty_patch_rate_mean", "schema_warning_rate_mean",
    ):
        v = rollup.get(k)
        lines.append(f"| `{k}` | {v if v is not None else '—'} |")

    lines.extend([
        "",
        "## Per-task",
        "",
        "| task | n_total | mean_judge | pass@1 | stable | pass² | user_corr μ | runtime s | out tok | coverage | empty% | warn% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in per_task:
        lines.append("| " + " | ".join([
            f"`{r['task']}`",
            f"{r['n_total']}",
            _fmt(r['mean_judge']),
            _fmt(r['pass_at_1']),
            _fmt(r['stable_pass_rate']),
            _fmt(r['pass_sq']),
            _fmt(r['user_correction_mean']),
            _num(r['runtime_sec_mean']),
            _num(r['output_tokens_mean']),
            _fmt(r['coverage_mean']),
            _pct(r['empty_patch_rate']),
            _pct(r['schema_warning_rate']),
        ]) + " |")
    return "\n".join(lines) + "\n"


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _pct(x: Any) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def _num(x: Any) -> str:
    if x is None:
        return "—"
    return f"{x:,.0f}"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trials-root", type=Path, required=True, action="append",
                    help="Directory holding <task>__<id> trial dirs. Repeat to aggregate "
                         "across multiple cohorts as k replicates of the same (task, agent, sim).")
    ap.add_argument("--tasks-root", type=Path,
                    default=REPO_ROOT / "tasks",
                    help="Directory holding canonical task dirs (default: tasks/)")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory to write plan, step summaries, and aggregate report")
    ap.add_argument("--model-tag", default="",
                    help="Label for this (agent, sim) cohort in the report header")

    # Verdict file names — pass once for the same name across all trials roots,
    # or pass repeated values to match the order of --trials-root (per-cohort overrides).
    ap.add_argument("--judge-out-name", action="append", default=None,
                    help="default: judge_verdict.json")
    ap.add_argument("--coverage-out-name", action="append", default=None,
                    help="default: intent_coverage_verdict.json")

    # Skip / force / per-step concurrency.
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--skip-intent-coverage", action="store_true")
    ap.add_argument("--skip-tag-messages", action="store_true")
    ap.add_argument("--force-tag-messages", action="store_true")
    ap.add_argument("--tag-workers", type=int, default=50)
    ap.add_argument("--tag-model", default=kg.CANONICAL_TAG_MODEL,
                    help="LLM model for message tagging (pinned for reproducibility)")
    ap.add_argument(
        "--tag-backend",
        choices=("litellm", "vertex-gateway"),
        default="litellm",
        help="Transport for the primary tagger (default: public LiteLLM provider).",
    )
    ap.add_argument(
        "--user-correction-source",
        choices=("single", "threeway"),
        default="single",
        help="U-Corr labels from the released single-tagger protocol, or an "
        "optional local three-model ensemble (default: single).",
    )
    ap.add_argument(
        "--tag-judge-b-model",
        default=kg.ENSEMBLE_TAG_JUDGE_B,
        help="Optional ensemble Judge-B model (default: Claude Opus 4.6).",
    )
    ap.add_argument(
        "--tag-arbiter-model",
        default=kg.ENSEMBLE_TAG_ARBITER,
        help="Optional ensemble disagreement arbiter (default: GPT-5.5).",
    )
    ap.add_argument(
        "--tag-arbiter-proxy",
        default="http://127.0.0.1:4220/v1",
        help="Chat Completions endpoint for the threeway GPT arbiter.",
    )
    ap.add_argument(
        "--tag-arbiter-auto-start",
        action="store_true",
        help="For a free loopback arbiter URL, temporarily start an authenticated "
        "bundled Codex-OAuth proxy; occupied ports fail closed.",
    )
    ap.add_argument(
        "--tag-arbiter-auth-json",
        type=Path,
        default=None,
        help="Account-owned Codex auth file for --tag-arbiter-auto-start "
        "(default: CODEX_HOST_AUTH_JSON or ~/.codex/auth.json).",
    )
    ap.add_argument(
        "--tag-arbiter-client-auth-file",
        type=Path,
        default=None,
        help="Private bearer-token file for a manually managed arbiter proxy. "
        "The token is read in-process and never added to argv or logs.",
    )
    ap.add_argument(
        "--tag-judge-b-sidecar",
        type=Path,
        default=None,
        help="Threeway Judge-B cache (default: <output-dir>/u_corr_judge_b.json).",
    )
    ap.add_argument("--only-aggregate", action="store_true",
                    help="Skip all three steps and just aggregate existing verdicts")
    ap.add_argument("--force-correctness", action="store_true")
    ap.add_argument("--force-intent-coverage", action="store_true")
    ap.add_argument("--correctness-workers", type=int, default=20)
    ap.add_argument("--intent-coverage-workers", type=int, default=5)
    ap.add_argument("--intent-coverage-model", default=None,
                    help="LLM model for intent_coverage (default: package default)")
    ap.add_argument("--denom-tasks", type=int, default=None,
                    help="Fixed task-count denominator for pass-rate metrics (default: "
                         "#task dirs under --tasks-root, e.g. 109). Missing / all-infra "
                         "tasks then count as 0.")
    ap.add_argument(
        "--require-complete",
        action="store_true",
        help="Return nonzero if any pipeline step fails, a substantive patch is "
        "unjudged, message tags are absent, or expected task/replicate counts differ.",
    )
    ap.add_argument(
        "--expected-replicates",
        type=int,
        default=None,
        help="With --require-complete, require this many trials per task.",
    )
    ap.add_argument(
        "--expected-judge-model",
        default=DEFAULT_EXPECTED_JUDGE_MODEL,
        help="With --require-complete, require substantive patches to have been "
        "scored by this judge (provider prefixes are normalized; default: "
        f"{DEFAULT_EXPECTED_JUDGE_MODEL}).",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.tag_arbiter_auto_start and args.tag_arbiter_client_auth_file:
        ap.error(
            "--tag-arbiter-client-auth-file is for a manually managed proxy; "
            "auto-start generates a fresh per-run credential"
        )
    if (
        args.tag_arbiter_client_auth_file
        and args.user_correction_source != "threeway"
    ):
        ap.error("--tag-arbiter-client-auth-file requires threeway U-Corr")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    jobs = discover_jobs(
        args.trials_root, args.tasks_root,
        coverage_names=args.coverage_out_name,
        judge_names=args.judge_out_name,
    )
    if not jobs:
        logger.error("no trial/task pairs found under %s ↔ %s",
                     args.trials_root, args.tasks_root)
        return 2
    logger.info("paired %d trial dirs across %d unique tasks (cohorts: %s)",
                len(jobs), len({j["task"] for j in jobs}),
                sorted({j["cohort"] for j in jobs}))

    if args.only_aggregate:
        args.skip_correctness = args.skip_intent_coverage = args.skip_tag_messages = True

    # Plans are step-CLI-format; per-cohort verdict filenames go through each
    # job's `*_out_name` field already so a single plan with mixed cohorts
    # works (each step CLI writes to the per-job out_name).
    # If you need to actually RUN a step across mixed cohorts, the step CLI
    # uses ONE out_name across the whole plan — write one plan per distinct
    # out_name in that case.
    distinct_judge = sorted({j["judge_out_name"] for j in jobs})
    distinct_cov   = sorted({j["coverage_out_name"] for j in jobs})

    def _plan_path(prefix: str, name: str, multi: bool) -> Path:
        return out / (f"plan_{prefix}_{Path(name).stem}.json" if multi
                      else f"plan_{prefix}.json")

    plan_correct_paths = [
        write_plan([j for j in jobs if j["judge_out_name"] == n],
                   _plan_path("correctness", n, len(distinct_judge) > 1), n)
        for n in distinct_judge
    ]
    plan_cov_paths = [
        write_plan([j for j in jobs if j["coverage_out_name"] == n],
                   _plan_path("intent_coverage", n, len(distinct_cov) > 1), n)
        for n in distinct_cov
    ]

    pipeline_step_failures: list[str] = []
    if not args.skip_correctness:
        for p in plan_correct_paths:
            rc = run_step_correctness(
                p, out / f"summary_{p.stem}.json",
                workers=args.correctness_workers, force=args.force_correctness,
                extra=[],
            )
            if rc != 0:
                logger.error("step 1 (correctness) failed rc=%d on %s — continuing", rc, p.name)
                pipeline_step_failures.append(f"correctness:{p.name}:rc={rc}")

    if not args.skip_intent_coverage:
        for p in plan_cov_paths:
            rc = run_step_intent_coverage(
                p, out / f"summary_{p.stem}.json",
                workers=args.intent_coverage_workers, force=args.force_intent_coverage,
                model=args.intent_coverage_model, extra=[],
            )
            if rc != 0:
                logger.error("step 2 (intent_coverage) failed rc=%d on %s — continuing", rc, p.name)
                pipeline_step_failures.append(f"intent_coverage:{p.name}:rc={rc}")

    if not args.skip_tag_messages:
        # A strict row must be reproducible even for the released single-tagger
        # path; legacy tag arrays without model/prompt provenance are re-tagged.
        require_tag_provenance = (
            args.require_complete or args.user_correction_source == "threeway"
        )
        rc = run_step_tag_messages(
            args.trials_root,
            model=args.tag_model,
            workers=args.tag_workers,
            force=args.force_tag_messages,
            require_provenance=require_tag_provenance,
            backend=args.tag_backend,
        )
        if rc != 0:
            logger.error("step 2b (tag_messages) failed rc=%d — continuing", rc)
            pipeline_step_failures.append(f"tag_messages:rc={rc}")
        if args.user_correction_source == "threeway":
            judge_b_sidecar = (
                args.tag_judge_b_sidecar.resolve()
                if args.tag_judge_b_sidecar
                else out / "u_corr_judge_b.json"
            )
            rc = run_step_tag_messages(
                args.trials_root,
                model=args.tag_judge_b_model,
                workers=args.tag_workers,
                force=args.force_tag_messages,
                require_provenance=True,
                output_sidecar=judge_b_sidecar,
                backend="litellm",
            )
            if rc != 0:
                logger.error(
                    "step 2c (Judge-B tagging) failed rc=%d — continuing", rc
                )
                pipeline_step_failures.append(f"tag_judge_b:rc={rc}")
            try:
                manual_client_token = (
                    _read_external_arbiter_token(
                        args.tag_arbiter_proxy,
                        args.tag_arbiter_client_auth_file,
                    )
                    if args.tag_arbiter_client_auth_file
                    else None
                )
                with _managed_arbiter_proxy(
                    proxy_url=args.tag_arbiter_proxy,
                    auto_start=args.tag_arbiter_auto_start,
                    auth_json=args.tag_arbiter_auth_json,
                    log_path=out / "u_corr_arbiter_proxy.log",
                ) as arbiter_client_token:
                    rc = run_step_adjudicate_3way(
                        args.trials_root,
                        judge_b_sidecar=judge_b_sidecar,
                        judge_a_model=args.tag_model,
                        judge_b_model=args.tag_judge_b_model,
                        arbiter_model=args.tag_arbiter_model,
                        arbiter_proxy=args.tag_arbiter_proxy,
                        workers=args.tag_workers,
                        force=args.force_tag_messages,
                        client_token=arbiter_client_token or manual_client_token,
                    )
            except RuntimeError as exc:
                logger.error("step 2d arbiter proxy failed: %s", exc)
                rc = 2
            if rc != 0:
                logger.error(
                    "step 2d (three-way adjudication) failed rc=%d — continuing", rc
                )
                pipeline_step_failures.append(f"tag_adjudicate_3way:rc={rc}")

    # Aggregate — read every per-trial verdict and group by task.
    logger.info("aggregating per-task metrics from %d trials", len(jobs))
    flat = [
        join_trial_artefacts(j, args.user_correction_source) for j in jobs
    ]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for t in flat:
        by_task[t["task"]].append(t)

    denom_tasks = args.denom_tasks
    if denom_tasks is None and args.tasks_root.is_dir():
        denom_tasks = sum(1 for d in args.tasks_root.iterdir() if d.is_dir()) or None
    per_task = aggregate_per_task(by_task)
    rollup   = cross_task_rollup(per_task, denom_tasks)
    completeness = completeness_issues(
        jobs,
        expected_replicates=args.expected_replicates,
        expected_tasks=denom_tasks if args.require_complete else None,
        expected_judge_model=(
            args.expected_judge_model if args.require_complete else None
        ),
        require_cost_data=args.require_complete,
        user_correction_source=args.user_correction_source,
        expected_tag_model=(
            args.tag_model
            if args.require_complete
            else None
        ),
        expected_tag_judge_b_model=args.tag_judge_b_model,
        expected_tag_arbiter_model=args.tag_arbiter_model,
    )
    completeness = [*pipeline_step_failures, *completeness]

    report = {
        "model_tag": args.model_tag,
        "trials_roots": [str(p.resolve()) for p in args.trials_root],
        "tasks_root": str(args.tasks_root.resolve()),
        "cohorts": sorted({j["cohort"] for j in jobs}),
        "n_trials": len(flat),
        "success_threshold": SUCCESS_THRESHOLD,
        "expected_judge_model": args.expected_judge_model,
        "user_correction_source": args.user_correction_source,
        "tag_model": args.tag_model,
        "tag_backend": args.tag_backend,
        "tag_judge_b_model": args.tag_judge_b_model,
        "tag_arbiter_model": args.tag_arbiter_model,
        "cross_task": rollup,
        "complete": not completeness,
        "canonical_u_corr_protocol_complete": (
            args.require_complete
            and not completeness
            and args.user_correction_source == "single"
            and kg.normalized_model_id(args.tag_model)
            == kg.normalized_model_id(kg.CANONICAL_TAG_MODEL)
        ),
        "completeness_issues": completeness,
        "per_task": per_task,
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out / "per_trial.json").write_text(json.dumps(flat, indent=2, ensure_ascii=False))
    (out / "eval_report.md").write_text(render_markdown(per_task, rollup, args))

    logger.info("wrote eval_report.{json,md} + per_trial.json under %s", out)
    print(f"\n→ {out / 'eval_report.md'}")
    if args.require_complete and completeness:
        logger.error(
            "benchmark completeness gate failed with %d issue(s): %s",
            len(completeness),
            completeness[:10],
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
