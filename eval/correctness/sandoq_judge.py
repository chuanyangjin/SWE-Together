"""Host-side agentic judge over a sandoq OCI-run session (SWE-Together harness).

Drop-in alternative to ``eval.correctness.sandbox.run_judge_in_e2b`` /
``podman_judge.run_judge_in_podman`` for the sandoq backend. The judge MODEL
(Opus via the metagen x2p gateway) runs HOST-SIDE through litellm; its single
``bash`` tool execs into a sandoq session (``SandoqEnvironment``) that holds the
patched task workspace.

Parity: signature/return-compatible with ``run_judge_in_e2b`` (same
``JudgeInputs`` in, ``JudgeRunResult`` out). To stay in lockstep with the
validated podman judge we **reuse its helpers verbatim** —
``_apply_patch`` / ``_drop_inputs`` / ``_first_message`` / ``_run_loop`` /
``_judge_model`` only ever call ``BaseEnvironment`` methods
(``exec``/``upload_file``/``upload_dir``), so they work unchanged over a
``SandoqEnvironment``. The only differences here are the environment object
(sandoq, not podman) and that there is **no per-trial vfs store** to set up or
tear down — sandoq is a remote service.

Why the model stays host-side: the x2p gateway is not reachable from inside the
sandoq pod (verified unreachable from cluster nodes without the login relay), so
as with podman we drive the agentic loop from the host and proxy each tool call
through the bearer-authenticated outer ``/v1/exec`` API and into the nested
Podman + gVisor task container.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "external" / "harbor" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from harbor.models.task.config import TaskConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from eval.correctness.sandbox import (  # noqa: E402
    JUDGE_MAX_TURNS,
    JUDGE_TIMEOUT_SEC,
    JudgeInputs,
    JudgeRunResult,
)

# Reuse the validated podman-judge helpers (env-agnostic — they only touch the
# BaseEnvironment surface). This keeps the two host-side judges byte-identical in
# their prompts, patch-apply, input-drop, and agentic loop.
from eval.correctness.podman_judge import (  # noqa: E402
    DeterministicPatchApplyError,
    INPUTS_DIR,
    _apply_patch,
    _deterministic_patch_failure_verdict,
    _drop_inputs,
    _first_message,
    _judge_model,
    _run_loop,
    _run_loop_with_goal_id_repair,
)
from sandoq_env import SandoqEnvironment  # noqa: E402

log = logging.getLogger(__name__)

TASKS_DIR = REPO_ROOT / "tasks"


async def run_judge_in_sandoq(
    task_name: str,
    trial_id: str,
    inputs: JudgeInputs,
    oauth_token: str | None = None,
    *,
    timeout_sec: int = JUDGE_TIMEOUT_SEC,
    max_turns: int = JUDGE_MAX_TURNS,
    api_key: str | None = None,
) -> JudgeRunResult:
    """Run the agentic judge in a sandoq OCI-run session, host-side model.

    Signature-compatible with ``run_judge_in_e2b`` / ``run_judge_in_podman``.
    ``oauth_token`` / ``api_key`` are accepted for parity but ignored: like the
    podman judge, this authenticates via ``ANTHROPIC_API_KEY`` +
    ``ANTHROPIC_BASE_URL`` (the metagen gateway) read from the environment.
    """
    model = _judge_model()
    task_dir = TASKS_DIR / task_name
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return JudgeRunResult(
            verdict={"error": "task_toml_missing", "task": task_name},
            stdout="",
            stderr="",
            exit_code=1,
            sandbox_id="",
            judge_model=model,
        )

    env_cfg = TaskConfig.model_validate_toml(task_toml.read_text()).environment
    # The judge always needs egress (patch apply + test.sh installs), regardless
    # of the task's own allow_internet — matches the E2B/podman judge.
    env_cfg.allow_internet = True
    if not env_cfg.docker_image:
        return JudgeRunResult(
            verdict={
                "error": "no_docker_image",
                "task": task_name,
                "detail": "sandoq judge boots the prebuilt image; task.toml has none.",
            },
            stdout="",
            stderr="",
            exit_code=1,
            sandbox_id="",
            judge_model=model,
        )

    session_id = f"{trial_id}--judge"
    host_tmp_root = os.environ.get("SANDOQ_JUDGE_TMPDIR", "/tmp")
    with tempfile.TemporaryDirectory(prefix="judge-", dir=host_tmp_root) as tmp:
        env = SandoqEnvironment(
            environment_dir=task_dir / "environment",
            environment_name=task_name,
            session_id=session_id,
            trial_paths=TrialPaths(trial_dir=Path(tmp)),
            task_env_config=env_cfg,
        )
        sandbox_id = ""
        try:
            await env.start(force_build=False)
            sandbox_id = env._sandbox_id or ""
            repo_hint = await _apply_patch(env, inputs, Path(tmp))
            await _drop_inputs(env, inputs, Path(tmp))
            first_message = _first_message(inputs, repo_hint)
            output_path = (
                f"{INPUTS_DIR}/canonical_goals.json"
                if inputs.phase == 1
                else f"{INPUTS_DIR}/verdict.json"
            )
            verdict, transcript, exit_code = await _run_loop_with_goal_id_repair(
                env,
                inputs,
                first_message,
                output_path,
                timeout_sec,
                max_turns,
                run_loop=_run_loop,
            )
            return JudgeRunResult(
                verdict=verdict,
                stdout=transcript,
                stderr="",
                exit_code=exit_code,
                sandbox_id=sandbox_id,
                judge_model=model,
            )
        except DeterministicPatchApplyError as e:
            verdict = _deterministic_patch_failure_verdict(inputs, e)
            if verdict is not None:
                log.info(
                    "sandoq deterministic patch rejection for %s/%s: %s",
                    task_name,
                    trial_id,
                    e,
                )
                return JudgeRunResult(
                    verdict=verdict,
                    stdout=e.stdout,
                    stderr=e.stderr,
                    exit_code=0,
                    sandbox_id=sandbox_id,
                    judge_model=model,
                )
            log.warning("sandoq Phase-1 patch rejection for %s/%s: %s", task_name, trial_id, e)
            return JudgeRunResult(
                verdict={"error": "sandoq_judge_failed", "exception": str(e)[:500]},
                stdout=e.stdout,
                stderr=e.stderr,
                exit_code=1,
                sandbox_id=sandbox_id,
                judge_model=model,
            )
        except Exception as e:
            log.warning("sandoq judge failed for %s/%s: %s", task_name, trial_id, e)
            return JudgeRunResult(
                verdict={"error": "sandoq_judge_failed", "exception": str(e)[:500]},
                stdout="",
                stderr=str(e)[:500],
                exit_code=1,
                sandbox_id=sandbox_id,
                judge_model=model,
            )
        finally:
            try:
                await env.stop(delete=True)
            except Exception as e:
                log.warning("judge session stop failed (%s): %s", session_id, e)
