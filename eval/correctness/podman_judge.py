"""Host-side agentic judge over a podman container (SWE-Together on-cluster harness).

Drop-in alternative to ``eval.correctness.sandbox.run_judge_in_e2b`` for hosts
with no E2B and no in-container model access. The judge MODEL (Opus via the
metagen x2p gateway) runs HOST-SIDE through litellm; its single ``bash`` tool
execs into a podman container (``PodmanEnvironment``) that holds the patched task
workspace.

Parity with the E2B judge — so ``run_batch.py`` / ``generate_task_goals.py`` can
dispatch to either backend:
  * same ``JudgeInputs`` in, same ``JudgeRunResult`` out;
  * the Phase 1/2 system prompts + first messages are reused VERBATIM, including
    their ``/tmp/judge_inputs/`` layout and the "write the rubric/verdict to a
    file" output convention — we read that file back out of the container.

Why host-side model: the x2p gateway is not reachable from inside the container
(it hangs), so unlike E2B we cannot run ``claude --print`` in-sandbox. Instead we
drive the agentic loop from the host and proxy each tool call in via
``podman exec``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "external" / "harbor" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import litellm  # noqa: E402

from harbor.models.task.config import TaskConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from eval.correctness.sandbox import (  # noqa: E402
    JUDGE_MAX_TURNS,
    JUDGE_TIMEOUT_SEC,
    JudgeInputs,
    JudgeRunResult,
)
from podman_env import PodmanEnvironment  # noqa: E402

log = logging.getLogger(__name__)

TASKS_DIR = REPO_ROOT / "tasks"
INPUTS_DIR = "/tmp/judge_inputs"

# The gateway rejects some litellm default params (e.g. temperature for reasoning
# models); let litellm strip whatever the provider refuses.
litellm.drop_params = True

# Per-tool-call output returned to the model is capped so a chatty test.sh can't
# blow the context window.
_MAX_TOOL_OUTPUT = 20000

# Printed by the in-container apply harness only after the normal, --recount,
# and --recount -C1 attempts have all rejected the submitted patch.  Transport,
# image, upload, repo-discovery, and missing-git failures never print it, which
# lets callers distinguish an invalid agent artifact from judge infrastructure.
_PATCH_REJECTED_SENTINEL = "__SWE_TOGETHER_PATCH_APPLY_REJECTED__="


class DeterministicPatchApplyError(RuntimeError):
    """The task image was healthy, but git rejected the agent patch three times."""

    def __init__(
        self,
        *,
        return_code: int,
        repo_path: str,
        stdout: str,
        stderr: str,
    ) -> None:
        self.return_code = return_code
        self.repo_path = repo_path
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"agent patch rejected by git after three apply modes "
            f"(rc={return_code}, repo={repo_path})"
        )


def _deterministic_patch_failure_verdict(
    inputs: JudgeInputs, error: DeterministicPatchApplyError
) -> dict | None:
    """Build a valid Phase-2 zero without falsely claiming an LLM decision.

    ``run_batch`` has already validated the frozen rubric before invoking a
    judge, but this helper still fails closed when called directly with malformed
    input.  Phase 1 cannot use this fallback: an unappliable oracle means no
    trustworthy rubric can be generated and must remain an error.
    """
    if inputs.phase != 2:
        return None
    try:
        rubric = json.loads(inputs.canonical_goals_json)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(rubric, dict):
        return None
    goals = rubric.get("completeness_goals")
    if not isinstance(goals, list) or not goals:
        return None
    goal_ids = [goal.get("id") if isinstance(goal, dict) else None for goal in goals]
    if (
        any(not isinstance(goal_id, str) or not goal_id.strip() for goal_id in goal_ids)
        or len(goal_ids) != len(set(goal_ids))
    ):
        return None

    evidence = (
        "The submitted agent patch was rejected by git in the pristine task "
        "repository after the normal, --recount, and --recount -C1 apply modes; "
        "this goal therefore has no applied implementation to evaluate."
    )
    return {
        "judge_score": 0.0,
        "verdict": "incorrect",
        "rubric_source": "canonical_goals.json",
        "goal_results": [
            {"id": goal_id, "met": False, "evidence": evidence}
            for goal_id in goal_ids
        ],
        "judge_notes": (
            "Deterministic pre-judge result: the agent patch could not be applied "
            "to the task image, so the model judge was not invoked."
        ),
        "verdict_source": "swe-together.patch-apply-gate/v1",
        "judge_invoked": False,
        "patch_apply": {
            "classification": "deterministic_agent_patch_rejection",
            "attempts": ["normal", "recount", "recount_context_1"],
            "return_code": error.return_code,
            "repo_path": error.repo_path,
            "stderr_tail": error.stderr[-2000:],
        },
    }

_BASH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command inside the task workspace container (as root). "
                "Returns combined stdout+stderr and the exit code. Use it for "
                "everything: read files (cat), search (grep/find/ls), run "
                "test.sh, and WRITE your required output file "
                "(cat > /tmp/judge_inputs/<file> <<'EOF' ... EOF)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run (bash -c).",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def _judge_model() -> str:
    """The host-side judge model, pinned to the paper's Opus 4.6 default."""
    return os.environ.get("JUDGE_PODMAN_MODEL", "anthropic/claude-opus-4-6")


def _clean_agent_patch(patch: str) -> str:
    """Reduce a multi-repo repo_diff capture to one applyable diff.

    ``src/user_agent/repo_diff.py`` concatenates EVERY git repo it finds under
    the container, each prefixed with ``=== <path> (cumulative vs harbor-base)
    ===``. When the container has a stray repo (e.g. a temp ``/tmp/tmp.XXXX/repo``
    alongside the real ``/workspace``), ``final.patch`` becomes several diffs +
    header lines — not a single applyable patch (the ``===`` lines and duplicate
    per-file diffs make ``git apply`` fail). Split on the headers and keep ONE
    section — prefer ``/workspace``, then any non-``/tmp`` repo, then the largest
    — dropping the header lines so what's left is a clean diff. No-op when the
    capture is already a single clean diff (no ``=== `` headers).

    Git's default diff renders binary changes as ``Binary files ... differ``
    without embedding their bytes. Such blocks can never be applied and are
    commonly generated by downloaded wheel/cache junk. Drop only those
    placeholder blocks; retain real ``GIT binary patch`` payloads.
    """
    import re as _re

    def drop_unapplyable_binary_blocks(text: str) -> str:
        chunks = _re.split(r"(?=^diff --git )", text, flags=_re.MULTILINE)
        return "".join(
            chunk
            for chunk in chunks
            if not (
                chunk.startswith("diff --git ")
                and _re.search(r"^Binary files .+ differ\s*$", chunk, _re.MULTILINE)
                and not _re.search(r"^GIT binary patch\s*$", chunk, _re.MULTILINE)
            )
        )

    hdr = _re.compile(r"^=== (.*?) \(cumulative vs harbor-base\) ===\s*$")
    if not any(hdr.match(ln) for ln in patch.splitlines()):
        return drop_unapplyable_binary_blocks(patch)
    sections: list[tuple[str, list[str]]] = []
    cur_path, cur = None, []
    for ln in patch.splitlines(keepends=True):
        m = hdr.match(ln)
        if m:
            if cur:
                sections.append((cur_path or "", cur))
            cur_path, cur = m.group(1), []
        else:
            cur.append(ln)
    if cur:
        sections.append((cur_path or "", cur))
    sections = [
        (p, c)
        for p, c in sections
        if any(line.startswith("diff --git") for line in c)
    ]
    if not sections:
        return drop_unapplyable_binary_blocks(patch)

    def _rank(item: tuple[str, list[str]]):
        p, c = item
        pref = 0 if p.startswith("/workspace") else (1 if not p.startswith("/tmp") else 2)
        return (pref, -sum(len(line) for line in c))

    return drop_unapplyable_binary_blocks(
        "".join(min(sections, key=_rank)[1])
    )


def _extract_json(text: str) -> dict:
    """Parse a JSON object, tolerating leading/trailing prose or code fences."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("no JSON object found", text, 0)


# ── container setup (patch apply + input drop) ─────────────────────────────────


async def _apply_patch(env: PodmanEnvironment, inputs: JudgeInputs, host_tmp: Path) -> str:
    """Apply the phase-appropriate patch in-container; return the repo path.

    Mirrors ``sandbox.run_judge_in_e2b``'s repo-discovery + apply step: phase 1
    applies ``oracle.patch`` (judge the reference state), phase 2 applies
    ``agent.patch`` (judge what the agent did). Both run as root with
    ``safe.directory='*'`` so repos chowned to a non-root user apply cleanly.
    """
    patch_to_apply = inputs.oracle_patch if inputs.phase == 1 else inputs.agent_patch
    skip_apply = inputs.phase == 1 and not (patch_to_apply or "").strip()

    patch_file = host_tmp / "apply.patch"
    patch_file.write_text(_clean_agent_patch(patch_to_apply or ""))
    await env.upload_file(patch_file, "/tmp/agent.patch")

    discover = (
        "set -e; "
        'ROOTS="/workspace /opt /home /app /repo /tmp /entire-cli /entireio-cli /no-magic"; '
        'if [ -n "${HARBOR_REPO_PATHS:-}" ]; then '
        '  ROOTS="$ROOTS $(echo "$HARBOR_REPO_PATHS" | tr ":" " ")"; '
        "fi; "
        'EXISTING=""; '
        'for r in $ROOTS; do [ -e "$r" ] && EXISTING="$EXISTING $r"; done; '
        'if [ -z "$EXISTING" ]; then echo "NO_REPO_ROOTS_EXIST" >&2; exit 1; fi; '
        # Prefer the top-level /workspace repository. Some tasks contain a git
        # submodule (for example /workspace/nunchaku); an unordered `find | head`
        # can select that child even though final.patch is rooted at /workspace.
        'if [ -e /workspace/.git ]; then REPO=/workspace; else '
        "REPO=$(find $EXISTING -maxdepth 3 -name .git \\( -type d -o -type f \\) "
        "2>/dev/null | awk '{ p=$0; n=gsub(\"/\",\"/\",p); print n, $0 }' "
        "| sort -n | head -1 | cut -d' ' -f2- | xargs -r dirname); fi; "
        'if [ -z "$REPO" ]; then echo "NO_GIT_REPO_FOUND" >&2; exit 1; fi; '
        'cd "$REPO" && echo "applying to $(pwd)" && '
    )
    if skip_apply:
        # No diffable oracle (Phase 1 fallback) — leave the buggy state, just
        # make it world-readable for the judge.
        discover += 'chmod -R a+rwX "$REPO" 2>/dev/null || true'
    else:
        # Apply, capture rc, chmod best-effort, then propagate the apply rc so a
        # genuine apply failure surfaces (unlike a trailing `|| true`). Fall back
        # to --recount (fixes @@ line-count drift from the repo_diff capture that
        # otherwise trips "corrupt patch at line N") then to fuzz (-C1) before
        # giving up — only a truly unappliable patch then fails (→ scored 0).
        # Use a shell function so safe.directory="*" stays quoted (an unquoted
        # `$GA` word-splits and GLOBS the `*` against cwd → bogus git args).
        # safe.directory is also set globally in PodmanEnvironment.start, so this
        # is belt-and-suspenders. Fall back to --recount (fixes @@ line-count
        # drift) then fuzz (-C1); only a truly unappliable patch then fails (→0).
        discover += (
            'if ! command -v git >/dev/null 2>&1; then '
            'echo "GIT_NOT_FOUND" >&2; exit 127; fi; '
            'A() { git -c safe.directory="*" apply --whitespace=nowarn "$@" /tmp/agent.patch; }; '
            # ``discover`` starts with ``set -e``. Disable errexit while trying
            # the progressively more tolerant apply modes; otherwise the first
            # failure exits before ``rc=$?`` and neither fallback ever runs.
            'set +e; '
            'A 2>/tmp/ga.err; rc=$?; '
            'if [ $rc -ne 0 ]; then A --recount 2>>/tmp/ga.err; rc=$?; fi; '
            'if [ $rc -ne 0 ]; then A --recount -C1 2>>/tmp/ga.err; rc=$?; fi; '
            'set -e; '
            'chmod -R a+rwX "$REPO" 2>/dev/null || true; '
            'if [ $rc -ne 0 ]; then '
            f'printf "{_PATCH_REJECTED_SENTINEL}%s\\n" "$rc" >&2; '
            'cat /tmp/ga.err >&2; fi; exit "$rc"'
        )

    res = await env.exec(discover, timeout_sec=180)
    repo_hint = "/workspace"
    repo_discovered = False
    for line in (res.stdout or "").splitlines():
        if line.startswith("applying to "):
            repo_hint = line[len("applying to ") :].strip()
            repo_discovered = True
            break
    if res.return_code != 0:
        output = "\n".join(part for part in (res.stdout, res.stderr) if part)
        rejected_rc: int | None = None
        for line in output.splitlines():
            if line.startswith(_PATCH_REJECTED_SENTINEL):
                try:
                    rejected_rc = int(line[len(_PATCH_REJECTED_SENTINEL) :])
                except ValueError:
                    rejected_rc = None
                break
        # git apply uses 1 for a contextual rejection and 128 for a malformed
        # patch. Signal deaths (e.g. 137/OOM) remain infrastructure errors even
        # if the shell survived long enough to print the sentinel.
        if (
            rejected_rc in (1, 128)
            and res.return_code == rejected_rc
            and repo_discovered
        ):
            raise DeterministicPatchApplyError(
                return_code=rejected_rc,
                repo_path=repo_hint,
                stdout=res.stdout or "",
                stderr=res.stderr or "",
            )
        raise RuntimeError(
            f"patch apply failed (rc={res.return_code}): "
            f"stdout={(res.stdout or '')[-500:]!r} stderr={(res.stderr or '')[-500:]!r}"
        )
    return repo_hint


async def _drop_inputs(env: PodmanEnvironment, inputs: JudgeInputs, host_tmp: Path) -> None:
    """Populate /tmp/judge_inputs/ in the container exactly like the E2B judge."""
    d = host_tmp / "judge_inputs"
    (d / "tests").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text(inputs.readme or "")
    (d / "user_simulation_prompt.md").write_text(inputs.user_sim_prompt or "")
    (d / "oracle.patch").write_text(inputs.oracle_patch or "")
    (d / "agent.patch").write_text(inputs.agent_patch or "")
    (d / "test.sh").write_text(inputs.test_sh or "")
    (d / "judge_system.md").write_text(inputs.system_prompt or "")
    if inputs.user_dialogue:
        (d / "user_dialogue.md").write_text(inputs.user_dialogue)
    if inputs.phase == 2 and inputs.canonical_goals_json:
        (d / "canonical_goals.json").write_text(inputs.canonical_goals_json)
    for filename, content in (inputs.tests_files or {}).items():
        try:
            (d / "tests" / filename).write_bytes(content)
        except OSError as e:
            log.debug("skip tests/%s: %s", filename, e)

    await env.upload_dir(d, INPUTS_DIR)
    await env.exec(f"chmod +x {INPUTS_DIR}/tests/test.sh 2>/dev/null || true", timeout_sec=15)


def _first_message(inputs: JudgeInputs, repo_hint: str) -> str:
    """The per-phase first user message — copied verbatim from the E2B judge."""
    if inputs.phase == 1:
        if (inputs.oracle_patch or "").strip():
            return (
                f"Begin by reading {INPUTS_DIR}/README.md and "
                f"{INPUTS_DIR}/user_simulation_prompt.md to understand the task. "
                f"Then read {INPUTS_DIR}/oracle.patch (the reference solution, "
                f"already applied to {repo_hint}) and explore the workspace. "
                f"You may run the canonical test.sh to see which F2P tests the "
                f"oracle satisfies. Decompose the task into completeness goals "
                f"and write the FROZEN rubric to {INPUTS_DIR}/canonical_goals.json."
            )
        return (
            f"Begin by reading {INPUTS_DIR}/README.md and "
            f"{INPUTS_DIR}/user_simulation_prompt.md to understand the task. "
            f"This task has NO canonical oracle patch (the original session's "
            f"tool_use inputs were stripped of diffable content). Instead, "
            f"read {INPUTS_DIR}/user_dialogue.md which contains: (a) the "
            f"per-turn user intents extracted from the original session, and "
            f"(b) the verbatim user messages. Derive goals from what the user "
            f"explicitly asked for + corrections they made + tests in "
            f"{INPUTS_DIR}/test.sh — those F2P tests are the empirical "
            f"definition of 'completed'. The workspace at {repo_hint} is in "
            f"the BUGGY pre-fix state (no oracle applied), so use it for "
            f"context on the codebase shape but not as evidence of the "
            f"correct fix. Decompose into completeness goals and write the "
            f"FROZEN rubric to {INPUTS_DIR}/canonical_goals.json."
        )
    if inputs.phase == 2:
        return (
            f"Begin by reading {INPUTS_DIR}/canonical_goals.json — this is "
            f"the FROZEN rubric. DO NOT re-derive goals; for each goal in "
            f"the rubric, mark met:true/false with concrete evidence. Then "
            f"read {INPUTS_DIR}/README.md and {INPUTS_DIR}/user_simulation_prompt.md "
            f"for context, inspect the agent's patch at {INPUTS_DIR}/agent.patch "
            f"(already applied to {repo_hint}), explore the workspace, and "
            f"optionally run tests. Write your verdict to "
            f"{INPUTS_DIR}/verdict.json."
        )
    raise ValueError(f"unsupported judge phase={inputs.phase!r} (use 1 or 2).")


# ── host-side agentic loop ─────────────────────────────────────────────────────


async def _bash_exec(env: PodmanEnvironment, command: str, timeout_sec: int) -> str:
    try:
        res = await env.exec(command, cwd="/tmp", timeout_sec=timeout_sec)
    except Exception as e:  # podman timeout / crash — report to the model, keep going
        return f"(exit=124) command failed to run: {e}"
    parts = []
    if res.stdout:
        parts.append(res.stdout)
    if res.stderr:
        parts.append("[stderr]\n" + res.stderr)
    out = "\n".join(parts).strip()
    if len(out) > _MAX_TOOL_OUTPUT:
        out = out[:_MAX_TOOL_OUTPUT] + f"\n...[truncated {len(out) - _MAX_TOOL_OUTPUT} chars]"
    return f"(exit={res.return_code})\n{out}" if out else f"(exit={res.return_code})"


async def _output_exists(env: PodmanEnvironment, path: str) -> bool:
    r = await env.exec(f"test -f {shlex.quote(path)} && echo __OK__", timeout_sec=15)
    return "__OK__" in (r.stdout or "")


async def _completion_with_retry(**kwargs):
    """acompletion with one retry on transient errors (gateway blips)."""
    try:
        return await litellm.acompletion(**kwargs)
    except Exception as e:
        log.warning("judge acompletion error (retrying once): %s", str(e)[:200])
        await asyncio.sleep(3)
        return await litellm.acompletion(**kwargs)


async def _run_loop(
    env: PodmanEnvironment,
    system_prompt: str,
    first_message: str,
    output_path: str,
    timeout_sec: int,
    max_turns: int,
) -> tuple[dict, str, int]:
    """Drive Opus (host-side) with a bash tool until it writes ``output_path``.

    Returns (verdict_dict, transcript, exit_code).
    """
    model = _judge_model()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_base = os.environ.get("ANTHROPIC_BASE_URL") or None
    messages: list[dict] = [{"role": "user", "content": first_message}]
    transcript: list[str] = []
    deadline = time.monotonic() + timeout_sec
    exit_code = 0

    for turn in range(max_turns):
        remaining = deadline - time.monotonic()
        if remaining <= 5:
            transcript.append(f"[budget] wall-clock exhausted at turn {turn}")
            exit_code = 124
            break
        if turn == max(1, max_turns - 5):
            # Some large refactors keep the judge exploring until the hard
            # turn cap even after it has already reasoned through every goal.
            # Give it an explicit finalization budget while there are still
            # enough turns to write and, if needed, repair the JSON file.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Only 5 judge turns remain. Stop further exploration now. "
                        f"Using the evidence already collected, write the required "
                        f"STRICT JSON verdict to {output_path} with the bash tool, "
                        "verify that the file exists, and stop."
                    ),
                }
            )
            transcript.append("[budget] five-turn finalization warning")
        try:
            resp = await _completion_with_retry(
                model=model,
                api_key=api_key,
                api_base=api_base,
                messages=[{"role": "system", "content": system_prompt}, *messages],
                tools=_BASH_TOOL,
                tool_choice="auto",
                max_tokens=16384,
                timeout=min(remaining, 300),
            )
        except Exception as e:
            transcript.append(f"[llm-error] {str(e)[:300]}")
            exit_code = 1
            break

        msg = resp.choices[0].message
        tool_calls = list(msg.tool_calls or [])

        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)
        if msg.content:
            transcript.append(f"[assistant] {msg.content[:400]}")

        if not tool_calls:
            # Model stopped calling tools. Done iff the output file is present.
            if await _output_exists(env, output_path):
                break
            transcript.append("[nudge] no tool call and no output file yet")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"You have not written {output_path} yet. Complete the "
                        f"task now: write the required STRICT JSON to "
                        f"{output_path} using the bash tool, then stop."
                    ),
                }
            )
            continue

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            command = args.get("command", "")
            transcript.append(f"[bash] {command[:300]}")
            remaining = max(5, int(deadline - time.monotonic()))
            result = await _bash_exec(env, command, timeout_sec=remaining)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": "bash", "content": result}
            )

        if await _output_exists(env, output_path):
            break
    else:
        transcript.append(f"[budget] max turns exhausted ({max_turns})")
        exit_code = 2

    # Read the output file back out of the container.
    verdict: dict
    try:
        r = await env.exec(f"cat {shlex.quote(output_path)}", timeout_sec=30)
        if r.return_code == 0 and (r.stdout or "").strip():
            verdict = _extract_json(r.stdout)
        else:
            verdict = {
                "error": "verdict_read_failed",
                "expected_path": output_path,
                "judge_exit_code": exit_code,
                "transcript_tail": "\n".join(transcript)[-2000:],
            }
    except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
        verdict = {
            "error": "verdict_parse_failed",
            "exception": str(e)[:300],
            "expected_path": output_path,
            "transcript_tail": "\n".join(transcript)[-2000:],
        }
    return verdict, "\n".join(transcript), exit_code


def _phase2_expected_goal_ids(inputs: JudgeInputs) -> list[str]:
    """Return the frozen Phase-2 goal IDs, or an empty list if unavailable."""
    if inputs.phase != 2 or not inputs.canonical_goals_json:
        return []
    try:
        rubric = json.loads(inputs.canonical_goals_json)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(rubric, dict):
        return []
    goals = rubric.get("completeness_goals")
    if not isinstance(goals, list) or not goals:
        return []
    goal_ids = [goal.get("id") if isinstance(goal, dict) else None for goal in goals]
    if (
        any(not isinstance(goal_id, str) or not goal_id.strip() for goal_id in goal_ids)
        or len(goal_ids) != len(set(goal_ids))
    ):
        return []
    return goal_ids


def _phase2_goal_ids_need_repair(inputs: JudgeInputs, verdict: object) -> bool:
    """Detect an exact-ID coverage failure without guessing how rows map."""
    expected = _phase2_expected_goal_ids(inputs)
    if not expected or not isinstance(verdict, dict):
        return False
    results = verdict.get("goal_results")
    if not isinstance(results, list):
        return False
    observed = [
        result.get("id") if isinstance(result, dict) else None for result in results
    ]
    return (
        any(not isinstance(goal_id, str) or not goal_id for goal_id in observed)
        or len(observed) != len(set(observed))
        or len(observed) != len(expected)
        or set(observed) != set(expected)
    )


def _goal_id_repair_message(
    inputs: JudgeInputs, rejected_path: str, output_path: str
) -> str:
    """Build a fail-closed schema-repair request with the exact frozen IDs."""
    expected = _phase2_expected_goal_ids(inputs)
    return (
        "Your previously written Phase-2 verdict failed the exact "
        "goal_results ID coverage gate. Read the frozen rubric again at "
        f"{INPUTS_DIR}/canonical_goals.json and the rejected verdict at "
        f"{rejected_path}. Write a fresh complete STRICT JSON verdict to "
        f"{output_path}. The only permitted goal IDs, each required exactly "
        f"once, are: {json.dumps(expected)}. Do not blindly relabel rows by "
        "array position. Match each met decision and its evidence to the "
        "corresponding frozen rubric goal; when the rejected evidence cannot "
        "be matched unambiguously, re-inspect the workspace and reassess that "
        "goal. Recompute judge_score and verdict from the repaired goal_results, "
        "verify the file exists, and stop."
    )


async def _run_loop_with_goal_id_repair(
    env,
    inputs: JudgeInputs,
    first_message: str,
    output_path: str,
    timeout_sec: int,
    max_turns: int,
    *,
    run_loop=_run_loop,
) -> tuple[dict, str, int]:
    """Run the judge and make one explicit semantic retry on wrong goal IDs.

    The host never rewrites or positionally remaps model output. The rejected
    verdict is moved aside and Opus must produce a new verdict after seeing the
    exact frozen IDs. Any second malformed result remains malformed and is
    rejected by ``run_batch``'s normal strict schema gate.
    """
    verdict, transcript, exit_code = await run_loop(
        env,
        inputs.system_prompt,
        first_message,
        output_path,
        timeout_sec,
        max_turns,
    )
    if not _phase2_goal_ids_need_repair(inputs, verdict):
        return verdict, transcript, exit_code

    rejected_path = f"{output_path}.invalid-goal-ids"
    move = await env.exec(
        f"mv -- {shlex.quote(output_path)} {shlex.quote(rejected_path)}",
        timeout_sec=30,
    )
    if move.return_code != 0:
        return (
            verdict,
            transcript + "\n[schema-repair] could not preserve rejected verdict",
            exit_code,
        )

    log.warning(
        "Phase-2 goal IDs mismatched the frozen rubric; requesting one "
        "exact-ID semantic repair"
    )
    repaired, repair_transcript, repair_exit_code = await run_loop(
        env,
        inputs.system_prompt,
        _goal_id_repair_message(inputs, rejected_path, output_path),
        output_path,
        min(timeout_sec, 600),
        max_turns,
    )
    return (
        repaired,
        transcript + "\n[schema-repair] exact-ID retry\n" + repair_transcript,
        repair_exit_code,
    )


# ── public entrypoint (mirrors run_judge_in_e2b) ───────────────────────────────


async def run_judge_in_podman(
    task_name: str,
    trial_id: str,
    inputs: JudgeInputs,
    oauth_token: str | None = None,
    *,
    timeout_sec: int = JUDGE_TIMEOUT_SEC,
    max_turns: int = JUDGE_MAX_TURNS,
    api_key: str | None = None,
) -> JudgeRunResult:
    """Run the agentic judge in a podman container, host-side model.

    Signature-compatible with ``run_judge_in_e2b``. ``oauth_token`` / ``api_key``
    are accepted for parity but ignored: the podman judge authenticates via
    ``ANTHROPIC_API_KEY`` + ``ANTHROPIC_BASE_URL`` (the metagen gateway) read
    from the environment.
    """
    model = _judge_model()
    task_dir = TASKS_DIR / task_name
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return JudgeRunResult(
            verdict={"error": "task_toml_missing", "task": task_name},
            stdout="", stderr="", exit_code=1, sandbox_id="", judge_model=model,
        )

    env_cfg = TaskConfig.model_validate_toml(task_toml.read_text()).environment
    # The judge always needs egress (patch apply + test.sh installs), regardless
    # of the task's own allow_internet — matches the E2B judge's allow_internet.
    env_cfg.allow_internet = True
    if not env_cfg.docker_image:
        return JudgeRunResult(
            verdict={
                "error": "no_docker_image",
                "task": task_name,
                "detail": "podman judge pulls the prebuilt image; task.toml has none.",
            },
            stdout="", stderr="", exit_code=1, sandbox_id="", judge_model=model,
        )

    session_id = f"{trial_id}--judge"
    host_tmp_root = os.environ.get("HARBOR_PODMAN_TMPDIR", "/tmp")
    with tempfile.TemporaryDirectory(prefix="judge-", dir=host_tmp_root) as tmp:
        env = PodmanEnvironment(
            environment_dir=task_dir / "environment",
            environment_name=task_name,
            session_id=session_id,
            trial_paths=TrialPaths(trial_dir=Path(tmp)),
            task_env_config=env_cfg,
        )
        # Per-trial ISOLATED podman store. Concurrent judges sharing one vfs
        # store race on shared image layers (a finishing trial's rmi removes a
        # layer another trial is mid-pull on → "layer not known" / "image not
        # known" / "container has already been removed"). Give each trial its own
        # throwaway store under /dev/shm and delete it in finally, so trials
        # never touch each other's layers.
        import hashlib
        import shutil

        # Short hashed dir: podman rejects a runroot path >50 chars (socket
        # sun_path limit), and "/dev/shm/judge-<trial_id>--judge/run" overflows.
        _store_base = os.environ.get("SWE_PODMAN_STORE_BASE", "/dev/shm")
        _tag = hashlib.md5(session_id.encode()).hexdigest()[:10]
        _trial_store = os.path.join(_store_base, f"j{_tag}")
        env._store = os.path.join(_trial_store, "store")
        env._runroot = os.path.join(_trial_store, "run")
        env._tmpdir = os.path.join(_trial_store, "tmp")
        for _d in (env._store, env._runroot, env._tmpdir):
            os.makedirs(_d, exist_ok=True)
        container_id = ""
        try:
            await env.start(force_build=False)
            container_id = env._container_id or ""
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
                sandbox_id=container_id,
                judge_model=model,
            )
        except DeterministicPatchApplyError as e:
            verdict = _deterministic_patch_failure_verdict(inputs, e)
            if verdict is not None:
                log.info(
                    "podman deterministic patch rejection for %s/%s: %s",
                    task_name,
                    trial_id,
                    e,
                )
                return JudgeRunResult(
                    verdict=verdict,
                    stdout=e.stdout,
                    stderr=e.stderr,
                    # The judge completed successfully by applying the
                    # deterministic pre-judge rule. Preserve git's rc inside
                    # verdict.patch_apply instead of marking this as infra.
                    exit_code=0,
                    sandbox_id=container_id,
                    judge_model=model,
                )
            log.warning(
                "podman Phase-1 patch rejection for %s/%s: %s",
                task_name,
                trial_id,
                e,
            )
            return JudgeRunResult(
                verdict={"error": "podman_judge_failed", "exception": str(e)[:500]},
                stdout=e.stdout,
                stderr=e.stderr,
                exit_code=1,
                sandbox_id=container_id,
                judge_model=model,
            )
        except Exception as e:
            log.warning("podman judge failed for %s/%s: %s", task_name, trial_id, e)
            return JudgeRunResult(
                verdict={"error": "podman_judge_failed", "exception": str(e)[:500]},
                stdout="", stderr=str(e)[:500], exit_code=1,
                sandbox_id=container_id, judge_model=model,
            )
        finally:
            try:
                await env.stop(delete=True)
            except Exception as e:
                log.warning("judge container stop failed (%s): %s", session_id, e)
            shutil.rmtree(_trial_store, ignore_errors=True)
