"""Canonical correctness step — Phase 1 (rubric, once per task) + Phase 2 (per trial).

This is the agentic judge for `eval/eval_design.md` Step 1. Two phases:

  **Phase 1** — `generate_task_goals.generate_one` — runs **once per task**:
      reads task spec + oracle patch, derives per-task `canonical_goals.json`
      (frozen rubric of weighted goals). Same rubric is re-used across every
      cohort + replicate, so judge_score deltas reflect agent quality rather
      than judge decomposition noise. Cached at
      `tasks/<task>/canonical_goals.json`.

  **Phase 2** — per-trial scoring — runs **once per trial**:
      reads the frozen rubric, marks each goal `met: true/false` against the
      agent's patch, and writes `judge_verdict.json` with
      `judge_score = sum(weight × met)` mechanically derived.

This runner reads a plan file describing many (trial, task, out_name) jobs.
At startup it runs Phase 1 for any unique task lacking a rubric, then Phase 2
in an asyncio.Semaphore-bounded sandbox pool.

Usage (matches the legacy single-pass CLI for orchestrator drop-in):
    .venv/bin/python -m eval.correctness.run_batch --plan plan.json --workers 50

Plan file shape (JSON list):
    [
      {"trial_dir": "<abs path>", "task_dir": "<abs path>",
       "out_name": "judge_verdict.json"},
      ...
    ]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from eval.correctness.sandbox import (  # noqa: E402
    JudgeInputs,
    judge_timeout_for_task,
    run_judge_in_e2b,
)
from eval.correctness.generate_task_goals import generate_one as _phase1_generate_one
from eval.correctness._env import load_dotenv  # shared .env loader
from eval.patch_utils import patch_text_has_changes


def _judge_runner():
    """Select the judge backend per ``JUDGE_ENV``.

    ``JUDGE_ENV=podman`` → the host-side Opus loop over a local podman container
    (on-cluster harness; must run inside judge_local.sh's unshare session).
    ``JUDGE_ENV=sandoq`` → the same host-side Opus loop, but its bash tool execs
    into a remote sandoq OCI-run session (no local container runtime).
    Anything else → the canonical E2B sandbox judge. Imported lazily so the E2B
    path doesn't pull in litellm/podman_env and vice-versa.
    """
    judge_env = os.environ.get("JUDGE_ENV")
    if judge_env == "podman":
        from eval.correctness.podman_judge import run_judge_in_podman

        return run_judge_in_podman
    if judge_env == "sandoq":
        from eval.correctness.sandoq_judge import run_judge_in_sandoq

        return run_judge_in_sandoq
    return run_judge_in_e2b

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE2_PROMPT = (
    REPO_ROOT / "eval" / "correctness" / "prompts" / "judge_phase2_system.md"
).read_text()

log = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_tests_files(task_dir: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    td = task_dir / "tests"
    if not td.exists():
        return out
    for f in td.iterdir():
        if f.is_file():
            try:
                out[f.name] = f.read_bytes()
            except Exception:
                pass
    return out


# ``rationale`` is useful judge context but three released frozen rubrics predate
# that field. The scoring-critical schema is goal identity/text/tier/weight.
_RUBRIC_REQUIRED_FIELDS = {"id", "goal", "tier", "weight"}
_VERDICT_BUCKETS = {"equivalent", "partial", "incorrect", "gameable"}


def _rubric_issues(rubric: object) -> list[str]:
    """Validate the frozen scoring rubric before it can affect a verdict."""
    if not isinstance(rubric, dict):
        return ["rubric_not_object"]
    if "error" in rubric:
        return ["rubric_contains_error"]
    goals = rubric.get("completeness_goals")
    if not isinstance(goals, list) or not goals:
        return ["rubric_goals_missing"]

    issues: list[str] = []
    ids: list[str] = []
    weights: list[float] = []
    has_core = False
    for index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            issues.append(f"rubric_goal_{index}_not_object")
            continue
        missing = _RUBRIC_REQUIRED_FIELDS - set(goal)
        if missing:
            issues.append(
                f"rubric_goal_{index}_missing:{','.join(sorted(missing))}"
            )
        goal_id = goal.get("id")
        if not isinstance(goal_id, str) or not goal_id.strip():
            issues.append(f"rubric_goal_{index}_invalid_id")
        else:
            ids.append(goal_id)
        if not isinstance(goal.get("goal"), str) or not goal["goal"].strip():
            issues.append(f"rubric_goal_{index}_invalid_goal")
        if not isinstance(goal.get("tier"), str) or not goal["tier"].strip():
            issues.append(f"rubric_goal_{index}_invalid_tier")
        weight = goal.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or not 0.0 <= float(weight) <= 1.0
        ):
            issues.append(f"rubric_goal_{index}_invalid_weight")
        else:
            weights.append(float(weight))
        has_core = has_core or goal.get("tier") == "core"

    if len(ids) != len(set(ids)):
        issues.append("rubric_duplicate_goal_ids")
    if len(weights) == len(goals) and not math.isclose(
        sum(weights), 1.0, rel_tol=0.0, abs_tol=0.01
    ):
        issues.append(f"rubric_weight_sum:{sum(weights):.6g}")
    if not has_core:
        issues.append("rubric_missing_core_goal")
    return issues


def _goal_result_issues(rubric: dict, goal_results: object) -> list[str]:
    """Require one boolean decision with evidence for every frozen goal."""
    if not isinstance(goal_results, list):
        return ["goal_results_not_list"]
    expected = [goal["id"] for goal in rubric["completeness_goals"]]
    observed: list[str] = []
    issues: list[str] = []
    for index, result in enumerate(goal_results):
        if not isinstance(result, dict):
            issues.append(f"goal_result_{index}_not_object")
            continue
        goal_id = result.get("id")
        if not isinstance(goal_id, str) or not goal_id:
            issues.append(f"goal_result_{index}_invalid_id")
        else:
            observed.append(goal_id)
        if type(result.get("met")) is not bool:  # bool only; 0/1 and strings fail.
            issues.append(f"goal_result_{index}_met_not_bool")
        evidence = result.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            issues.append(f"goal_result_{index}_missing_evidence")
    if len(observed) != len(set(observed)):
        issues.append("duplicate_goal_result_ids")
    if len(observed) != len(expected) or set(observed) != set(expected):
        issues.append("goal_result_ids_do_not_match_rubric")
    return issues


def _derive_judge_score(
    rubric: dict, goal_results: list[dict], *, gameable: bool = False
) -> tuple[float, str]:
    """Mechanically derive the score after exact schema validation."""
    if gameable:
        return 0.0, "gameable"
    weight_by_id = {
        goal["id"]: float(goal["weight"])
        for goal in rubric["completeness_goals"]
    }
    # ``_goal_result_issues`` guarantees exact IDs and real JSON booleans.
    met_by_id = {result["id"]: result["met"] for result in goal_results}
    score = round(
        sum(weight * (1.0 if met_by_id[goal_id] else 0.0)
            for goal_id, weight in weight_by_id.items()),
        2,
    )
    if score >= 0.85:
        return score, "equivalent"
    if score >= 0.30:
        return score, "partial"
    return score, "incorrect"


def _raw_phase2_verdict_issues(rubric: dict, verdict: object) -> list[str]:
    """Validate model output before adding host provenance and derived fields."""
    issues = _rubric_issues(rubric)
    if issues:
        return issues
    if not isinstance(verdict, dict):
        return ["verdict_not_object"]
    if "error" in verdict:
        return ["verdict_contains_error"]
    score = verdict.get("judge_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        issues.append("invalid_judge_score")
    if verdict.get("verdict") not in _VERDICT_BUCKETS:
        issues.append("invalid_verdict_bucket")
    if verdict.get("rubric_source") != "canonical_goals.json":
        issues.append("invalid_rubric_source")
    issues.extend(_goal_result_issues(rubric, verdict.get("goal_results")))
    return issues


def _normalize_phase2_verdict(
    rubric: dict, verdict: object
) -> tuple[dict[str, Any], list[str]]:
    """Return a schema-checked, mechanically scored model verdict."""
    issues = _raw_phase2_verdict_issues(rubric, verdict)
    if issues:
        return {"error": "invalid_phase2_verdict", "schema_issues": issues}, issues
    assert isinstance(verdict, dict)  # Proven above; narrows for type checkers.
    normalized = dict(verdict)
    normalized["judge_reported_score"] = verdict["judge_score"]
    normalized["judge_reported_verdict"] = verdict["verdict"]
    score, bucket = _derive_judge_score(
        rubric,
        verdict["goal_results"],
        gameable=verdict["verdict"] == "gameable",
    )
    normalized["judge_score"] = score
    normalized["verdict"] = bucket
    return normalized, []


def _phase2_verdict_issues(
    rubric: object,
    verdict: object,
    *,
    task_name: str | None = None,
    trial_id: str | None = None,
    expected_judge_model: str | None = None,
) -> list[str]:
    """Validate a persisted Phase-2 verdict for reuse/headline scoring."""
    issues = _rubric_issues(rubric)
    if issues:
        return issues
    assert isinstance(rubric, dict)
    if not isinstance(verdict, dict):
        return ["verdict_not_object"]
    if "error" in verdict:
        return ["verdict_contains_error"]
    issues.extend(_goal_result_issues(rubric, verdict.get("goal_results")))
    if issues:
        return issues

    score = verdict.get("judge_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        issues.append("invalid_judge_score")
    elif not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
        issues.append("invalid_judge_score")

    reported_gameable = verdict.get("judge_reported_verdict") == "gameable"
    final_gameable = verdict.get("verdict") == "gameable"
    expected_score, expected_bucket = _derive_judge_score(
        rubric,
        verdict["goal_results"],
        gameable=reported_gameable or final_gameable,
    )
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        if not math.isclose(
            float(score), expected_score, rel_tol=0.0, abs_tol=1e-9
        ):
            issues.append(
                f"judge_score_mismatch:{float(score):.6g}!={expected_score:.6g}"
            )
    if verdict.get("verdict") != expected_bucket:
        issues.append(
            f"verdict_bucket_mismatch:{verdict.get('verdict')!r}!={expected_bucket}"
        )
    if verdict.get("rubric_source") != "canonical_goals.json":
        issues.append("invalid_rubric_source")
    if verdict.get("judge_phase") != 2:
        issues.append("invalid_judge_phase")
    if verdict.get("rubric_n_goals") != len(rubric["completeness_goals"]):
        issues.append("rubric_goal_count_mismatch")
    if task_name is not None and verdict.get("task") != task_name:
        issues.append("verdict_task_mismatch")
    if trial_id is not None and verdict.get("trial_id") != trial_id:
        issues.append("verdict_trial_id_mismatch")
    if expected_judge_model is not None:
        observed = str(verdict.get("judge_model") or "").lower().rstrip("/")
        expected = str(expected_judge_model).lower().rstrip("/")
        observed = observed.rsplit("/", 1)[-1].rsplit(":", 1)[-1].replace("_", "-")
        expected = expected.rsplit("/", 1)[-1].rsplit(":", 1)[-1].replace("_", "-")
        if not observed or observed != expected:
            issues.append("judge_model_mismatch")
    return issues


async def _ensure_rubrics(task_dirs: set[Path], oauth_token: str,
                          api_key: str | None, workers: int,
                          force: bool) -> list[dict]:
    """Phase 1 pre-pass — generate canonical_goals.json for any task missing one.

    Phase 1 is **once per task, frozen**. Without --force-rubric, tasks that
    already have a rubric are skipped (no LLM call) so this is a near-noop on
    well-warmed task suites."""
    pending = sorted(
        td for td in task_dirs
        if force or not (td / "canonical_goals.json").exists()
    )
    if not pending:
        log.info("Phase 1: all %d tasks already have canonical_goals.json", len(task_dirs))
        return []
    log.info("Phase 1: generating rubrics for %d task(s) (workers=%d, force=%s)",
             len(pending), workers, force)
    sem = asyncio.Semaphore(workers)

    async def _bounded(td: Path) -> dict:
        async with sem:
            return await _phase1_generate_one(td, oauth_token, api_key, force)

    return await asyncio.gather(*[_bounded(td) for td in pending])


def _valid_phase2_verdict(
    path: Path,
    rubric: object,
    *,
    task_name: str | None = None,
    trial_id: str | None = None,
    expected_judge_model: str | None = None,
) -> bool:
    """True only for a reusable, successfully scored Phase-2 verdict."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return not _phase2_verdict_issues(
        rubric,
        value,
        task_name=task_name,
        trial_id=trial_id,
        expected_judge_model=expected_judge_model,
    )


async def _phase2_one(job: dict, oauth_token: str, sem: asyncio.Semaphore,
                      api_key: str | None, force: bool) -> dict:
    trial_dir = Path(job["trial_dir"]).expanduser()
    task_dir = Path(job["task_dir"]).expanduser()
    out_name = job.get("out_name") or "judge_verdict.json"
    out_path = trial_dir / out_name
    result: dict[str, Any] = {
        "trial_dir": str(trial_dir),
        "task_dir": str(task_dir),
        "out_name": out_name,
    }

    rubric_path = task_dir / "canonical_goals.json"
    if not rubric_path.exists():
        result["status"] = "skipped_no_rubric"
        result["reason"] = f"Phase 1 did not produce {rubric_path.relative_to(REPO_ROOT)}"
        return result
    rubric_text = rubric_path.read_text()
    try:
        rubric = json.loads(rubric_text)
    except json.JSONDecodeError as e:
        result["status"] = "rubric_parse_error"
        result["error"] = str(e)
        return result
    rubric_validation_issues = _rubric_issues(rubric)
    if rubric_validation_issues:
        result["status"] = "rubric_invalid"
        result["error"] = ", ".join(rubric_validation_issues)[:500]
        return result

    agent_patch_p = trial_dir / "agent" / "final.patch"
    if not agent_patch_p.exists():
        result["status"] = "skipped_no_patch"
        return result
    agent_patch = agent_patch_p.read_text()
    if not patch_text_has_changes(agent_patch):
        result["status"] = "skipped_empty_patch"
        result["patch_bytes"] = len(agent_patch)
        return result

    if not force and _valid_phase2_verdict(
        out_path,
        rubric,
        task_name=task_dir.name,
        trial_id=trial_dir.name,
    ):
        result["status"] = "skipped_existing"
        return result

    readme = (task_dir / "README.md").read_text() if (task_dir / "README.md").exists() else ""
    usp = task_dir / "user_simulation_prompt.md"
    user_sim = usp.read_text() if usp.exists() else ""
    test_sh_p = task_dir / "tests" / "test.sh"
    test_sh_text = test_sh_p.read_text() if test_sh_p.exists() else ""

    inputs = JudgeInputs(
        readme=readme,
        user_sim_prompt=user_sim,
        oracle_patch="",  # not used in Phase 2
        agent_patch=agent_patch,
        test_sh=test_sh_text,
        system_prompt=PHASE2_PROMPT,
        tests_files=_load_tests_files(task_dir),
        phase=2,
        canonical_goals_json=rubric_text,
    )

    timeout = judge_timeout_for_task(task_dir.name)
    async with sem:
        # Recheck after waiting for a worker slot.  A second resumable shard may
        # have completed this verdict while this coroutine was queued; checking
        # only before the semaphore permits duplicate model calls and a
        # nondeterministic last-writer-wins overwrite.
        if not force and _valid_phase2_verdict(
            out_path,
            rubric,
            task_name=task_dir.name,
            trial_id=trial_dir.name,
        ):
            result["status"] = "skipped_existing"
            return result
        t0 = time.time()
        log.info("start %s out=%s timeout=%ds", trial_dir.name, out_name, timeout)
        try:
            sb = await _judge_runner()(
                task_name=task_dir.name,
                trial_id=trial_dir.name,
                inputs=inputs,
                oauth_token=oauth_token,
                timeout_sec=timeout,
                api_key=api_key,
            )
        except Exception as e:
            result["status"] = "sandbox_failed"
            result["error"] = str(e)[:500]
            log.warning("sandbox_failed %s: %s", trial_dir.name, result["error"])
            return result
        elapsed = round(time.time() - t0, 1)

    verdict, schema_issues = _normalize_phase2_verdict(rubric, sb.verdict)
    if schema_issues:
        log.warning(
            "invalid Phase-2 verdict %s: %s",
            trial_dir.name,
            ", ".join(schema_issues),
        )

    verdict.setdefault("task", task_dir.name)
    verdict.setdefault("trial_id", trial_dir.name)
    reward_p = trial_dir / "verifier" / "reward.txt"
    if reward_p.exists():
        try:
            verdict["test_reward_raw"] = float(reward_p.read_text().strip())
        except ValueError:
            pass
    js = verdict.get("judge_score")
    tr = verdict.get("test_reward_raw")
    if js is not None and tr is not None:
        d = float(js) - float(tr)
        verdict["score_delta"] = round(d, 4)
        verdict["direction"] = ("unchanged" if abs(d) < 1e-6
                                else "upgrade" if d > 0 else "downgrade")
    verdict["judge_elapsed_sec"] = elapsed
    verdict["sandbox_id"] = sb.sandbox_id
    verdict["judge_exit_code"] = sb.exit_code
    if sb.judge_model:
        verdict["judge_model"] = sb.judge_model
    verdict["judge_phase"] = 2
    verdict["rubric_n_goals"] = len(rubric.get("completeness_goals", []))

    _atomic_write_json(out_path, verdict)
    result["status"] = "ok" if "error" not in verdict else "verdict_error"
    result["judge_score"] = verdict.get("judge_score")
    result["test_reward_raw"] = verdict.get("test_reward_raw")
    result["verdict"] = verdict.get("verdict")
    result["direction"] = verdict.get("direction")
    result["elapsed_sec"] = elapsed
    log.info("done %s score=%s verdict=%s elapsed=%.1fs",
             trial_dir.name, result.get("judge_score"),
             result.get("verdict"), elapsed)
    return result


async def amain() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-name", default="judge_verdict.json",
                    help="default filename inside each trial dir "
                         "(plan entries' own out_name takes precedence)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run Phase 2 over existing per-trial verdict files. "
                         "Does NOT regenerate the frozen Phase 1 rubric — "
                         "use --force-rubric for that.")
    ap.add_argument("--force-rubric", action="store_true",
                    help="Re-run Phase 1 even when canonical_goals.json exists. "
                         "Use only when intentionally rebuilding rubrics.")
    ap.add_argument("--phase1-workers", type=int, default=5,
                    help="Concurrency cap for Phase 1 pre-pass (default: 5).")
    ap.add_argument("--skip-phase1", action="store_true",
                    help="Skip the Phase 1 pre-pass entirely; trials whose task "
                         "is missing canonical_goals.json will report "
                         "skipped_no_rubric.")
    ap.add_argument("--summary", type=Path, default=None,
                    help="Write JSON run-summary to this path "
                         "(default: <plan>.summary.json)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    load_dotenv()
    judge_env = os.environ.get("JUDGE_ENV")
    # podman + sandoq are both host-side gateway judges (no E2B); they differ only
    # in where the bash tool execs (local container vs remote sandoq session).
    hostside_judge = judge_env in ("podman", "sandoq")
    if not hostside_judge and not os.environ.get("E2B_API_KEY"):
        print("ERROR: E2B_API_KEY not set", file=sys.stderr)
        return 2
    api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if hostside_judge:
        # Host-side loop authenticates via the gateway (ANTHROPIC_API_KEY +
        # ANTHROPIC_BASE_URL); OAuth/E2B are not used.
        if not (api_key and os.environ.get("ANTHROPIC_BASE_URL")):
            print(f"ERROR: JUDGE_ENV={judge_env} needs ANTHROPIC_API_KEY + "
                  "ANTHROPIC_BASE_URL (the gateway)", file=sys.stderr)
            return 2
        log.info("judge backend: %s (host-side %s via %s)",
                 judge_env,
                 os.environ.get("JUDGE_PODMAN_MODEL", "anthropic/claude-opus-4-6"),
                 os.environ["ANTHROPIC_BASE_URL"])
        if judge_env == "sandoq":
            # eval.run_eval/launch.py invoke this module directly, bypassing
            # judge_sandoq.sh's shell preflight. Reject the whole batch before
            # any per-trial error verdicts are written when the bearer token is
            # missing or has unsafe permissions.
            src_dir = REPO_ROOT / "src"
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            from sandoq_env import _read_token

            try:
                _read_token()
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
    else:
        if not (api_key or oauth):
            print("ERROR: need ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN",
                  file=sys.stderr)
            return 2
        auth_kind = ("ANTHROPIC_API_KEY (pay-per-token)" if api_key
                     else "CLAUDE_CODE_OAUTH_TOKEN (subscription)")
        log.info("judge auth: %s", auth_kind)

    if not args.plan.exists():
        print(f"ERROR: plan not found: {args.plan}", file=sys.stderr)
        return 2
    jobs = json.loads(args.plan.read_text())
    if not isinstance(jobs, list):
        print("ERROR: plan must be a JSON list", file=sys.stderr)
        return 2
    for j in jobs:
        if not j.get("out_name"):
            j["out_name"] = args.out_name

    # Phase 1 pre-pass — generate any missing rubrics. Cached on disk under
    # `tasks/<task>/canonical_goals.json`; on a warmed suite this is a
    # near-noop (all tasks report `skipped_existing`).
    phase1_results: list[dict] = []
    if not args.skip_phase1:
        task_dirs = {Path(j["task_dir"]).expanduser() for j in jobs}
        phase1_results = await _ensure_rubrics(
            task_dirs, oauth, api_key, args.phase1_workers, args.force_rubric,
        )
        p1_tally = Counter(r.get("status", "?") for r in phase1_results)
        if p1_tally:
            log.info("Phase 1 tally: %s", dict(p1_tally))

    # Phase 2 — per-trial scoring.
    log.info("Phase 2: %d trial job(s), workers=%d", len(jobs), args.workers)
    sem = asyncio.Semaphore(args.workers)
    t0 = time.time()
    tasks = [
        asyncio.create_task(_phase2_one(j, oauth, sem, api_key, args.force))
        for j in jobs
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    summary_path = args.summary or args.plan.with_suffix(".summary.json")
    summary = {
        "plan": str(args.plan),
        "workers": args.workers,
        "phase1_workers": args.phase1_workers,
        "force": args.force,
        "force_rubric": args.force_rubric,
        "skip_phase1": args.skip_phase1,
        "elapsed_sec": round(elapsed, 1),
        "n_jobs": len(jobs),
        "phase1_results": phase1_results,
        "phase2_results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    p2_tally = Counter(r.get("status", "?") for r in results)
    print(f"\nwrote summary to {summary_path}")
    print(f"\nDone in {elapsed:.1f}s")
    if phase1_results:
        p1_tally = Counter(r.get("status", "?") for r in phase1_results)
        print(f"  Phase 1: {dict(p1_tally)}")
    for k, v in sorted(p2_tally.items()):
        print(f"  Phase 2 {k}: {v}")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
