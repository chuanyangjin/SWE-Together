#!/usr/bin/env python3
"""Strict action-stage completeness and provenance validator for the Sushi lane."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from eval.table2_metrics import get_wall_minutes  # noqa: E402
from eval_infra_sentinel import classify_trial, write_sidecar  # noqa: E402


EXPECTED_MODEL = "openai/sushi_0803_step575"
EXPECTED_ENV_IMPORT = "podman_env:PodmanEnvironment"
EXPECTED_AGENT_IMPORT = "user_agent.agents.user_enabled_opencode:UserEnabledOpenCode"
EXPECTED_AGENT_NAME = "user-enabled-opencode"
EXPECTED_AGENT_VERSION = "1.15.13"
EXPECTED_USER_MODEL = "anthropic/claude-opus-4-8"
EXPECTED_USER_TEMPERATURE = 0.5
EXPECTED_TIMEOUT_SEC = 4800.0


@dataclass(frozen=True)
class TrialInspection:
    """One independently computed trial result, aggregated in input order."""

    scoped: bool = False
    counted_task: str | None = None
    wall_minutes: float | None = None
    raw_total: int | None = None
    errors: tuple[str, ...] = ()


def task_name(config: dict) -> str:
    task = config.get("task") or {}
    return Path(str(task.get("path") or "")).name


def valid_reward(result: dict) -> bool:
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    reward = rewards.get("reward")
    return (
        isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and math.isfinite(float(reward))
    )


def raw_token_totals(trial_dir: Path) -> tuple[int, int, int]:
    """Return (step_finish records, total tokens, output tokens) from raw JSONL."""
    records = total = output = 0
    for path in sorted((trial_dir / "agent").glob("opencode.txt.turn-*")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "step_finish":
                continue
            tokens = (event.get("part") or {}).get("tokens") or {}
            raw_total = tokens.get("total")
            raw_output = tokens.get("output")
            if not isinstance(raw_total, int) or not isinstance(raw_output, int):
                continue
            records += 1
            total += raw_total
            output += raw_output
    return records, total, output


def inspect_trial(
    trial_dir: Path,
    *,
    expected_model: str,
    expected_tasks: frozenset[str],
    scope_expected_tasks: bool,
) -> TrialInspection:
    """Validate one trial without mutating shared aggregation state."""

    config_path = trial_dir / "config.json"
    result_path = trial_dir / "result.json"
    errors: list[str] = []
    if not config_path.is_file():
        if not scope_expected_tasks:
            errors.append(f"{trial_dir.name}: missing config.json")
        return TrialInspection(errors=tuple(errors))
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return TrialInspection(
            errors=(f"{trial_dir.name}: invalid JSON: {exc}",)
        )
    name = task_name(config)
    if scope_expected_tasks and name not in expected_tasks:
        return TrialInspection()
    if not result_path.is_file():
        return TrialInspection(
            scoped=True,
            errors=(f"{trial_dir.name}: missing result.json",),
        )
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return TrialInspection(
            scoped=True,
            errors=(f"{trial_dir.name}: invalid result JSON: {exc}",),
        )

    model = (config.get("agent") or {}).get("model_name")
    agent = config.get("agent") or {}
    agent_kwargs = agent.get("kwargs") or {}
    env_import = (config.get("environment") or {}).get("import_path")
    if model != expected_model:
        errors.append(
            f"{trial_dir.name}: action model {model!r} != {expected_model!r}"
        )
    if env_import != EXPECTED_ENV_IMPORT:
        errors.append(
            f"{trial_dir.name}: environment.import_path "
            f"{env_import!r} != {EXPECTED_ENV_IMPORT!r}"
        )
    if agent.get("import_path") != EXPECTED_AGENT_IMPORT:
        errors.append(f"{trial_dir.name}: wrong action-agent import path")
    if agent_kwargs.get("user_model_name") != EXPECTED_USER_MODEL:
        errors.append(f"{trial_dir.name}: wrong user simulator model")
    if agent_kwargs.get("user_temperature") != EXPECTED_USER_TEMPERATURE:
        errors.append(
            f"{trial_dir.name}: wrong/missing user simulator temperature"
        )
    if agent.get("override_timeout_sec") != EXPECTED_TIMEOUT_SEC:
        errors.append(f"{trial_dir.name}: action timeout is not 4800 seconds")
    agent_info = result.get("agent_info") or {}
    if agent_info.get("name") != EXPECTED_AGENT_NAME:
        errors.append(f"{trial_dir.name}: action runtime is not OpenCode")
    if agent_info.get("version") != EXPECTED_AGENT_VERSION:
        errors.append(
            f"{trial_dir.name}: OpenCode runtime version is not 1.15.13"
        )

    token_records, raw_total, raw_output = raw_token_totals(trial_dir)
    valid_raw_total: int | None = raw_total
    if token_records < 1 or raw_total < 1 or raw_output < 1:
        errors.append(
            f"{trial_dir.name}: missing positive raw OpenCode token records"
        )
        valid_raw_total = None
    minutes = get_wall_minutes(result, trial_dir)
    valid_minutes: float | None = minutes
    if minutes is None or not math.isfinite(minutes) or minutes < 0:
        errors.append(
            f"{trial_dir.name}: missing/invalid action wall-clock minutes"
        )
        valid_minutes = None

    # Always recompute with the current sentinel implementation and refresh
    # the sidecar; never let a stale cached verdict qualify a resume.
    infra_verdict = classify_trial(trial_dir)
    write_sidecar(trial_dir, infra_verdict)
    evidence = infra_verdict.evidence or {}
    if (
        infra_verdict.status != "ok"
        or infra_verdict.version != 2
        or int(evidence.get("assistant_turn_count") or 0) < 1
    ):
        errors.append(f"{trial_dir.name}: missing/failing fresh infra verdict")
    if not valid_reward(result):
        errors.append(f"{trial_dir.name}: missing/non-finite verifier reward")

    return TrialInspection(
        scoped=True,
        counted_task=name,
        wall_minutes=valid_minutes,
        raw_total=valid_raw_total,
        errors=tuple(errors),
    )


def inspect_trials(
    trial_dirs: Iterable[Path],
    *,
    expected_model: str,
    expected_tasks: frozenset[str],
    scope_expected_tasks: bool,
    workers: int = 1,
) -> list[TrialInspection]:
    """Inspect trials serially by default or concurrently, preserving order."""

    if workers < 1:
        raise ValueError("workers must be positive")
    inspect = partial(
        inspect_trial,
        expected_model=expected_model,
        expected_tasks=expected_tasks,
        scope_expected_tasks=scope_expected_tasks,
    )
    if workers == 1:
        return [inspect(trial_dir) for trial_dir in trial_dirs]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(inspect, trial_dirs))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, default=Path("tasks"))
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--expected-model", default=EXPECTED_MODEL)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel trial readers (default: 1, preserving serial behavior).",
    )
    parser.add_argument("--expected-task", action="append", default=None)
    parser.add_argument(
        "--scope-expected-tasks",
        action="store_true",
        help="Ignore trial directories outside explicit --expected-task values.",
    )
    args = parser.parse_args()

    if args.scope_expected_tasks and not args.expected_task:
        parser.error("--scope-expected-tasks requires at least one --expected-task")
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.expected_task:
        expected_tasks = sorted(set(args.expected_task))
    else:
        expected_tasks = sorted(
            path.name
            for path in args.tasks_root.iterdir()
            if path.is_dir()
            and (path / "task.toml").exists()
            and (path / "instruction.md").exists()
            and (path / "tests" / "test.sh").exists()
        )
    errors: list[str] = []
    counts: Counter[str] = Counter()
    wall_minutes: list[float] = []
    cohort_raw_tokens: list[int] = []
    trial_dirs = sorted(path for path in args.trials_root.iterdir() if path.is_dir())
    scoped_trial_dirs = 0
    inspections = inspect_trials(
        trial_dirs,
        expected_model=args.expected_model,
        expected_tasks=frozenset(expected_tasks),
        scope_expected_tasks=args.scope_expected_tasks,
        workers=args.workers,
    )
    for inspection in inspections:
        scoped_trial_dirs += int(inspection.scoped)
        if inspection.counted_task is not None:
            counts[inspection.counted_task] += 1
        if inspection.raw_total is not None:
            cohort_raw_tokens.append(inspection.raw_total)
        if inspection.wall_minutes is not None:
            wall_minutes.append(inspection.wall_minutes)
        errors.extend(inspection.errors)

    for name in expected_tasks:
        if counts[name] != args.replicates:
            errors.append(f"{name}: trial count {counts[name]} != {args.replicates}")
    unexpected = sorted(set(counts) - set(expected_tasks))
    if unexpected:
        errors.append("unexpected task names: " + ", ".join(unexpected))

    summary = {
        "status": "pass" if not errors else "fail",
        "expected_tasks": len(expected_tasks),
        "expected_trials": len(expected_tasks) * args.replicates,
        "observed_trial_dirs": len(trial_dirs),
        "scoped_trial_dirs": scoped_trial_dirs,
        "exact_model": args.expected_model,
        "exact_environment_import_path": EXPECTED_ENV_IMPORT,
        "exact_agent_runtime": f"{EXPECTED_AGENT_NAME}@{EXPECTED_AGENT_VERSION}",
        "exact_user_model": EXPECTED_USER_MODEL,
        "exact_user_temperature": EXPECTED_USER_TEMPERATURE,
        "exact_agent_timeout_sec": EXPECTED_TIMEOUT_SEC,
        "mean_action_wall_minutes": (
            sum(wall_minutes) / len(wall_minutes) if wall_minutes else None
        ),
        "mean_raw_tokens_per_trial": (
            sum(cohort_raw_tokens) / len(cohort_raw_tokens) if cohort_raw_tokens else None
        ),
        "error_count": len(errors),
        "errors": errors[:100],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
