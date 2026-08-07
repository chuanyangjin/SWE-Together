#!/usr/bin/env python3
"""Archive non-results/infra failures and verify an exact trial matrix.

Exit 0 means every task has exactly ``--replicates`` valid trials after the
archive pass. Exit 10 means the move succeeded but valid deficits remain, so a
resumable ``--skip-existing`` run should be launched. Other failures exit 2.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval_infra_sentinel import classify_trial, write_sidecar  # noqa: E402

_LOCK_FILENAME = ".archive-invalid-trials.lock"


def _task_name(trial_name: str, tasks: list[str]) -> str | None:
    prefix = trial_name.rsplit("__", 1)[0]
    if prefix in tasks:
        return prefix
    candidates = [name for name in tasks if name.startswith(prefix)]
    return max(candidates, key=len) if candidates else None


def _invalid_reason(trial: Path) -> str | None:
    result_path = trial / "result.json"
    try:
        result = json.loads(result_path.read_text())
    except FileNotFoundError:
        return "missing_result"
    except (OSError, json.JSONDecodeError):
        return "invalid_result"
    if not isinstance(result, dict):
        return "invalid_result"
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if not isinstance(rewards, dict) or not rewards:
        return "missing_rewards"
    reward = rewards.get("reward")
    try:
        valid_reward = (
            isinstance(reward, (int, float))
            and not isinstance(reward, bool)
            and math.isfinite(float(reward))
        )
    except (TypeError, ValueError):
        valid_reward = False
    if not valid_reward:
        return "invalid_reward"
    verdict = classify_trial(trial)
    write_sidecar(trial, verdict)
    if verdict.status == "infra_failed":
        return f"infra_failed:{verdict.reason}"
    return None


def _archive_destination(archive_root: Path, trial_name: str) -> Path:
    """Return a non-destructive destination even after a cancellation race.

    A late Harbor writer can recreate an active trial directory after an audit
    already moved the original directory.  Preserve both copies instead of
    blocking every subsequent resumable repair on the name collision.
    """
    destination = archive_root / trial_name
    if not destination.exists():
        return destination
    index = 1
    while True:
        candidate = archive_root / f"{trial_name}.duplicate-{index}"
        if not candidate.exists():
            return candidate
        index += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args(argv)
    if args.replicates <= 0:
        parser.error("--replicates must be positive")

    trials_root = args.trials_root.resolve()
    archive_root = args.archive_root.resolve()
    tasks_root = args.tasks_root.resolve()
    if not trials_root.is_dir() or not tasks_root.is_dir():
        parser.error("--trials-root and --tasks-root must be directories")
    if (
        trials_root == archive_root
        or trials_root in archive_root.parents
        or archive_root in trials_root.parents
    ):
        parser.error("active and archive roots must be separate, non-nested paths")
    archive_root.mkdir(parents=True, exist_ok=True)
    # The repair/finalize entrypoints also take a pipeline-wide lock, but keep
    # this script safe when invoked directly. Lock both resources in a stable
    # order: the active-root lock serializes competing audits even when callers
    # choose different archives, while the archive lock protects destination
    # selection when multiple active roots share one archive.
    lock_paths = sorted(
        {
            trials_root / _LOCK_FILENAME,
            archive_root / _LOCK_FILENAME,
        },
        key=lambda path: str(path),
    )
    archive_locks = [path.open("a+") for path in lock_paths]
    for lock in archive_locks:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    tasks = sorted(path.name for path in tasks_root.iterdir() if path.is_dir())

    moved: list[tuple[str, str]] = []
    for trial in sorted(trials_root.glob("*__*")):
        if not trial.is_dir():
            continue
        reason = _invalid_reason(trial)
        if not reason:
            continue
        destination = _archive_destination(archive_root, trial.name)
        shutil.move(str(trial), str(destination))
        moved.append((trial.name, reason))
        suffix = (
            f" destination={destination.name}"
            if destination.name != trial.name
            else ""
        )
        print(f"ARCHIVED {trial.name} reason={reason}{suffix}")

    counts: Counter[str] = Counter()
    unknown: list[str] = []
    for trial in sorted(trials_root.glob("*__*")):
        if not trial.is_dir():
            continue
        task = _task_name(trial.name, tasks)
        if task is None:
            unknown.append(trial.name)
        else:
            counts[task] += 1

    deficits = {
        task: args.replicates - counts.get(task, 0)
        for task in tasks
        if counts.get(task, 0) != args.replicates
    }
    active = sum(counts.values())
    expected = len(tasks) * args.replicates
    print(
        f"MATRIX active={active} expected={expected} tasks={len(tasks)} "
        f"moved={len(moved)} mismatched_tasks={len(deficits)} unknown={len(unknown)}"
    )
    for task, deficit in sorted(deficits.items()):
        print(f"DEFICIT {task} delta={deficit:+d}")
    for name in unknown:
        print(f"UNKNOWN {name}")
    if any(delta < 0 for delta in deficits.values()):
        print(
            "ERROR: active matrix contains excess valid trials; refusing to "
            "guess which result to archive",
            file=sys.stderr,
        )
        result = 2
    else:
        result = 0 if not deficits and not unknown and active == expected else 10
    for lock in reversed(archive_locks):
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
