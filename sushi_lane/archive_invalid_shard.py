#!/usr/bin/env python3
"""Recoverably archive invalid cells owned by one Sushi shard.

By default, active directories (no result.json) and every other shard are
immutable. A shard-lock holder may explicitly archive its stale incomplete
directories before a repair pass.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sushi_lane"))

from eval_infra_sentinel import classify_trial, write_sidecar  # noqa: E402
from shard_tasks import validate_manifest  # noqa: E402


def _valid_reward(result: object) -> bool:
    if not isinstance(result, dict):
        return False
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    return (
        isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and math.isfinite(float(reward))
    )


def _destination(archive_root: Path, name: str) -> Path:
    candidate = archive_root / name
    index = 0
    while candidate.exists():
        index += 1
        candidate = archive_root / f"{name}.duplicate-{index}"
    return candidate


def archive_invalid(
    trials_root: Path,
    tasks_root: Path,
    manifest_path: Path,
    shard_index: int,
    archive_root: Path,
    *,
    archive_incomplete: bool = False,
    force_archive_names: set[str] | None = None,
) -> list[tuple[str, str]]:
    manifest = json.loads(manifest_path.read_text())
    shards = validate_manifest(manifest, tasks_root)
    owned = set(shards[shard_index])
    archive_root.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[str, str]] = []

    for trial in sorted(trials_root.iterdir()):
        config_path = trial / "config.json"
        result_path = trial / "result.json"
        if not trial.is_dir() or not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            # Without trusted ownership provenance, fail closed without moving.
            continue
        task_config = config.get("task")
        if not isinstance(task_config, dict):
            # Without trusted ownership provenance, fail closed without moving.
            continue
        task = Path(str(task_config.get("path") or "")).name
        if task not in owned:
            continue

        reason: str | None = None
        if force_archive_names and trial.name in force_archive_names:
            reason = "sanitizer_race_exposure"
        elif not result_path.is_file():
            if archive_incomplete:
                reason = "incomplete"
            else:
                continue
        else:
            try:
                result = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                reason = "invalid_result"
            else:
                if not _valid_reward(result):
                    reason = "invalid_reward"
        if reason is None:
            verdict = classify_trial(trial)
            write_sidecar(trial, verdict)
            if verdict.status != "ok":
                reason = f"{verdict.status}:{verdict.reason}"
        if reason is None:
            continue
        destination = _destination(archive_root, trial.name)
        shutil.move(str(trial), str(destination))
        moved.append((trial.name, reason))
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument(
        "--archive-incomplete",
        action="store_true",
        help="Also archive owned no-result directories; use only while holding the shard producer lock.",
    )
    parser.add_argument("--force-archive-list", type=Path)
    args = parser.parse_args()
    if not 0 <= args.shard_index < 4:
        parser.error("--shard-index must be 0 through 3")
    force_archive_names: set[str] | None = None
    if args.force_archive_list:
        force_archive_names = {
            line.strip()
            for line in args.force_archive_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if any(Path(name).name != name for name in force_archive_names):
            parser.error("--force-archive-list contains a non-basename entry")
    moved = archive_invalid(
        args.trials_root,
        args.tasks_root,
        args.manifest,
        args.shard_index,
        args.archive_root,
        archive_incomplete=args.archive_incomplete,
        force_archive_names=force_archive_names,
    )
    for name, reason in moved:
        print(f"ARCHIVED shard={args.shard_index} trial={name} reason={reason}")
    print(f"SUSHI_SHARD_ARCHIVE_DONE shard={args.shard_index} moved={len(moved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
