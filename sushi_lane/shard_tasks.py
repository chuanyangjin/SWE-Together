#!/usr/bin/env python3
"""Build and validate deterministic, disjoint canonical SWE-Together shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_TASKS = 109
METHOD = "sorted_round_robin_index_modulo"


def canonical_tasks(tasks_root: Path) -> list[str]:
    tasks = sorted(
        path.name
        for path in tasks_root.iterdir()
        if path.is_dir()
        and (path / "task.toml").is_file()
        and (path / "instruction.md").is_file()
        and (path / "tests" / "test.sh").is_file()
    )
    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            f"canonical task count {len(tasks)} != expected {EXPECTED_TASKS}"
        )
    return tasks


def build_manifest(tasks_root: Path, shard_count: int) -> dict[str, Any]:
    if shard_count < 2:
        raise ValueError("shard count must be at least 2")
    tasks = canonical_tasks(tasks_root)
    shards = [tasks[index::shard_count] for index in range(shard_count)]
    digest = hashlib.sha256("\n".join(tasks).encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "expected_tasks": EXPECTED_TASKS,
        "canonical_tasks_sha256": digest,
        "shard_count": shard_count,
        "shards": [
            {"index": index, "task_count": len(shard), "tasks": shard}
            for index, shard in enumerate(shards)
        ],
    }


def validate_manifest(
    manifest: Any, tasks_root: Path, expected_shards: int | None = None
) -> list[list[str]]:
    if not isinstance(manifest, dict):
        raise RuntimeError("shard manifest is not an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("shard manifest schema version is invalid")
    if manifest.get("method") != METHOD:
        raise RuntimeError("shard manifest method is invalid")
    if manifest.get("expected_tasks") != EXPECTED_TASKS:
        raise RuntimeError("shard manifest expected-task count is invalid")
    shard_count = manifest.get("shard_count")
    if not isinstance(shard_count, int) or shard_count < 2:
        raise RuntimeError("shard manifest shard count is invalid")
    if expected_shards is not None and shard_count != expected_shards:
        raise RuntimeError(
            f"shard manifest count {shard_count} != expected {expected_shards}"
        )
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != shard_count:
        raise RuntimeError("shard manifest shard rows are invalid")

    shards: list[list[str]] = []
    for expected_index, row in enumerate(raw_shards):
        if not isinstance(row, dict) or row.get("index") != expected_index:
            raise RuntimeError("shard manifest indices are invalid")
        tasks = row.get("tasks")
        if (
            not isinstance(tasks, list)
            or not all(isinstance(task, str) and task for task in tasks)
            or row.get("task_count") != len(tasks)
        ):
            raise RuntimeError("shard manifest task row is invalid")
        shards.append(tasks)

    canonical = canonical_tasks(tasks_root)
    flattened = [task for shard in shards for task in shard]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("shard manifest contains duplicate task ownership")
    if sorted(flattened) != canonical:
        raise RuntimeError("shard manifest union does not equal canonical tasks")
    expected_digest = hashlib.sha256("\n".join(canonical).encode()).hexdigest()
    if manifest.get("canonical_tasks_sha256") != expected_digest:
        raise RuntimeError("shard manifest canonical-task digest is invalid")
    for index, shard in enumerate(shards):
        if shard != canonical[index::shard_count]:
            raise RuntimeError("shard manifest does not match deterministic method")
    return shards


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=Path("tasks"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--index", type=int)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    args = parser.parse_args()

    if args.write_manifest:
        if args.manifest or args.index is not None:
            parser.error("--write-manifest cannot be combined with --manifest/--index")
        manifest = build_manifest(args.tasks_root, args.shards)
        _atomic_write(args.write_manifest, manifest)
        print(
            json.dumps(
                {
                    "output": str(args.write_manifest),
                    "shards": args.shards,
                    "tasks": EXPECTED_TASKS,
                },
                sort_keys=True,
            )
        )
        return 0

    if not args.manifest or args.index is None:
        parser.error("selection requires --manifest and --index")
    manifest = json.loads(args.manifest.read_text())
    shards = validate_manifest(manifest, args.tasks_root, args.shards)
    if not 0 <= args.index < len(shards):
        parser.error(f"--index must be in [0, {len(shards) - 1}]")
    selected = shards[args.index]
    print(",".join(selected) if args.format == "csv" else json.dumps(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
