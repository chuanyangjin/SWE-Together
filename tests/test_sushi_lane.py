from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sushi_lane"))

from register_relay_client import update_registry  # noqa: E402
from archive_invalid_shard import archive_invalid  # noqa: E402
from secure_tcp_relay import _validate_registry  # noqa: E402
from shard_tasks import build_manifest, validate_manifest  # noqa: E402
from scripts.build_missing_judge_plan import build_plan  # noqa: E402


def test_four_shards_are_exact_disjoint_canonical_union() -> None:
    manifest = build_manifest(REPO / "tasks", 4)
    shards = validate_manifest(manifest, REPO / "tasks", expected_shards=4)
    flattened = [task for shard in shards for task in shard]

    assert [len(shard) for shard in shards] == [28, 27, 27, 27]
    assert len(flattened) == len(set(flattened)) == 109
    assert manifest["canonical_tasks_sha256"] == (
        "92260b84f512d13b8e18d67deafb50570066f126f841184210468e4682e73d61"
    )


def test_shard_manifest_rejects_duplicate_ownership() -> None:
    manifest = build_manifest(REPO / "tasks", 4)
    tampered = copy.deepcopy(manifest)
    tampered["shards"][1]["tasks"][0] = tampered["shards"][0]["tasks"][0]

    with pytest.raises(RuntimeError, match="duplicate|union"):
        validate_manifest(tampered, REPO / "tasks", expected_shards=4)


def test_relay_registry_updates_atomically_and_owner_only(tmp_path: Path) -> None:
    registry = tmp_path / "clients.json"
    update_registry(registry, "job-a", "192.0.2.1")
    update_registry(registry, "job-b", "192.0.2.2")

    assert registry.stat().st_mode & 0o777 == 0o600
    assert _validate_registry(registry) == {
        "job-a": "192.0.2.1",
        "job-b": "192.0.2.2",
    }

    update_registry(registry, "job-a", None)
    assert json.loads(registry.read_text())["clients"] == {"job-b": "192.0.2.2"}


def test_relay_registry_rejects_permissive_mode_and_symlink(tmp_path: Path) -> None:
    registry = tmp_path / "clients.json"
    update_registry(registry, "job-a", "192.0.2.1")
    registry.chmod(0o640)
    with pytest.raises(RuntimeError, match="group/world"):
        _validate_registry(registry)

    registry.chmod(0o600)
    link = tmp_path / "linked.json"
    link.symlink_to(registry)
    with pytest.raises(OSError):
        _validate_registry(link)
    assert os.path.islink(link)


def test_relay_registry_keeps_colocated_source_until_last_label_removed(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "clients.json"
    shared_ip = "192.0.2.44"
    update_registry(registry, "job-a", shared_ip)
    update_registry(registry, "job-b", shared_ip)

    update_registry(registry, "job-a", None)
    clients = _validate_registry(registry)
    assert clients == {"job-b": shared_ip}
    assert shared_ip in set(clients.values())

    update_registry(registry, "job-b", None)
    assert _validate_registry(registry) == {}


def test_completed_only_judge_plan_excludes_mutable_active_trial(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks"
    task = tasks / "example-task"
    task.mkdir(parents=True)
    (task / "canonical_goals.json").write_text("{}\n")
    trials = tmp_path / "trials"
    trial = trials / "example-task__abc123"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "final.patch").write_text(
        "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -0,0 +1 @@\n+x\n"
    )

    assert build_plan(
        trials,
        tasks,
        "judge_verdict_opus46.json",
        "anthropic/claude-opus-4-6",
        completed_only=True,
    ) == []

    (trial / "result.json").write_text("{}\n")
    plan = build_plan(
        trials,
        tasks,
        "judge_verdict_opus46.json",
        "anthropic/claude-opus-4-6",
        completed_only=True,
    )
    assert [Path(row["trial_dir"]).name for row in plan] == [trial.name]


def test_valid_completed_judge_plan_excludes_reward_and_infra_failures(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks"
    task = tasks / "example-task"
    task.mkdir(parents=True)
    (task / "canonical_goals.json").write_text("{}\n")
    trials = tmp_path / "trials"

    def write_trial(name: str, reward: float | None, *, sim_error: bool = False) -> Path:
        trial = trials / name
        (trial / "agent").mkdir(parents=True)
        (trial / "agent" / "final.patch").write_text(
            "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -0,0 +1 @@\n+x\n"
        )
        (trial / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": reward}}})
        )
        if sim_error:
            episode = trial / "agent" / "episode-1"
            episode.mkdir()
            (episode / "user_decision.json").write_text(
                json.dumps({"raw_response": "error: "})
            )
        return trial

    valid = write_trial("example-task__valid", 1.0)
    write_trial("example-task__badreward", None)
    write_trial("example-task__badinfra", 1.0, sim_error=True)

    plan = build_plan(
        trials,
        tasks,
        "judge_verdict_opus46.json",
        "anthropic/claude-opus-4-6",
        completed_only=True,
        valid_completed_only=True,
    )
    assert [Path(row["trial_dir"]).name for row in plan] == [valid.name]


def test_shard_archive_moves_only_owned_completed_invalid_trials(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = build_manifest(REPO / "tasks", 4)
    manifest_path.write_text(json.dumps(manifest))
    owned_task = manifest["shards"][0]["tasks"][0]
    foreign_task = manifest["shards"][1]["tasks"][0]
    trials = tmp_path / "trials"
    archive = tmp_path / "archive"

    def write_trial(name: str, task: str, *, completed: bool) -> Path:
        trial = trials / name
        trial.mkdir(parents=True)
        (trial / "config.json").write_text(
            json.dumps({"task": {"path": str(REPO / "tasks" / task)}})
        )
        if completed:
            (trial / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": None}}})
            )
        return trial

    owned_invalid = write_trial("owned__invalid", owned_task, completed=True)
    owned_active = write_trial("owned__active", owned_task, completed=False)
    foreign_invalid = write_trial("foreign__invalid", foreign_task, completed=True)

    moved = archive_invalid(
        trials,
        REPO / "tasks",
        manifest_path,
        shard_index=0,
        archive_root=archive,
    )

    assert moved == [(owned_invalid.name, "invalid_reward")]
    assert not owned_invalid.exists()
    assert (archive / owned_invalid.name / "result.json").is_file()
    assert owned_active.is_dir()
    assert foreign_invalid.is_dir()

    moved = archive_invalid(
        trials,
        REPO / "tasks",
        manifest_path,
        shard_index=0,
        archive_root=archive,
        force_archive_names={owned_active.name, foreign_invalid.name},
    )
    assert moved == [(owned_active.name, "sanitizer_race_exposure")]
    assert not owned_active.exists()
    assert (archive / owned_active.name / "config.json").is_file()
    assert foreign_invalid.is_dir()
