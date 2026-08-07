from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sushi_lane import validate_action_cohort as validator


def make_valid_trial(root: Path, name: str, task: str) -> Path:
    trial = root / name
    agent_dir = trial / "agent"
    agent_dir.mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps(
            {
                "task": {"path": f"/tasks/{task}"},
                "environment": {"import_path": validator.EXPECTED_ENV_IMPORT},
                "agent": {
                    "model_name": "openai/test-model",
                    "import_path": validator.EXPECTED_AGENT_IMPORT,
                    "override_timeout_sec": validator.EXPECTED_TIMEOUT_SEC,
                    "kwargs": {
                        "user_model_name": validator.EXPECTED_USER_MODEL,
                        "user_temperature": validator.EXPECTED_USER_TEMPERATURE,
                    },
                },
            }
        )
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "agent_info": {
                    "name": validator.EXPECTED_AGENT_NAME,
                    "version": validator.EXPECTED_AGENT_VERSION,
                },
                "agent_execution": {
                    "started_at": "2026-08-07T00:00:00Z",
                    "finished_at": "2026-08-07T00:01:00Z",
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )
    (agent_dir / "opencode.txt.turn-0").write_text(
        json.dumps(
            {
                "type": "step_finish",
                "part": {
                    "reason": "stop",
                    "tokens": {"total": 11, "output": 3},
                },
            }
        )
        + "\n"
    )
    (agent_dir / "final.patch").write_text(
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    return trial


def inspection_kwargs(tasks: set[str]) -> dict[str, object]:
    return {
        "expected_model": "openai/test-model",
        "expected_tasks": frozenset(tasks),
        "scope_expected_tasks": False,
    }


def test_parallel_validator_matches_serial_and_preserves_input_order(
    tmp_path: Path,
) -> None:
    trials = [
        make_valid_trial(tmp_path, f"trial-{index}", f"task-{index}")
        for index in range(8)
    ]
    ordered = list(reversed(trials))
    kwargs = inspection_kwargs({f"task-{index}" for index in range(8)})

    serial = validator.inspect_trials(ordered, workers=1, **kwargs)
    parallel = validator.inspect_trials(ordered, workers=4, **kwargs)

    assert parallel == serial
    assert [row.counted_task for row in parallel] == [
        f"task-{index}" for index in reversed(range(8))
    ]
    assert all(not row.errors for row in parallel)


def test_validator_default_is_serial_and_rejects_nonpositive_workers(
    tmp_path: Path,
) -> None:
    trial = make_valid_trial(tmp_path, "trial", "task")
    kwargs = inspection_kwargs({"task"})

    with patch.object(
        validator,
        "ThreadPoolExecutor",
        side_effect=AssertionError("serial default used a pool"),
    ):
        rows = validator.inspect_trials([trial], **kwargs)

    assert rows[0].counted_task == "task"
    assert not rows[0].errors
    with pytest.raises(ValueError, match="workers must be positive"):
        validator.inspect_trials([trial], workers=0, **kwargs)
