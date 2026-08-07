from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from eval.correctness import podman_judge, run_batch, sandoq_judge
from eval.correctness.sandbox import JudgeInputs


TASK_NAME = "agent-swarm-task-4a881b"
MODEL = "anthropic/claude-opus-4-6"


def _rubric() -> dict:
    return {
        "completeness_goals": [
            {
                "id": "goal_core",
                "goal": "Implement the requested behavior",
                "tier": "core",
                "weight": 0.7,
            },
            {
                "id": "goal_secondary",
                "goal": "Add regression coverage",
                "tier": "secondary",
                "weight": 0.3,
            },
        ]
    }


def _inputs(*, phase: int = 2) -> JudgeInputs:
    return JudgeInputs(
        readme="task",
        user_sim_prompt="simulate",
        oracle_patch="diff --git a/a b/a\n" if phase == 1 else "",
        agent_patch="diff --git a/a b/a\n",
        test_sh="#!/bin/sh\ntrue\n",
        system_prompt="judge",
        tests_files={},
        phase=phase,
        canonical_goals_json=json.dumps(_rubric()),
    )


def _rejection() -> podman_judge.DeterministicPatchApplyError:
    return podman_judge.DeterministicPatchApplyError(
        return_code=1,
        repo_path="/workspace",
        stdout="applying to /workspace\n",
        stderr="error: patch failed: source.py:1",
    )


class PatchFailureClassificationTests(unittest.TestCase):
    def test_apply_helper_requires_post_attempt_sentinel(self) -> None:
        async def run(stderr: str, return_code: int = 1):
            environment = SimpleNamespace(
                upload_file=AsyncMock(),
                exec=AsyncMock(
                    return_value=SimpleNamespace(
                        return_code=return_code,
                        stdout="applying to /workspace\n",
                        stderr=stderr,
                    )
                ),
            )
            with tempfile.TemporaryDirectory() as tmp:
                return await podman_judge._apply_patch(
                    environment, _inputs(), Path(tmp)
                )

        sentinel = podman_judge._PATCH_REJECTED_SENTINEL + "1\n"
        with self.assertRaises(podman_judge.DeterministicPatchApplyError):
            asyncio.run(run(sentinel + "error: patch does not apply"))

        # A repo-discovery/container/exec error may share rc=1, but without the
        # private post-three-attempt marker it must remain infrastructure.
        with self.assertRaisesRegex(RuntimeError, "patch apply failed"):
            asyncio.run(run("NO_GIT_REPO_FOUND"))

        # A killed git process is infrastructure even if its supervising shell
        # survives long enough to emit the marker.
        with self.assertRaisesRegex(RuntimeError, "patch apply failed"):
            asyncio.run(
                run(podman_judge._PATCH_REJECTED_SENTINEL + "137\n", 137)
            )

        # If infrastructure dies after the sentinel is written, the outer rc
        # no longer matches git's rc and must fail closed as infrastructure.
        with self.assertRaisesRegex(RuntimeError, "patch apply failed"):
            asyncio.run(run(sentinel + "error: patch does not apply", 137))

    def test_phase2_fallback_is_raw_schema_valid_and_phase1_fails_closed(self) -> None:
        verdict = podman_judge._deterministic_patch_failure_verdict(
            _inputs(), _rejection()
        )
        self.assertIsNotNone(verdict)
        assert verdict is not None
        self.assertEqual(run_batch._raw_phase2_verdict_issues(_rubric(), verdict), [])
        self.assertEqual(verdict["judge_score"], 0.0)
        self.assertEqual(verdict["verdict"], "incorrect")
        self.assertFalse(verdict["judge_invoked"])
        self.assertEqual(
            [row["id"] for row in verdict["goal_results"]],
            ["goal_core", "goal_secondary"],
        )
        self.assertTrue(all(row["met"] is False for row in verdict["goal_results"]))
        self.assertIsNone(
            podman_judge._deterministic_patch_failure_verdict(
                _inputs(phase=1), _rejection()
            )
        )


class HostJudgeBackendParityTests(unittest.TestCase):
    def _assert_valid_zero(self, result, sandbox_id: str) -> None:
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.sandbox_id, sandbox_id)
        self.assertEqual(result.judge_model, MODEL)
        self.assertEqual(result.verdict["judge_score"], 0.0)
        self.assertEqual(result.verdict["verdict"], "incorrect")
        self.assertEqual(
            run_batch._raw_phase2_verdict_issues(_rubric(), result.verdict), []
        )
        self.assertFalse(result.verdict["judge_invoked"])

    def test_podman_deterministic_rejection_is_valid_zero_with_provenance(self) -> None:
        environment = SimpleNamespace(
            _container_id="podman-container",
            start=AsyncMock(),
            stop=AsyncMock(),
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"SWE_PODMAN_STORE_BASE": tmp}
        ), patch.object(
            podman_judge, "PodmanEnvironment", Mock(return_value=environment)
        ), patch.object(
            podman_judge, "_judge_model", return_value=MODEL
        ), patch.object(
            podman_judge, "_apply_patch", AsyncMock(side_effect=_rejection())
        ), patch.object(
            podman_judge, "_run_loop", AsyncMock()
        ) as run_loop:
            result = asyncio.run(
                podman_judge.run_judge_in_podman(
                    TASK_NAME, "trial-podman", _inputs()
                )
            )

        self._assert_valid_zero(result, "podman-container")
        run_loop.assert_not_awaited()
        environment.stop.assert_awaited_once_with(delete=True)

    def test_sandoq_deterministic_rejection_is_valid_zero_with_provenance(self) -> None:
        environment = SimpleNamespace(
            _sandbox_id="sandoq-session",
            start=AsyncMock(),
            stop=AsyncMock(),
        )
        with patch.object(
            sandoq_judge, "SandoqEnvironment", Mock(return_value=environment)
        ), patch.object(
            sandoq_judge, "_judge_model", return_value=MODEL
        ), patch.object(
            sandoq_judge, "_apply_patch", AsyncMock(side_effect=_rejection())
        ), patch.object(
            sandoq_judge, "_run_loop", AsyncMock()
        ) as run_loop:
            result = asyncio.run(
                sandoq_judge.run_judge_in_sandoq(
                    TASK_NAME, "trial-sandoq", _inputs()
                )
            )

        self._assert_valid_zero(result, "sandoq-session")
        run_loop.assert_not_awaited()
        environment.stop.assert_awaited_once_with(delete=True)

    def test_start_image_or_transport_failure_remains_error_on_both_backends(self) -> None:
        for backend, module, id_field, expected_error in (
            (
                "podman",
                podman_judge,
                "_container_id",
                "podman_judge_failed",
            ),
            (
                "sandoq",
                sandoq_judge,
                "_sandbox_id",
                "sandoq_judge_failed",
            ),
        ):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                environment = SimpleNamespace(
                    **{
                        id_field: "",
                        "start": AsyncMock(
                            side_effect=RuntimeError("image pull transport failed")
                        ),
                        "stop": AsyncMock(),
                    }
                )
                env_name = (
                    "PodmanEnvironment" if backend == "podman" else "SandoqEnvironment"
                )
                env = {"SWE_PODMAN_STORE_BASE": tmp} if backend == "podman" else {}
                with patch.dict(os.environ, env), patch.object(
                    module, env_name, Mock(return_value=environment)
                ), patch.object(module, "_judge_model", return_value=MODEL):
                    runner = (
                        module.run_judge_in_podman
                        if backend == "podman"
                        else module.run_judge_in_sandoq
                    )
                    result = asyncio.run(runner(TASK_NAME, "trial-infra", _inputs()))

                self.assertEqual(result.exit_code, 1)
                self.assertEqual(result.verdict["error"], expected_error)
                self.assertNotIn("judge_score", result.verdict)
                self.assertEqual(result.judge_model, MODEL)
                environment.stop.assert_awaited_once_with(delete=True)


if __name__ == "__main__":
    unittest.main()
