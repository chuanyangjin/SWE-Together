from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external" / "harbor" / "src"))

import runner  # noqa: E402
import sandoq_env  # noqa: E402
import sandoq_probe  # noqa: E402
from eval.correctness import sandoq_judge  # noqa: E402
from eval.correctness.sandbox import JudgeInputs  # noqa: E402


class SandoqRegressionTests(unittest.TestCase):
    def test_control_plane_probe_does_not_read_or_send_token(self) -> None:
        lease = {
            "sessionId": "sid",
            "portUrls": {"exec": "https://session.example/"},
        }
        sandoq_probe._ACTIVE_SESSION_ID = None
        with patch.object(sandoq_probe, "_read_token") as read_token, patch.object(
            sandoq_probe, "install_cleanup_handlers"
        ), patch.object(
            sandoq_probe, "create_session", return_value=lease
        ), patch.object(
            sandoq_probe, "wait_health"
        ), patch.object(
            sandoq_probe, "_request", return_value=(401, {})
        ) as request, patch.object(
            sandoq_probe, "_delete_outer_sync"
        ) as delete, patch.object(
            sandoq_probe, "outer_exec"
        ) as authenticated_exec, patch(
            "builtins.print"
        ):
            rc = sandoq_probe.main(["--control-plane-only"])

        self.assertEqual(rc, 0)
        read_token.assert_not_called()
        authenticated_exec.assert_not_called()
        request.assert_called_once()
        self.assertEqual(request.call_args.args[1], "https://session.example/v1/exec")
        delete.assert_called_once()
        self.assertIsNone(sandoq_probe._ACTIVE_SESSION_ID)

    def test_session_registry_lock_is_reentrant_for_signal_cleanup(self) -> None:
        lock = sandoq_env._sessions_lock
        with lock:
            acquired_again = lock.acquire(blocking=False)
            self.assertTrue(
                acquired_again,
                "SIGTERM cleanup must be able to re-enter the session registry lock",
            )
            if acquired_again:
                lock.release()

    def test_legacy_runner_uses_sandoq_import_path_and_build_multiplier(self) -> None:
        result = SimpleNamespace(verifier_result=None, exception_info=None)
        trial = SimpleNamespace(run=AsyncMock(return_value=result))
        args = SimpleNamespace(
            model="anthropic/test-model",
            user_model=None,
            agent_type="opencode",
            user_context_chars=3000,
            call_user_on_completion=True,
            setup_timeout=None,
            agent_timeout=4800,
            env_type="sandoq",
            keep=False,
            trials_dir="trials/unit-sandoq-runner",
            env_path="unused",
        )

        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            task_dir.mkdir()
            missing_tar = Path(tmp) / "missing.tar"
            with patch.dict(
                os.environ, {"SANDOQ_BUILD_TIMEOUT_MULTIPLIER": "4.5"}
            ), patch.object(
                runner,
                "resolve_model",
                return_value=("anthropic/test-model", "test-key", "ANTHROPIC_API_KEY"),
            ), patch.object(
                runner, "resolve_task_dir", return_value=task_dir
            ), patch.object(
                runner, "load_analysis", return_value={}
            ), patch.object(
                runner, "load_user_messages", return_value=[]
            ), patch.object(
                runner, "Trial", return_value=trial
            ) as trial_class, patch.object(
                runner, "_extract_gt_session_duration", return_value=None
            ), patch.object(
                runner, "_write_timing"
            ), patch.object(
                runner, "_copy_sim_prompts_to_trials"
            ), patch.object(
                runner, "_build_trajectories"
            ), patch.object(
                runner, "_auto_upload_traces"
            ), patch(
                "builtins.print"
            ):
                asyncio.run(runner.run_single_task("task", missing_tar, args))

        config = trial_class.call_args.kwargs["config"]
        self.assertEqual(
            config.environment.import_path, "sandoq_env:SandoqEnvironment"
        )
        self.assertIsNone(config.environment.type)
        self.assertEqual(config.environment_build_timeout_multiplier, 4.5)
        trial.run.assert_awaited_once_with()

    def test_sandoq_judge_success_path_starts_runs_and_stops_environment(self) -> None:
        fake_environment = SimpleNamespace(
            _sandbox_id="remote-session",
            start=AsyncMock(),
            stop=AsyncMock(),
        )
        environment_class = Mock(return_value=fake_environment)
        apply_patch = AsyncMock(return_value="/workspace")
        drop_inputs = AsyncMock()
        run_loop = AsyncMock(
            return_value=({"judge_score": 1.0, "verdict": "pass"}, "transcript", 0)
        )
        inputs = JudgeInputs(
            readme="task",
            user_sim_prompt="simulate",
            oracle_patch="",
            agent_patch="diff --git a/a b/a\n",
            test_sh="#!/bin/sh\ntrue\n",
            system_prompt="judge",
            tests_files={},
            phase=2,
            canonical_goals_json='{"goals": []}',
        )

        with patch.object(
            sandoq_judge, "SandoqEnvironment", environment_class
        ), patch.object(
            sandoq_judge, "_judge_model", return_value="anthropic/claude-opus-4-6"
        ), patch.object(
            sandoq_judge, "_apply_patch", apply_patch
        ), patch.object(
            sandoq_judge, "_drop_inputs", drop_inputs
        ), patch.object(
            sandoq_judge, "_first_message", return_value="first message"
        ) as first_message, patch.object(
            sandoq_judge, "_run_loop", run_loop
        ):
            result = asyncio.run(
                sandoq_judge.run_judge_in_sandoq(
                    "agent-swarm-task-4a881b",
                    "trial-id",
                    inputs,
                    timeout_sec=123,
                    max_turns=4,
                )
            )

        fake_environment.start.assert_awaited_once_with(force_build=False)
        apply_patch.assert_awaited_once()
        self.assertIs(apply_patch.await_args.args[0], fake_environment)
        drop_inputs.assert_awaited_once()
        self.assertIs(drop_inputs.await_args.args[0], fake_environment)
        first_message.assert_called_once_with(inputs, "/workspace")
        run_loop.assert_awaited_once_with(
            fake_environment,
            "judge",
            "first message",
            f"{sandoq_judge.INPUTS_DIR}/verdict.json",
            123,
            4,
        )
        fake_environment.stop.assert_awaited_once_with(delete=True)
        self.assertEqual(result.sandbox_id, "remote-session")
        self.assertEqual(result.judge_model, "anthropic/claude-opus-4-6")
        self.assertEqual(result.verdict["judge_score"], 1.0)
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
