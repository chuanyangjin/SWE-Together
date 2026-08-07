from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from eval.correctness import podman_judge, run_batch
from scripts import build_missing_judge_plan


def _rubric() -> dict:
    return {
        "completeness_goals": [
            {
                "id": "goal_0",
                "goal": "Implement the behavior",
                "tier": "core",
                "weight": 1.0,
                "rationale": "Required",
            }
        ]
    }


def _verdict(
    *,
    task: str = "task",
    trial: str = "task__trial",
    model: str = "claude-opus-4-6",
    met: object = True,
) -> dict:
    return {
        "judge_score": 1.0 if met is True else 0.0,
        "verdict": "equivalent" if met is True else "incorrect",
        "rubric_source": "canonical_goals.json",
        "goal_results": [
            {"id": "goal_0", "met": met, "evidence": "source.py:1"}
        ],
        "task": task,
        "trial_id": trial,
        "judge_model": model,
        "judge_phase": 2,
        "rubric_n_goals": 1,
    }


class _Gate:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __aenter__(self):
        self.entered.set()
        await self.release.wait()
        return self

    async def __aexit__(self, *_args) -> None:
        return None


class JudgeResumeTests(unittest.TestCase):
    def test_patch_cleaner_drops_unapplyable_binary_placeholder_only(self) -> None:
        raw = (
            "=== /workspace/repo (cumulative vs harbor-base) ===\n"
            "diff --git a/cache.whl b/cache.whl\n"
            "new file mode 100644\n"
            "Binary files /dev/null and b/cache.whl differ\n"
            "diff --git a/source.py b/source.py\n"
            "--- a/source.py\n"
            "+++ b/source.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )

        cleaned = podman_judge._clean_agent_patch(raw)

        self.assertNotIn("cache.whl", cleaned)
        self.assertIn("diff --git a/source.py b/source.py", cleaned)
        self.assertIn("+new", cleaned)

    def test_host_judge_warns_then_marks_max_turn_exhaustion(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))]
        )
        environment = SimpleNamespace(
            exec=AsyncMock(
                return_value=SimpleNamespace(return_code=1, stdout="", stderr="")
            )
        )
        with patch.object(
            podman_judge, "_completion_with_retry", AsyncMock(return_value=response)
        ), patch.object(
            podman_judge, "_output_exists", AsyncMock(return_value=False)
        ):
            verdict, transcript, exit_code = asyncio.run(
                podman_judge._run_loop(
                    environment,
                    "system",
                    "first",
                    "/tmp/verdict.json",
                    timeout_sec=60,
                    max_turns=2,
                )
            )

        self.assertEqual(verdict["error"], "verdict_read_failed")
        self.assertEqual(exit_code, 2)
        self.assertIn("five-turn finalization warning", transcript)
        self.assertIn("max turns exhausted (2)", transcript)

    def test_goal_id_repair_is_semantic_retry_not_positional_remap(self) -> None:
        rubric = {
            "completeness_goals": [
                {
                    "id": "goal_0",
                    "goal": "First behavior",
                    "tier": "core",
                    "weight": 0.5,
                },
                {
                    "id": "goal_1",
                    "goal": "Second behavior",
                    "tier": "core",
                    "weight": 0.5,
                },
            ]
        }
        inputs = SimpleNamespace(
            phase=2,
            canonical_goals_json=json.dumps(rubric),
            system_prompt="judge",
        )
        rejected = {
            "goal_results": [
                {"id": "goal_1", "met": True, "evidence": "first"},
                {"id": "goal_2", "met": False, "evidence": "second"},
            ]
        }
        repaired = {
            "goal_results": [
                {"id": "goal_0", "met": True, "evidence": "first"},
                {"id": "goal_1", "met": False, "evidence": "second"},
            ]
        }
        environment = SimpleNamespace(
            exec=AsyncMock(return_value=SimpleNamespace(return_code=0))
        )
        run_loop = AsyncMock(
            side_effect=[
                (rejected, "initial", 0),
                (repaired, "repair", 0),
            ]
        )

        verdict, transcript, exit_code = asyncio.run(
            podman_judge._run_loop_with_goal_id_repair(
                environment,
                inputs,
                "first",
                "/tmp/judge_inputs/verdict.json",
                1200,
                40,
                run_loop=run_loop,
            )
        )

        self.assertEqual(verdict, repaired)
        self.assertEqual(exit_code, 0)
        self.assertIn("exact-ID retry", transcript)
        self.assertEqual(run_loop.await_count, 2)
        repair_prompt = run_loop.await_args_list[1].args[2]
        self.assertIn('["goal_0", "goal_1"]', repair_prompt)
        self.assertIn("Do not blindly relabel rows by array position", repair_prompt)
        environment.exec.assert_awaited_once()

    def test_goal_id_repair_does_not_retry_valid_or_non_list_rows(self) -> None:
        inputs = SimpleNamespace(
            phase=2,
            canonical_goals_json=json.dumps(_rubric()),
            system_prompt="judge",
        )
        for verdict in (_verdict(), {"goal_results": "not-a-list"}):
            with self.subTest(verdict=verdict):
                environment = SimpleNamespace(exec=AsyncMock())
                run_loop = AsyncMock(return_value=(verdict, "initial", 0))
                observed, _, _ = asyncio.run(
                    podman_judge._run_loop_with_goal_id_repair(
                        environment,
                        inputs,
                        "first",
                        "/tmp/judge_inputs/verdict.json",
                        1200,
                        40,
                        run_loop=run_loop,
                    )
                )
                self.assertEqual(observed, verdict)
                run_loop.assert_awaited_once()
                environment.exec.assert_not_awaited()

    def test_error_or_missing_score_is_not_a_reusable_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict = Path(tmp) / "judge_verdict.json"
            verdict.write_text(json.dumps({"error": "verdict_read_failed"}))
            self.assertFalse(run_batch._valid_phase2_verdict(verdict, _rubric()))
            verdict.write_text(json.dumps({"judge_model": "claude-opus-4-6"}))
            self.assertFalse(run_batch._valid_phase2_verdict(verdict, _rubric()))
            verdict.write_text(json.dumps({"judge_score": 0.0}))
            self.assertFalse(run_batch._valid_phase2_verdict(verdict, _rubric()))
            verdict.write_text(json.dumps(_verdict()))
            self.assertTrue(
                run_batch._valid_phase2_verdict(
                    verdict,
                    _rubric(),
                    task_name="task",
                    trial_id="task__trial",
                )
            )
            verdict.write_text(json.dumps({"judge_score": 2.0}))
            self.assertFalse(run_batch._valid_phase2_verdict(verdict, _rubric()))

    def test_repair_plan_validates_score_and_expected_model(self) -> None:
        self.assertFalse(
            build_missing_judge_plan._reusable_verdict(
                {"judge_score": 2.0, "judge_model": "claude-opus-4-6"},
                "anthropic/claude-opus-4-6",
                _rubric(),
                task_name="task",
                trial_id="task__trial",
            )
        )
        self.assertFalse(
            build_missing_judge_plan._reusable_verdict(
                _verdict(model="claude-opus-4-8"),
                "anthropic/claude-opus-4-6",
                _rubric(),
                task_name="task",
                trial_id="task__trial",
            )
        )
        self.assertTrue(
            build_missing_judge_plan._reusable_verdict(
                _verdict(),
                "anthropic/claude-opus-4-6",
                _rubric(),
                task_name="task",
                trial_id="task__trial",
            )
        )

    def test_phase2_schema_rejects_non_boolean_met_and_invalid_rubric(self) -> None:
        bad_verdict = _verdict(met="false")
        issues = run_batch._phase2_verdict_issues(_rubric(), bad_verdict)
        self.assertIn("goal_result_0_met_not_bool", issues)
        self.assertIn(
            "rubric_duplicate_goal_ids",
            run_batch._rubric_issues(
                {
                    "completeness_goals": [
                        *_rubric()["completeness_goals"],
                        _rubric()["completeness_goals"][0],
                    ]
                }
            ),
        )

    def test_gameable_override_survives_mechanical_derivation(self) -> None:
        raw = _verdict()
        raw["judge_score"] = 0.0
        raw["verdict"] = "gameable"
        normalized, issues = run_batch._normalize_phase2_verdict(_rubric(), raw)
        self.assertEqual(issues, [])
        self.assertEqual(normalized["judge_score"], 0.0)
        self.assertEqual(normalized["verdict"], "gameable")
        normalized.update(
            {
                "task": "task",
                "trial_id": "task__trial",
                "judge_model": "claude-opus-4-6",
                "judge_phase": 2,
                "rubric_n_goals": 1,
            }
        )
        self.assertEqual(
            run_batch._phase2_verdict_issues(
                _rubric(),
                normalized,
                task_name="task",
                trial_id="task__trial",
            ),
            [],
        )

    def test_queued_job_rechecks_verdict_after_worker_slot(self) -> None:
        async def scenario() -> tuple[dict, Mock]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                trial = root / "task__trial"
                task = root / "task"
                (trial / "agent").mkdir(parents=True)
                (task / "tests").mkdir(parents=True)
                (task / "canonical_goals.json").write_text(
                    json.dumps(_rubric())
                )
                (trial / "agent" / "final.patch").write_text(
                    "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+b\n"
                )
                gate = _Gate()
                judge = Mock()
                job = {
                    "trial_dir": str(trial),
                    "task_dir": str(task),
                    "out_name": "judge_verdict.json",
                }
                with patch.object(run_batch, "_judge_runner", return_value=judge):
                    pending = asyncio.create_task(
                        run_batch._phase2_one(job, "", gate, None, False)
                    )
                    await gate.entered.wait()
                    (trial / "judge_verdict.json").write_text(
                        json.dumps(_verdict())
                    )
                    gate.release.set()
                    return await pending, judge

        result, judge = asyncio.run(scenario())
        self.assertEqual(result["status"], "skipped_existing")
        judge.assert_not_called()

    def test_verdict_json_write_is_atomic_and_failure_preserves_old_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge_verdict.json"
            old = {"version": "old"}
            path.write_text(json.dumps(old))

            with patch.object(run_batch.os, "fsync", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    run_batch._atomic_write_json(path, {"version": "new"})

            self.assertEqual(json.loads(path.read_text()), old)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*")), [])

            new = {"version": "new", "value": 1}
            run_batch._atomic_write_json(path, new)
            self.assertEqual(json.loads(path.read_text()), new)


if __name__ == "__main__":
    unittest.main()
