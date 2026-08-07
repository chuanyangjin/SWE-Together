from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eval_infra_sentinel import (  # noqa: E402
    SIDECAR_VERSION,
    TrialSignals,
    _detect_no_agent_progress,
    classify_trial,
)


class InfraSentinelTests(unittest.TestCase):
    def test_user_sim_error_invalidates_even_a_real_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            agent = trial / "agent"
            episode = agent / "episode-1"
            episode.mkdir(parents=True)
            (agent / "final.patch").write_text(
                "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+b\n"
            )
            (episode / "user_decision.json").write_text(
                json.dumps(
                    {
                        "action": "no-op",
                        "raw_response": "error: transport unavailable",
                    }
                )
            )
            verdict = classify_trial(trial)
            self.assertEqual(verdict.status, "infra_failed")
            self.assertEqual(verdict.reason, "user_sim_error")

    def test_real_read_only_agent_activity_is_a_scored_failure(self) -> None:
        matched, _detail, _evidence = _detect_no_agent_progress(
            TrialSignals(
                assistant_turn_count=8,
                assistant_texts=["I investigated but could not find a safe change."],
                all_tool_calls=5,
                edit_tool_calls=0,
                patch_has_changes=False,
            )
        )
        self.assertFalse(matched)

        matched, _detail, _evidence = _detect_no_agent_progress(
            TrialSignals(assistant_turn_count=8)
        )
        self.assertTrue(matched)

    def test_setup_failure_is_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            (trial / "agent").mkdir()
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "exception_info": {
                            "exception_type": "RuntimeError",
                            "exception_message": "Agent setup failed with exit code 127",
                        }
                    }
                )
            )
            verdict = classify_trial(trial)
            self.assertEqual(verdict.status, "infra_failed")
            self.assertEqual(verdict.reason, "agent_setup_failure")
            self.assertEqual(verdict.version, SIDECAR_VERSION)

    def test_opencode_turn_file_auth_error_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            agent = trial / "agent"
            agent.mkdir()
            (agent / "final.patch").write_text(
                "=== /workspace/repo (cumulative vs harbor-base) ===\n"
            )
            event = {
                "type": "error",
                "error": {
                    "name": "APIError",
                    "data": {"message": "invalid x-api-key", "statusCode": 401},
                },
            }
            (agent / "opencode.txt.turn-0").write_text(json.dumps(event) + "\n")
            verdict = classify_trial(trial)
            self.assertEqual(verdict.status, "infra_failed")
            self.assertEqual(verdict.reason, "opencode_backend_error")

    def test_tiny_structural_patch_is_not_overridden_by_setup_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            agent = trial / "agent"
            agent.mkdir()
            (agent / "final.patch").write_text(
                "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+b\n"
            )
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "exception_info": {
                            "exception_type": "RuntimeError",
                            "exception_message": "Agent setup failed after useful work",
                        }
                    }
                )
            )
            self.assertEqual(classify_trial(trial).status, "ok")


if __name__ == "__main__":
    unittest.main()
