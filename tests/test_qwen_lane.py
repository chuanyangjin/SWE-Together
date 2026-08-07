from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import qwen35_4b_lane as lane  # noqa: E402


def _make_trial(root: Path, task: str, suffix: str, timeout: float) -> Path:
    trial = root / f"{task}__{suffix}"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    (agent / "final.patch").write_text(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    (agent / "opencode.txt").write_text(
        json.dumps(
            {
                "type": "step_finish",
                "part": {
                    "id": f"finish-{suffix}",
                    "tokens": {"output": 2, "reasoning": 3},
                },
            }
        )
        + "\n"
    )
    config = {
        "agent": {
            "model_name": lane.EXPECTED_ACTION_MODEL,
            "override_timeout_sec": timeout,
            "kwargs": {
                "user_model_name": lane.EXPECTED_USER_MODEL,
                "user_temperature": lane.EXPECTED_USER_TEMPERATURE,
            },
        },
        "environment": {"import_path": lane.EXPECTED_ENVIRONMENT_IMPORT},
    }
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "config": config,
                "verifier_result": {"rewards": {"reward": 0.0}},
                "agent_execution": {
                    "started_at": "2026-08-04T00:00:00Z",
                    "finished_at": "2026-08-04T00:00:01Z",
                },
                "exception_info": None,
            }
        )
    )
    return trial


class QwenLaneTests(unittest.TestCase):
    def test_protocol_match_rejects_missing_or_malformed_numeric_fields(self) -> None:
        base = (
            lane.EXPECTED_ACTION_MODEL,
            lane.EXPECTED_USER_MODEL,
            lane.EXPECTED_USER_TEMPERATURE,
            lane.EXPECTED_AGENT_TIMEOUT,
            lane.EXPECTED_ENVIRONMENT_IMPORT,
        )
        self.assertTrue(lane._protocol_matches(base))
        self.assertFalse(lane._protocol_matches((*base[:2], None, *base[3:])))
        self.assertFalse(lane._protocol_matches((*base[:3], "bad", base[4])))

    def test_producer_lock_is_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lock = Path(raw) / "producer.lock"
            with lane._exclusive_producer_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "lock is held"):
                    with lane._exclusive_producer_lock(lock):
                        pass

    def test_prepare_resume_writes_transaction_manifest_before_final_state(self) -> None:
        tasks = lane._expected_tasks()
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            root = base / "trials"
            root.mkdir()
            valid = _make_trial(
                root, tasks[0], "valid", lane.EXPECTED_AGENT_TIMEOUT
            )
            invalid = _make_trial(root, tasks[1], "old", 1800.0)

            report = lane.prepare_resume(root, base / "quarantine", k=1)

            self.assertTrue(valid.exists())
            self.assertFalse(invalid.exists())
            self.assertEqual(report["state"], "complete")
            self.assertEqual(report["quarantined"], 1)
            self.assertEqual(report["moves"][0]["state"], "moved")
            manifest = json.loads(
                Path(report["quarantine_root"], "quarantine_manifest.json").read_text()
            )
            self.assertEqual(manifest["state"], "complete")
            self.assertTrue(Path(manifest["moves"][0]["destination"]).exists())


if __name__ == "__main__":
    unittest.main()
