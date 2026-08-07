from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts import archive_invalid_trials


class ArchiveInvalidTrialsTests(unittest.TestCase):
    @staticmethod
    def _valid_trial(root: Path, name: str) -> None:
        trial = root / name
        (trial / "agent").mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
        )
        (trial / "agent" / "final.patch").write_text(
            "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+b\n"
        )

    def test_archives_incomplete_and_accepts_exact_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks"
            active = root / "active"
            archive = root / "archive"
            for task in ("task-a", "task-b"):
                (tasks / task).mkdir(parents=True)
            active.mkdir()
            self._valid_trial(active, "task-a__aaaaaaa")
            self._valid_trial(active, "task-a__bbbbbbb")
            self._valid_trial(active, "task-b__ccccccc")
            self._valid_trial(active, "task-b__ddddddd")
            incomplete = active / "task-a__eeeeeee"
            incomplete.mkdir()
            (incomplete / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": None}})
            )
            invalid = active / "task-b__fffffff"
            invalid.mkdir()
            (invalid / "result.json").write_text(
                json.dumps(
                    {"verifier_result": {"rewards": {"reward": "NaN"}}}
                )
            )

            rc = archive_invalid_trials.main(
                [
                    "--trials-root",
                    str(active),
                    "--archive-root",
                    str(archive),
                    "--tasks-root",
                    str(tasks),
                    "--replicates",
                    "2",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertFalse(incomplete.exists())
            self.assertTrue((archive / incomplete.name).is_dir())
            self.assertFalse(invalid.exists())
            self.assertTrue((archive / invalid.name).is_dir())
            self.assertEqual(len(list(active.glob("*__*"))), 4)

    def test_archive_collision_preserves_late_recreated_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks"
            active = root / "active"
            archive = root / "archive"
            (tasks / "task-a").mkdir(parents=True)
            active.mkdir()
            archive.mkdir()
            self._valid_trial(active, "task-a__aaaaaaa")
            self._valid_trial(active, "task-a__bbbbbbb")
            (archive / "task-a__latewrite").mkdir()
            late = active / "task-a__latewrite"
            late.mkdir()

            rc = archive_invalid_trials.main(
                [
                    "--trials-root",
                    str(active),
                    "--archive-root",
                    str(archive),
                    "--tasks-root",
                    str(tasks),
                    "--replicates",
                    "2",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertFalse(late.exists())
            self.assertTrue((archive / "task-a__latewrite").is_dir())
            self.assertTrue(
                (archive / "task-a__latewrite.duplicate-1").is_dir()
            )

    def test_direct_cli_waits_for_archive_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks"
            active = root / "active"
            archive = root / "archive"
            (tasks / "task-a").mkdir(parents=True)
            active.mkdir()
            archive.mkdir()
            self._valid_trial(active, "task-a__aaaaaaa")

            with (
                active / archive_invalid_trials._LOCK_FILENAME
            ).open("a+") as held_lock:
                fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX)
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(archive_invalid_trials.__file__).resolve()),
                        "--trials-root",
                        str(active),
                        "--archive-root",
                        str(archive),
                        "--tasks-root",
                        str(tasks),
                        "--replicates",
                        "1",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                time.sleep(0.2)
                self.assertIsNone(process.poll())
                fcntl.flock(held_lock.fileno(), fcntl.LOCK_UN)

            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stdout + stderr)


if __name__ == "__main__":
    unittest.main()
