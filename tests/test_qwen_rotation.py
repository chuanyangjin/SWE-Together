from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import rotate_qwen35_bearer as rotation


class QwenRotationTests(unittest.TestCase):
    def test_secure_read_requires_private_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            key = root / "key"
            key.write_bytes(b"private-value")
            key.chmod(0o644)
            with self.assertRaises(RuntimeError):
                rotation._secure_read(key, 64)
            key.chmod(0o600)
            self.assertEqual(rotation._secure_read(key, 64), b"private-value")
            link = root / "link"
            link.symlink_to(key)
            with self.assertRaises(OSError):
                rotation._secure_read(link, 64)

    def test_atomic_private_bytes_overwrites_with_mode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "credential"
            target.write_bytes(b"old")
            target.chmod(0o644)
            rotation._atomic_private_bytes(target, b"new")
            self.assertEqual(target.read_bytes(), b"new")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)

    def test_rotation_wait_requires_repeated_new_accept_old_reject(self) -> None:
        statuses = [200, 401, 200, 401, 200, 401]
        with (
            patch.object(rotation, "_models_status", side_effect=statuses),
            patch.object(rotation.time, "sleep", return_value=None),
        ):
            old_status, new_status = rotation._wait_for_rotation(
                "http://qwen.invalid", "old-private-value", "new-private-value"
            )
        self.assertEqual(old_status, 401)
        self.assertEqual(new_status, 200)

    def test_rotation_accepts_litellm_invalid_bearer_400(self) -> None:
        statuses = [200, 400, 200, 400, 200, 400]
        with (
            patch.object(rotation, "_models_status", side_effect=statuses),
            patch.object(rotation.time, "sleep", return_value=None),
        ):
            old_status, new_status = rotation._wait_for_rotation(
                "http://qwen.invalid", "old-private-value", "new-private-value"
            )
        self.assertEqual(old_status, 400)
        self.assertEqual(new_status, 200)

    def test_exclusive_lock_rejects_an_active_holder(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "rotation.lock"
            with rotation._exclusive_lock(path):
                with self.assertRaises(RuntimeError):
                    with rotation._exclusive_lock(path):
                        self.fail("nested nonblocking lock unexpectedly succeeded")
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_persistent_source_rejects_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "proxy.sbatch"
            config = root / "litellm_config.sh"
            batch.write_text("read_proxy_master_key\n")
            config.write_text("read_proxy_master_key\n")
            with (
                patch.object(rotation, "PROXY_BATCH_SOURCE", batch),
                patch.object(rotation, "CONFIG_GENERATOR_SOURCE", config),
            ):
                rotation._validate_persistent_key_source()
                config.write_text("read_proxy_master_key\nsk-model-proxy-key\n")
                with self.assertRaises(RuntimeError):
                    rotation._validate_persistent_key_source()


if __name__ == "__main__":
    unittest.main()
