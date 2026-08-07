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

from harbor.environments.base import ExecResult  # noqa: E402
from harbor.environments.factory import EnvironmentFactory  # noqa: E402
from harbor.models.environment_type import EnvironmentType  # noqa: E402
from harbor.models.task.config import EnvironmentConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

import sandoq_env  # noqa: E402


class TokenContractTests(unittest.TestCase):
    def test_token_requires_regular_mode_0600_and_supports_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token = Path(tmp) / "token"
            token.write_text("first\n")
            token.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "mode-0600"):
                sandoq_env._read_token(token)
            token.chmod(0o600)
            self.assertEqual(sandoq_env._read_token(token), "first")
            token.write_text("rotated\n")
            self.assertEqual(sandoq_env._read_token(token), "rotated")

    def test_token_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.write_text("secret\n")
            target.chmod(0o600)
            link = Path(tmp) / "token"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                sandoq_env._read_token(link)


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def _unstarted_environment() -> sandoq_env.SandoqEnvironment:
        environment = object.__new__(sandoq_env.SandoqEnvironment)
        environment.task_env_config = SimpleNamespace(
            docker_image="registry.example/task:1",
            allow_internet=True,
            cpus=4,
            memory_mb=4096,
            storage_mb=10240,
        )
        environment.environment_name = "task"
        environment.logger = Mock()
        environment._base = "https://sandoq.example"
        environment._outer_environment = "oci-runner"
        environment._lease = "1h"
        environment._owner = "tester"
        environment._create_deadline = 1
        environment._pull_timeout = 1200
        environment._sandbox_id = None
        environment._exec_url = None
        environment._resolved_digest = None
        environment._nested_ready = False
        return environment

    def test_command_response_normalization(self) -> None:
        self.assertEqual(
            sandoq_env._normalize_command_response(
                {"result": {"stdout": "ok", "stderr": "", "exitCode": 0}}
            ),
            ("ok", "", 0, False),
        )
        self.assertEqual(
            sandoq_env._normalize_command_response({"exit_code": -1}),
            ("", "", -1, True),
        )
        with self.assertRaisesRegex(RuntimeError, "no integer exit code"):
            sandoq_env._normalize_command_response({"stdout": "missing"})

    def test_nested_command_quotes_cwd_and_environment(self) -> None:
        wrapped = sandoq_env._wrap_command(
            "printf '%s' \"$VALUE\"", "/workspace/a b", {"VALUE": "x y"}
        )
        self.assertIn("cd '/workspace/a b'", wrapped)
        self.assertIn("export VALUE='x y'", wrapped)
        with self.assertRaisesRegex(ValueError, "Invalid environment variable"):
            sandoq_env._wrap_command("true", None, {"BAD; touch /tmp/x": "1"})

    def test_exec_url_must_be_https_and_cannot_contain_credentials(self) -> None:
        self.assertEqual(
            sandoq_env._validated_exec_url("https://session.example/port"),
            "https://session.example/port/",
        )
        for value in (
            "http://session.example",
            "https://token@session.example",
            "https://session.example/?redirect=evil",
        ):
            with self.assertRaisesRegex(RuntimeError, "must be an HTTPS URL"):
                sandoq_env._validated_exec_url(value)

    def test_outer_exec_uses_authenticated_v1_contract(self) -> None:
        environment = object.__new__(sandoq_env.SandoqEnvironment)
        environment._exec_url = "https://session.example/"
        environment._sandbox_id = "sid"
        calls: list[tuple] = []

        def request(*args):
            calls.append(args)
            return 200, {"stdout": "ok", "stderr": "", "exit_code": 0}

        with patch.object(sandoq_env, "_request", request), patch.object(
            sandoq_env, "_read_token", return_value="rotated-token"
        ):
            result = asyncio.run(environment._outer_exec("echo ok", timeout_sec=7))

        self.assertEqual(result.return_code, 0)
        self.assertEqual(calls[0][1], "https://session.example/v1/exec")
        self.assertEqual(
            calls[0][2],
            {"command": ["bash", "-lc", "echo ok"], "timeout": 7},
        )
        self.assertEqual(calls[0][4], {"Authorization": "Bearer rotated-token"})

    def test_outer_timeout_is_rescuable_and_kills_nested_process_group(self) -> None:
        environment = object.__new__(sandoq_env.SandoqEnvironment)
        environment._nested_ready = True
        environment.logger = Mock()
        environment.environment_name = "task"
        environment._outer_exec = AsyncMock(
            side_effect=[
                TimeoutError("server command timeout"),
                ExecResult(stdout=None, stderr=None, return_code=0),
            ]
        )

        with self.assertRaises(TimeoutError):
            asyncio.run(environment._nested_exec("sleep 300", timeout_sec=1))

        self.assertEqual(environment._outer_exec.await_count, 2)
        run_command = environment._outer_exec.await_args_list[0].args[0]
        cleanup_command = environment._outer_exec.await_args_list[1].args[0]
        self.assertIn("set -m", run_command)
        self.assertIn("harbor_child=$!", run_command)
        self.assertIn(".cancel", run_command)
        self.assertIn("mv -f", run_command)
        self.assertLess(run_command.index("if [ -e"), run_command.index("sleep 300"))
        self.assertIn(": >", cleanup_command)
        self.assertIn("while [ ! -s", cleanup_command)
        self.assertIn("kill -TERM", cleanup_command)
        self.assertIn("kill -KILL", cleanup_command)

        transport = object.__new__(sandoq_env.SandoqEnvironment)
        transport._exec_url = "https://session.example/"
        transport._sandbox_id = "sid"
        with patch.object(
            sandoq_env,
            "_request",
            return_value=(0, {"error": "ReadTimeout: deadline exceeded"}),
        ), patch.object(transport, "_auth_headers", return_value={}):
            with self.assertRaises(TimeoutError):
                asyncio.run(transport._outer_exec("sleep 300", timeout_sec=1))

    def test_cancelled_exec_publishes_tombstone_before_pid_cleanup(self) -> None:
        environment = object.__new__(sandoq_env.SandoqEnvironment)
        environment._nested_ready = True
        environment.logger = Mock()
        environment.environment_name = "task"
        calls: list[str] = []

        async def exercise() -> None:
            started = asyncio.Event()
            never = asyncio.Event()

            async def outer(command: str, timeout_sec: int):
                del timeout_sec
                calls.append(command)
                if len(calls) == 1:
                    started.set()
                    await never.wait()
                return ExecResult(stdout=None, stderr=None, return_code=0)

            environment._outer_exec = outer
            task = asyncio.create_task(
                environment._nested_exec("touch /tmp/must-not-run-late")
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(exercise())
        self.assertEqual(len(calls), 2)
        self.assertIn("must-not-run-late", calls[0])
        self.assertIn(".cancel", calls[0])
        self.assertIn(": >", calls[1])
        self.assertIn("while [ ! -s", calls[1])

    def test_bootstrap_uses_nested_podman_runsc_and_validates_digest(self) -> None:
        environment = object.__new__(sandoq_env.SandoqEnvironment)
        environment.task_env_config = SimpleNamespace(
            allow_internet=True,
            cpus=4,
            memory_mb=8192,
            storage_mb=10240,
        )
        digest = "sha256:" + "a" * 64
        environment._outer_exec = AsyncMock(
            return_value=ExecResult(
                stdout=f"OCI_RESOLVED_DIGEST={digest}\n", stderr=None, return_code=0
            )
        )
        environment._nested_exec = AsyncMock(
            return_value=ExecResult(stdout=None, stderr=None, return_code=0)
        )
        environment._nested_container_env = Mock(return_value={})
        environment._pull_timeout = 1200

        actual = asyncio.run(environment._bootstrap_nested("registry.example/task:1"))

        self.assertEqual(actual, digest)
        bootstrap = environment._outer_exec.await_args.args[0]
        self.assertIn("podman pull --quiet registry.example/task:1", bootstrap)
        self.assertIn("--runtime runsc", bootstrap)
        self.assertIn("--user 0:0", bootstrap)
        self.assertIn("--cpus 4", bootstrap)
        self.assertIn("--memory 8192m", bootstrap)
        self.assertNotIn("--volume", bootstrap)
        self.assertIn("/home/runner/.swe-together-transfer", bootstrap)
        self.assertIn('test "$actual" = "$digest"', bootstrap)
        validation = environment._nested_exec.await_args.args[0]
        self.assertIn("df -Pm /", validation)
        self.assertIn('-ge 10240', validation)
        self.assertIn("/logs/agent", validation)
        self.assertIn("/installed-agent", validation)

    def test_short_leases_renew_before_expiry(self) -> None:
        self.assertEqual(sandoq_env._lease_renewal_interval("3h"), 300.0)
        self.assertEqual(sandoq_env._lease_renewal_interval("2m"), 40.0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            sandoq_env._lease_renewal_interval("0s")

        sandoq_env._active_sessions.clear()
        with patch.object(sandoq_env.time, "monotonic", return_value=100.0):
            sandoq_env._register_session("short", "https://sandoq.example", "2m")
        try:
            state = sandoq_env._active_sessions["short"]
            self.assertEqual(state.renewal_interval, 40.0)
            self.assertEqual(state.next_renewal, 140.0)
        finally:
            sandoq_env._active_sessions.clear()

    def test_http_client_lock_is_reentrant_for_signal_cleanup(self) -> None:
        lock = sandoq_env._client_lock
        with lock:
            acquired_again = lock.acquire(blocking=False)
            self.assertTrue(acquired_again)
            if acquired_again:
                lock.release()

    def test_unspecified_harbor_exec_timeout_uses_long_transport_default(self) -> None:
        environment = object.__new__(sandoq_env.SandoqEnvironment)
        environment._default_exec_timeout = 3600
        environment._persistent_env = {}
        environment._nested_exec = AsyncMock(
            return_value=ExecResult(stdout=None, stderr=None, return_code=0)
        )
        asyncio.run(environment.exec("long verifier", timeout_sec=None))
        self.assertEqual(
            environment._nested_exec.await_args.kwargs["timeout_sec"], 3600
        )

    def test_delete_confirms_http_404(self) -> None:
        responses = Mock(side_effect=[(204, {}), (200, {}), (404, {})])
        with patch.object(sandoq_env, "_request", responses), patch.object(
            sandoq_env.time, "sleep", Mock()
        ):
            sandoq_env._delete_outer_sync("https://sandoq.example", "sid", 1)
        self.assertEqual(
            [call.args[0] for call in responses.call_args_list],
            ["DELETE", "GET", "GET"],
        )

    def test_failed_stop_retains_session_for_retry_or_exit_cleanup(self) -> None:
        environment = object.__new__(sandoq_env.SandoqEnvironment)
        environment._base = "https://sandoq.example"
        environment._sandbox_id = "sid"
        environment._exec_url = "https://session.example/"
        environment._nested_ready = True
        with patch.object(
            sandoq_env, "_delete_outer_sync", side_effect=RuntimeError("transient")
        ):
            with self.assertRaisesRegex(RuntimeError, "transient"):
                asyncio.run(environment.stop(delete=True))
        self.assertEqual(environment._sandbox_id, "sid")
        self.assertTrue(environment._nested_ready)

    def test_startup_failure_and_cancellation_delete_outer_session(self) -> None:
        lease = {
            "sessionId": "sid",
            "portUrls": {"exec": "https://session.example/"},
        }
        for failure in (RuntimeError("auth failed"), asyncio.CancelledError()):
            environment = self._unstarted_environment()
            environment._wait_for_command_server = AsyncMock(side_effect=failure)
            environment.stop = AsyncMock()
            with patch.object(sandoq_env, "_request", return_value=(201, lease)), patch.object(
                sandoq_env, "_register_session"
            ), patch.object(sandoq_env, "_ensure_renewer"), patch.object(
                sandoq_env, "_ensure_exit_cleanup"
            ):
                with self.assertRaises(type(failure)):
                    asyncio.run(environment.start(force_build=False))
            environment.stop.assert_awaited_once_with(delete=True)

    def test_insecure_lease_exec_url_is_deleted_before_any_command(self) -> None:
        environment = self._unstarted_environment()
        environment.stop = AsyncMock()
        lease = {
            "sessionId": "sid",
            "portUrls": {"exec": "http://attacker.invalid/"},
        }
        with patch.object(sandoq_env, "_request", return_value=(201, lease)), patch.object(
            sandoq_env, "_register_session"
        ), patch.object(sandoq_env, "_ensure_renewer"), patch.object(
            sandoq_env, "_ensure_exit_cleanup"
        ):
            with self.assertRaisesRegex(RuntimeError, "must be an HTTPS URL"):
                asyncio.run(environment.start(force_build=False))
        environment.stop.assert_awaited_once_with(delete=True)

    def test_upload_uses_private_outer_staging_and_podman_cp(self) -> None:
        environment = object.__new__(sandoq_env.SandoqEnvironment)
        environment._stage_upload = AsyncMock()
        environment._nested_exec = AsyncMock(
            return_value=ExecResult(stdout=None, stderr=None, return_code=0)
        )
        environment._outer_exec = AsyncMock(
            return_value=ExecResult(stdout=None, stderr=None, return_code=0)
        )
        asyncio.run(environment._upload_bytes(b"patch", "/tmp/agent.patch"))
        outer_path = environment._stage_upload.await_args.args[0]
        self.assertTrue(
            outer_path.startswith("/home/runner/.swe-together-transfer/")
        )
        self.assertNotIn("shared", outer_path)
        outer_commands = [call.args[0] for call in environment._outer_exec.await_args_list]
        self.assertIn("podman cp", outer_commands[0])
        self.assertIn("task:/tmp/agent.patch", outer_commands[0])
        self.assertIn("rm -f", outer_commands[-1])

    def test_download_uses_podman_cp_without_nested_shared_mount(self) -> None:
        environment = object.__new__(sandoq_env.SandoqEnvironment)
        environment._stage_download = AsyncMock(return_value=b"result")
        environment._outer_exec = AsyncMock(
            return_value=ExecResult(stdout=None, stderr=None, return_code=0)
        )
        data = asyncio.run(environment._download_bytes("/workspace/result.txt"))
        self.assertEqual(data, b"result")
        commands = [call.args[0] for call in environment._outer_exec.await_args_list]
        self.assertIn("podman cp", commands[0])
        self.assertIn("task:/workspace/result.txt", commands[0])
        self.assertNotIn("/shared", "\n".join(commands))

    def test_registered_session_cleanup_and_sigterm_hook(self) -> None:
        sandoq_env._active_sessions.clear()
        sandoq_env._register_session("sid", "https://sandoq.example", "1h")
        with patch.object(sandoq_env, "_delete_outer_sync") as delete:
            sandoq_env._cleanup_registered_sessions(timeout=2)
        delete.assert_called_once()
        self.assertEqual(delete.call_args.args[:2], ("https://sandoq.example", "sid"))
        self.assertGreater(delete.call_args.args[2], 0)
        self.assertLessEqual(delete.call_args.args[2], 2)
        self.assertNotIn("sid", sandoq_env._active_sessions)

        installed: dict[str, object] = {}
        old_atexit = sandoq_env._atexit_registered
        old_sigterm = sandoq_env._sigterm_registered
        try:
            sandoq_env._atexit_registered = False
            sandoq_env._sigterm_registered = False
            with patch.object(sandoq_env.atexit, "register"), patch.object(
                sandoq_env.signal, "getsignal", return_value=sandoq_env.signal.SIG_IGN
            ), patch.object(
                sandoq_env.signal,
                "signal",
                side_effect=lambda _sig, handler: installed.setdefault("handler", handler),
            ), patch.object(sandoq_env, "_cleanup_registered_sessions") as cleanup:
                sandoq_env._ensure_exit_cleanup()
                installed["handler"](sandoq_env.signal.SIGTERM, None)
                cleanup.assert_called_once_with(timeout=10.0)
        finally:
            sandoq_env._atexit_registered = old_atexit
            sandoq_env._sigterm_registered = old_sigterm

    def test_sigterm_install_can_retry_after_worker_thread_failure(self) -> None:
        old_atexit = sandoq_env._atexit_registered
        old_sigterm = sandoq_env._sigterm_registered
        try:
            sandoq_env._atexit_registered = False
            sandoq_env._sigterm_registered = False
            with patch.object(sandoq_env.atexit, "register"), patch.object(
                sandoq_env.signal, "getsignal", return_value=sandoq_env.signal.SIG_DFL
            ), patch.object(
                sandoq_env.signal, "signal", side_effect=[ValueError("worker"), None]
            ) as install:
                sandoq_env._ensure_exit_cleanup()
                self.assertFalse(sandoq_env._sigterm_registered)
                sandoq_env._ensure_exit_cleanup()
                self.assertTrue(sandoq_env._sigterm_registered)
                self.assertEqual(install.call_count, 2)
        finally:
            sandoq_env._atexit_registered = old_atexit
            sandoq_env._sigterm_registered = old_sigterm

    def test_renewal_failure_is_reported(self) -> None:
        sandoq_env._active_sessions.clear()
        sandoq_env._register_session("sid", "https://sandoq.example", "3h")
        try:
            with patch.object(
                sandoq_env, "_request", return_value=(503, {"error": "busy"})
            ), self.assertLogs(sandoq_env.__name__, level="WARNING") as logs:
                self.assertFalse(sandoq_env._renew_sessions_once())
            self.assertIn("lease renewal failed", "\n".join(logs.output))
        finally:
            sandoq_env._active_sessions.clear()


class RunnerIntegrationTests(unittest.TestCase):
    def test_environment_type_sandoq_is_factory_wired_and_honors_overrides(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = EnvironmentConfig(
                docker_image="registry.example/task:1",
                cpus=4,
                memory_mb=4096,
                storage_mb=10240,
            )
            with patch.object(sandoq_env, "_read_token", return_value="token"):
                environment = EnvironmentFactory.create_environment(
                    type=EnvironmentType.SANDOQ,
                    environment_dir=root,
                    environment_name="task",
                    session_id="trial",
                    trial_paths=TrialPaths(trial_dir=root / "trial"),
                    task_env_config=config,
                    override_cpus=2,
                    override_memory_mb=3072,
                    override_storage_mb=5120,
                    suppress_override_warnings=True,
                )

        self.assertIsInstance(environment, sandoq_env.SandoqEnvironment)
        flags, storage_mb = environment._nested_resource_flags()
        self.assertEqual(flags, ["--cpus", "2", "--memory", "3072m"])
        self.assertEqual(storage_mb, 5120)

    def test_sandoq_trial_gets_nested_pull_start_timeout(self) -> None:
        # Import lazily: run_eval adjusts sys.path for Harbor and its local modules.
        import run_eval

        task_dir = REPO / "tasks" / "agent-swarm-task-4a881b"
        with patch.dict(os.environ, {"SANDOQ_BUILD_TIMEOUT_MULTIPLIER": "3"}):
            os.environ.pop("TRIAL_BUDGET_SEC", None)
            config = run_eval.build_trial_config(
                task_dir=task_dir,
                action_model="anthropic/claude-opus-4-8",
                user_model="anthropic/claude-opus-4-8",
                user_key="test",
                user_api_base="https://example.invalid",
                agent_env={"ANTHROPIC_API_KEY": "test"},
                trials_dir=REPO / "trials" / "unit-test",
                env_type="sandoq",
                agent_timeout=4800,
                user_context_chars=3000,
                call_user_on_completion=True,
                agent_type="opencode",
                reasoning_effort="high",
            )
        self.assertEqual(
            config.environment.import_path, "sandoq_env:SandoqEnvironment"
        )
        self.assertEqual(config.environment_build_timeout_multiplier, 3.0)
        self.assertEqual(config.agent.kwargs["trial_budget_sec"], 4740)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRIAL_BUDGET_SEC", None)
            inherited = run_eval.build_trial_config(
                task_dir=task_dir,
                action_model="anthropic/claude-opus-4-8",
                user_model="anthropic/claude-opus-4-8",
                user_key="test",
                user_api_base="https://example.invalid",
                agent_env={"ANTHROPIC_API_KEY": "test"},
                trials_dir=REPO / "trials" / "unit-test",
                env_type="sandoq",
                agent_timeout=None,
                user_context_chars=3000,
                call_user_on_completion=True,
                agent_type="opencode",
                reasoning_effort="high",
            )
        self.assertEqual(inherited.agent.kwargs["trial_budget_sec"], 3540)


if __name__ == "__main__":
    unittest.main()
