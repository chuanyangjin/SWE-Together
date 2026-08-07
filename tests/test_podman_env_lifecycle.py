from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external" / "harbor" / "src"))

from harbor.environments.base import ExecResult  # noqa: E402

from podman_env import (  # noqa: E402
    _ENV_FILE_PLACEHOLDER,
    PodmanEnvironment,
)
from user_agent.exec_helpers import exec_with_budget  # noqa: E402


class _FakePodman:
    """Small image/container model with Podman 3.x force-rmi semantics."""

    def __init__(self, image: str) -> None:
        self.image = image
        self.image_exists = True
        self.containers: dict[str, str] = {}
        self.commands: list[tuple[str, list[str]]] = []
        self.pause_exists_for: str | None = None
        self.exists_entered = asyncio.Event()
        self.release_exists = asyncio.Event()

    async def run(
        self,
        owner: str,
        cmd: list[str],
        check: bool = True,
        **_kwargs,
    ) -> ExecResult:
        self.commands.append((owner, cmd))
        stdout: str | None = None
        stderr: str | None = None
        return_code = 0

        if cmd[:2] == ["image", "exists"]:
            # Snapshot the result before pausing. This opens the exact old race:
            # stop(A) could force-rmi after start(B)'s successful check but
            # before start(B)'s podman run.
            return_code = 0 if self.image_exists else 1
            if owner == self.pause_exists_for:
                self.exists_entered.set()
                await self.release_exists.wait()
        elif cmd[0] == "pull":
            self.image_exists = True
        elif cmd[0] == "run":
            name = cmd[cmd.index("--name") + 1]
            image = cmd[-3]
            if not self.image_exists:
                return_code = 125
                stderr = "image not known"
            else:
                self.containers[name] = image
                stdout = f"id-{name}\n"
        elif cmd[0] == "rm":
            name = cmd[-1]
            if self.containers.pop(name, None) is None:
                return_code = 1
                stderr = "no such container"
        elif cmd[0] == "rmi":
            users = [name for name, value in self.containers.items() if value == cmd[-1]]
            if users and "-f" not in cmd:
                return_code = 2
                stderr = "image is in use"
            else:
                if "-f" in cmd:
                    for name in users:
                        del self.containers[name]
                self.image_exists = False

        result = ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
        )
        if check and return_code:
            raise RuntimeError(stderr or "podman failed")
        return result


class PodmanLifecycleRaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Locks are process-global in production; isolate this event loop/test.
        PodmanEnvironment._image_pull_locks = {}

    @staticmethod
    def _environment(
        name: str, image: str, podman: _FakePodman
    ) -> PodmanEnvironment:
        environment = object.__new__(PodmanEnvironment)
        environment.environment_name = name
        environment.task_env_config = SimpleNamespace(
            docker_image=image,
            allow_internet=True,
        )
        environment._container = f"hb-{name}"
        environment._container_id = None
        environment.logger = Mock()
        environment._podman = lambda *args: list(args)
        environment._container_env = Mock(return_value={})

        async def run(cmd: list[str], **kwargs) -> ExecResult:
            return await podman.run(name, cmd, **kwargs)

        environment._run = AsyncMock(side_effect=run)
        environment.exec = AsyncMock(
            return_value=ExecResult(stdout=None, stderr=None, return_code=0)
        )
        return environment

    async def test_same_image_start_stop_rmi_is_atomic_and_non_destructive(self) -> None:
        image = "registry.example/task:1"
        podman = _FakePodman(image)
        first = self._environment("first", image, podman)
        second = self._environment("second", image, podman)

        with patch.dict(os.environ, {"HARBOR_PODMAN_RMI": "1"}):
            await first.start(force_build=False)

            podman.pause_exists_for = "second"
            second_start = asyncio.create_task(second.start(force_build=False))
            await asyncio.wait_for(podman.exists_entered.wait(), timeout=1)

            # stop(first) races precisely between second's image check and run.
            first_stop = asyncio.create_task(first.stop(delete=True))
            await asyncio.sleep(0)
            self.assertFalse(
                any(cmd[0] == "rmi" for _, cmd in podman.commands),
                "rmi must wait for the same-image start critical section",
            )

            podman.release_exists.set()
            await asyncio.wait_for(
                asyncio.gather(second_start, first_stop), timeout=1
            )

            self.assertIn(second._container, podman.containers)
            rmi_commands = [cmd for _, cmd in podman.commands if cmd[0] == "rmi"]
            self.assertEqual(len(rmi_commands), 1)
            self.assertNotIn("-f", rmi_commands[0])

            await second.stop(delete=True)
            self.assertFalse(podman.image_exists)
            self.assertNotIn(second._container, podman.containers)

    async def test_start_passes_container_secret_via_env_file(self) -> None:
        image = "registry.example/task:1"
        podman = _FakePodman(image)
        environment = self._environment("secret-start", image, podman)
        secret = "unit-container-secret-must-not-appear-in-argv"
        environment._container_env = Mock(return_value={"OPENAI_API_KEY": secret})

        await environment.start(force_build=False)

        run_call = next(
            call
            for call in environment._run.await_args_list
            if call.args[0][0] == "run"
        )
        self.assertNotIn(secret, "\0".join(run_call.args[0]))
        self.assertNotIn("OPENAI_API_KEY", run_call.args[0])
        self.assertIn(_ENV_FILE_PLACEHOLDER, run_call.args[0])
        self.assertEqual(
            run_call.kwargs["container_env"], {"OPENAI_API_KEY": secret}
        )

    async def test_environment_timeout_becomes_rescuable_cap_result(self) -> None:
        environment = SimpleNamespace(
            exec=AsyncMock(side_effect=TimeoutError("podman command timed out"))
        )
        exec_input = SimpleNamespace(command="agent run", cwd="/repo", env={})

        result, timed_out = await exec_with_budget(
            environment, exec_input, start_time=time.monotonic()
        )

        self.assertTrue(timed_out)
        self.assertEqual(result.return_code, -1)
        self.assertIn("exec capped", result.stderr)

    async def test_proxied_anthropic_target_is_not_added_to_no_proxy(self) -> None:
        environment = object.__new__(PodmanEnvironment)
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_BASE_URL": "http://gateway.internal/v1",
                "OPENAI_BASE_URL": "http://qwen.internal/v1",
                "LITELLM_PROXY_MODEL": "claude-opus-test",
                "HARBOR_PODMAN_NO_PROXY": "ghcr.io",
            },
            clear=False,
        ):
            hosts = set(environment._no_proxy_hosts().split(","))
        self.assertNotIn("gateway.internal", hosts)
        self.assertIn("qwen.internal", hosts)
        self.assertIn("ghcr.io", hosts)

    async def test_secure_qwen_mode_forwards_only_loopback_placeholder(self) -> None:
        environment = object.__new__(PodmanEnvironment)
        environment.task_env_config = SimpleNamespace(allow_internet=True)
        secure = {
            "SWE_QWEN_LOOPBACK_PROXY": "1",
            "OPENAI_BASE_URL": "http://127.0.0.1:43123/v1",
            "OPENAI_API_KEY": "qwen-loopback-placeholder",
            "HTTP_PROXY": "http://filtering-proxy:8080",
            "HARBOR_PODMAN_FORWARD_ENV": "ANTHROPIC_API_KEY,SAFE_TRACE",
            "ANTHROPIC_API_KEY": "must-never-enter-qwen-container",
            "SAFE_TRACE": "trace-ok",
            "HARBOR_PODMAN_NO_PROXY": "qwen.internal",
        }
        with patch.dict(os.environ, secure, clear=True):
            container_env = environment._container_env()

        self.assertEqual(
            container_env["OPENAI_BASE_URL"], "http://127.0.0.1:43123/v1"
        )
        self.assertEqual(
            container_env["OPENAI_API_KEY"], "qwen-loopback-placeholder"
        )
        self.assertEqual(container_env["SAFE_TRACE"], "trace-ok")
        self.assertNotIn("ANTHROPIC_API_KEY", container_env)
        self.assertEqual(container_env["no_proxy"], "127.0.0.1,localhost")
        self.assertNotIn("qwen.internal", str(container_env))

    async def test_secure_qwen_mode_rejects_direct_endpoint_or_real_key(self) -> None:
        environment = object.__new__(PodmanEnvironment)
        environment.task_env_config = SimpleNamespace(allow_internet=True)
        base = {
            "SWE_QWEN_LOOPBACK_PROXY": "1",
            "OPENAI_BASE_URL": "http://127.0.0.1:43123/v1",
            "OPENAI_API_KEY": "qwen-loopback-placeholder",
        }
        for override in (
            {"OPENAI_BASE_URL": "http://qwen.internal:8100/v1"},
            {"OPENAI_API_KEY": "real-looking-deployment-bearer"},
        ):
            with patch.dict(os.environ, {**base, **override}, clear=True):
                with self.assertRaises(RuntimeError):
                    environment._container_env()

    async def test_exec_timeout_terminates_inner_process_group(self) -> None:
        environment = object.__new__(PodmanEnvironment)
        environment._container = "hb-timeout-test"
        environment.environment_name = "timeout-test"
        environment.logger = Mock()
        environment._merge_env = Mock(return_value={})
        environment._podman = lambda *args: list(args)
        environment._run = AsyncMock(
            side_effect=[
                TimeoutError("podman command timed out"),
                ExecResult(stdout=None, stderr=None, return_code=0),
            ]
        )

        with self.assertRaises(TimeoutError):
            await environment.exec("sleep 300", timeout_sec=1)

        self.assertEqual(environment._run.await_count, 2)
        run_command = environment._run.await_args_list[0].args[0]
        cleanup_command = environment._run.await_args_list[1].args[0]
        self.assertIn("set -m", run_command[-1])
        self.assertIn("harbor_child=$!", run_command[-1])
        self.assertIn("kill -TERM -- \"-$harbor_pid\"", cleanup_command[-1])
        self.assertIn("kill -KILL -- \"-$harbor_pid\"", cleanup_command[-1])

    async def test_exec_passes_secret_via_env_file(self) -> None:
        environment = object.__new__(PodmanEnvironment)
        environment._container = "hb-secret-test"
        environment.environment_name = "secret-test"
        environment.logger = Mock()
        secret = "unit-secret-must-not-appear-in-argv"
        environment._merge_env = Mock(return_value={"PROXY_API_KEY": secret})
        environment._podman = lambda *args: list(args)
        environment._run = AsyncMock(
            return_value=ExecResult(stdout=None, stderr=None, return_code=0)
        )

        await environment.exec("true", timeout_sec=5)

        command = environment._run.await_args.args[0]
        self.assertNotIn(secret, "\0".join(command))
        self.assertNotIn("PROXY_API_KEY", command)
        self.assertIn(_ENV_FILE_PLACEHOLDER, command)
        self.assertEqual(
            environment._run.await_args.kwargs["container_env"],
            {"PROXY_API_KEY": secret},
        )

    async def test_run_keeps_host_path_and_uses_anonymous_env_file(self) -> None:
        class _Process:
            returncode = 0

            async def communicate(self):
                return b"ok", b""

        environment = object.__new__(PodmanEnvironment)
        environment.environment_name = "anonymous-env-test"
        secret = "unit-secret-never-in-host-environ-or-argv"
        command = ["podman", "run", "--env-file", _ENV_FILE_PLACEHOLDER, "image"]
        container_env = {"PATH": "/container/bin", "PROXY_API_KEY": secret}

        with patch.dict(
            os.environ,
            {"PATH": "/usr/bin:/bin", "PROXY_API_KEY": "host-secret"},
            clear=True,
        ), patch(
            "podman_env.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_Process()),
        ) as spawn:
            result = await environment._run(command, container_env=container_env)

        self.assertEqual(result.return_code, 0)
        spawned_command = spawn.await_args.args
        spawned_kwargs = spawn.await_args.kwargs
        self.assertNotIn(secret, "\0".join(spawned_command))
        self.assertNotIn(_ENV_FILE_PLACEHOLDER, spawned_command)
        self.assertRegex(spawned_command[3], r"^/proc/self/fd/\d+$")
        self.assertEqual(spawned_kwargs["env"]["PATH"], "/usr/bin:/bin")
        self.assertNotIn("PROXY_API_KEY", spawned_kwargs["env"])
        self.assertEqual(spawned_kwargs["pass_fds"], (int(spawned_command[3][14:]),))

    async def test_run_env_file_descriptor_is_readable_by_child(self) -> None:
        environment = object.__new__(PodmanEnvironment)
        environment.environment_name = "env-fd-integration-test"

        result = await environment._run(
            [
                "/bin/sh",
                "-c",
                'cat "$1"',
                "env-reader",
                _ENV_FILE_PLACEHOLDER,
            ],
            container_env={"SAFE_TEST_VALUE": "value=with-equals"},
        )

        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.stdout, "SAFE_TEST_VALUE=value=with-equals\n")


if __name__ == "__main__":
    unittest.main()
