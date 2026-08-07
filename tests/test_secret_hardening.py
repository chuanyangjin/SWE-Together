from __future__ import annotations

import asyncio
import http.server
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external" / "harbor" / "src"))

import run_eval  # noqa: E402
from harbor.models.trial.config import AgentConfig  # noqa: E402
from harbor.utils.redaction import (  # noqa: E402
    redact_artifact_text,
    redacted_model_dump_json,
)
from proxies.litellm_proxy import (  # noqa: E402
    _node_proxy_script_source,
    launch_litellm_proxy,
)
from proxies.oauth_proxy import (  # noqa: E402
    AUTH_APP_KEY,
    build_app as build_oauth_app,
    handle_health as oauth_health,
    read_client_auth_token,
    run_proxy as run_oauth_proxy,
)
from scripts import sanitize_traces  # noqa: E402


class HarborPersistenceTests(unittest.TestCase):
    def test_model_dump_redacts_copy_without_mutating_runtime_config(self) -> None:
        secret = "unit-secret-persistence-value"
        config = AgentConfig(
            name="opencode",
            model_name="anthropic/example",
            env={
                "ANTHROPIC_AUTH_TOKEN": secret,
                "PROXY_FALLBACK_KEY": "unit-fallback-persistence-value",
                "ANTHROPIC_BASE_URL": "https://gateway.example/v1",
            },
        )

        serialized = redacted_model_dump_json(config, indent=2)

        self.assertNotIn(secret, serialized)
        self.assertEqual(json.loads(serialized)["env"]["ANTHROPIC_AUTH_TOKEN"], "<redacted>")
        self.assertEqual(
            json.loads(serialized)["env"]["PROXY_FALLBACK_KEY"], "<redacted>"
        )
        self.assertIn("https://gateway.example/v1", serialized)
        self.assertEqual(config.env["ANTHROPIC_AUTH_TOKEN"], secret)

    def test_capture_time_text_redacts_command_env_and_authorization(self) -> None:
        secret = "unit-command-artifact-secret"
        raw = (
            f'echo {{"apiKey":"{secret}"}}\n'
            f"Authorization: Bearer {secret}\n"
        )
        with patch.dict(os.environ, {}, clear=True):
            persisted = redact_artifact_text(raw, {"OPENAI_API_KEY": secret})
        self.assertNotIn(secret, persisted)
        self.assertGreaterEqual(persisted.count("<redacted>"), 2)

    def test_secure_qwen_run_env_fails_closed(self) -> None:
        valid = {
            "SWE_QWEN_LOOPBACK_PROXY": "1",
            "OPENAI_BASE_URL": "http://127.0.0.1:43123/v1",
            "OPENAI_API_KEY": "qwen-loopback-placeholder",
        }
        with patch.dict(os.environ, valid, clear=True):
            run_eval.validate_qwen_loopback_proxy("openai/Qwen3.5-4B")

        for disabled in ({}, {"SWE_QWEN_LOOPBACK_PROXY": "0"}):
            with patch.dict(os.environ, disabled, clear=True):
                with self.assertRaises(ValueError):
                    run_eval.validate_qwen_loopback_proxy("openai/Qwen3.5-4B")

        with patch.dict(os.environ, valid, clear=True):
            with self.assertRaises(ValueError):
                run_eval.validate_qwen_loopback_proxy("openai/not-qwen")

        for override in (
            {"OPENAI_API_KEY": "real-looking-deployment-bearer"},
            {"OPENAI_BASE_URL": "http://qwen.internal:8100/v1"},
        ):
            with patch.dict(os.environ, {**valid, **override}, clear=True):
                with self.assertRaises(ValueError):
                    run_eval.validate_qwen_loopback_proxy("openai/Qwen3.5-4B")

    def test_qwen_launcher_never_parses_or_exports_deployment_key(self) -> None:
        launcher = (REPO / "qwen35_4b_repro.sbatch").read_text()
        self.assertNotIn('info["api_key"]', launcher)
        self.assertNotIn("service_key", launcher)
        self.assertIn("src/proxies/qwen_host_proxy.py", launcher)
        self.assertIn("/usr/bin/env -i", launcher)
        self.assertIn("OPENAI_API_KEY=qwen-loopback-placeholder", launcher)
        self.assertIn("#SBATCH --exclusive", launcher)
        self.assertIn("#SBATCH --export=NIL", launcher)
        self.assertIn('>"$proxy_log" 2>&1 8>&- &', launcher)
        self.assertIn('setsid bash run_local.sh "${args[@]}" &', launcher)
        self.assertIn("Qwen host proxy exited during evaluation", launcher)
        self.assertIn("sushi_lane/register_relay_client.py", launcher)
        self.assertIn("probe_relay_acl allowed", launcher)
        self.assertIn("probe_relay_acl denied", launcher)
        self.assertNotIn('${SWE_EGRESS_PROXY:-', launcher)
        capture_source = (
            REPO / "src/user_agent/agents/user_enabled_opencode.py"
        ).read_text()
        self.assertIn(
            "redact_artifact_text(exec_input.command, exec_input.env)",
            capture_source,
        )
        self.assertIn(
            "redact_artifact_text(resume_cmd.command, resume_cmd.env)",
            capture_source,
        )

    def test_qwen_proxy_scrubbed_exec_drops_parent_api_keys(self) -> None:
        stale_openai = "unit-stale-openai-parent-key"
        stale_qwen = "unit-stale-qwen-parent-key"
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            info = root / "proxy_info.json"
            ready = root / "ready.json"
            info.write_text(
                json.dumps(
                    {
                        "url": "http://127.0.0.1:9",
                        "api_key": "unit-file-only-deployment-key",
                        "model": "Qwen3.5-4B",
                    }
                )
            )
            info.chmod(0o600)
            process = subprocess.Popen(
                [
                    "/usr/bin/env",
                    "-i",
                    "PATH=/usr/bin:/bin",
                    "PYTHONNOUSERSITE=1",
                    "PYTHONUNBUFFERED=1",
                    str(REPO / ".venv/bin/python"),
                    str(REPO / "src/proxies/qwen_host_proxy.py"),
                    "--proxy-info",
                    str(info),
                    "--port",
                    "0",
                    "--ready-file",
                    str(ready),
                ],
                env={
                    **os.environ,
                    "OPENAI_API_KEY": stale_openai,
                    "QWEN_API_KEY": stale_qwen,
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                for _ in range(100):
                    if ready.exists():
                        break
                    if process.poll() is not None:
                        self.fail("scrubbed Qwen proxy exited before readiness")
                    time.sleep(0.02)
                self.assertTrue(ready.exists())
                environment = Path(f"/proc/{process.pid}/environ").read_bytes()
                self.assertNotIn(stale_openai.encode(), environment)
                self.assertNotIn(stale_qwen.encode(), environment)
                self.assertNotIn(b"OPENAI_API_KEY=", environment)
                self.assertNotIn(b"QWEN_API_KEY=", environment)
            finally:
                process.terminate()
                process.wait(timeout=5)
            self.assertFalse(ready.exists())

    def test_qwen_finalizer_is_secure_and_stage_ordered(self) -> None:
        finalizer = (REPO / "qwen35_4b_finalize.sbatch").read_text()
        self.assertIn("#SBATCH --exclusive", finalizer)
        self.assertIn("#SBATCH --export=NIL", finalizer)
        self.assertIn("sushi_lane/register_relay_client.py", finalizer)
        self.assertIn("probe_relay_acl allowed", finalizer)
        self.assertIn("probe_relay_acl denied", finalizer)
        self.assertIn("anthropic/claude-opus-4-6", finalizer)
        self.assertIn("gemini/gemini-3.1-pro-preview", finalizer)
        judge = finalizer.index("bash judge_local.sh")
        tags = finalizer.index("scripts/run_vertex_tagger.py")
        metrics = finalizer.index("bash qwen35_4b_metrics.sh")
        self.assertLess(judge, tags)
        self.assertLess(tags, metrics)


class ProxyPersistenceTests(unittest.TestCase):
    def test_node_proxy_keeps_header_values_out_of_curl_argv(self) -> None:
        source = _node_proxy_script_source(
            proxy_port=4210,
            target_url="https://gateway.example/api",
            proxy_model="claude-example",
            is_openrouter_target=False,
            instance_id="unit-instance",
        )
        self.assertNotIn('args.push("-H", name + ": " + value)', source)
        self.assertIn('"--header", "@/proc/self/fd/3"', source)
        self.assertIn('stdio: ["pipe", "pipe", "pipe", headerReadFd]', source)
        read_index = source.index('headerReadFd = fs.openSync(headerPath, "r")')
        unlink_index = source.index("fs.unlinkSync(headerPath)")
        write_index = source.index("fs.writeFileSync(headerWriteFd, headerText")
        self.assertLess(read_index, unlink_index)
        self.assertLess(unlink_index, write_index)

    def test_node_proxy_sigkill_leaves_no_named_header_file(self) -> None:
        if not shutil.which("node") or not shutil.which("curl"):
            self.skipTest("node and curl are required")

        request_reached_upstream = threading.Event()
        release_request = threading.Event()

        class SlowUpstream(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib handler API
                length = int(self.headers.get("content-length", "0"))
                self.rfile.read(length)
                request_reached_upstream.set()
                release_request.wait(timeout=5)
                body = b'{}'
                self.send_response(200)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SlowUpstream)
        upstream_thread = threading.Thread(
            target=upstream.serve_forever, daemon=True
        )
        upstream_thread.start()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            proxy_port = reservation.getsockname()[1]

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "proxy.js"
            script.write_text(
                _node_proxy_script_source(
                    proxy_port=proxy_port,
                    target_url=f"http://127.0.0.1:{upstream.server_port}",
                    proxy_model="claude-example",
                    is_openrouter_target=False,
                    instance_id="sigkill-test",
                )
            )
            process = subprocess.Popen(
                ["node", str(script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env={
                    **os.environ,
                    "TMPDIR": tmp,
                    "PROXY_API_KEY": "unit-sigkill-secret",
                },
            )

            def send_request() -> None:
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{proxy_port}/v1/messages",
                        data=b'{}',
                        headers={"content-type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(request, timeout=5).read()
                except OSError:
                    pass

            request_thread = threading.Thread(target=send_request, daemon=True)
            try:
                health_url = f"http://127.0.0.1:{proxy_port}/health"
                for _ in range(50):
                    try:
                        urllib.request.urlopen(health_url, timeout=0.2).read()
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    self.fail("node proxy did not become healthy")
                request_thread.start()
                self.assertTrue(request_reached_upstream.wait(timeout=3))
                self.assertEqual(list(Path(tmp).glob(".swe-proxy-headers-*")), [])
                os.kill(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                self.assertEqual(list(Path(tmp).glob(".swe-proxy-headers-*")), [])
            finally:
                release_request.set()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if process.poll() is None:
                    process.wait(timeout=5)
                upstream.shutdown()
                upstream.server_close()

    def test_node_proxy_survives_curl_spawn_failure(self) -> None:
        node = shutil.which("node")
        if not node:
            self.skipTest("node is required")

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "proxy.js"
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                proxy_port = reservation.getsockname()[1]
            script.write_text(
                _node_proxy_script_source(
                    proxy_port=proxy_port,
                    target_url="http://127.0.0.1:9",
                    proxy_model="claude-example",
                    is_openrouter_target=False,
                    instance_id="missing-curl-test",
                )
            )
            process = subprocess.Popen(
                [node, str(script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={
                    **os.environ,
                    "PATH": tmp,
                    "TMPDIR": tmp,
                    "PROXY_API_KEY": "unit-spawn-failure-secret",
                },
            )
            try:
                health_url = f"http://127.0.0.1:{proxy_port}/health"
                for _ in range(50):
                    try:
                        urllib.request.urlopen(health_url, timeout=0.2).read()
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    self.fail("node proxy did not become healthy")

                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy_port}/v1/messages",
                    data=b'{}',
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=3).read()
                self.assertEqual(raised.exception.code, 502)
                self.assertIsNone(process.poll())
                self.assertEqual(list(Path(tmp).glob(".swe-proxy-headers-*")), [])
            finally:
                process.terminate()
                process.wait(timeout=5)

    def test_oauth_health_omits_account_and_token_metadata(self) -> None:
        response = asyncio.run(
            oauth_health(SimpleNamespace(app={AUTH_APP_KEY: object()}))
        )
        payload = json.loads(response.text)
        self.assertTrue(payload["ok"])
        self.assertNotIn("account_id", payload)
        self.assertNotIn("access_token_len", payload)

    def test_oauth_proxy_requires_bearer_on_every_route(self) -> None:
        token = "A" * 48

        async def exercise() -> None:
            client = TestClient(
                TestServer(build_oauth_app(object(), client_auth_token=token))
            )
            await client.start_server()
            try:
                for method, path in (
                    ("GET", "/health"),
                    ("GET", "/v1/models"),
                    ("POST", "/v1/chat/completions"),
                    ("POST", "/v1/responses"),
                ):
                    missing = await client.request(method, path, json={})
                    self.assertEqual(missing.status, 401, path)
                    wrong = await client.request(
                        method,
                        path,
                        json={},
                        headers={"Authorization": "Bearer wrong"},
                    )
                    self.assertEqual(wrong.status, 401, path)

                headers = {"Authorization": f"Bearer {token}"}
                health = await client.get("/health", headers=headers)
                self.assertEqual(health.status, 200)
                health_payload = await health.json()
                self.assertEqual(
                    health_payload["service"], "swe-together-oauth-proxy"
                )
                self.assertTrue(health_payload["client_auth"])
                models = await client.get("/v1/models", headers=headers)
                self.assertEqual(models.status, 200)
            finally:
                await client.close()

        asyncio.run(exercise())

    def test_oauth_client_token_file_must_be_private_and_strong(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid"
            valid.write_text("B" * 48)
            valid.chmod(0o600)
            self.assertEqual(read_client_auth_token(valid), "B" * 48)

            public = root / "public"
            public.write_text("C" * 48)
            public.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "group/world"):
                read_client_auth_token(public)

            short = root / "short"
            short.write_text("short")
            short.chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, "too short"):
                read_client_auth_token(short)

            non_ascii = root / "non-ascii"
            non_ascii.write_text("é" * 48)
            non_ascii.chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, "exactly one token"):
                read_client_auth_token(non_ascii)

            link = root / "link"
            link.symlink_to(valid)
            with self.assertRaisesRegex(RuntimeError, "non-symlink"):
                read_client_auth_token(link)

    def test_oauth_unauthenticated_mode_requires_explicit_sandbox_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth = Path(tmp) / "auth.json"
            auth.write_text(
                json.dumps(
                    {
                        "tokens": {
                            "access_token": "dummy-access-token",
                            "account_id": "dummy-account",
                        }
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "opt in"):
                asyncio.run(run_oauth_proxy("127.0.0.1", 0, auth))
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                asyncio.run(
                    run_oauth_proxy(
                        "0.0.0.0",
                        0,
                        auth,
                        allow_unauthenticated_loopback=True,
                    )
                )

    def test_isolated_sandbox_launchers_use_explicit_unauthenticated_opt_in(self) -> None:
        for relative in (
            "src/user_agent/agents/user_enabled_opencode.py",
            "src/user_agent/agents/user_enabled_mini_swe_agent.py",
        ):
            source = (REPO / relative).read_text()
            self.assertIn("--allow-unauthenticated-loopback", source, relative)

    def test_generated_proxy_reads_keys_from_exec_environment(self) -> None:
        primary = "unit-primary-proxy-secret"
        fallback = "unit-fallback-proxy-secret"

        class FakeEnvironment:
            def __init__(self) -> None:
                self.exec_env: dict[str, str] | None = None

            async def upload_file(self, source_path: Path, target_path: str) -> None:
                self.uploaded_source = source_path
                self.uploaded_target = target_path

            async def exec(self, *, command: str, env: dict[str, str]):
                self.exec_env = env
                return SimpleNamespace(return_code=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "LITELLM_PROXY_MODEL": "anthropic/claude-example",
                "LITELLM_PROXY_PORT": "4210",
                "PROXY_TARGET_URL": "https://gateway.example/api",
                "PROXY_API_KEY": primary,
                "PROXY_FALLBACK_URL": "https://fallback.example/api",
                "PROXY_FALLBACK_KEY": fallback,
                "PROXY_FALLBACK_MODEL": "anthropic/fallback",
            },
            clear=False,
        ):
            environment = FakeEnvironment()
            started = asyncio.run(
                launch_litellm_proxy(environment, Path(tmp))
            )
            source = (Path(tmp) / "model_proxy.py").read_text()

        self.assertTrue(started)
        self.assertNotIn(primary, source)
        self.assertNotIn(fallback, source)
        self.assertIn('os.environ.get("PROXY_API_KEY", "")', source)
        self.assertIn("https://gateway.example/api", source)
        self.assertEqual(
            environment.exec_env,
            {"PROXY_API_KEY": primary, "PROXY_FALLBACK_KEY": fallback},
        )


class TraceSanitizerTests(unittest.TestCase):
    def test_parallel_tree_matches_serial_bytes_and_counts(self) -> None:
        known = "unit-parallel-sanitizer-secret"

        def populate(root: Path) -> None:
            root.mkdir()
            for index in range(24):
                (root / f"trace-{index:02d}.json").write_text(
                    json.dumps(
                        {
                            "api_key": f"literal-{index}",
                            "payload": known,
                            "index": index,
                        }
                    )
                )
                (root / f"trace-{index:02d}.log").write_text(
                    f"Authorization: Bearer token-{index}\npayload={known}\n"
                )
            (root / "artifact.bin").write_bytes(b"\x00" + known.encode())

        def snapshot(root: Path) -> dict[str, bytes]:
            return {
                str(path.relative_to(root)): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file() and not path.is_symlink()
            }

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"UNIT_API_KEY": known}, clear=False
        ):
            serial_root = Path(tmp) / "serial"
            parallel_root = Path(tmp) / "parallel"
            populate(serial_root)
            populate(parallel_root)

            serial_counts = sanitize_traces.sanitize_tree(serial_root)
            parallel_counts = sanitize_traces.sanitize_tree(
                parallel_root, workers=8
            )

            self.assertEqual(parallel_counts, serial_counts)
            self.assertEqual(snapshot(parallel_root), snapshot(serial_root))

    def test_sanitizer_default_does_not_construct_a_thread_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trials"
            root.mkdir()
            (root / "trace.txt").write_text("plain text\n")
            with patch.object(
                sanitize_traces,
                "ThreadPoolExecutor",
                side_effect=AssertionError("serial default used a pool"),
            ):
                self.assertEqual(
                    sanitize_traces.sanitize_tree(root),
                    (1, 0),
                )

    def test_tree_scrubs_json_text_and_known_values_but_skips_binary_and_links(self) -> None:
        known = "unit-known-environment-secret"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"UNIT_API_KEY": known}, clear=False
        ):
            root = Path(tmp) / "trials"
            root.mkdir()
            json_path = root / "config.json"
            json_path.write_text(
                json.dumps(
                    {
                        "env": {"apiKey": "json-only-secret"},
                        "base_url": "https://gateway.example/v1",
                        "message": f"known={known}",
                    }
                )
            )
            text_path = root / "agent.log"
            text_path.write_text(
                "OPENAI_API_KEY='text-only-secret'\n"
                'API_KEY = os.environ.get("PROXY_API_KEY", "")\n'
                "Authorization: Bearer header-only-secret\n"
                f"payload={known}\n"
            )
            binary_path = root / "artifact.bin"
            binary = b"\x00" + known.encode()
            binary_path.write_bytes(binary)
            outside = Path(tmp) / "outside.txt"
            outside.write_text(f"UNIT_API_KEY={known}\n")
            (root / "outside-link").symlink_to(outside)
            patch_path = root / "final.patch"
            patch_path.write_text(
                'diff --git a/a.py b/a.py\n+API_KEY = "fixture-not-a-host-secret"\n'
                f'# accidental real value: {known}\n'
            )

            scanned, changed = sanitize_traces.sanitize_tree(root)
            changed_again = sanitize_traces.sanitize_tree(root)[1]

            persisted = json.loads(json_path.read_text())
            text = text_path.read_text()
            self.assertGreaterEqual(scanned, 3)
            self.assertEqual(changed, 3)
            self.assertEqual(changed_again, 0)
            self.assertEqual(persisted["env"]["apiKey"], "<redacted>")
            self.assertEqual(persisted["base_url"], "https://gateway.example/v1")
            self.assertNotIn(known, persisted["message"])
            self.assertNotIn("text-only-secret", text)
            self.assertNotIn("header-only-secret", text)
            self.assertNotIn(known, text)
            self.assertIn('API_KEY = os.environ.get("PROXY_API_KEY", "")', text)
            self.assertEqual(binary_path.read_bytes(), binary)
            self.assertIn(known, outside.read_text())
            sanitized_patch = patch_path.read_text()
            self.assertIn('API_KEY = "fixture-not-a-host-secret"', sanitized_patch)
            self.assertNotIn(known, sanitized_patch)

    def test_runner_sanitizes_without_bucket_and_fails_if_sanitizer_missing(self) -> None:
        known = "unit-runner-local-secret"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"UNIT_API_KEY": known}, clear=False
        ):
            os.environ.pop("BUCKET_NAME", None)
            trials = Path(tmp) / "trials"
            trials.mkdir()
            artifact = trials / "trace.txt"
            artifact.write_text(f"payload={known}\n")

            run_eval._sanitize_and_upload(trials)

            self.assertNotIn(known, artifact.read_text())

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_eval, "REPO_ROOT", Path(tmp)
        ):
            trials = Path(tmp) / "trials"
            trials.mkdir()
            with self.assertRaises(FileNotFoundError):
                run_eval._sanitize_and_upload(trials)

    def test_runner_scoped_sanitizer_never_mutates_active_or_foreign_trials(self) -> None:
        known = "unit-scoped-concurrency-secret"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"UNIT_API_KEY": known}, clear=False
        ):
            os.environ.pop("BUCKET_NAME", None)
            trials = Path(tmp) / "trials"

            def trial(name: str, task: str, completed: bool) -> Path:
                root = trials / name
                root.mkdir(parents=True)
                (root / "config.json").write_text(
                    json.dumps({"task": {"path": f"/tasks/{task}"}})
                )
                (root / "trace.txt").write_text(f"payload={known}\n")
                if completed:
                    (root / "result.json").write_text("{}\n")
                return root

            owned_completed = trial("owned__done", "owned", True)
            owned_active = trial("owned__active", "owned", False)
            foreign_completed = trial("foreign__done", "foreign", True)

            run_eval._sanitize_and_upload(
                trials, ["owned"], completed_only=True
            )

            self.assertNotIn(known, (owned_completed / "trace.txt").read_text())
            self.assertIn(known, (owned_active / "trace.txt").read_text())
            self.assertIn(known, (foreign_completed / "trace.txt").read_text())


if __name__ == "__main__":
    unittest.main()
