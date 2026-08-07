from __future__ import annotations

import asyncio
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external" / "harbor" / "src"))

from eval import table2_metrics  # noqa: E402
from eval import run_eval as eval_run_eval  # noqa: E402
from eval.patch_utils import patch_text_has_changes  # noqa: E402
from eval.user_behavior import user_metrics  # noqa: E402
import run_eval  # noqa: E402
import launch  # noqa: E402
from eval.correctness import podman_judge  # noqa: E402
from harbor.agents.installed.opencode import OpenCode as InstalledOpenCode  # noqa: E402
from proxies.litellm_proxy import _node_proxy_script_source  # noqa: E402
from user_agent.agents.user_enabled_opencode import UserEnabledOpenCode  # noqa: E402


class Table2MetricTests(unittest.TestCase):
    def test_table2_script_supports_direct_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [sys.executable, str(REPO / "eval" / "table2_metrics.py"), "--help"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Compute SWE-Together", completed.stdout)

    def test_boolean_and_string_scores_are_not_numeric_evidence(self) -> None:
        for value in (True, False, "0.9"):
            result = {"verifier_result": {"rewards": {"reward": value}}}
            self.assertIsNone(table2_metrics.get_reward(result))
            with tempfile.TemporaryDirectory() as tmp:
                trial = Path(tmp)
                (trial / "judge_verdict.json").write_text(
                    json.dumps({"judge_score": value})
                )
                self.assertEqual(
                    table2_metrics.get_judge(trial), (0.0, False)
                )

    def test_no_patch_cannot_inherit_a_stale_positive_judge_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks"
            trials = root / "trials"
            (tasks / "task").mkdir(parents=True)
            trial = trials / "task__trial"
            (trial / "agent").mkdir(parents=True)
            (trial / "result.json").write_text("{}")
            (trial / "agent" / "final.patch").write_text("")
            (trial / "judge_verdict.json").write_text(
                json.dumps({"judge_score": 1.0})
            )

            records, incomplete = table2_metrics._collect_trials_detailed(
                [trials], tasks
            )

            self.assertEqual(incomplete, [])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["judge_score"], 0.0)
            self.assertFalse(records[0]["is_judged"])
            self.assertFalse(records[0]["has_substantive_patch"])

    def test_cost_parsers_reject_negative_or_non_numeric_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            agent = trial / "agent"
            agent.mkdir()
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "agent_execution": {
                            "started_at": "2026-01-02T00:00:00Z",
                            "finished_at": "2026-01-01T00:00:00Z",
                        }
                    }
                )
            )
            (trial / "timing.json").write_text(
                json.dumps({"trial_wall_clock_sec": "60"})
            )
            event = {
                "type": "step_finish",
                "part": {
                    "id": "bad-finish",
                    "tokens": {"output": -1, "reasoning": 2},
                },
            }
            (agent / "opencode.txt").write_text(json.dumps(event) + "\n")
            result = json.loads((trial / "result.json").read_text())
            self.assertIsNone(table2_metrics.get_wall_minutes(result, trial))
            self.assertIsNone(eval_run_eval._trial_runtime_sec(trial))
            self.assertIsNone(table2_metrics.sum_tokens(trial))
            self.assertIsNone(eval_run_eval._trial_output_tokens(trial))

    def test_tagged_trial_with_no_followups_has_zero_user_correction(self) -> None:
        self.assertIsNone(user_metrics.metrics_from_rows(None)["user_correction"])
        self.assertEqual(user_metrics.metrics_from_rows([])["user_correction"], 0.0)

    def test_header_only_patch_is_empty_but_tiny_real_diff_is_not(self) -> None:
        headers = (
            "=== /tmp/repo (cumulative vs harbor-base) ===\n\n"
            "=== /workspace/repo (cumulative vs harbor-base) ===\n"
        )
        tiny = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+b\n"
        self.assertFalse(patch_text_has_changes(headers))
        self.assertTrue(patch_text_has_changes(tiny))

    def test_user_correction_uses_tags_not_raw_message_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            (trial / "intent_coverage_verdict.json").write_text(
                json.dumps(
                    {
                        "trial_msg_tags": [
                            {"tags": ["question"]},
                            {"tags": ["request", "correction"]},
                            {"tags": ["question", "nudge"]},
                        ]
                    }
                )
            )
            self.assertEqual(table2_metrics.get_user_correction(trial), 1.2)

    def test_pass_at_one_remains_equally_task_weighted_when_incomplete(self) -> None:
        def record(task: str, score: float) -> dict:
            return {
                "task": task,
                "judge_score": score,
                "s": int(score >= table2_metrics.TAU),
                "user_correction": 0.0,
                "tokens": 1,
                "minutes": 1.0,
                "reward": score,
                "is_judged": True,
            }

        result = table2_metrics.compute_metrics(
            [record("task_a", 1.0), record("task_a", 1.0), record("task_b", 0.0)],
            k=2,
        )
        # Paper formula: mean(task_a sbar=1, task_b sbar=0), not 2/3 trials.
        self.assertEqual(result["aggregates"]["pass@1"], 0.5)

    def test_strict_single_tag_provenance_checks_model_and_episode_indices(self) -> None:
        provenance = user_metrics.tagging_provenance(
            user_metrics.CANONICAL_TAG_MODEL
        )
        verdict = {
            "trial_msg_tags": [{"trial_idx": 7, "tags": ["request"]}],
            "message_tagging": provenance,
        }
        issues = user_metrics.user_correction_provenance_issues(
            verdict,
            source="single",
            expected_tag_model=user_metrics.CANONICAL_TAG_MODEL,
            expected_trial_indices={1},
        )
        self.assertTrue(
            any(issue.startswith("tag_indices_mismatch:") for issue in issues)
        )
        self.assertIn(
            "missing_tagging_provenance",
            user_metrics.user_correction_provenance_issues(
                {"trial_msg_tags": []},
                source="single",
                expected_tag_model=user_metrics.CANONICAL_TAG_MODEL,
                expected_trial_indices=set(),
            ),
        )

    def test_tokens_are_output_plus_reasoning_only_across_turn_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = Path(tmp) / "agent"
            agent.mkdir()
            (agent / "opencode.txt.turn-0").write_text(
                json.dumps(
                    {
                        "type": "step_finish",
                        "part": {
                            "id": "finish-0",
                            "tokens": {"input": 1000, "output": 20, "reasoning": 30}
                        },
                    }
                )
                + "\n"
            )
            duplicate = json.dumps(
                {
                    "type": "step_finish",
                    "part": {
                        "id": "finish-0",
                        "tokens": {"input": 1000, "output": 20, "reasoning": 30},
                    },
                }
            )
            new_event = json.dumps(
                    {
                        "type": "step-finish",
                        "part": {
                            "id": "finish-1",
                            "tokens": {"input": 500, "output": 7, "reasoning": 3},
                        },
                    }
            )
            (agent / "opencode.txt.turn-1").write_text(
                duplicate + "\n" + new_event + "\n"
            )
            self.assertEqual(table2_metrics.sum_tokens(Path(tmp)), 60)
            self.assertEqual(eval_run_eval._trial_output_tokens(Path(tmp)), 60)

    def test_strict_collector_rejects_empty_result_and_stale_infra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks"
            trials = root / "trials"
            (tasks / "task").mkdir(parents=True)
            trial = trials / "task__trial"
            (trial / "agent").mkdir(parents=True)
            (trial / "result.json").write_text("{}")
            (trial / "trial_infra.json").write_text(
                json.dumps({"status": "ok", "version": 1})
            )
            (trial / "intent_coverage_verdict.json").write_text(
                json.dumps({"trial_msg_tags": []})
            )
            (trial / "timing.json").write_text(
                json.dumps({"trial_wall_clock_sec": 1})
            )
            (trial / "agent" / "opencode.txt").write_text(
                json.dumps(
                    {
                        "type": "step_finish",
                        "part": {"id": "finish", "tokens": {"output": 1}},
                    }
                )
                + "\n"
            )

            records, incomplete = table2_metrics._collect_trials_detailed(
                [trials], tasks
            )
            issues = table2_metrics.completeness_issues(
                records,
                incomplete_dirs=incomplete,
                tasks_root=tasks,
                k=1,
                expected_tasks=1,
                expected_judge_model="anthropic/claude-opus-4-6",
            )
            self.assertIn(f"invalid_result:{trial.resolve()}", issues)
            self.assertIn(
                f"infra_version:{trial.resolve()}:1!=expected_2", issues
            )


class ResumeTests(unittest.TestCase):
    def test_node_proxy_fallback_health_and_model_rewrite(self) -> None:
        if not shutil.which("node") or not shutil.which("curl"):
            self.skipTest("node and curl are required")

        observed: dict[str, object] = {}

        class Upstream(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib handler API
                length = int(self.headers.get("content-length", "0"))
                observed["body"] = json.loads(self.rfile.read(length))
                observed["api_key"] = self.headers.get("x-api-key")
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            proxy_port = reservation.getsockname()[1]

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "proxy.js"
            script.write_text(
                _node_proxy_script_source(
                    proxy_port=proxy_port,
                    target_url=f"http://127.0.0.1:{upstream.server_port}",
                    proxy_model="claude-opus-test",
                    is_openrouter_target=False,
                    instance_id="unit-instance",
                )
            )
            process = subprocess.Popen(
                ["node", str(script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={**os.environ, "PROXY_API_KEY": "unit-key"},
            )
            try:
                health_url = f"http://127.0.0.1:{proxy_port}/health"
                for _ in range(50):
                    try:
                        with urllib.request.urlopen(health_url, timeout=0.2) as response:
                            health = json.loads(response.read())
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    self.fail("node proxy did not become healthy")
                self.assertEqual(health["instance"], "unit-instance")

                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy_port}/v1/messages",
                    data=json.dumps({"model": "placeholder"}).encode(),
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(json.loads(response.read()), {"ok": True})
                self.assertEqual(observed["body"], {"model": "claude-opus-test"})
                self.assertEqual(observed["api_key"], "unit-key")
            finally:
                process.terminate()
                process.wait(timeout=5)
                upstream.shutdown()
                upstream.server_close()

    def test_run_eval_closes_litellm_clients_before_event_loop_exit(self) -> None:
        with patch.object(run_eval, "main", new=AsyncMock(return_value=0)):
            with patch(
                "litellm.close_litellm_async_clients", new=AsyncMock()
            ) as close_clients:
                rc = asyncio.run(run_eval._main_with_client_cleanup())
        self.assertEqual(rc, 0)
        close_clients.assert_awaited_once()

    def test_launcher_budget_finishes_before_outer_agent_timeout(self) -> None:
        command = (
            "unset TRIAL_BUDGET_SEC; "
            "source scripts/wrapper_budget.sh; "
            "set_swe_wrapper_budget_from_args --agent-timeout 2400; "
            "printf %s \"$TRIAL_BUDGET_SEC\""
        )
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=REPO,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "2340")

    def test_canonical_launcher_exports_matching_trial_budget(self) -> None:
        plan = {
            "trials_root": "trials/canonical_full109",
            "replicates": [1],
            "user_model": "gemini/gemini-3.1-pro-preview",
            "user_temperature": 0.5,
        }
        cfg = {
            "opus": {
                "model": "openrouter/anthropic/claude-opus-4-8",
                "agent_timeout": 4800,
            }
        }
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRIAL_BUDGET_SEC", None)
            with patch.object(launch, "_run", return_value=0) as run:
                launch.stage_run(plan, cfg, "sandoq", True)
        self.assertEqual(run.call_args.args[2], {"TRIAL_BUDGET_SEC": "4740"})
        command = run.call_args.args[0]
        self.assertEqual(
            command[command.index("--user-model") + 1],
            "gemini/gemini-3.1-pro-preview",
        )
        self.assertEqual(
            command[command.index("--user-temperature") + 1], "0.5"
        )

        with patch.dict(os.environ, {"TRIAL_BUDGET_SEC": "4500"}):
            with patch.object(launch, "_run", return_value=0) as run:
                launch.stage_run(plan, cfg, "sandoq", True)
        self.assertIsNone(run.call_args.args[2])

    def test_opencode_config_patch_uses_guaranteed_node_not_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            config = home / ".config" / "opencode" / "opencode.json"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps(
                    {
                        "provider": {
                            "anthropic": {"models": {"claude-sonnet-4-6": {}}},
                            "openrouter": {"models": {"anthropic/opus": {}}},
                        }
                    }
                )
            )
            agent = object.__new__(UserEnabledOpenCode)
            agent._using_proxied_provider = True
            agent._litellm_proxy_port = 23456
            agent._disallowed_tools = "webfetch,websearch"
            command = agent._opencode_thinking_patch_command()
            self.assertIsNotNone(command)
            self.assertNotIn("python3", command)
            completed = subprocess.run(
                ["bash", "-c", command],
                env={**os.environ, "HOME": str(home)},
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            updated = json.loads(config.read_text())
            self.assertEqual(
                updated["provider"]["anthropic"]["options"]["baseURL"],
                "http://localhost:23456/v1",
            )
            self.assertTrue(
                updated["provider"]["anthropic"]["models"]
                ["claude-sonnet-4-6"]["reasoning"]
            )
            self.assertEqual(
                updated["provider"]["openrouter"]["models"]
                ["anthropic/opus"]["variants"]["high"]["reasoning"]["effort"],
                "high",
            )
            self.assertEqual(updated["permission"]["tools"]["webfetch"], "deny")

            command_input = SimpleNamespace(
                command=(
                    ". ~/.nvm/nvm.sh; opencode --model=anthropic/test "
                    "run --format=json -- task"
                ),
                env={},
            )
            agent._reasoning_effort = "high"
            rewritten = agent._inject_opencode_flags([command_input])[0].command
            self.assertIn("JSEOF\n) && (. ~/.nvm/nvm.sh; opencode", rewritten)
            self.assertEqual(
                command_input.env["ANTHROPIC_BASE_URL"],
                "http://localhost:23456",
            )

    def test_opencode_node_bootstrap_does_not_require_xz(self) -> None:
        template = (
            REPO
            / "external/harbor/src/harbor/agents/installed/install-opencode.sh.j2"
        ).read_text()
        self.assertIn(".tar.gz", template)
        self.assertIn("-xzf /tmp/node.tar.gz", template)
        self.assertNotIn("-xJf /tmp/node.tar.xz", template)

    def test_opencode_resume_reuses_custom_openai_provider_ref(self) -> None:
        agent = object.__new__(UserEnabledOpenCode)
        agent._inner = SimpleNamespace(
            model_name="openai/Qwen3.5-4B",
            _run_model_ref=lambda: "vllm-openai-compatible/Qwen3.5-4B",
        )
        agent._inner_run_env = {}
        agent._reasoning_effort = "high"

        command = agent._build_resume_command("session-id", "continue").command

        self.assertIn(
            "opencode --model=vllm-openai-compatible/Qwen3.5-4B run", command
        )
        self.assertNotIn("opencode --model=openai/Qwen3.5-4B run", command)

    def test_opencode_secure_qwen_config_contains_only_placeholder(self) -> None:
        agent = object.__new__(InstalledOpenCode)
        agent.model_name = "openai/Qwen3.5-4B"
        agent.mcp_servers = []
        secure = {
            "SWE_QWEN_LOOPBACK_PROXY": "1",
            "OPENAI_BASE_URL": "http://127.0.0.1:43123/v1",
            "OPENAI_API_KEY": "qwen-loopback-placeholder",
        }
        with patch.dict(os.environ, secure, clear=True):
            command = agent._build_register_config_command()
        self.assertIsNotNone(command)
        self.assertIn("qwen-loopback-placeholder", command)
        self.assertIn("http://127.0.0.1:43123/v1", command)

        with patch.dict(
            os.environ,
            {**secure, "OPENAI_API_KEY": "real-looking-deployment-bearer"},
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                agent._build_register_config_command()

    def test_opencode_materializes_combined_host_log_for_token_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = object.__new__(UserEnabledOpenCode)
            agent.logs_dir = Path(tmp)
            agent._cumulative_output = [
                '{"type":"step_start"}\n',
                '{"type":"step_finish","part":{"tokens":{"output":7}}}\n',
            ]

            agent._sync_combined_opencode_log()

            combined = (Path(tmp) / "opencode.txt").read_text()
            self.assertIn('"type":"step_start"', combined)
            self.assertIn('"output":7', combined)

    def test_launcher_stops_after_first_executed_stage_failure(self) -> None:
        plan = {
            "trials_root": "trials/canonical_full109",
            "replicates": [1, 2],
        }
        with patch.object(launch, "_run", return_value=7) as run:
            rc = launch.stage_run(
                plan,
                {"opus": {"model": "openrouter/anthropic/claude-opus-4-8"}},
                "sandoq",
                True,
            )
        self.assertEqual(rc, 7)
        self.assertEqual(run.call_count, 1)

    def test_completed_trial_counter_supports_replicate_deficits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for suffix in ("aaaaaaa", "bbbbbbb"):
                trial = root / f"task-name__{suffix}"
                trial.mkdir()
                (trial / "result.json").write_text(
                    json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
                )
            with patch.object(
                run_eval,
                "classify_or_load",
                return_value=SimpleNamespace(status="ok", reason=""),
            ):
                self.assertEqual(
                    run_eval.count_completed_trials("task-name", root), 2
                )

    def test_launcher_propagates_sandoq_to_judge_and_pins_paper_judge(self) -> None:
        plan = {
            "trials_root": "trials/canonical_full109",
            "tasks_root": "tasks",
            "replicates": [1, 2],
        }
        with patch.object(launch, "_run", return_value=0) as run:
            launch.stage_judge(plan, {"opus": {}}, "results", "sandoq", True)
        self.assertEqual(
            run.call_args.args[2],
            {
                "JUDGE_ENV": "sandoq",
                "JUDGE_PODMAN_MODEL": "anthropic/claude-opus-4-6",
            },
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JUDGE_PODMAN_MODEL", None)
            self.assertEqual(
                podman_judge._judge_model(), "anthropic/claude-opus-4-6"
            )

    def test_completeness_gate_rejects_unjudged_substantive_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "task__trial"
            (trial / "agent").mkdir(parents=True)
            (trial / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
            )
            (trial / "trial_infra.json").write_text(
                json.dumps({"version": 2, "status": "ok"})
            )
            (trial / "agent" / "final.patch").write_text(
                "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+b\n"
            )
            (trial / "intent_coverage_verdict.json").write_text(
                json.dumps({"trial_msg_tags": []})
            )
            jobs = [
                {
                    "trial_dir": str(trial),
                    "task": "task",
                    "judge_out_name": "judge_verdict.json",
                    "coverage_out_name": "intent_coverage_verdict.json",
                }
            ]
            with patch.object(eval_run_eval, "_is_infra_failed", return_value=False):
                issues = eval_run_eval.completeness_issues(
                    jobs, expected_replicates=1, expected_tasks=1
                )
            self.assertEqual(issues, ["judge_incomplete:task__trial"])

    def test_completeness_gate_rejects_wrong_judge_and_missing_costs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "task__trial"
            (trial / "agent").mkdir(parents=True)
            (trial / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
            )
            (trial / "trial_infra.json").write_text(
                json.dumps({"version": 2, "status": "ok"})
            )
            (trial / "agent" / "final.patch").write_text(
                "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+b\n"
            )
            (trial / "judge_verdict.json").write_text(
                json.dumps(
                    {
                        "judge_score": 1.0,
                        "judge_model": "anthropic/claude-opus-4-8",
                    }
                )
            )
            (trial / "intent_coverage_verdict.json").write_text(
                json.dumps({"trial_msg_tags": []})
            )
            jobs = [
                {
                    "trial_dir": str(trial),
                    "task": "task",
                    "judge_out_name": "judge_verdict.json",
                    "coverage_out_name": "intent_coverage_verdict.json",
                }
            ]
            with patch.object(eval_run_eval, "_is_infra_failed", return_value=False):
                issues = eval_run_eval.completeness_issues(
                    jobs,
                    expected_replicates=1,
                    expected_tasks=1,
                    expected_judge_model="claude-opus-4-6",
                    require_cost_data=True,
                )
            self.assertIn("runtime_incomplete:task__trial", issues)
            self.assertIn("tokens_incomplete:task__trial", issues)
            self.assertIn(
                "judge_model:task__trial:anthropic/claude-opus-4-8"
                "!=expected_claude-opus-4-6",
                issues,
            )

    def test_completeness_gate_rejects_out_of_range_judge_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "task__trial"
            (trial / "agent").mkdir(parents=True)
            (trial / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
            )
            (trial / "trial_infra.json").write_text(
                json.dumps({"version": 2, "status": "ok"})
            )
            (trial / "agent" / "final.patch").write_text(
                "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+b\n"
            )
            (trial / "judge_verdict.json").write_text(
                json.dumps(
                    {
                        "judge_score": 2.0,
                        "judge_model": "anthropic/claude-opus-4-6",
                    }
                )
            )
            (trial / "intent_coverage_verdict.json").write_text(
                json.dumps({"trial_msg_tags": []})
            )
            jobs = [
                {
                    "trial_dir": str(trial),
                    "task": "task",
                    "judge_out_name": "judge_verdict.json",
                    "coverage_out_name": "intent_coverage_verdict.json",
                }
            ]
            with patch.object(eval_run_eval, "_is_infra_failed", return_value=False):
                issues = eval_run_eval.completeness_issues(
                    jobs,
                    expected_replicates=1,
                    expected_tasks=1,
                    expected_judge_model="anthropic/claude-opus-4-6",
                )
            self.assertEqual(issues, ["judge_invalid_score:task__trial"])

    def test_completeness_gate_fails_closed_on_result_and_infra_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "task__trial"
            (trial / "agent").mkdir(parents=True)
            (trial / "result.json").write_text("{}")
            (trial / "trial_infra.json").write_text(
                json.dumps({"version": 1, "status": "ok"})
            )
            (trial / "intent_coverage_verdict.json").write_text(
                json.dumps({"trial_msg_tags": []})
            )
            jobs = [{"trial_dir": str(trial), "task": "task"}]
            with patch.object(
                eval_run_eval,
                "_classify_infra_fresh",
                side_effect=RuntimeError("unreadable"),
            ):
                issues = eval_run_eval.completeness_issues(
                    jobs, expected_replicates=1, expected_tasks=1
                )
            self.assertIn("invalid_result:task__trial", issues)
            self.assertIn(
                "infra_version:task__trial:1!=expected_2", issues
            )
            self.assertIn(
                "infra_fresh_status:task__trial:unavailable", issues
            )


if __name__ == "__main__":
    unittest.main()
