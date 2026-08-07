from __future__ import annotations

import asyncio
import json
import socket
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "harbor" / "src"))

import launch  # noqa: E402
from eval import run_eval as eval_run_eval  # noqa: E402
from eval import table2_metrics  # noqa: E402
from eval.user_behavior import adjudicate_3way, tag_messages, user_metrics as kg  # noqa: E402
from scripts import run_vertex_tagger  # noqa: E402


class _FakeArbiter:
    def __init__(self, votes: dict[int, list[str]]):
        self.votes = votes
        self.calls: list[list[dict]] = []

    async def tag(self, messages: list[dict]) -> dict[int, list[str]]:
        self.calls.append(messages)
        return self.votes


class ThreewayAdjudicationTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        a_tags: list[str],
        b_tags: list[str] | None,
    ) -> tuple[Path, Path]:
        trial = root / "fake-task__abcdefg"
        trial.mkdir()
        (trial / "result.json").write_text("{}")
        (trial / "intent_coverage_verdict.json").write_text(
            json.dumps(
                {
                    "trial_msg_tags": [
                        {"trial_idx": 1, "tags": a_tags, "frustration": 0}
                    ],
                    "message_tagging": kg.tagging_provenance(
                        kg.PAPER_TAG_JUDGE_A
                    ),
                }
            )
        )
        trials = {}
        if b_tags is not None:
            trials[trial.name] = {"1": b_tags}
        sidecar = root / "judge_b.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": kg.TAGGING_SCHEMA_VERSION,
                    "message_tagging": kg.tagging_provenance(
                        kg.PAPER_TAG_JUDGE_B
                    ),
                    "trials": trials,
                }
            )
        )
        return trial, sidecar

    @staticmethod
    def _sim_messages(_trial_dir: Path, _task_dir: Path | None) -> list[dict]:
        return [
            {"trial_idx": 0, "text": "initial"},
            {"trial_idx": 1, "text": "please fix that"},
        ]

    def test_agreement_needs_no_arbiter_and_writes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial, sidecar = self._fixture(
                root,
                a_tags=["request", "correction"],
                b_tags=["request", "correction"],
            )
            arbiter = _FakeArbiter({})
            with patch.object(
                adjudicate_3way, "load_trial_sim_msgs", self._sim_messages
            ):
                counts = asyncio.run(
                    adjudicate_3way.run_adjudication(
                        [root],
                        judge_b_sidecar=sidecar,
                        require_provenance=True,
                        arbiter=arbiter,
                    )
                )
            verdict = json.loads(
                (trial / "intent_coverage_verdict.json").read_text()
            )
            self.assertEqual(counts["ok"], 1)
            self.assertEqual(arbiter.calls, [])
            self.assertEqual(verdict["user_correction_3way"], 1.0)
            self.assertEqual(
                kg.user_correction_provenance_issues(
                    verdict, source="threeway"
                ),
                [],
            )

    def test_disagreement_uses_only_explicit_arbiter_vote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial, sidecar = self._fixture(
                root, a_tags=["request"], b_tags=["request", "correction"]
            )
            arbiter = _FakeArbiter({1: ["correction"]})
            with patch.object(
                adjudicate_3way, "load_trial_sim_msgs", self._sim_messages
            ):
                counts = asyncio.run(
                    adjudicate_3way.run_adjudication(
                        [root],
                        judge_b_sidecar=sidecar,
                        require_provenance=True,
                        arbiter=arbiter,
                    )
                )
            verdict = json.loads(
                (trial / "intent_coverage_verdict.json").read_text()
            )
            self.assertEqual(counts["arbitrated"], 1)
            self.assertEqual([m["trial_idx"] for m in arbiter.calls[0]], [1])
            self.assertIn("correction", verdict["trial_msg_tags_3way"][0]["tags"])

    def test_missing_judge_b_vote_clears_stale_output_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial, sidecar = self._fixture(
                root, a_tags=["request"], b_tags=None
            )
            verdict_path = trial / "intent_coverage_verdict.json"
            verdict = json.loads(verdict_path.read_text())
            verdict.update(
                {
                    "trial_msg_tags_3way": [],
                    "user_correction_3way": 0.0,
                    "message_tagging_3way": {"stale": True},
                }
            )
            verdict_path.write_text(json.dumps(verdict))
            with patch.object(
                adjudicate_3way, "load_trial_sim_msgs", self._sim_messages
            ):
                counts = asyncio.run(
                    adjudicate_3way.run_adjudication(
                        [root],
                        judge_b_sidecar=sidecar,
                        require_provenance=True,
                        force=True,
                        arbiter=_FakeArbiter({}),
                    )
                )
            persisted = json.loads(verdict_path.read_text())
            self.assertEqual(counts["err"], 1)
            self.assertNotIn("trial_msg_tags_3way", persisted)
            self.assertNotIn("message_tagging_3way", persisted)


class TagSidecarTests(unittest.TestCase):
    def test_v1_tagging_provenance_fails_current_strict_validation(self) -> None:
        provenance = kg.tagging_provenance(kg.CANONICAL_TAG_MODEL)
        provenance["schema_version"] = 1
        verdict = {
            "trial_msg_tags": [
                {"trial_idx": 1, "tags": ["request"], "frustration": 0}
            ],
            "message_tagging": provenance,
        }

        self.assertIn(
            "tagging_schema_version",
            kg.tagging_provenance_issues(
                provenance, kg.CANONICAL_TAG_MODEL
            ),
        )
        self.assertIn(
            "tagging_schema_version",
            kg.user_correction_provenance_issues(
                verdict,
                source="single",
                expected_tag_model=kg.CANONICAL_TAG_MODEL,
                expected_trial_indices={1},
            ),
        )

    def test_v1_sidecar_requires_force_and_force_regenerates_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trials"
            trial = root / "task__abcdefg"
            trial.mkdir(parents=True)
            (trial / "result.json").write_text("{}")
            output = Path(tmp) / "tags.json"
            legacy_provenance = kg.tagging_provenance(kg.PAPER_TAG_JUDGE_B)
            legacy_provenance["schema_version"] = 1
            output.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "message_tagging": legacy_provenance,
                        "tag_transport": {"backend": "litellm"},
                        "trials": {trial.name: {}},
                        "trial_rows": {trial.name: []},
                        "trial_input_sha256": {
                            trial.name: kg.tag_input_sha256([])
                        },
                    }
                )
            )

            with self.assertRaisesRegex(
                RuntimeError, "provenance mismatch"
            ):
                asyncio.run(
                    tag_messages.run_sidecar_batch(
                        [root],
                        model=kg.PAPER_TAG_JUDGE_B,
                        workers=1,
                        output=output,
                    )
                )

            with patch.object(
                tag_messages, "_make_llm", return_value=object()
            ), patch.object(
                tag_messages, "tag_one", AsyncMock(return_value=[])
            ):
                counts = asyncio.run(
                    tag_messages.run_sidecar_batch(
                        [root],
                        model=kg.PAPER_TAG_JUDGE_B,
                        workers=1,
                        output=output,
                        force=True,
                    )
                )

            regenerated = json.loads(output.read_text())
            self.assertEqual(counts, {"ok": 1, "skip": 0, "err": 0})
            self.assertEqual(
                regenerated["schema_version"], kg.TAGGING_SCHEMA_VERSION
            )
            self.assertEqual(
                regenerated["message_tagging"]["schema_version"],
                kg.TAGGING_SCHEMA_VERSION,
            )

    def test_v1_sidecar_fails_table2_envelope_validation(self) -> None:
        provenance = kg.tagging_provenance(kg.CANONICAL_TAG_MODEL)
        provenance["schema_version"] = 1
        issues = table2_metrics._tag_sidecar_issues(
            {
                "schema_version": 1,
                "message_tagging": provenance,
                "tag_transport": {"backend": "vertex-gateway"},
                "trials": {},
                "trial_rows": {},
                "trial_input_sha256": {},
            },
            kg.CANONICAL_TAG_MODEL,
        )
        self.assertIn("tag_sidecar:schema_version", issues)
        self.assertIn("tag_sidecar:tagging_schema_version", issues)

    def test_tagger_retries_partial_message_rows(self) -> None:
        class LLM:
            def __init__(self) -> None:
                self.calls = 0
                self.prompts = []

            async def call(self, prompt: str):
                self.calls += 1
                self.prompts.append(prompt)
                results = (
                    [{"trial_idx": 1, "tags": ["request"], "frustration": 0}]
                    if self.calls == 1
                    else [
                        {"trial_idx": 1, "tags": ["request"], "frustration": 0},
                        {"trial_idx": 2, "tags": ["question"], "frustration": 0},
                    ]
                )
                return type("Response", (), {"content": json.dumps({"results": results})})()

        llm = LLM()
        with patch.object(tag_messages.asyncio, "sleep", AsyncMock()):
            rows = asyncio.run(
                tag_messages._ask(llm, "system", "user", {1, 2})
            )

        self.assertEqual(set(rows), {1, 2})
        self.assertEqual(llm.calls, 2)
        self.assertEqual(llm.prompts[0], "system\n\nuser")
        self.assertEqual(
            llm.prompts[1],
            "system\n\nuser\n\n"
            + tag_messages._TAG_RETRY_VALIDATION_FEEDBACK,
        )

    def test_tagger_retry_feedback_corrects_missing_base_without_local_repair(
        self,
    ) -> None:
        class LLM:
            def __init__(self) -> None:
                self.prompts = []

            async def call(self, prompt: str):
                self.prompts.append(prompt)
                row = (
                    {
                        "trial_idx": 1,
                        "tags": ["correction"],
                        "frustration": 0,
                    }
                    if len(self.prompts) == 1
                    else {
                        "trial_idx": 1,
                        "tags": ["request", "correction"],
                        "frustration": 0,
                    }
                )
                return type(
                    "Response",
                    (),
                    {"content": json.dumps({"results": [row]})},
                )()

        llm = LLM()
        with patch.object(tag_messages.asyncio, "sleep", AsyncMock()):
            rows = asyncio.run(
                tag_messages._ask(llm, "system", "user", {1})
            )

        self.assertEqual(llm.prompts[0], "system\n\nuser")
        self.assertNotIn("Validation correction", llm.prompts[0])
        self.assertIn("Validation correction", llm.prompts[1])
        self.assertIn("at least one base speech-act tag", llm.prompts[1])
        self.assertEqual(rows[1]["tags"], ["correction", "request"])

    def test_tagger_fails_closed_on_repeated_duplicate_rows(self) -> None:
        class LLM:
            async def call(self, _prompt: str):
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "results": [
                                    {
                                        "trial_idx": 1,
                                        "tags": ["request"],
                                        "frustration": 0,
                                    },
                                    {
                                        "trial_idx": 1,
                                        "tags": ["question"],
                                        "frustration": 0,
                                    },
                                ]
                            }
                        )
                    },
                )()

        with patch.object(tag_messages.asyncio, "sleep", AsyncMock()):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                asyncio.run(
                    tag_messages._ask(LLM(), "system", "user", {1, 2})
                )

    def test_tagger_retries_then_rejects_structurally_invalid_rows(self) -> None:
        invalid_rows = {
            "null_tags": {"trial_idx": 1, "tags": None, "frustration": 0},
            "non_list_tags": {
                "trial_idx": 1,
                "tags": "request",
                "frustration": 0,
            },
            "unknown_only": {
                "trial_idx": 1,
                "tags": ["not-a-real-tag"],
                "frustration": 0,
            },
            "duplicate_tags": {
                "trial_idx": 1,
                "tags": ["request", "request"],
                "frustration": 0,
            },
            "missing_base_act": {
                "trial_idx": 1,
                "tags": ["correction"],
                "frustration": 0,
            },
            "string_frustration": {
                "trial_idx": 1,
                "tags": ["request"],
                "frustration": "0",
            },
            "out_of_range_frustration": {
                "trial_idx": 1,
                "tags": ["request"],
                "frustration": 2,
            },
        }

        for case, invalid_row in invalid_rows.items():
            with self.subTest(case=case):
                class LLM:
                    def __init__(self) -> None:
                        self.calls = 0
                        self.prompts = []

                    async def call(self, prompt: str):
                        self.calls += 1
                        self.prompts.append(prompt)
                        return type(
                            "Response",
                            (),
                            {
                                "content": json.dumps(
                                    {"results": [invalid_row]}
                                )
                            },
                        )()

                llm = LLM()
                with patch.object(tag_messages.asyncio, "sleep", AsyncMock()):
                    with self.assertRaisesRegex(
                        ValueError, "structurally invalid"
                    ):
                        asyncio.run(
                            tag_messages._ask(
                                llm, "system", "user", {1}, attempts=3
                            )
                        )
                self.assertEqual(llm.calls, 3)
                self.assertEqual(llm.prompts[0], "system\n\nuser")
                self.assertNotIn("Validation correction", llm.prompts[0])
                self.assertIn("Validation correction", llm.prompts[1])
                self.assertIn("Validation correction", llm.prompts[2])

    def test_tagger_accepts_valid_multilabel_rows_without_loss(self) -> None:
        class LLM:
            def __init__(self) -> None:
                self.calls = 0

            async def call(self, _prompt: str):
                self.calls += 1
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "results": [
                                    {
                                        "trial_idx": 1,
                                        "tags": [
                                            "request",
                                            "correction",
                                            "nudge",
                                        ],
                                        "frustration": True,
                                    },
                                    {
                                        "trial_idx": 2,
                                        "tags": ["question", "context"],
                                        "frustration": 0,
                                    },
                                ]
                            }
                        )
                    },
                )()

        llm = LLM()
        rows = asyncio.run(
            tag_messages._ask(llm, "system", "user", {1, 2})
        )
        self.assertEqual(llm.calls, 1)
        self.assertEqual(
            rows,
            {
                1: {
                    "trial_idx": 1,
                    "tags": ["correction", "nudge", "request"],
                    "frustration": 1,
                },
                2: {
                    "trial_idx": 2,
                    "tags": ["context", "question"],
                    "frustration": 0,
                },
            },
        )

    def test_sidecar_is_versioned_resumable_and_model_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trials"
            trial = root / "task__abcdefg"
            trial.mkdir(parents=True)
            (trial / "result.json").write_text("{}")
            output = Path(tmp) / "judge_b.json"
            rows = [{"trial_idx": 1, "tags": ["request"], "frustration": 0}]
            with patch.object(tag_messages, "_make_llm", return_value=object()), patch.object(
                tag_messages, "tag_one", AsyncMock(return_value=rows)
            ):
                counts = asyncio.run(
                    tag_messages.run_sidecar_batch(
                        [root],
                        model=kg.PAPER_TAG_JUDGE_B,
                        workers=1,
                        output=output,
                    )
                )
            sidecar = json.loads(output.read_text())
            self.assertEqual(counts["ok"], 1)
            self.assertEqual(
                sidecar["message_tagging"],
                kg.tagging_provenance(kg.PAPER_TAG_JUDGE_B),
            )
            self.assertEqual(sidecar["trials"][trial.name], {"1": ["request"]})
            self.assertEqual(sidecar["trial_rows"][trial.name], rows)
            self.assertEqual(
                sidecar["trial_input_sha256"][trial.name],
                kg.tag_input_sha256([]),
            )
            self.assertEqual(sidecar["tag_transport"]["backend"], "litellm")

    def test_sidecar_retags_when_normalized_message_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trials"
            trial = root / "task__abcdefg"
            episode = trial / "agent" / "episode-1"
            episode.mkdir(parents=True)
            task_dir = Path(tmp) / "task"
            task_dir.mkdir()
            (task_dir / "instruction.md").write_text("initial instruction")
            (trial / "result.json").write_text("{}")
            (trial / "intent_coverage_verdict.json").write_text(
                json.dumps({"trial_dir": str(trial), "task_dir": str(task_dir)})
            )
            decision = episode / "user_decision.json"
            decision.write_text(
                json.dumps({"has_message": True, "content": "first request"})
            )
            output = Path(tmp) / "tags.json"
            tag_one = AsyncMock(
                return_value=[
                    {"trial_idx": 1, "tags": ["request"], "frustration": 0}
                ]
            )
            with patch.object(tag_messages, "_make_llm", return_value=object()), patch.object(
                tag_messages, "tag_one", tag_one
            ):
                first = asyncio.run(
                    tag_messages.run_sidecar_batch(
                        [root], model=kg.PAPER_TAG_JUDGE_B, workers=1, output=output
                    )
                )
                decision.write_text(
                    json.dumps({"has_message": True, "content": "changed request"})
                )
                second = asyncio.run(
                    tag_messages.run_sidecar_batch(
                        [root], model=kg.PAPER_TAG_JUDGE_B, workers=1, output=output
                    )
                )

            self.assertEqual(first["ok"], 1)
            self.assertEqual(second["ok"], 1)
            self.assertEqual(second["skip"], 0)
            self.assertEqual(tag_one.await_count, 2)

    def test_sidecar_projects_to_strict_single_tagger_verdict(self) -> None:
        trial_name = "task__abcdefg"
        rows = [
            {
                "trial_idx": 1,
                "tags": ["request", "nudge"],
                "frustration": 0,
            }
        ]
        sidecar = {
            "schema_version": kg.TAGGING_SCHEMA_VERSION,
            "message_tagging": kg.tagging_provenance(kg.CANONICAL_TAG_MODEL),
            "trials": {trial_name: {"1": ["request", "nudge"]}},
            "trial_rows": {trial_name: rows},
        }
        verdict = table2_metrics._sidecar_trial_verdict(sidecar, trial_name)
        self.assertEqual(
            kg.metrics_from_verdict(verdict, "single")["user_correction"],
            0.2,
        )
        self.assertEqual(
            kg.user_correction_provenance_issues(
                verdict,
                source="single",
                expected_tag_model=kg.CANONICAL_TAG_MODEL,
                expected_trial_indices={1},
            ),
            [],
        )

    def test_vertex_gateway_tagger_uses_native_generate_content(self) -> None:
        class Response:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "candidates": [
                        {"content": {"parts": [{"text": '{"results": []}'}]}}
                    ]
                }

        class Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.post_args = None
                self.closed = False

            async def post(self, url, **kwargs):
                self.post_args = (url, kwargs)
                return Response()

            async def aclose(self):
                self.closed = True

        client = Client()
        with patch.object(
            tag_messages,
            "_vertex_attribution_headers",
            return_value={name: "test-value" for name in tag_messages._VERTEX_ATTRIBUTION_HEADERS},
        ), patch.object(tag_messages.httpx, "AsyncClient", return_value=client):
            llm = tag_messages._VertexGatewayLLM(
                kg.CANONICAL_TAG_MODEL, temperature=0.0
            )
            response = asyncio.run(llm.call("tag these messages"))
            asyncio.run(llm.aclose())
        self.assertEqual(response.content, '{"results": []}')
        self.assertIn(
            "/v1beta1/projects/devai-mea-egeit/locations/global/",
            client.post_args[0],
        )
        self.assertIn("gemini-3.1-pro-preview:generateContent", client.post_args[0])
        self.assertEqual(
            client.post_args[1]["json"]["generationConfig"]["temperature"],
            0.0,
        )
        self.assertEqual(
            client.post_args[1]["json"]["generationConfig"]["responseMimeType"],
            "application/json",
        )
        self.assertEqual(
            client.post_args[1]["json"]["generationConfig"]["responseSchema"]["type"],
            "OBJECT",
        )
        self.assertTrue(client.closed)

    def test_vertex_gateway_requires_all_attribution_header_names(self) -> None:
        partial = (
            "X-Meta-AI-Gateway-Calling-Product: test\n"
            "X-Meta-AI-Gateway-App-Instance-Id: test"
        )
        with patch.dict(
            tag_messages.os.environ,
            {"ANTHROPIC_CUSTOM_HEADERS": partial},
            clear=False,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "X-Meta-AI-Gateway-Trace-Id"
            ):
                tag_messages._vertex_attribution_headers()

    def test_vertex_gateway_model_pin_and_safe_http_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned"):
            tag_messages._VertexGatewayLLM("gemini/wrong-model", 0.0)

        class Response:
            status_code = 403
            text = "must-not-leak-secret-or-metadata"

        class Client:
            async def post(self, _url, **_kwargs):
                return Response()

            async def aclose(self):
                return None

        with patch.object(
            tag_messages,
            "_vertex_attribution_headers",
            return_value={name: "test-value" for name in tag_messages._VERTEX_ATTRIBUTION_HEADERS},
        ), patch.object(tag_messages.httpx, "AsyncClient", return_value=Client()):
            llm = tag_messages._VertexGatewayLLM(
                kg.CANONICAL_TAG_MODEL, temperature=0.0
            )
            with self.assertRaisesRegex(RuntimeError, "Vertex gateway HTTP 403") as cm:
                asyncio.run(llm.call("tag these messages"))
        self.assertNotIn("must-not-leak", str(cm.exception))

    def test_main_evaluator_propagates_vertex_backend(self) -> None:
        with patch.object(eval_run_eval, "_run_subprocess", return_value=0) as run:
            rc = eval_run_eval.run_step_tag_messages(
                [Path("trials/opus_k2")],
                model=kg.CANONICAL_TAG_MODEL,
                workers=7,
                force=False,
                backend=tag_messages.VERTEX_GATEWAY_BACKEND,
            )
        self.assertEqual(rc, 0)
        command = run.call_args.args[0]
        self.assertEqual(
            command[command.index("--backend") + 1],
            tag_messages.VERTEX_GATEWAY_BACKEND,
        )

    def test_private_vertex_wrapper_loads_only_allowlisted_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "attribution"
            source.write_text(
                "X-Meta-AI-Gateway-Calling-Product: test-product\n"
                "X-Meta-AI-Gateway-App-Instance-Id: test-instance\n"
                "X-Meta-AI-Gateway-Trace-Id: test-trace\n"
                "Authorization: must-not-be-loaded\n"
            )
            source.chmod(0o600)
            loaded = run_vertex_tagger.load_attribution_source(source)
            self.assertEqual(
                set(loaded), tag_messages._VERTEX_ATTRIBUTION_HEADERS
            )
            source.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "group/world"):
                run_vertex_tagger.load_attribution_source(source)

    def test_private_vertex_wrapper_forces_backend_without_argv_values(self) -> None:
        forwarded = [
            "--",
            "--trials-root",
            "trials/example",
            "--model",
            kg.CANONICAL_TAG_MODEL,
        ]
        with patch.object(
            run_vertex_tagger, "_install_attribution"
        ) as install, patch.object(
            tag_messages, "main", return_value=0
        ) as tag_main, patch.object(
            run_vertex_tagger.sys, "argv", ["wrapper"]
        ):
            self.assertEqual(run_vertex_tagger.main(forwarded), 0)
            install.assert_called_once_with(None)
            tag_main.assert_called_once_with()
            self.assertEqual(
                run_vertex_tagger.sys.argv[1:3],
                ["--backend", tag_messages.VERTEX_GATEWAY_BACKEND],
            )


class ThreewayMetricAndLauncherTests(unittest.TestCase):
    def test_metric_selects_threeway_rows_not_cached_single_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            verdict = {
                "trial_msg_tags": [{"trial_idx": 1, "tags": ["request"]}],
                "user_correction": 99,
                "trial_msg_tags_3way": [
                    {"trial_idx": 1, "tags": ["request", "correction"]}
                ],
            }
            (trial / "intent_coverage_verdict.json").write_text(json.dumps(verdict))
            self.assertEqual(table2_metrics.get_user_correction(trial, "single"), 0.0)
            self.assertEqual(table2_metrics.get_user_correction(trial, "threeway"), 1.0)

    def test_threeway_strict_provenance_rejects_wrong_model_or_missing_vote(self) -> None:
        provenance = {
            "schema_version": kg.THREEWAY_SCHEMA_VERSION,
            "source_field": "trial_msg_tags_3way",
            "judge_a": kg.tagging_provenance(kg.PAPER_TAG_JUDGE_A),
            "judge_b": kg.tagging_provenance("anthropic/wrong-model"),
            "arbiter": {
                "model": kg.PAPER_TAG_ARBITER,
                "temperature": None,
                "prompt_sha256": kg.TAG_PROMPT_SHA256,
            },
            "judge_b_sidecar_sha256": "a" * 64,
            "disputed_trial_indices": [1],
            "arbiter_votes": {},
            "legacy_unverified": False,
        }
        issues = kg.user_correction_provenance_issues(
            {
                "trial_msg_tags_3way": [
                    {"trial_idx": 1, "tags": ["request"]}
                ],
                "message_tagging_3way": provenance,
            },
            source="threeway",
        )
        self.assertTrue(any("judge_b_tagging_model" in issue for issue in issues))
        self.assertIn("missing_arbiter_vote:1", issues)

    def test_launcher_explicitly_plumbs_optional_threeway_models(self) -> None:
        plan = {
            "trials_root": "trials/canonical_full109",
            "tasks_root": "tasks",
            "replicates": [1, 2],
            "user_correction": {
                "source": "threeway",
                "judge_a_model": kg.PAPER_TAG_JUDGE_A,
                "judge_b_model": kg.PAPER_TAG_JUDGE_B,
                "arbiter_model": kg.PAPER_TAG_ARBITER,
                "arbiter_proxy": "http://127.0.0.1:4220/v1",
                "arbiter_auto_start": True,
            },
        }
        with patch.object(launch, "_run", return_value=0) as run:
            launch.stage_judge(plan, {"opus": {}}, "results", "sandoq", True)
        command = run.call_args.args[0]
        for flag, value in (
            ("--user-correction-source", "threeway"),
            ("--tag-model", kg.PAPER_TAG_JUDGE_A),
            ("--tag-judge-b-model", kg.PAPER_TAG_JUDGE_B),
            ("--tag-arbiter-model", kg.PAPER_TAG_ARBITER),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--tag-arbiter-auto-start", command)

    def test_canonical_plan_uses_released_single_gemini_tagger(self) -> None:
        plan = json.loads((REPO / "canonical_full109.json").read_text())
        self.assertEqual(plan["user_correction"]["source"], "single")
        self.assertEqual(
            plan["user_correction"]["tag_model"], kg.CANONICAL_TAG_MODEL
        )
        with patch.object(launch, "_run", return_value=0) as run:
            launch.stage_judge(
                plan, {"opencode_opus48": {}}, "results", "sandoq", True
            )
        command = run.call_args.args[0]
        self.assertEqual(
            command[command.index("--user-correction-source") + 1], "single"
        )
        self.assertEqual(
            command[command.index("--tag-model") + 1], kg.CANONICAL_TAG_MODEL
        )
        self.assertNotIn("--tag-arbiter-auto-start", command)


class ArbiterProxyLifecycleTests(unittest.TestCase):
    def test_health_requires_authenticated_bundled_identity(self) -> None:
        class Response:
            status_code = 200

            def __init__(self, payload: dict):
                self.payload = payload

            def json(self) -> dict:
                return self.payload

        token = "health-token-" + "h" * 40
        valid = {
            "ok": True,
            "service": "swe-together-oauth-proxy",
            "client_auth": True,
        }
        with patch.object(
            eval_run_eval.httpx, "get", return_value=Response(valid)
        ) as get:
            self.assertTrue(
                eval_run_eval._arbiter_is_healthy(
                    "http://127.0.0.1:4220/health", token
                )
            )
        self.assertEqual(
            get.call_args.kwargs["headers"],
            {"Authorization": f"Bearer {token}"},
        )
        with patch.object(
            eval_run_eval.httpx,
            "get",
            return_value=Response({**valid, "service": "impostor"}),
        ):
            self.assertFalse(
                eval_run_eval._arbiter_is_healthy(
                    "http://127.0.0.1:4220/health", token
                )
            )

    def test_occupied_loopback_port_fails_without_probe_or_secret(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(
                eval_run_eval.httpx, "get"
            ) as health, patch.object(
                eval_run_eval.secrets, "token_urlsafe"
            ) as token_urlsafe, patch.object(
                eval_run_eval.subprocess, "Popen"
            ) as popen:
                with self.assertRaisesRegex(RuntimeError, "refusing to reuse"):
                    with eval_run_eval._managed_arbiter_proxy(
                        proxy_url=f"http://127.0.0.1:{port}/v1",
                        auto_start=True,
                        auth_json=Path(tmp) / "missing-auth.json",
                        log_path=Path(tmp) / "proxy.log",
                    ):
                        pass
            health.assert_not_called()
            token_urlsafe.assert_not_called()
            popen.assert_not_called()
        finally:
            listener.close()

    def test_started_proxy_is_terminated_and_waited(self) -> None:
        class FakeProcess:
            pid = 123

            def __init__(self) -> None:
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True
                return 0

            def kill(self) -> None:
                raise AssertionError("healthy process should terminate cleanly")

        class FakeListener:
            def __init__(self) -> None:
                self.closed = False

            def fileno(self) -> int:
                return 77

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            auth = root / "auth.json"
            auth.write_text("{}")
            auth.chmod(0o600)
            process = FakeProcess()
            listener = FakeListener()
            captured: dict[str, object] = {}

            def start(command, **_kwargs):
                token_path = Path(command[command.index("--client-auth-file") + 1])
                captured["path"] = token_path
                captured["token"] = token_path.read_text()
                captured["file_mode"] = stat.S_IMODE(token_path.stat().st_mode)
                captured["dir_mode"] = stat.S_IMODE(token_path.parent.stat().st_mode)
                return process

            def healthy(_url: str, token: str) -> bool:
                self.assertEqual(token, captured["token"])
                self.assertFalse(listener.closed)
                return True

            with patch.object(
                eval_run_eval, "_reserve_arbiter_listener", return_value=listener
            ), patch.object(
                eval_run_eval, "_arbiter_is_healthy", side_effect=healthy
            ), patch.object(
                eval_run_eval.subprocess, "Popen", side_effect=start
            ) as popen:
                with eval_run_eval._managed_arbiter_proxy(
                    proxy_url="http://localhost:4220/v1",
                    auto_start=True,
                    auth_json=auth,
                    log_path=root / "proxy.log",
                ) as client_token:
                    self.assertFalse(process.terminated)
                    self.assertFalse(listener.closed)
                    self.assertEqual(client_token, captured["token"])
                    self.assertGreaterEqual(len(client_token), 48)
            self.assertTrue(process.terminated)
            self.assertTrue(process.waited)
            self.assertTrue(listener.closed)
            self.assertEqual(captured["file_mode"], 0o600)
            self.assertEqual(captured["dir_mode"], 0o700)
            self.assertFalse(Path(captured["path"]).exists())
            command = popen.call_args.args[0]
            self.assertTrue(any(str(part).endswith("proxies/oauth_proxy.py") for part in command))
            self.assertIn(str(auth.resolve()), command)
            self.assertNotIn(captured["token"], command)
            self.assertEqual(popen.call_args.kwargs["pass_fds"], (77,))
            self.assertNotIn(str(captured["token"]), (root / "proxy.log").read_text())

    def test_reserved_listener_closes_on_pre_spawn_failure(self) -> None:
        class FakeListener:
            closed = False

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            auth = root / "auth.json"
            auth.write_text("{}")
            auth.chmod(0o600)
            listener = FakeListener()
            with patch.object(
                eval_run_eval, "_reserve_arbiter_listener", return_value=listener
            ), patch.object(
                eval_run_eval.secrets,
                "token_urlsafe",
                side_effect=RuntimeError("entropy unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "entropy unavailable"):
                    with eval_run_eval._managed_arbiter_proxy(
                        proxy_url="http://127.0.0.1:4220/v1",
                        auto_start=True,
                        auth_json=auth,
                        log_path=root / "proxy.log",
                    ):
                        pass
            self.assertTrue(listener.closed)

    def test_auto_start_rejects_external_arbiter_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                with eval_run_eval._managed_arbiter_proxy(
                    proxy_url="https://arbiter.example/v1",
                    auto_start=True,
                    auth_json=None,
                    log_path=Path(tmp) / "proxy.log",
                ):
                    pass

    def test_external_arbiter_is_allowed_when_auto_start_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            eval_run_eval.subprocess, "Popen"
        ) as popen, patch.object(
            eval_run_eval, "_reserve_arbiter_listener"
        ) as reserve, patch.object(
            eval_run_eval.secrets, "token_urlsafe"
        ) as token_urlsafe:
            with eval_run_eval._managed_arbiter_proxy(
                proxy_url="https://arbiter.example/v1",
                auto_start=False,
                auth_json=None,
                log_path=Path(tmp) / "proxy.log",
            ) as client_token:
                self.assertIsNone(client_token)
        popen.assert_not_called()
        reserve.assert_not_called()
        token_urlsafe.assert_not_called()

    def test_manual_external_bearer_uses_private_file_and_safe_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "arbiter-token"
            token = "manual-arbiter-token-" + "m" * 40
            token_path.write_text(token)
            token_path.chmod(0o600)
            self.assertEqual(
                eval_run_eval._read_external_arbiter_token(
                    "https://arbiter.example/v1", token_path
                ),
                token,
            )
            self.assertEqual(
                eval_run_eval._read_external_arbiter_token(
                    "http://127.0.0.1:4220/v1", token_path
                ),
                token,
            )
            with self.assertRaisesRegex(RuntimeError, "HTTPS"):
                eval_run_eval._read_external_arbiter_token(
                    "http://arbiter.example/v1", token_path
                )
            with self.assertRaisesRegex(RuntimeError, "embedded"):
                eval_run_eval._read_external_arbiter_token(
                    "https://user:password@arbiter.example/v1", token_path
                )

    def test_authenticated_arbiter_sends_bearer_without_url_or_body_leak(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "results": [
                                            {"trial_idx": 1, "tags": ["correction"]}
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, url: str, **kwargs):
                captured["url"] = url
                captured.update(kwargs)
                return FakeResponse()

        token = "unit-arbiter-bearer-" + "x" * 40
        with patch.object(
            eval_run_eval.httpx, "AsyncClient", return_value=FakeClient()
        ):
            tags = asyncio.run(
                eval_run_eval._BearerArbiterLLM(
                    "http://127.0.0.1:4220/v1", "gpt-5.5", token
                ).tag([{"trial_idx": 1, "text": "please fix it"}])
            )
        self.assertEqual(tags, {1: ["correction"]})
        self.assertEqual(
            captured["headers"], {"Authorization": f"Bearer {token}"}
        )
        self.assertNotIn(token, str(captured["url"]))
        self.assertNotIn(token, json.dumps(captured["json"]))


if __name__ == "__main__":
    unittest.main()
