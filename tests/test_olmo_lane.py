from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sushi_lane.sampling_proxy import PROFILE  # noqa: E402


LANE = REPO / "olmo_lane"
EXPECTED_CHECKPOINT = (
    "/checkpoint/ram/chuanyang/autodata/run_0716_single_turn/weights/step_500"
)
EXPECTED_SERVED_MODEL = "Olmo-0716-step500"
EXPECTED_ACTION_MODEL = f"openai/{EXPECTED_SERVED_MODEL}"


def read(name: str) -> str:
    return (LANE / name).read_text()


def test_olmo_protocol_pins_exact_checkpoint_identity_and_table2_stack() -> None:
    protocol = json.loads(read("protocol.json"))

    assert protocol["lane"] == "Olmo_0716_step500"
    assert protocol["checkpoint"] == EXPECTED_CHECKPOINT
    assert protocol["checkpoint_architecture"] == "Qwen3_5ForConditionalGeneration"
    assert protocol["served_model"] == EXPECTED_SERVED_MODEL
    assert protocol["action_model"] == EXPECTED_ACTION_MODEL
    assert protocol["action_agent"] == "opencode@1.15.13"
    assert protocol["environment_import_path"] == "podman_env:PodmanEnvironment"
    assert protocol["benchmark_tasks"] == 109
    assert protocol["replicates"] == 2
    assert protocol["agent_timeout_sec"] == 4800
    assert protocol["correctness_judge"] == "anthropic/claude-opus-4-6"
    assert protocol["message_tagger"] == {
        "model": "gemini/gemini-3.1-pro-preview",
        "backend": "vertex-gateway",
    }


def test_olmo_sampling_pin_matches_the_verified_loopback_proxy() -> None:
    protocol = json.loads(read("protocol.json"))

    assert protocol["sampling"] == {
        key: value for key, value in PROFILE.items() if key != "chat_template_kwargs"
    }
    assert protocol["chat_template_kwargs"] == PROFILE["chat_template_kwargs"]
    smoke = read("tool_call_smoke.py")
    for literal in (
        '"temperature": 0.6',
        '"top_p": 0.95',
        '"top_k": 20',
        '"presence_penalty": 1.5',
        '"repetition_penalty": 1.05',
        '"max_tokens": 32768',
        '"chat_template_kwargs": {"enable_thinking": False}',
    ):
        assert literal in smoke


def test_olmo_action_and_service_provenance_are_exact() -> None:
    service = read("serve_step500.sbatch")
    common = read("run_action_common.sh")
    pilot = read("run_pilot.sbatch")
    shard = read("run_full_shard.sbatch")
    finalizer = read("finalize_full_k2.sbatch")
    validator = (REPO / "sushi_lane" / "validate_action_cohort.py").read_text()

    assert f"readonly CHECKPOINT={EXPECTED_CHECKPOINT}" in service
    assert f"readonly SERVED_MODEL={EXPECTED_SERVED_MODEL}" in service
    assert 'config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]' in service
    assert 'config.get("model_type") != "qwen3_5"' in service
    assert "--default-chat-template-kwargs '{\"enable_thinking\": false}'" in service
    assert "--enable-auto-tool-choice" in service
    assert "--tool-call-parser qwen3_coder" in service
    assert "--model openai/Olmo-0716-step500" in common
    assert "--expected-model" in validator
    for launcher in (pilot, shard, finalizer):
        assert "--expected-model openai/Olmo-0716-step500" in launcher


def test_olmo_mutable_paths_are_lane_specific() -> None:
    scripts = [
        "serve_step500.sbatch",
        "tool_call_smoke.sbatch",
        "run_action_common.sh",
        "run_pilot.sbatch",
        "run_full_shard.sbatch",
        "rolling_judge.sbatch",
        "finalize_full_k2.sbatch",
    ]
    for name in scripts:
        source = read(name)
        assert "trials/sushi" not in source
        assert "sushi_lane/logs" not in source
        assert "sushi_lane/artifacts" not in source
        assert "sushi_lane/state/shard" not in source
        assert "sushi_lane/state/endpoint" not in source
        if name.endswith(".sbatch"):
            assert "/olmo_lane/logs/" in source

    common = read("run_action_common.sh")
    rolling = read("rolling_judge.sbatch")
    finalizer = read("finalize_full_k2.sbatch")
    for source in (common, rolling, finalizer):
        assert 'olmo_lane/state/egress_relay_clients.json' in source
        assert "readonly RELAY_PORT=48837" in source
        assert 'sushi_lane/state/egress_relay_clients.json' not in source


def test_olmo_shards_are_disjoint_and_archive_only_owned_cells() -> None:
    shard = read("run_full_shard.sbatch")

    assert "sushi_lane/shard_manifest_k4.json" in shard
    assert "sushi_lane/shard_tasks.py" in shard
    assert "--manifest \"$SHARD_MANIFEST\" --shard-index \"$SHARD_INDEX\"" in shard
    assert "--archive-root \"$INVALID_ARCHIVE\" --archive-incomplete" in shard
    assert "--scope-expected-tasks" in shard
    assert "olmo_lane/state/shard_${SHARD_INDEX}.lock" in shard
    assert "trials/olmo_0716_step500_k2_invalid_shard${SHARD_INDEX}" in shard
    assert "sanitize_traces.py" not in shard
    assert "sanitize_traces.py" not in read("run_action_common.sh")


def test_olmo_finalizer_excludes_producers_before_sanitizing() -> None:
    finalizer = read("finalize_full_k2.sbatch")

    producer_locks = finalizer.index("for shard in 0 1 2 3")
    action_validation = finalizer.index("validate_action_cohort.py")
    final_missing_check = finalizer.rindex("build_missing_judge_plan.py")
    sanitizer = finalizer.index("scripts/sanitize_traces.py")
    tagger = finalizer.index("scripts/run_vertex_tagger.py")
    metrics = finalizer.index("eval/table2_metrics.py")
    assert producer_locks < action_validation < final_missing_check < sanitizer < tagger < metrics
    assert 'if ! flock -n "$lock_fd"' in finalizer
    assert '--expected-judge-model anthropic/claude-opus-4-6' in finalizer
    assert '--expected-tag-model gemini/gemini-3.1-pro-preview' in finalizer
    assert "--k 2 --expected-tasks 109 --strict" in finalizer
    assert "--model 'Olmo_0716_step500'" in finalizer


def test_olmo_rolling_judge_only_snapshots_completed_cells() -> None:
    rolling = read("rolling_judge.sbatch")

    assert "--completed-only" in rolling
    assert "--valid-completed-only" in rolling
    assert "judge_verdict_opus46.json" in rolling
    assert "anthropic/claude-opus-4-6" in rolling
    assert "sanitize_traces.py" not in rolling
