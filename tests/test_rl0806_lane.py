from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LANE_DIR = REPO / "rl0806_lane"
sys.path.insert(0, str(LANE_DIR))

from lane_config import (  # noqa: E402
    CANONICAL_TASKS_SHA256,
    SHARD_MANIFEST_FILE_SHA256,
    ProtocolError,
    atomic_write_protocol,
    build_protocol,
    validate_checkpoint,
    validate_protocol,
)


def make_checkpoint(
    tmp_path: Path, *, run: str = "run_0806_v1", step: int = 987654
) -> Path:
    checkpoint = tmp_path / run / "weights" / f"step_{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "STABLE").touch()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
            }
        )
    )
    (checkpoint / "tokenizer.json").write_text("{}\n")
    shard = "model-00001-of-00001.safetensors"
    (checkpoint / shard).write_bytes(b"test-only-shard")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": shard}})
    )
    return checkpoint.resolve()


def read(name: str) -> str:
    return (LANE_DIR / name).read_text()


def test_protocol_binds_lane_checkpoint_and_canonical_benchmark(
    tmp_path: Path,
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    protocol = build_protocol(
        lane="run_0806_v1_step987654",
        checkpoint=checkpoint,
        relay_host="10.146.5.90",
        relay_port=31038,
    )

    assert protocol["checkpoint"] == str(checkpoint)
    assert protocol["served_model"] == "run_0806_v1_step987654"
    assert protocol["action_model"] == "openai/run_0806_v1_step987654"
    assert protocol["benchmark_tasks"] == 109
    assert protocol["replicates"] == 2
    assert protocol["action_agent"] == "opencode@1.15.13"
    assert protocol["correctness_judge"] == "anthropic/claude-opus-4-6"
    assert protocol["resources"]["service_partition"] == "h200"
    assert protocol["resources"]["service_qos"] == "h200_ram_high"
    assert protocol["resources"]["service_gpu"] == "1xH200 each"
    assert protocol["canonical_tasks_sha256"] == CANONICAL_TASKS_SHA256
    assert protocol["shard_manifest_file_sha256"] == (
        SHARD_MANIFEST_FILE_SHA256
    )
    manifest = Path(protocol["shard_manifest"])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        SHARD_MANIFEST_FILE_SHA256
    )
    assert protocol["trials_root"].endswith(
        "/trials/run_0806_v1_step987654_k2"
    )


def test_protocol_round_trip_rejects_identity_tampering(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    protocol = build_protocol(
        lane="run_0806_v1_step987654",
        checkpoint=checkpoint,
        relay_host="10.146.5.90",
        relay_port=31038,
    )
    protocol_path = tmp_path / "protocol.json"
    atomic_write_protocol(protocol_path, protocol)
    assert validate_protocol(protocol_path)["action_model"] == (
        "openai/run_0806_v1_step987654"
    )

    protocol["action_model"] = "openai/wrong-model"
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ProtocolError, match="action_model"):
        validate_protocol(protocol_path)


def test_checkpoint_rejects_cross_run_and_unindexed_or_symlink_shards(
    tmp_path: Path,
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    with pytest.raises(ProtocolError, match="under run_0806_v2"):
        validate_checkpoint("run_0806_v2_step987654", checkpoint)

    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.unlink()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    shard.symlink_to(outside)
    with pytest.raises(ProtocolError, match="non-symlink"):
        validate_checkpoint("run_0806_v1_step987654", checkpoint)

    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"bad": ["not", "a", "filename"]}})
    )
    with pytest.raises(ProtocolError, match="unsafe shard name"):
        validate_checkpoint("run_0806_v1_step987654", checkpoint)


def test_protocol_rejects_reserved_relay_port(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    with pytest.raises(ProtocolError, match="reserved"):
        build_protocol(
            lane="run_0806_v1_step987654",
            checkpoint=checkpoint,
            relay_host="10.146.5.90",
            relay_port=48837,
        )


def test_protocol_rejects_host_ephemeral_relay_port(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    with pytest.raises(ProtocolError, match="ephemeral range"):
        build_protocol(
            lane="run_0806_v1_step987654",
            checkpoint=checkpoint,
            relay_host="10.146.5.90",
            relay_port=48838,
        )


def test_launcher_dry_run_is_write_free_and_uses_isolated_roots(
    tmp_path: Path,
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    lane = "run_0806_v1_step987654"
    run_root = LANE_DIR / "runs" / lane
    trials_root = REPO / "trials" / f"{lane}_k2"
    assert not run_root.exists()
    assert not trials_root.exists()

    environment = os.environ.copy()
    environment["http_proxy"] = "http://127.0.0.1:41683"
    completed = subprocess.run(
        [
            "bash",
            str(LANE_DIR / "launch_lane.sh"),
            "--lane",
            lane,
            "--checkpoint",
            str(checkpoint),
            "--relay-port",
            "31038",
            "--dry-run",
        ],
        cwd=REPO,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "RL0806_DRY_RUN_OK" in completed.stdout
    assert f"run_root={run_root}" in completed.stdout
    assert f"trials_root={trials_root}" in completed.stdout
    assert completed.stdout.count("serve_step575.sbatch") == 4
    assert completed.stdout.count("--partition=h200") == 4
    assert completed.stdout.count("--qos=h200_ram_high") == 4
    assert completed.stdout.count("run_full_shard.sbatch") == 4
    assert "rolling dependency: after:<all-four-shards-start>" in completed.stdout
    assert "final dependency: afterok:<all-four-shards>:<rolling-watcher>" in (
        completed.stdout
    )
    assert not run_root.exists()
    assert not trials_root.exists()


def test_generic_scripts_keep_mutable_state_lane_local() -> None:
    launch = read("launch_lane.sh")
    common = read("common.sh")
    rolling = read("rolling_judge_watch.sbatch")
    finalizer = read("finalize_full_k2.sbatch")

    assert "sushi_lane/serve_step575.sbatch" in launch
    assert "sushi_lane/tool_call_smoke.sbatch" in launch
    assert "sushi_lane/run_full_shard.sbatch" in launch
    assert "BENCH_FORCE_ARCHIVE_LIST=/dev/null" in launch
    for source in (launch, common, rolling, finalizer):
        assert "trials/sushi" not in source
        assert "sushi_lane/state/" not in source
        assert "sushi_lane/logs/" not in source
        assert "sushi_lane/artifacts/" not in source

    assert "chmod 0600 \"$STATE_DIR/relay.pid\"" not in launch
    assert launch.index("trap cleanup EXIT") < launch.index(
        'setsid "$PY" "$REPO/sushi_lane/secure_tcp_relay.py"'
    )


def test_rolling_and_finalizer_enforce_safe_snapshot_ordering() -> None:
    rolling = read("rolling_judge_watch.sbatch")
    finalizer = read("finalize_full_k2.sbatch")

    assert "--completed-only" in rolling
    assert "--valid-completed-only" in rolling
    assert "shard_${shard}.lock" in rolling
    assert "sanitize_traces.py" not in rolling

    producer_locks = finalizer.index("for shard in 0 1 2 3")
    action_validation = finalizer.index("validate_action_cohort.py")
    final_missing_check = finalizer.rindex("build_missing_judge_plan.py")
    sanitizer = finalizer.index("scripts/sanitize_traces.py")
    tagger = finalizer.index("scripts/run_vertex_tagger.py")
    metrics = finalizer.index("eval/table2_metrics.py")
    assert (
        producer_locks
        < action_validation
        < final_missing_check
        < sanitizer
        < tagger
        < metrics
    )
    assert "--expected-judge-model anthropic/claude-opus-4-6" in finalizer
    assert "--expected-tag-model gemini/gemini-3.1-pro-preview" in finalizer
    assert "--k 2 --expected-tasks 109 --strict" in finalizer
    assert finalizer.count("--workers 16") == 2
    assert 'aggregates.get("n_trials") == 218' in finalizer
