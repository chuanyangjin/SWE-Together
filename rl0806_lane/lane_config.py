#!/usr/bin/env python3
"""Build and validate immutable run_0806 SWE-Together lane protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
LANE_RE = re.compile(r"^(run_0806_v[12])_step([0-9]+)$")
RUN_RE = re.compile(r"^run_0806_v[12]$")
RESERVED_RELAY_PORTS = {48835, 48836, 48837}
EPHEMERAL_RELAY_PORT_MIN = 32768
EPHEMERAL_RELAY_PORT_MAX = 60999
EXPECTED_ARCHITECTURES = ["Qwen3_5ForConditionalGeneration"]
EXPECTED_MODEL_TYPE = "qwen3_5"
SHARD_MANIFEST = REPO / "sushi_lane" / "shard_manifest_k4.json"
CANONICAL_TASKS_SHA256 = (
    "92260b84f512d13b8e18d67deafb50570066f126f841184210468e4682e73d61"
)
SHARD_MANIFEST_FILE_SHA256 = (
    "4003671192b741e42f09c2cf3ac6eacdd444254246474d1a317a07f8506613a6"
)


class ProtocolError(RuntimeError):
    """Raised when a lane protocol or checkpoint identity is unsafe."""


def _regular_non_symlink(path: Path, description: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProtocolError(f"{description} is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProtocolError(f"{description} must be a regular non-symlink file: {path}")


def validate_lane_name(lane: str) -> tuple[str, int]:
    match = LANE_RE.fullmatch(lane)
    if not match:
        raise ProtocolError(
            "lane must be exactly run_0806_v1_stepN or run_0806_v2_stepN"
        )
    return match.group(1), int(match.group(2))


def validate_checkpoint(lane: str, raw_checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    run_name, lane_step = validate_lane_name(lane)
    if not raw_checkpoint.is_absolute():
        raise ProtocolError("checkpoint must be an absolute path")
    try:
        checkpoint = raw_checkpoint.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"checkpoint is unavailable: {raw_checkpoint}") from exc
    if checkpoint != raw_checkpoint:
        raise ProtocolError("checkpoint path must already be canonical (no symlink aliases)")
    if "," in str(checkpoint) or "\n" in str(checkpoint):
        raise ProtocolError("checkpoint path contains an unsafe Slurm export character")
    if checkpoint.name != f"step_{lane_step}":
        raise ProtocolError(
            f"lane step {lane_step} does not match checkpoint directory {checkpoint.name!r}"
        )
    if checkpoint.parent.name != "weights" or checkpoint.parent.parent.name != run_name:
        raise ProtocolError(
            f"checkpoint must be under {run_name}/weights/step_{lane_step}"
        )
    if not RUN_RE.fullmatch(checkpoint.parent.parent.name):
        raise ProtocolError("checkpoint run is outside the run_0806_v1/v2 scope")

    stable = checkpoint / "STABLE"
    config_path = checkpoint / "config.json"
    tokenizer = checkpoint / "tokenizer.json"
    index = checkpoint / "model.safetensors.index.json"
    for path, description in (
        (stable, "STABLE marker"),
        (config_path, "checkpoint config"),
        (tokenizer, "tokenizer"),
        (index, "safetensors index"),
    ):
        _regular_non_symlink(path, description)
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("checkpoint config is not valid JSON") from exc
    if config.get("architectures") != EXPECTED_ARCHITECTURES:
        raise ProtocolError("checkpoint architecture is not Qwen3.5")
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ProtocolError("checkpoint model_type is not qwen3_5")

    try:
        index_payload = json.loads(index.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("checkpoint safetensors index is not valid JSON") from exc
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ProtocolError("checkpoint safetensors index has no weight map")
    raw_shard_names = list(weight_map.values())
    if not all(
        isinstance(name, str)
        and re.fullmatch(r"model-[A-Za-z0-9_.-]+\.safetensors", name)
        for name in raw_shard_names
    ):
        raise ProtocolError("checkpoint safetensors index contains an unsafe shard name")
    shard_names = set(raw_shard_names)
    actual_shards = {path.name for path in checkpoint.glob("model-*.safetensors")}
    if actual_shards != shard_names:
        raise ProtocolError("checkpoint safetensor shards do not exactly match their index")
    for name in sorted(shard_names):
        _regular_non_symlink(checkpoint / name, "model safetensor shard")
    return checkpoint, config


def _validate_relay(relay_host: str, relay_port: int) -> None:
    if relay_host != "10.146.5.90":
        raise ProtocolError(
            "relay host must be the current chuanyang-login-0 address 10.146.5.90"
        )
    if not 1024 <= relay_port <= 65535:
        raise ProtocolError("relay port must be between 1024 and 65535")
    if relay_port in RESERVED_RELAY_PORTS:
        raise ProtocolError(
            f"relay port {relay_port} is reserved by an existing benchmark lane"
        )
    if EPHEMERAL_RELAY_PORT_MIN <= relay_port <= EPHEMERAL_RELAY_PORT_MAX:
        raise ProtocolError(
            "relay port is inside the host ephemeral range 32768-60999"
        )


def _manifest_identity() -> tuple[str, str]:
    try:
        raw_manifest = SHARD_MANIFEST.read_bytes()
        manifest = json.loads(raw_manifest)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("canonical four-shard manifest is unavailable") from exc
    file_digest = hashlib.sha256(raw_manifest).hexdigest()
    if (
        manifest.get("schema_version") != 1
        or manifest.get("method") != "sorted_round_robin_index_modulo"
        or manifest.get("shard_count") != 4
        or manifest.get("expected_tasks") != 109
        or manifest.get("canonical_tasks_sha256") != CANONICAL_TASKS_SHA256
        or file_digest != SHARD_MANIFEST_FILE_SHA256
    ):
        raise ProtocolError("canonical four-shard manifest identity changed")
    shards = manifest.get("shards")
    if (
        not isinstance(shards, list)
        or [row.get("index") for row in shards if isinstance(row, dict)]
        != [0, 1, 2, 3]
        or [row.get("task_count") for row in shards if isinstance(row, dict)]
        != [28, 27, 27, 27]
    ):
        raise ProtocolError("canonical four-shard manifest shape changed")
    return CANONICAL_TASKS_SHA256, file_digest


def build_protocol(
    *,
    lane: str,
    checkpoint: Path,
    relay_host: str,
    relay_port: int,
) -> dict[str, Any]:
    checkpoint, _config = validate_checkpoint(lane, checkpoint)
    _validate_relay(relay_host, relay_port)
    canonical_digest, manifest_digest = _manifest_identity()
    run_root = REPO / "rl0806_lane" / "runs" / lane
    trials_root = REPO / "trials" / f"{lane}_k2"
    state_dir = run_root / "state"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lane": lane,
        "checkpoint": str(checkpoint),
        "checkpoint_architecture": EXPECTED_ARCHITECTURES[0],
        "checkpoint_model_type": EXPECTED_MODEL_TYPE,
        "served_model": lane,
        "action_model": f"openai/{lane}",
        "result_label": lane,
        "action_agent": "opencode@1.15.13",
        "environment_import_path": "podman_env:PodmanEnvironment",
        "benchmark_tasks": 109,
        "replicates": 2,
        "agent_timeout_sec": 4800,
        "context_limit": 131072,
        "output_limit": 32768,
        "chat_template_kwargs": {"enable_thinking": False},
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.05,
            "max_tokens": 32768,
        },
        "user_simulator": {
            "model": "anthropic/claude-opus-4-8",
            "requested_temperature": 0.5,
            "effective_temperature": None,
            "context_chars": 3000,
            "call_on_completion": True,
        },
        "correctness_judge": "anthropic/claude-opus-4-6",
        "message_tagger": {
            "model": "gemini/gemini-3.1-pro-preview",
            "backend": "vertex-gateway",
        },
        "run_root": str(run_root),
        "state_dir": str(state_dir),
        "logs_dir": str(run_root / "logs"),
        "artifacts_dir": str(run_root / "artifacts"),
        "trials_root": str(trials_root),
        "invalid_archive_base": str(REPO / "trials" / f"{lane}_k2_invalid_shard"),
        "shard_manifest": str(SHARD_MANIFEST),
        "canonical_tasks_sha256": canonical_digest,
        "shard_manifest_file_sha256": manifest_digest,
        "relay": {
            "host": relay_host,
            "port": relay_port,
            "registry": str(state_dir / "egress_relay_clients.json"),
        },
        "resources": {
            "service_replicas": 4,
            "service_partition": "h200",
            "service_qos": "h200_ram_high",
            "service_gpu": "1xH200 each",
            "action_shards": 4,
            "workers_per_shard": 16,
            "max_action_passes": 3,
        },
    }


def validate_protocol(path: Path) -> dict[str, Any]:
    _regular_non_symlink(path, "lane protocol")
    try:
        protocol = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("lane protocol is not valid JSON") from exc
    if not isinstance(protocol, dict) or protocol.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("lane protocol schema/version is invalid")
    lane = protocol.get("lane")
    checkpoint = protocol.get("checkpoint")
    relay = protocol.get("relay")
    if not isinstance(lane, str) or not isinstance(checkpoint, str):
        raise ProtocolError("lane protocol identity fields are invalid")
    if not isinstance(relay, dict):
        raise ProtocolError("lane protocol relay field is invalid")
    relay_host = relay.get("host")
    relay_port = relay.get("port")
    if not isinstance(relay_host, str) or not isinstance(relay_port, int):
        raise ProtocolError("lane protocol relay identity is invalid")
    expected = build_protocol(
        lane=lane,
        checkpoint=Path(checkpoint),
        relay_host=relay_host,
        relay_port=relay_port,
    )
    if set(protocol) != set(expected):
        raise ProtocolError("lane protocol fields do not exactly match the schema")
    generated_at = protocol.get("generated_at")
    if not isinstance(generated_at, str):
        raise ProtocolError("lane protocol generation timestamp is invalid")
    try:
        generated_time = datetime.fromisoformat(generated_at)
    except ValueError as exc:
        raise ProtocolError("lane protocol generation timestamp is invalid") from exc
    if generated_time.tzinfo is None:
        raise ProtocolError("lane protocol generation timestamp lacks a timezone")
    for key, expected_value in expected.items():
        if key == "generated_at":
            continue
        if protocol.get(key) != expected_value:
            raise ProtocolError(f"lane protocol field {key!r} does not match its identity")
    return protocol


def atomic_write_protocol(path: Path, protocol: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--lane", required=True)
    build.add_argument("--checkpoint", type=Path, required=True)
    build.add_argument("--relay-host", default="10.146.5.90")
    build.add_argument("--relay-port", type=int, required=True)
    build.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--protocol", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "build":
            protocol = build_protocol(
                lane=args.lane,
                checkpoint=args.checkpoint,
                relay_host=args.relay_host,
                relay_port=args.relay_port,
            )
            if args.output:
                atomic_write_protocol(args.output, protocol)
            print(json.dumps(protocol, indent=2, sort_keys=True))
        else:
            protocol = validate_protocol(args.protocol)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "lane": protocol["lane"],
                        "checkpoint": protocol["checkpoint"],
                        "action_model": protocol["action_model"],
                    },
                    sort_keys=True,
                )
            )
    except ProtocolError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
