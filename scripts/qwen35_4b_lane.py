#!/usr/bin/env python3
"""Audit and seed the Qwen3.5-4B SWE-Together k=2 evaluation lane.

This is intentionally lane-specific.  It keeps the legacy ``qwen_k2``
artifacts immutable while allowing any genuinely reusable cells
to be copied into a clean, provenance-tracked cohort.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval_infra_sentinel import classify_trial, write_sidecar  # noqa: E402


EXPECTED_ACTION_MODEL = "openai/Qwen3.5-4B"
EXPECTED_USER_MODEL = "anthropic/claude-opus-4-8"
EXPECTED_USER_TEMPERATURE = 0.5
EXPECTED_AGENT_TIMEOUT = 4800.0
EXPECTED_ENVIRONMENT_IMPORT = "podman_env:PodmanEnvironment"
DEFAULT_PRODUCER_LOCK = REPO_ROOT / "pipeline_logs/qwen35_4b_producer.lock"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _expected_tasks() -> list[str]:
    manifest = _load_json(REPO_ROOT / "canonical_full109.json")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
        raise RuntimeError("canonical_full109.json does not contain a string task list")
    return tasks


def _trial_task(trial: Path, result: dict[str, Any]) -> str:
    task = result.get("task_name")
    if isinstance(task, str) and task:
        return task
    return trial.name.rsplit("__", 1)[0]


def _trial_config(trial: Path, result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config")
    if isinstance(config, dict):
        return config
    return _load_json(trial / "config.json")


def _finite_reward(result: dict[str, Any]) -> float | None:
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    value = rewards.get("reward") if isinstance(rewards, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _wall_minutes(result: dict[str, Any]) -> float | None:
    execution = result.get("agent_execution")
    if not isinstance(execution, dict):
        return None
    try:
        started = datetime.fromisoformat(str(execution["started_at"]).replace("Z", "+00:00"))
        finished = datetime.fromisoformat(str(execution["finished_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    seconds = (finished - started).total_seconds()
    return seconds / 60.0 if seconds >= 0 else None


def _output_reasoning_tokens(trial: Path) -> int | None:
    agent_dir = trial / "agent"
    combined = agent_dir / "opencode.txt"
    paths = [combined] if combined.exists() else sorted(agent_dir.glob("opencode.txt.turn-*"))
    if not paths:
        return None
    seen: set[str] = set()
    found = False
    total = 0
    for path in paths:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            return None
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") not in {
                "step_finish", "step-finish"
            }:
                continue
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            event_id = str(part.get("id") or part.get("messageID") or "")
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else event.get("tokens")
            if not isinstance(tokens, dict):
                continue
            output = tokens.get("output", 0) or 0
            reasoning = tokens.get("reasoning", 0) or 0
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (output, reasoning)
            ):
                return None
            total += output + reasoning
            found = True
    return total if found else None


def _protocol_fields(config: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    agent = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    kwargs = agent.get("kwargs") if isinstance(agent.get("kwargs"), dict) else {}
    environment = (
        config.get("environment")
        if isinstance(config.get("environment"), dict)
        else {}
    )
    return (
        agent.get("model_name"),
        kwargs.get("user_model_name"),
        kwargs.get("user_temperature"),
        agent.get("override_timeout_sec"),
        environment.get("import_path"),
    )


def _protocol_matches(fields: tuple[Any, Any, Any, Any, Any]) -> bool:
    action_model, user_model, user_temperature, agent_timeout, environment = fields
    try:
        temperature_ok = float(user_temperature) == EXPECTED_USER_TEMPERATURE
        timeout_ok = float(agent_timeout) == EXPECTED_AGENT_TIMEOUT
    except (TypeError, ValueError):
        return False
    return (
        action_model == EXPECTED_ACTION_MODEL
        and user_model == EXPECTED_USER_MODEL
        and temperature_ok
        and timeout_ok
        and environment == EXPECTED_ENVIRONMENT_IMPORT
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_producer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"producer lock is held; refusing quarantine: {path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> dict[str, Any]:
    expected = set(_expected_tasks())
    trials = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    strict = Counter()
    relaxed = Counter()
    task_counts = Counter()
    models = Counter()
    user_models = Counter()
    user_temperatures = Counter()
    agent_timeouts = Counter()
    environment_imports = Counter()
    judge_models = Counter()
    tag_models = Counter()
    numeric_rewards = 0
    runtime_complete = 0
    token_complete = 0
    reusable = 0

    for trial in trials:
        result = _load_json(trial / "result.json")
        task = _trial_task(trial, result)
        task_counts[task] += 1
        config = _trial_config(trial, result)
        (
            action_model,
            user_model,
            user_temperature,
            agent_timeout,
            environment_import,
        ) = _protocol_fields(config)
        models[str(action_model)] += 1
        user_models[str(user_model)] += 1
        user_temperatures[str(user_temperature)] += 1
        agent_timeouts[str(agent_timeout)] += 1
        environment_imports[str(environment_import)] += 1
        reward = _finite_reward(result)
        numeric_rewards += reward is not None
        runtime_complete += _wall_minutes(result) is not None
        token_complete += _output_reasoning_tokens(trial) is not None

        strict_verdict = classify_trial(trial, strict=True)
        relaxed_verdict = classify_trial(trial, strict=False)
        strict[f"{strict_verdict.status}:{strict_verdict.reason}"] += 1
        relaxed[f"{relaxed_verdict.status}:{relaxed_verdict.reason}"] += 1
        reusable += (
            strict_verdict.status == "ok"
            and reward is not None
            and _protocol_matches(
                (
                    action_model,
                    user_model,
                    user_temperature,
                    agent_timeout,
                    environment_import,
                )
            )
            and task in expected
        )

        judge = _load_json(trial / "judge_verdict.json")
        if judge:
            judge_models[str(judge.get("judge_model"))] += 1
        for tag_name in ("intent_tag_verdict.json", "tag_verdict.json"):
            tag = _load_json(trial / tag_name)
            if tag:
                tag_models[str(tag.get("model") or tag.get("tag_model"))] += 1
                break

    expected_counts = Counter(task_counts.get(task, 0) for task in expected)
    unexpected = sorted(set(task_counts) - expected)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "trial_dirs": len(trials),
        "expected_tasks": len(expected),
        "expected_task_replicate_histogram": dict(sorted(expected_counts.items())),
        "unexpected_tasks": unexpected,
        "numeric_rewards": numeric_rewards,
        "runtime_complete": runtime_complete,
        "output_reasoning_tokens_complete": token_complete,
        "strict_fresh_infra": dict(strict),
        "relaxed_fresh_infra": dict(relaxed),
        "strict_protocol_reusable": reusable,
        "action_models": dict(models),
        "user_models": dict(user_models),
        "user_temperatures": dict(user_temperatures),
        "agent_timeouts": dict(agent_timeouts),
        "environment_imports": dict(environment_imports),
        "judge_models": dict(judge_models),
        "tag_models": dict(tag_models),
        "strict_complete_109x2": (
            len(trials) == 218
            and not unexpected
            and all(task_counts.get(task) == 2 for task in expected)
            and numeric_rewards == 218
            and runtime_complete == 218
            and token_complete == 218
            and strict.get("ok:", 0) == 218
            and models == Counter({EXPECTED_ACTION_MODEL: 218})
            and user_models == Counter({EXPECTED_USER_MODEL: 218})
            and user_temperatures == Counter({str(EXPECTED_USER_TEMPERATURE): 218})
            and agent_timeouts == Counter({str(EXPECTED_AGENT_TIMEOUT): 218})
            and environment_imports == Counter({EXPECTED_ENVIRONMENT_IMPORT: 218})
        ),
    }


def seed(source: Path, target: Path) -> dict[str, Any]:
    """Copy only strict-valid, protocol-matching legacy cells into target."""

    expected = set(_expected_tasks())
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for trial in sorted(p for p in source.iterdir() if p.is_dir()):
        result = _load_json(trial / "result.json")
        task = _trial_task(trial, result)
        config = _trial_config(trial, result)
        (
            action_model,
            user_model,
            user_temperature,
            agent_timeout,
            environment_import,
        ) = _protocol_fields(config)
        verdict = classify_trial(trial, strict=True)
        reward = _finite_reward(result)
        if not (
            verdict.status == "ok"
            and reward is not None
            and task in expected
            and _protocol_matches(
                (
                    action_model,
                    user_model,
                    user_temperature,
                    agent_timeout,
                    environment_import,
                )
            )
        ):
            continue

        destination = target / trial.name
        if destination.exists():
            continue
        shutil.copytree(trial, destination)
        legacy_judge = destination / "judge_verdict.json"
        if legacy_judge.exists():
            legacy_judge.rename(destination / "judge_verdict.legacy_opus48.json")
        provenance = {
            "source_root": str(source.resolve()),
            "source_trial": trial.name,
            "copied_at": datetime.now(timezone.utc).isoformat(),
            "selection": "classify_trial(strict=True) + finite reward + protocol match",
            "result_sha256": _sha256(trial / "result.json"),
            "reward": reward,
        }
        (destination / "source_trial_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        records.append({"task": task, "trial": trial.name, **provenance})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "target": str(target),
        "selection": "strict fresh infra + finite reward + exact model/user protocol",
        "copied": len(records),
        "records": records,
    }
    manifest_path = target / "_reuse_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _resume_issues(trial: Path, expected: set[str]) -> tuple[str, list[str]]:
    result = _load_json(trial / "result.json")
    task = _trial_task(trial, result)
    config = _trial_config(trial, result)
    (
        action_model,
        user_model,
        user_temperature,
        agent_timeout,
        environment_import,
    ) = _protocol_fields(config)
    verdict = classify_trial(trial, strict=True)
    issues: list[str] = []
    if task not in expected:
        issues.append("unexpected_task")
    if not result:
        issues.append("missing_or_invalid_result")
    if _finite_reward(result) is None:
        issues.append("missing_reward")
    if _wall_minutes(result) is None:
        issues.append("missing_runtime")
    if _output_reasoning_tokens(trial) is None:
        issues.append("missing_tokens")
    if verdict.status != "ok":
        issues.append(f"infra:{verdict.reason or verdict.status}")
    if action_model != EXPECTED_ACTION_MODEL:
        issues.append(f"action_model:{action_model}")
    if user_model != EXPECTED_USER_MODEL:
        issues.append(f"user_model:{user_model}")
    try:
        temperature_ok = float(user_temperature) == EXPECTED_USER_TEMPERATURE
    except (TypeError, ValueError):
        temperature_ok = False
    if not temperature_ok:
        issues.append(f"user_temperature:{user_temperature}")
    try:
        timeout_ok = float(agent_timeout) == EXPECTED_AGENT_TIMEOUT
    except (TypeError, ValueError):
        timeout_ok = False
    if not timeout_ok:
        issues.append(f"agent_timeout:{agent_timeout}")
    if environment_import != EXPECTED_ENVIRONMENT_IMPORT:
        issues.append(f"environment_import:{environment_import}")
    return task, issues


def prepare_resume(root: Path, quarantine_root: Path, k: int) -> dict[str, Any]:
    """Quarantine invalid/excess cells and stamp strict fresh sidecars.

    The move is recoverable and its manifest records every source/destination.
    Call only after the producing Slurm job has exited.
    """

    expected = set(_expected_tasks())
    candidates: dict[str, list[Path]] = {task: [] for task in expected}
    rejected: list[tuple[Path, str, list[str]]] = []
    for trial in sorted(p for p in root.iterdir() if p.is_dir() and "__" in p.name):
        task, issues = _resume_issues(trial, expected)
        if issues:
            rejected.append((trial, task, issues))
        else:
            candidates[task].append(trial)

    kept: list[Path] = []
    for task, trials in sorted(candidates.items()):
        kept.extend(trials[:k])
        for extra in trials[k:]:
            rejected.append((extra, task, [f"excess_valid_replicate:>{k}"]))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_root.mkdir(parents=True, exist_ok=True)
    if root.resolve().stat().st_dev != quarantine_root.resolve().stat().st_dev:
        raise RuntimeError("quarantine must share a filesystem with the trial root")
    destination_root = quarantine_root / stamp
    moves: list[dict[str, Any]] = [
        {
            "task": task,
            "trial": trial.name,
            "issues": issues,
            "source": str(trial),
            "destination": str(destination_root / trial.name),
            "state": "planned",
        }
        for trial, task, issues in rejected
    ]
    if rejected:
        destination_root.mkdir(parents=True, exist_ok=False)
    transaction_path = destination_root / "quarantine_manifest.json"
    transaction: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": "planned",
        "root": str(root),
        "quarantine_root": str(destination_root) if rejected else None,
        "k": k,
        "moves": moves,
    }
    if rejected:
        _write_json_atomic(transaction_path, transaction)
    for record in moves:
        source = Path(record["source"])
        destination = Path(record["destination"])
        if destination.exists():
            raise RuntimeError(f"quarantine destination already exists: {destination}")
        source.replace(destination)
        record["state"] = "moved"
        _write_json_atomic(transaction_path, transaction)

    for trial in kept:
        write_sidecar(trial, classify_trial(trial, strict=True))

    kept_counts = Counter()
    for trial in kept:
        result = _load_json(trial / "result.json")
        kept_counts[_trial_task(trial, result)] += 1
    missing = {
        task: k - kept_counts.get(task, 0)
        for task in sorted(expected)
        if kept_counts.get(task, 0) < k
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "quarantine_root": str(destination_root) if rejected else None,
        "k": k,
        "kept": len(kept),
        "quarantined": len(moves),
        "missing_trials": sum(missing.values()),
        "missing_by_task": missing,
        "moves": moves,
        "state": "complete",
    }
    if rejected:
        _write_json_atomic(transaction_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("root", type=Path)
    audit_parser.add_argument("--output", type=Path)
    seed_parser = subparsers.add_parser("seed")
    seed_parser.add_argument("source", type=Path)
    seed_parser.add_argument("target", type=Path)
    resume_parser = subparsers.add_parser("prepare-resume")
    resume_parser.add_argument("root", type=Path)
    resume_parser.add_argument("quarantine_root", type=Path)
    resume_parser.add_argument("--k", type=int, default=2)
    resume_parser.add_argument(
        "--producer-lock", type=Path, default=DEFAULT_PRODUCER_LOCK
    )
    args = parser.parse_args()

    if args.command == "audit":
        report = audit(args.root)
        encoded = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
        return 0
    if args.command == "seed":
        print(json.dumps(seed(args.source, args.target), indent=2))
        return 0
    if args.command == "prepare-resume":
        if args.k <= 0:
            parser.error("--k must be positive")
        try:
            with _exclusive_producer_lock(args.producer_lock):
                report = prepare_resume(args.root, args.quarantine_root, args.k)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
