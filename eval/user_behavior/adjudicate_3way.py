#!/usr/bin/env python3
"""Stage 3b — paper-protocol three-way User Correction adjudication.

The paper's headline U-Corr values reconcile:

* Judge A: Gemini 3.1 Pro single-pass tags stored as ``trial_msg_tags``;
* Judge B: Claude Opus 4.6 tags in a versioned sidecar;
* Arbiter: GPT-5.5, consulted only for A/B correction-layer disagreements.

The default single-tagger pipeline remains available for backward compatibility.
This command is fail-closed: incomplete votes or provenance never produce a
``trial_msg_tags_3way`` field that can pass the strict metric gate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "harbor" / "src"))

from eval.user_behavior import user_metrics as kg  # noqa: E402
from eval.user_behavior.coverage_one import (  # noqa: E402
    load_dotenv,
    load_trial_sim_msgs,
    parse_json,
)
from eval.user_behavior.tag_messages import _resolve_dirs  # noqa: E402

SYSTEM = (REPO / "eval" / "user_behavior" / "prompts" / "tag_messages_system.md").read_text()
VALID = kg.ALL_TAGS
LAYERS = ("correction", "nudge")


def _build_user(messages: list[dict]) -> str:
    return "Messages:\n" + "\n".join(
        f'trial_idx={message["trial_idx"]}: '
        f'{" ".join(message["text"].split())[:1000]}'
        for message in messages
    )


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class ArbiterLLM:
    """Minimal Chat Completions client for the Codex-OAuth GPT-5.5 proxy."""

    def __init__(self, base: str, model: str):
        self.base = base.rstrip("/")
        self.model = model

    async def tag(self, messages: list[dict]) -> dict[int, list[str]]:
        import httpx

        prompt = SYSTEM + "\n\n" + _build_user(messages)
        async with httpx.AsyncClient(timeout=240.0) as client:
            response = await client.post(
                f"{self.base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                raise RuntimeError(str(body["error"])[:300])
            parsed = parse_json(body["choices"][0]["message"]["content"])
            return {
                row["trial_idx"]: [
                    tag for tag in (row.get("tags") or []) if tag in VALID
                ]
                for row in parsed.get("results", [])
                if isinstance(row, dict) and isinstance(row.get("trial_idx"), int)
            }


def _rows_to_map(rows: Any, label: str) -> dict[int, list[str]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} rows are missing or not a list")
    output: dict[int, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("trial_idx"), int):
            raise ValueError(f"{label} has a row without integer trial_idx")
        index = row["trial_idx"]
        if index in output:
            raise ValueError(f"{label} has duplicate trial_idx={index}")
        tags = row.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"{label} trial_idx={index} has invalid tags")
        output[index] = [tag for tag in tags if tag in VALID]
    return output


def _sidecar_tag_map(value: Any) -> dict[int, list[str]]:
    if not isinstance(value, dict):
        raise ValueError("Judge-B trial entry is missing or not an object")
    output: dict[int, list[str]] = {}
    for raw_index, raw_tags in value.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Judge-B has invalid trial_idx={raw_index!r}") from exc
        if not isinstance(raw_tags, list) or not all(
            isinstance(tag, str) for tag in raw_tags
        ):
            raise ValueError(f"Judge-B trial_idx={index} has invalid tags")
        output[index] = [tag for tag in raw_tags if tag in VALID]
    return output


def _disputed(a_tags: list[str], b_tags: list[str]) -> bool:
    a, b = set(a_tags), set(b_tags)
    return any((label in a) != (label in b) for label in LAYERS)


def _reconcile_rows(
    a_rows: list[dict],
    b_tags: dict[int, list[str]],
    arbiter_tags: dict[int, list[str]],
    disputed: list[int],
) -> list[dict]:
    """Reconcile only corrective labels; preserve A's base tags/frustration."""
    disputed_set = set(disputed)
    output: list[dict] = []
    for row in a_rows:
        index = row["trial_idx"]
        a = set(row.get("tags") or [])
        b = set(b_tags[index])
        c = set(arbiter_tags.get(index, []))
        tags = set(a)
        for label in LAYERS:
            if (label in a) == (label in b):
                keep = label in a
            elif index in disputed_set:
                keep = label in c
            else:  # Defensive: every A/B disagreement must be in ``disputed``.
                raise ValueError(f"unadjudicated disagreement at trial_idx={index}")
            tags.discard(label)
            if keep:
                tags.add(label)
        output.append({**row, "tags": sorted(tags)})
    return output


def _clear_threeway(verdict: dict, out_field: str) -> bool:
    changed = False
    for field in (out_field, "user_correction_3way", "message_tagging_3way"):
        if field in verdict:
            verdict.pop(field, None)
            changed = True
    return changed


@dataclass
class TrialContext:
    verdict_path: Path
    verdict: dict
    a_rows: list[dict]
    b_tags: dict[int, list[str]]
    messages_by_index: dict[int, dict]
    disputed: list[int]


def _load_sidecar(
    path: Path,
    *,
    expected_model: str,
    require_provenance: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool, str]:
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    body = json.loads(raw_bytes)
    legacy = not (
        isinstance(body, dict)
        and isinstance(body.get("trials"), dict)
        and "message_tagging" in body
    )
    if legacy:
        if require_provenance:
            raise ValueError("legacy Judge-B sidecar is not allowed with --require-provenance")
        if not isinstance(body, dict):
            raise ValueError("Judge-B sidecar must be a JSON object")
        return body, kg.tagging_provenance(expected_model), True, digest

    issues = kg.tagging_provenance_issues(body.get("message_tagging"), expected_model)
    if body.get("schema_version") != kg.TAGGING_SCHEMA_VERSION:
        issues.append("sidecar_schema_version")
    if issues:
        raise ValueError(f"Judge-B sidecar provenance mismatch: {', '.join(issues)}")
    return body["trials"], body["message_tagging"], False, digest


def _build_context(
    verdict_path: Path,
    judge_b_trials: dict[str, Any],
    *,
    judge_a_model: str,
    require_provenance: bool,
) -> TrialContext:
    try:
        verdict = json.loads(verdict_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid verdict JSON: {exc}") from exc
    if not isinstance(verdict, dict) or "trial_msg_tags" not in verdict:
        raise ValueError("Judge-A trial_msg_tags are missing")
    if require_provenance:
        issues = kg.tagging_provenance_issues(
            verdict.get("message_tagging"), judge_a_model
        )
        if issues:
            raise ValueError(f"Judge-A provenance mismatch: {', '.join(issues)}")

    a_rows = verdict["trial_msg_tags"]
    a_tags = _rows_to_map(a_rows, "Judge-A")
    if verdict_path.parent.name not in judge_b_trials:
        raise ValueError("Judge-B sidecar has no entry for this trial")
    b_tags = _sidecar_tag_map(judge_b_trials[verdict_path.parent.name])

    trial_dir, task_dir = _resolve_dirs(str(verdict_path), verdict)
    messages = [
        message
        for message in load_trial_sim_msgs(trial_dir, task_dir)
        if message.get("trial_idx") != 0
    ]
    messages_by_index = {
        message["trial_idx"]: message
        for message in messages
        if isinstance(message.get("trial_idx"), int)
    }
    expected = set(messages_by_index)
    if set(a_tags) != expected:
        raise ValueError(
            f"Judge-A indices {sorted(a_tags)} != simulator indices {sorted(expected)}"
        )
    if set(b_tags) != expected:
        raise ValueError(
            f"Judge-B indices {sorted(b_tags)} != simulator indices {sorted(expected)}"
        )
    disputed = sorted(
        index for index in expected if _disputed(a_tags[index], b_tags[index])
    )
    return TrialContext(
        verdict_path=verdict_path,
        verdict=verdict,
        a_rows=a_rows,
        b_tags=b_tags,
        messages_by_index=messages_by_index,
        disputed=disputed,
    )


async def run_adjudication(
    trials_roots: list[Path],
    *,
    judge_b_sidecar: Path,
    judge_a_model: str = kg.PAPER_TAG_JUDGE_A,
    judge_b_model: str = kg.PAPER_TAG_JUDGE_B,
    arbiter_model: str = kg.PAPER_TAG_ARBITER,
    arbiter_proxy: str = "http://127.0.0.1:4220/v1",
    workers: int = 5,
    out_field: str = "trial_msg_tags_3way",
    require_provenance: bool = False,
    force: bool = False,
    arbiter: ArbiterLLM | None = None,
) -> dict[str, int]:
    judge_b_trials, judge_b_provenance, legacy, sidecar_hash = _load_sidecar(
        judge_b_sidecar,
        expected_model=judge_b_model,
        require_provenance=require_provenance,
    )
    verdict_paths = sorted(
        trial / "intent_coverage_verdict.json"
        for root in trials_roots
        for trial in root.glob("*__*")
        if trial.is_dir() and (trial / "result.json").exists()
    )
    trial_names = [path.parent.name for path in verdict_paths]
    if len(trial_names) != len(set(trial_names)):
        raise ValueError("duplicate trial directory names across --trials-root values")

    counts = {"ok": 0, "skip": 0, "err": 0, "arbitrated": 0}
    contexts: list[TrialContext] = []
    for verdict_path in verdict_paths:
        try:
            existing = json.loads(verdict_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
        if not force and out_field in existing:
            issues = kg.user_correction_provenance_issues(
                existing,
                source="threeway",
                expected_tag_model=judge_a_model,
                expected_judge_b_model=judge_b_model,
                expected_arbiter_model=arbiter_model,
            )
            recorded_sidecar_hash = (
                (existing.get("message_tagging_3way") or {}).get(
                    "judge_b_sidecar_sha256"
                )
                if isinstance(existing.get("message_tagging_3way"), dict)
                else None
            )
            if (
                (not issues and recorded_sidecar_hash == sidecar_hash)
                or not require_provenance
            ):
                counts["skip"] += 1
                continue
        try:
            contexts.append(
                _build_context(
                    verdict_path,
                    judge_b_trials,
                    judge_a_model=judge_a_model,
                    require_provenance=require_provenance,
                )
            )
        except ValueError as exc:
            if _clear_threeway(existing, out_field):
                _write_json_atomic(verdict_path, existing)
            counts["err"] += 1
            print(f"ERROR {verdict_path.parent.name}: {exc}", file=sys.stderr)

    arbiter = arbiter or ArbiterLLM(arbiter_proxy, arbiter_model)
    semaphore = asyncio.Semaphore(workers)

    async def arbitrate(context: TrialContext) -> tuple[TrialContext, dict[int, list[str]] | Exception]:
        if not context.disputed:
            return context, {}
        disputed_messages = [
            context.messages_by_index[index] for index in context.disputed
        ]
        async with semaphore:
            for attempt in range(3):
                try:
                    votes = await arbiter.tag(disputed_messages)
                    missing = set(context.disputed) - set(votes)
                    if missing:
                        raise ValueError(f"arbiter omitted trial_idx={sorted(missing)}")
                    return context, votes
                except Exception as exc:  # noqa: BLE001 - retry API/parse failures
                    if attempt == 2:
                        return context, exc
                    await asyncio.sleep(5)
        raise AssertionError("unreachable")

    results = await asyncio.gather(*(arbitrate(context) for context in contexts))
    for context, votes_or_error in results:
        verdict = context.verdict
        if isinstance(votes_or_error, Exception):
            _clear_threeway(verdict, out_field)
            _write_json_atomic(context.verdict_path, verdict)
            counts["err"] += 1
            print(
                f"ERROR {context.verdict_path.parent.name}: arbiter failed: "
                f"{votes_or_error}",
                file=sys.stderr,
            )
            continue
        votes = votes_or_error
        rows = _reconcile_rows(
            context.a_rows, context.b_tags, votes, context.disputed
        )
        judge_a_issues = kg.tagging_provenance_issues(
            verdict.get("message_tagging"), judge_a_model
        )
        verdict[out_field] = rows
        verdict["user_correction_3way"] = kg.metrics_from_rows(rows)["user_correction"]
        verdict["message_tagging_3way"] = {
            "schema_version": kg.THREEWAY_SCHEMA_VERSION,
            "source_field": out_field,
            "judge_a": verdict.get("message_tagging")
            if isinstance(verdict.get("message_tagging"), dict)
            else kg.tagging_provenance(judge_a_model),
            "judge_b": judge_b_provenance,
            "arbiter": {
                "model": arbiter_model,
                # The Codex-OAuth proxy does not expose a portable temperature
                # knob; record that explicitly instead of inventing one.
                "temperature": None,
                "prompt_sha256": kg.TAG_PROMPT_SHA256,
            },
            "judge_b_sidecar_sha256": sidecar_hash,
            "disputed_trial_indices": context.disputed,
            "arbiter_votes": {
                str(index): sorted(set(votes.get(index, [])) & set(LAYERS))
                for index in context.disputed
            },
            "legacy_unverified": legacy or bool(judge_a_issues),
        }
        _write_json_atomic(context.verdict_path, verdict)
        counts["ok"] += 1
        counts["arbitrated"] += int(bool(context.disputed))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-root", action="append", required=True, type=Path)
    parser.add_argument(
        "--judge-b-sidecar",
        "--judgeb-sidecar",
        dest="judge_b_sidecar",
        required=True,
        type=Path,
    )
    parser.add_argument("--judge-a-model", default=kg.PAPER_TAG_JUDGE_A)
    parser.add_argument("--judge-b-model", default=kg.PAPER_TAG_JUDGE_B)
    parser.add_argument("--arbiter-model", default=kg.PAPER_TAG_ARBITER)
    parser.add_argument("--arbiter-proxy", default="http://127.0.0.1:4220/v1")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out-field", default="trial_msg_tags_3way")
    parser.add_argument("--require-provenance", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv(REPO)
    try:
        counts = asyncio.run(
            run_adjudication(
                args.trials_root,
                judge_b_sidecar=args.judge_b_sidecar,
                judge_a_model=args.judge_a_model,
                judge_b_model=args.judge_b_model,
                arbiter_model=args.arbiter_model,
                arbiter_proxy=args.arbiter_proxy,
                workers=args.workers,
                out_field=args.out_field,
                require_provenance=args.require_provenance,
                force=args.force,
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"three-way adjudication: {counts}", flush=True)
    return 2 if counts["err"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
