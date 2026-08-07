#!/usr/bin/env python3
"""Compute SWE-Together "Table 2" metrics from one or more trials roots.

Reads Harbor trials directories (one subdir per trial, named
``<taskname>__<randomsuffix>``, with a task appearing ``k`` times when run with
``k`` replicates) and writes a JSON summary. ``--strict`` proves that all metric
inputs are complete; action-model/user-simulator/sandbox provenance must still
be compared with the paper separately. Without ``--strict`` the output is
deliberately labelled diagnostic/noncanonical.

Usage:
    python eval/table2_metrics.py \
      --trials-dir trials/opus_part1 --trials-dir trials/opus_part2 \
      --model "Claude Opus 4.8" --k 2 --strict

Partial/in-progress directories are excluded from diagnostic arithmetic but
reported. They make the strict completeness gate fail; they are never silently
accepted into a metric-complete row. The released benchmark pipeline uses the
single Gemini tagger; ``threeway`` is an optional ensemble sensitivity.

Metric definitions (paper, threshold tau = 0.85):
    Per task t, per replicate r:
        j_{t,r} = judge_score  (from judge_verdict.json; 0.0 if absent/error)
        s_{t,r} = 1 if j_{t,r} >= tau else 0
        jbar_t  = mean over t's replicates of j
        sbar_t  = mean over t's replicates of s
    pass@1       = mean over tasks of sbar_t
    SSR          = fraction of tasks with jbar_t >= tau
    pass^k       = fraction of tasks where ALL present replicates pass (s==1)
    mean_judge   = mean over tasks of jbar_t
    U-Corr       = mean over tasks of (mean over replicates of
                   #correction + 0.2*#nudge from the selected single/threeway
                   message-tag source)
    tok_per_task = mean over tasks of (mean over replicates of output+reasoning
                   tokens) [None-tokens skipped]
    min_per_task = mean over tasks of (mean over replicates of trial wall-minutes) [None-minutes skipped]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from eval.patch_utils import patch_file_has_changes
    from eval.user_behavior import user_metrics as kg
except ModuleNotFoundError:  # Direct ``python eval/table2_metrics.py`` fallback.
    from patch_utils import patch_file_has_changes
    from user_behavior import user_metrics as kg
from eval_infra_sentinel import (  # noqa: E402 - src path setup above
    SIDECAR_VERSION,
    classify_trial,
)
from eval.correctness.run_batch import (  # noqa: E402
    _phase2_verdict_issues,
    _rubric_issues,
)

TAU = 0.85
DEFAULT_TASKS_ROOT = REPO_ROOT / "tasks"
DEFAULT_EXPECTED_TASKS = 109
DEFAULT_EXPECTED_JUDGE_MODEL = "claude-opus-4-6"


# --------------------------------------------------------------------------- #
# Robust JSON helpers
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any | None:
    """Read + parse a JSON file, returning None on any error (never raises)."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Yield every top-level JSON object in *text*.

    Handles JSON-lines, concatenated JSON, and interleaved non-JSON preamble
    (e.g. opencode's "Performing one time database migration..." banner). Non-
    JSON regions are skipped to the next newline. Only dict objects are yielded.
    """
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        # Skip leading whitespace.
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            # Not JSON here -> skip to next line.
            nl = text.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
            continue
        if isinstance(obj, dict):
            yield obj
        i = end


# --------------------------------------------------------------------------- #
# Canonical task-name resolution (Harbor truncates dir names to 32 chars)
# --------------------------------------------------------------------------- #
def _task_dir_names(tasks_root: Path) -> list[str]:
    if not tasks_root.is_dir():
        return []
    return sorted(p.name for p in tasks_root.iterdir() if p.is_dir())


def canonical_task_name(trial_dirname: str, task_names: list[str]) -> str:
    """Map a trial dir name to its canonical task name.

    task = trial_dirname.rsplit("__", 1)[0]. Harbor truncates task names to 32
    chars in dir names, so: exact match against a tasks/ subdir wins, else the
    unique tasks/ subdir that startswith the (possibly truncated) prefix, else
    fall back to the raw prefix.
    """
    prefix = trial_dirname.rsplit("__", 1)[0]
    if prefix in task_names:
        return prefix
    matches = [t for t in task_names if t.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return prefix


# --------------------------------------------------------------------------- #
# Per-trial field extraction
# --------------------------------------------------------------------------- #
def get_reward(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    try:
        r = (result.get("verifier_result") or {}).get("rewards") or {}
        val = r.get("reward")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return None
        score = float(val)
        return score if math.isfinite(score) else None
    except (TypeError, ValueError):
        return None


def _iso_to_dt(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_wall_minutes(
    result: dict[str, Any] | None, trial_dir: Path | None = None
) -> float | None:
    if not isinstance(result, dict):
        result = {}
    ae = result.get("agent_execution") or {}
    start = _iso_to_dt(ae.get("started_at"))
    finish = _iso_to_dt(ae.get("finished_at"))
    if start is not None and finish is not None:
        secs = (finish - start).total_seconds()
        if secs >= 0:
            return secs / 60.0

    # Older runs recorded only the trial-level timer sidecar.  It is still a
    # real wall-clock measurement and prevents a complete historical run from
    # being rejected merely because Harbor omitted agent_execution timestamps.
    timing = _load_json(trial_dir / "timing.json") if trial_dir else None
    if isinstance(timing, dict):
        value = timing.get("trial_wall_clock_sec")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        secs = float(value)
        if math.isfinite(secs) and secs >= 0:
            return secs / 60.0
    return None


def get_judge(
    trial_dir: Path, out_name: str = "judge_verdict.json"
) -> tuple[float, bool]:
    """Return (judge_score, is_real_verdict).

    Absent file, unparseable JSON, an "error" key, or a missing/invalid
    judge_score all mean the agent produced no substantive work -> score 0.0
    and is_real_verdict False.
    """
    d = _load_json(trial_dir / out_name)
    if not isinstance(d, dict) or "error" in d:
        return 0.0, False
    val = d.get("judge_score")
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return 0.0, False
    score = float(val)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return 0.0, False
    return score, True


def _normalized_model_id(value: Any) -> str:
    """Normalize model IDs while ignoring provider/router prefixes."""
    if value is None:
        return ""
    normalized = str(value).strip().lower().rstrip("/")
    if not normalized:
        return ""
    normalized = normalized.rsplit("/", 1)[-1]
    # Providers also commonly use ``provider:model`` rather than a path.
    if ":" in normalized:
        normalized = normalized.rsplit(":", 1)[-1]
    return normalized.replace("_", "-")


def get_user_correction(
    trial_dir: Path,
    source: str = "single",
    verdict: dict[str, Any] | None = None,
) -> float | None:
    """Read canonical User Correction from multi-label message tags.

    Raw simulator interventions are not U-Corr: neutral questions, requests,
    approvals, and context messages carry zero weight.  The canonical metric is
    ``#correction + 0.2 * #nudge`` from the explicitly selected tag field.
    """
    if verdict is None:
        verdict = _load_json(trial_dir / "intent_coverage_verdict.json")
    if not isinstance(verdict, dict):
        return None
    # Always derive from selected rows. Cached scalar fields may come from a
    # different tagger and must never override the requested source.
    value = kg.metrics_from_verdict(verdict, source)["user_correction"]
    return float(value) if value is not None else None


def _sidecar_trial_verdict(sidecar: Any, trial_name: str) -> dict[str, Any] | None:
    """Project one immutable tag sidecar entry into the normal verdict shape."""
    if not isinstance(sidecar, dict):
        return None
    rows_by_trial = sidecar.get("trial_rows")
    if not isinstance(rows_by_trial, dict) or trial_name not in rows_by_trial:
        return None
    rows = rows_by_trial.get(trial_name)
    if not isinstance(rows, list):
        return None
    return {
        "trial_msg_tags": rows,
        "message_tagging": sidecar.get("message_tagging"),
    }


def _tag_sidecar_issues(
    sidecar: Any, expected_model: str | None
) -> list[str]:
    """Validate the sidecar envelope once; per-trial rows are checked later."""
    if not isinstance(sidecar, dict):
        return ["tag_sidecar:not_an_object"]
    issues: list[str] = []
    if sidecar.get("schema_version") != kg.TAGGING_SCHEMA_VERSION:
        issues.append("tag_sidecar:schema_version")
    if not isinstance(sidecar.get("trials"), dict):
        issues.append("tag_sidecar:missing_trials")
    if not isinstance(sidecar.get("trial_rows"), dict):
        issues.append("tag_sidecar:missing_trial_rows")
    if not isinstance(sidecar.get("trial_input_sha256"), dict):
        issues.append("tag_sidecar:missing_trial_input_sha256")
    transport = sidecar.get("tag_transport")
    if not isinstance(transport, dict) or not isinstance(
        transport.get("backend"), str
    ):
        issues.append("tag_sidecar:missing_tag_transport")
    if expected_model:
        issues.extend(
            f"tag_sidecar:{issue}"
            for issue in kg.tagging_provenance_issues(
                sidecar.get("message_tagging"), expected_model
            )
        )
    return issues


def _expected_sim_message_indices(trial_dir: Path, task_dir: Path) -> set[int]:
    """Reconstruct the follow-up indices consumed by the production tagger."""
    instruction = task_dir / "instruction.md"
    next_index = 0
    try:
        if instruction.read_text(errors="replace").strip():
            next_index = 1
    except OSError:
        pass

    indices: set[int] = set()
    for episode in sorted((trial_dir / "agent").glob("episode-*")):
        decision = _load_json(episode / "user_decision.json")
        if not isinstance(decision, dict) or not decision.get("has_message"):
            continue
        content = decision.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if next_index != 0:  # Turn 0 is the initial instruction, never tagged.
            indices.add(next_index)
        next_index += 1
    return indices


def _expected_tag_input_sha256(trial_dir: Path, task_dir: Path) -> str:
    """Rebuild the exact normalized follow-up payload hashed by the tagger."""
    instruction = task_dir / "instruction.md"
    next_index = 0
    try:
        if instruction.read_text(errors="replace").strip():
            next_index = 1
    except OSError:
        pass

    messages: list[dict[str, Any]] = []
    for episode in sorted((trial_dir / "agent").glob("episode-*")):
        decision = _load_json(episode / "user_decision.json")
        if not isinstance(decision, dict) or not decision.get("has_message"):
            continue
        content = decision.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if next_index != 0:
            messages.append({"trial_idx": next_index, "text": content})
        next_index += 1
    return kg.tag_input_sha256(messages)


def sum_tokens(trial_dir: Path) -> int | None:
    """Sum trial tokens across all agent/opencode.txt.turn-* files.

    For each opencode ``step_finish`` (a.k.a. "step-finish") event, tokens live
    at event["part"]["tokens"] (matching harbor opencode.py; falls back to
    event["tokens"]). Table 2 reports output+reasoning only; input/cache tokens
    are excluded. Returns the summed output+reasoning, or None if no
    step_finish event is parseable (agent produced no measurable work).
    """
    agent_dir = trial_dir / "agent"
    if not agent_dir.is_dir():
        return None
    combined = agent_dir / "opencode.txt"
    turn_files = [combined] if combined.exists() else sorted(
        agent_dir.glob("opencode.txt.turn-*")
    )
    if not turn_files:
        return None
    total = 0
    found = False
    invalid = False
    seen_finish_ids: set[str] = set()
    for tf in turn_files:
        try:
            text = tf.read_text()
        except OSError:
            invalid = True
            continue
        for event in _iter_json_objects(text):
            if event.get("type") not in ("step_finish", "step-finish"):
                continue
            part = event.get("part")
            finish_id = (
                part.get("id") or part.get("messageID")
                if isinstance(part, dict)
                else None
            )
            if finish_id and finish_id in seen_finish_ids:
                continue
            if finish_id:
                seen_finish_ids.add(finish_id)
            tok = part.get("tokens") if isinstance(part, dict) else None
            if not isinstance(tok, dict):
                tok = event.get("tokens")
            if not isinstance(tok, dict):
                continue
            out = tok.get("output", 0) or 0
            rea = tok.get("reasoning", 0) or 0
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in (out, rea)
            ):
                invalid = True
                continue
            total += out + rea
            found = True
    return total if found and not invalid else None


def patch_bytes(trial_dir: Path) -> int | None:
    p = trial_dir / "agent" / "final.patch"
    try:
        return p.stat().st_size if p.exists() else None
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _collect_trials_detailed(
    trials_dirs: list[Path],
    tasks_root: Path,
    judge_out_name: str = "judge_verdict.json",
    u_corr_source: str = "single",
    expected_tag_model: str | None = None,
    expected_tag_judge_b_model: str = kg.ENSEMBLE_TAG_JUDGE_B,
    expected_tag_arbiter_model: str = kg.ENSEMBLE_TAG_ARBITER,
    tag_sidecar: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect records across roots and return their incomplete directory paths."""
    task_names = _task_dir_names(tasks_root)
    records: list[dict[str, Any]] = []
    incomplete: list[str] = []
    seen_trial_dirs: set[Path] = set()
    for trials_dir in sorted(trials_dirs, key=lambda path: str(path)):
        for d in sorted(trials_dir.iterdir(), key=lambda path: path.name):
            if not (d.is_dir() and "__" in d.name):
                continue
            resolved_d = d.resolve()
            if resolved_d in seen_trial_dirs:
                continue
            seen_trial_dirs.add(resolved_d)
            result_path = d / "result.json"
            if not result_path.exists():
                incomplete.append(str(resolved_d))
                continue
            result = _load_json(result_path)
            task_name = canonical_task_name(d.name, task_names)
            task_dir = tasks_root / task_name
            has_substantive_patch = patch_file_has_changes(
                d / "agent" / "final.patch"
            )
            # A missing/empty patch is always a correctness failure.  Ignore
            # any stale or accidentally copied verdict rather than allowing a
            # no-patch cell to inherit a positive score.
            if has_substantive_patch:
                judge_score, is_judged = get_judge(d, judge_out_name)
            else:
                judge_score, is_judged = 0.0, False
            judge = _load_json(d / judge_out_name)
            rubric = _load_json(task_dir / "canonical_goals.json")
            infra = _load_json(d / "trial_infra.json")
            try:
                infra_fresh_status = classify_trial(d).status
            except Exception:
                # Strict mode must fail closed if current artifacts cannot be
                # classified; diagnostic mode still emits the partial record.
                infra_fresh_status = None
            user_verdict = (
                _sidecar_trial_verdict(tag_sidecar, d.name)
                if tag_sidecar is not None
                else _load_json(d / "intent_coverage_verdict.json")
            )
            tag_input_issues: list[str] = []
            if tag_sidecar is not None:
                sidecar_hashes = tag_sidecar.get("trial_input_sha256")
                observed_hash = (
                    sidecar_hashes.get(d.name)
                    if isinstance(sidecar_hashes, dict)
                    else None
                )
                if observed_hash != _expected_tag_input_sha256(d, task_dir):
                    tag_input_issues.append("tag_input_sha256")
            rewards = (
                (result.get("verifier_result") or {}).get("rewards")
                if isinstance(result, dict)
                else None
            )
            reward = get_reward(result)
            records.append(
                {
                    "task": task_name,
                    "trial": d.name,
                    "trial_dir": str(resolved_d),
                    "trials_root": str(trials_dir),
                    "result_valid": (
                        isinstance(result, dict)
                        and isinstance(rewards, dict)
                        and bool(rewards)
                        and reward is not None
                        and math.isfinite(reward)
                    ),
                    "infra_status": (
                        infra.get("status") if isinstance(infra, dict) else None
                    ),
                    "infra_version": (
                        infra.get("version") if isinstance(infra, dict) else None
                    ),
                    "infra_fresh_status": infra_fresh_status,
                    "judge_score": judge_score,
                    "s": 1 if judge_score >= TAU else 0,
                    "is_judged": is_judged,
                    "judge_model": (
                        judge.get("judge_model") if isinstance(judge, dict) else None
                    ),
                    "judge_validation_issues": (
                        _phase2_verdict_issues(
                            rubric,
                            judge,
                            task_name=task_name,
                            trial_id=d.name,
                        )
                        if has_substantive_patch
                        else []
                    ),
                    "reward": reward,
                    "user_correction": get_user_correction(
                        d, u_corr_source, user_verdict
                    ),
                    "u_corr_source": u_corr_source,
                    "u_corr_provenance_issues": (
                        kg.user_correction_provenance_issues(
                            user_verdict,
                            source=u_corr_source,
                            expected_tag_model=expected_tag_model,
                            expected_judge_b_model=expected_tag_judge_b_model,
                            expected_arbiter_model=expected_tag_arbiter_model,
                            expected_trial_indices=_expected_sim_message_indices(
                                d, task_dir
                            ),
                        )
                        + tag_input_issues
                    ),
                    "tokens": sum_tokens(d),
                    "minutes": get_wall_minutes(result, d),
                    "patch_bytes": patch_bytes(d),
                    "has_substantive_patch": has_substantive_patch,
                }
            )
    records.sort(key=lambda row: (row["task"], row["trial_dir"]))
    return records, sorted(incomplete)


def collect_trials(trials_dir: Path, tasks_root: Path) -> tuple[list[dict], int]:
    """Return (completed trial records, n_incomplete_skipped).

    This single-root wrapper is retained for callers importing the original
    public helper.  The CLI uses :func:`_collect_trials_detailed` so it can
    report the exact incomplete directories across repeated ``--trials-dir``
    arguments.
    """
    records, incomplete = _collect_trials_detailed([trials_dir], tasks_root)
    return records, len(incomplete)


def compute_metrics(records: list[dict], k: int) -> dict[str, Any]:
    # Group replicates by canonical task.
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)

    per_task: dict[str, Any] = {}
    jbar_list, sbar_list, ucorr_list = [], [], []
    tok_task_means, min_task_means, reward_task_means = [], [], []
    ssr_hits = pass_k_hits = 0
    n_tasks_full_k = 0

    for task, reps in sorted(by_task.items()):
        j_vals = [r["judge_score"] for r in reps]
        s_vals = [r["s"] for r in reps]
        user_corrections = [
            r["user_correction"] for r in reps if r["user_correction"] is not None
        ]
        toks = [r["tokens"] for r in reps if r["tokens"] is not None]
        mins = [r["minutes"] for r in reps if r["minutes"] is not None]
        rewards = [r["reward"] for r in reps if r["reward"] is not None]

        jbar = _mean(j_vals) or 0.0
        sbar = _mean([float(s) for s in s_vals]) or 0.0
        all_pass = all(s == 1 for s in s_vals) and len(s_vals) == k
        tok_mean = _mean([float(t) for t in toks])
        min_mean = _mean(mins)
        reward_mean = _mean(rewards)

        jbar_list.append(jbar)
        sbar_list.append(sbar)
        user_correction_mean = _mean(user_corrections)
        if user_correction_mean is not None:
            ucorr_list.append(user_correction_mean)
        if tok_mean is not None:
            tok_task_means.append(tok_mean)
        if min_mean is not None:
            min_task_means.append(min_mean)
        if reward_mean is not None:
            reward_task_means.append(reward_mean)
        if jbar >= TAU:
            ssr_hits += 1
        if all_pass:
            pass_k_hits += 1
        if len(reps) == k:
            n_tasks_full_k += 1

        per_task[task] = {
            "n_replicates": len(reps),
            "jbar": jbar,
            "sbar": sbar,
            "all_pass": all_pass,
            "mean_user_correction": user_correction_mean,
            "mean_tokens": tok_mean,
            "mean_minutes": min_mean,
            "mean_reward": reward_mean,
            "replicates": reps,
        }

    n_tasks = len(by_task)
    n_trials = len(records)
    aggregates = {
        # Keep the paper's equal task weighting even in diagnostic/incomplete
        # cohorts where tasks may have different observed replicate counts.
        "pass@1": _mean(sbar_list) or 0.0,
        "SSR": (ssr_hits / n_tasks) if n_tasks else 0.0,
        "pass^k": (pass_k_hits / n_tasks) if n_tasks else 0.0,
        "mean_judge": _mean(jbar_list) or 0.0,
        "u_corr": _mean(ucorr_list),
        "tok_per_task": _mean(tok_task_means),
        "min_per_task": _mean(min_task_means),
        "avg_reward": _mean(reward_task_means),  # informational (not a Table-2 column)
        "n_tasks": n_tasks,
        "n_trials": n_trials,
        "n_judged": sum(1 for r in records if r["is_judged"]),
        "n_tasks_with_full_k": n_tasks_full_k,
    }
    return {"aggregates": aggregates, "per_task": per_task}


def completeness_issues(
    records: list[dict[str, Any]],
    *,
    incomplete_dirs: list[str],
    tasks_root: Path,
    k: int,
    expected_tasks: int,
    expected_judge_model: str,
    extra_issues: list[str] | None = None,
) -> list[str]:
    """Return every gap in the selected Table-2 metric inputs."""
    issues: list[str] = list(extra_issues or [])
    task_names = _task_dir_names(tasks_root)
    expected_names = set(task_names)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task"], []).append(record)

    if len(task_names) != expected_tasks:
        issues.append(
            f"tasks_dir_count:{len(task_names)}!=expected_{expected_tasks}"
        )
    if len(by_task) != expected_tasks:
        issues.append(f"task_count:{len(by_task)}!=expected_{expected_tasks}")

    observed_names = set(by_task)
    for task in sorted(expected_names - observed_names):
        issues.append(f"missing_task:{task}")
    for task in sorted(observed_names - expected_names):
        issues.append(f"unknown_task:{task}")

    for task in task_names:
        rubric = _load_json(tasks_root / task / "canonical_goals.json")
        for issue in _rubric_issues(rubric):
            issues.append(f"invalid_rubric:{task}:{issue}")

    for task, reps in sorted(by_task.items()):
        if len(reps) != k:
            issues.append(f"replicate_count:{task}:{len(reps)}!=expected_{k}")

    for path in incomplete_dirs:
        issues.append(f"incomplete_trial:{path}")

    normalized_expected_model = _normalized_model_id(expected_judge_model)
    for record in records:
        trial = record["trial_dir"]
        if not record["result_valid"]:
            issues.append(f"invalid_result:{trial}")
        if record["infra_version"] != SIDECAR_VERSION:
            observed = record["infra_version"]
            if observed is None:
                observed = "missing"
            issues.append(
                f"infra_version:{trial}:{observed}!=expected_{SIDECAR_VERSION}"
            )
        if record["infra_status"] != "ok":
            observed = record["infra_status"] or "missing"
            issues.append(f"infra_status:{trial}:{observed}")
        if record["infra_fresh_status"] != "ok":
            observed = record["infra_fresh_status"] or "unavailable"
            issues.append(f"infra_fresh_status:{trial}:{observed}")
        if record["minutes"] is None:
            issues.append(f"runtime_incomplete:{trial}")
        if record["tokens"] is None:
            issues.append(f"tokens_incomplete:{trial}")
        if record["user_correction"] is None:
            issues.append(f"u_corr_incomplete:{trial}")
        for issue in record.get("u_corr_provenance_issues") or []:
            issues.append(f"u_corr_provenance:{trial}:{issue}")

        if record["has_substantive_patch"]:
            if not record["is_judged"]:
                issues.append(f"judge_incomplete:{trial}")
            else:
                for issue in record.get("judge_validation_issues") or []:
                    issues.append(f"judge_invalid:{trial}:{issue}")
                if _normalized_model_id(record["judge_model"]) != normalized_expected_model:
                    observed = record["judge_model"] or "missing"
                    issues.append(
                        f"judge_model:{trial}:{observed}"
                        f"!=expected_{expected_judge_model}"
                    )
    return issues


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _fmt_tok(x: float | None) -> str:
    return "-" if x is None else f"{x / 1000.0:.1f}k"


def _fmt_min(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def _fmt_float(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def format_row(model: str, agg: dict[str, Any]) -> tuple[str, str]:
    header = (
        f"{'Model':<20} | {'pass@1':>7} | {'SSR':>7} | {'pass^k':>7} | "
        f"{'Mean judge':>10} | {'U-Corr':>6} | {'Tok./task':>9} | {'Min./task':>9}"
    )
    row = (
        f"{model:<20} | {_fmt_pct(agg['pass@1']):>7} | {_fmt_pct(agg['SSR']):>7} | "
        f"{_fmt_pct(agg['pass^k']):>7} | {agg['mean_judge']:>10.3f} | "
        f"{_fmt_float(agg['u_corr']):>6} | {_fmt_tok(agg['tok_per_task']):>9} | "
        f"{_fmt_min(agg['min_per_task']):>9}"
    )
    return header, row


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_output_path(trials_dirs: list[Path]) -> Path:
    if len(trials_dirs) == 1:
        return trials_dirs[0] / "table2_metrics.json"

    identities: list[str] = []
    for root in trials_dirs:
        try:
            identities.append(str(root.relative_to(REPO_ROOT)))
        except ValueError:
            identities.append(str(root))
    digest = hashlib.sha256("\n".join(identities).encode()).hexdigest()[:12]
    common = Path(os.path.commonpath([str(path) for path in trials_dirs]))
    # Never choose the filesystem root as an implicit output destination.
    if common == Path(common.anchor):
        common = REPO_ROOT
    return common / f"table2_metrics_{digest}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute SWE-Together Table-2 metrics from trial roots."
    )
    parser.add_argument(
        "--trials-dir",
        action="append",
        required=True,
        help="Path to a trials root; repeat to combine split/cohort roots.",
    )
    parser.add_argument("--model", required=True, help="Model label for the printed row.")
    parser.add_argument("--k", type=int, default=2, help="Replicates per task (default: 2).")
    parser.add_argument(
        "--tasks-dir",
        default=None,
        help="Path to tasks/ for canonical name resolution "
        "(default: the repository's tasks/ directory).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require complete selected-source metric inputs; return 3 on any gap.",
    )
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=DEFAULT_EXPECTED_TASKS,
        help=f"Expected task count for the completeness gate (default: {DEFAULT_EXPECTED_TASKS}).",
    )
    parser.add_argument(
        "--expected-judge-model",
        default=DEFAULT_EXPECTED_JUDGE_MODEL,
        help="Required judge for every substantive patch; provider prefixes are ignored "
        f"(default: {DEFAULT_EXPECTED_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--judge-out-name",
        default="judge_verdict.json",
        help="Per-trial correctness verdict filename (default: judge_verdict.json).",
    )
    parser.add_argument(
        "--u-corr-source",
        choices=("single", "threeway"),
        default="single",
        help="User Correction source (default: canonical single tagger; "
        "threeway is an optional ensemble).",
    )
    parser.add_argument(
        "--tag-sidecar",
        default=None,
        help="Immutable single-tagger sidecar from tag_messages --output-sidecar; "
        "when set, source trial verdicts are not read for U-Corr.",
    )
    parser.add_argument(
        "--expected-tag-model",
        default=None,
        help="Required single/Judge-A tagging model. Strict mode defaults to "
        "the canonical Gemini 3.1 Pro tagger.",
    )
    parser.add_argument(
        "--expected-tag-judge-b-model",
        default=kg.ENSEMBLE_TAG_JUDGE_B,
        help="Required optional-ensemble Judge-B model (default: Claude Opus 4.6).",
    )
    parser.add_argument(
        "--expected-tag-arbiter-model",
        default=kg.ENSEMBLE_TAG_ARBITER,
        help="Required optional-ensemble arbiter model (default: GPT-5.5).",
    )
    parser.add_argument(
        "--output",
        "--output-path",
        dest="output",
        default=None,
        help="JSON output path (default: inside one root, or a stable hashed name for multiple roots).",
    )
    args = parser.parse_args(argv)

    if args.k <= 0:
        print("error: --k must be positive", file=sys.stderr)
        return 2
    if args.expected_tasks <= 0:
        print("error: --expected-tasks must be positive", file=sys.stderr)
        return 2
    if not _normalized_model_id(args.expected_judge_model):
        print("error: --expected-judge-model must not be empty", file=sys.stderr)
        return 2

    # Sort and deduplicate aliases of the same root so repeated CLI arguments
    # cannot accidentally double-count trials or change the default filename.
    trials_dirs = sorted(
        {Path(raw).resolve() for raw in args.trials_dir}, key=lambda path: str(path)
    )
    missing_roots = [path for path in trials_dirs if not path.is_dir()]
    if missing_roots:
        for path in missing_roots:
            print(f"error: trials dir not found: {path}", file=sys.stderr)
        return 1

    if args.tasks_dir:
        tasks_root = Path(args.tasks_dir).resolve()
    else:
        tasks_root = DEFAULT_TASKS_ROOT
    if not tasks_root.is_dir():
        print(f"error: tasks dir not found: {tasks_root}", file=sys.stderr)
        return 1

    if Path(args.judge_out_name).name != args.judge_out_name:
        print("error: --judge-out-name must be a filename", file=sys.stderr)
        return 2
    if args.tag_sidecar and args.u_corr_source != "single":
        print("error: --tag-sidecar currently supports --u-corr-source single", file=sys.stderr)
        return 2
    expected_tag_model = args.expected_tag_model
    if not expected_tag_model and (args.strict or args.u_corr_source == "threeway"):
        expected_tag_model = kg.CANONICAL_TAG_MODEL
    tag_sidecar_path = Path(args.tag_sidecar).resolve() if args.tag_sidecar else None
    tag_sidecar = _load_json(tag_sidecar_path) if tag_sidecar_path else None
    if tag_sidecar_path and tag_sidecar is None:
        print(f"error: invalid tag sidecar: {tag_sidecar_path}", file=sys.stderr)
        return 1
    sidecar_issues = (
        _tag_sidecar_issues(tag_sidecar, expected_tag_model)
        if tag_sidecar_path
        else []
    )
    records, incomplete_dirs = _collect_trials_detailed(
        trials_dirs,
        tasks_root,
        args.judge_out_name,
        args.u_corr_source,
        expected_tag_model,
        args.expected_tag_judge_b_model,
        args.expected_tag_arbiter_model,
        tag_sidecar,
    )
    result = compute_metrics(records, args.k)
    agg = result["aggregates"]
    issues = completeness_issues(
        records,
        incomplete_dirs=incomplete_dirs,
        tasks_root=tasks_root,
        k=args.k,
        expected_tasks=args.expected_tasks,
        expected_judge_model=args.expected_judge_model,
        extra_issues=sidecar_issues,
    )
    metric_complete = args.strict and not issues
    ensemble_u_corr_models_match = (
        kg.normalized_model_id(expected_tag_model)
        == kg.normalized_model_id(kg.CANONICAL_TAG_MODEL)
        and kg.normalized_model_id(args.expected_tag_judge_b_model)
        == kg.normalized_model_id(kg.ENSEMBLE_TAG_JUDGE_B)
        and kg.normalized_model_id(args.expected_tag_arbiter_model)
        == kg.normalized_model_id(kg.ENSEMBLE_TAG_ARBITER)
    )
    ensemble_u_corr_protocol_complete = (
        metric_complete
        and args.u_corr_source == "threeway"
        and ensemble_u_corr_models_match
    )
    canonical_u_corr_protocol_complete = (
        metric_complete
        and args.u_corr_source == "single"
        and kg.normalized_model_id(expected_tag_model)
        == kg.normalized_model_id(kg.CANONICAL_TAG_MODEL)
    )

    header, row = format_row(args.model, agg)
    if canonical_u_corr_protocol_complete:
        print(
            "STRICT COMPLETE — complete Table-2 metric inputs including the "
            "released single-Gemini U-Corr protocol "
            "(action/sandbox provenance is assessed separately)"
        )
    elif ensemble_u_corr_protocol_complete:
        print(
            "STRICT COMPLETE — complete Table-2 metric inputs "
            "including the optional three-model U-Corr ensemble "
            "(action/sandbox provenance is assessed separately)"
        )
    elif metric_complete:
        if args.u_corr_source == "single":
            detail = (
                "selected single-tagger metric inputs are complete, but the "
                "tag-model pin differs from the released Gemini protocol"
            )
        else:
            detail = "selected three-way inputs are complete, but model pins differ"
        print(f"STRICT COMPLETE — {detail}")
    elif args.strict:
        print("STRICT INCOMPLETE — metrics below are diagnostic/noncanonical")
    else:
        print("DIAGNOSTIC/NONCANONICAL — rerun with --strict for a metric completeness gate")
    print(header)
    print("-" * len(header))
    print(row)
    print()
    print(
        f"n_tasks={agg['n_tasks']}  n_trials={agg['n_trials']}  "
        f"n_judged={agg['n_judged']}  n_tasks_with_full_k(k={args.k})={agg['n_tasks_with_full_k']}  "
        f"n_incomplete_skipped={len(incomplete_dirs)}"
    )
    if issues:
        print(f"completeness_issues={len(issues)}")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  - ... {len(issues) - 20} more (see JSON output)")

    summary = {
        "model": args.model,
        "trials_dir": str(trials_dirs[0]) if len(trials_dirs) == 1 else None,
        "trials_dirs": [str(path) for path in trials_dirs],
        "tasks_dir": str(tasks_root),
        "k": args.k,
        "tau": TAU,
        "strict": args.strict,
        "expected_tasks": args.expected_tasks,
        "expected_judge_model": args.expected_judge_model,
        "judge_out_name": args.judge_out_name,
        "u_corr_source": args.u_corr_source,
        "tag_sidecar": str(tag_sidecar_path) if tag_sidecar_path else None,
        "tag_transport": (
            tag_sidecar.get("tag_transport")
            if isinstance(tag_sidecar, dict)
            else None
        ),
        "expected_tag_model": expected_tag_model,
        "expected_tag_judge_b_model": args.expected_tag_judge_b_model,
        "expected_tag_arbiter_model": args.expected_tag_arbiter_model,
        "metric_complete": metric_complete,
        "canonical_u_corr_protocol_complete": canonical_u_corr_protocol_complete,
        "ensemble_u_corr_protocol_complete": ensemble_u_corr_protocol_complete,
        "ensemble_u_corr_models_match": ensemble_u_corr_models_match,
        # Completeness alone cannot establish canonical infrastructure/model
        # provenance. Keep this explicit instead of emitting a misleading bool.
        "canonical_eligible": None,
        "status": (
            "strict_complete"
            if metric_complete
            else "strict_incomplete"
            if args.strict
            else "diagnostic_noncanonical"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_incomplete_skipped": len(incomplete_dirs),
        "incomplete_trial_dirs": incomplete_dirs,
        "completeness_issues": issues,
        "row": row,
        "row_is_canonical": None,
        "aggregates": agg,
        "strict_aggregates": agg if metric_complete else None,
        "per_task": result["per_task"],
    }
    out_path = Path(args.output).resolve() if args.output else _default_output_path(trials_dirs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {out_path}")
    return 3 if args.strict and issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
