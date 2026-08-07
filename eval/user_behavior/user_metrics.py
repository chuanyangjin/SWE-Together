"""Single source of truth for the message-tag taxonomy + the User Correction metric.

Per sim message (stored in intent_coverage_verdict.json :: trial_msg_tags):
  - tags: list[str]   multi-label, independent presence (>=1 base act, >=0 corrective)
  - frustration: int  orthogonal affect axis (0/1)

User Correction = #correction + 0.2·#nudge  (agent-driven corrective pushback), derived
in code (no hard-coded weights anywhere else). Imported by tag_messages.py,
eval/run_eval.py, and every aggregation so the taxonomy + weight live in ONE place.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

# ── base speech acts (every message has >=1) ─────────────────────────────────
BASE = {"request", "question", "verification", "workflow", "approval", "context"}

# ── corrective layer (>=0 per message) ───────────────────────────────────────
EXPLICIT_CORRECTIVE = {"correction"}   # asserts the agent erred (redirect/reminder fold in)
IMPLICIT_CORRECTIVE = {"nudge"}        # only implies the agent erred
CORRECTIVE = EXPLICIT_CORRECTIVE | IMPLICIT_CORRECTIVE

ALL_TAGS = BASE | CORRECTIVE
AFFECT = {"frustration"}               # separate axis, can co-occur with anything

# ── User Correction metric (agent-driven) ────────────────────────────────────
W_NUDGE = 0.2          # explicit correction counts 1.0; implicit nudge counts 0.2

# Tagging provenance. The released benchmark pipeline uses the canonical single
# Gemini tagger. ``adjudicate_3way.py`` is an optional local ensemble and is not
# part of the protocol specified by the paper.
# Version 2 adds fail-closed validation of the model-produced row schema before
# normalization.  Version-1 artifacts may contain rows synthesized from malformed
# model output (for example ``tags: null`` becoming ``request``), so they are not
# provenance-compatible even though the persisted row shape looks the same.
TAGGING_SCHEMA_VERSION = 2
THREEWAY_SCHEMA_VERSION = 1
TAG_PROMPT_SHA256 = hashlib.sha256(
    (Path(__file__).parent / "prompts" / "tag_messages_system.md").read_bytes()
).hexdigest()
CANONICAL_TAG_MODEL = "gemini/gemini-3.1-pro-preview"
ENSEMBLE_TAG_JUDGE_B = "anthropic/claude-opus-4-6"
ENSEMBLE_TAG_ARBITER = "gpt-5.5"

# Legacy import names retained for the optional ensemble module and old plans.
PAPER_TAG_JUDGE_A = CANONICAL_TAG_MODEL
PAPER_TAG_JUDGE_B = ENSEMBLE_TAG_JUDGE_B
PAPER_TAG_ARBITER = ENSEMBLE_TAG_ARBITER


def tag_input_text(messages: list[dict[str, Any]]) -> str:
    """Return the exact per-trial user payload consumed by the tagger."""
    lines = [
        f'trial_idx={message["trial_idx"]}: '
        f'{" ".join(str(message.get("text") or "").split())[:1000]}'
        for message in messages
    ]
    return "Messages:\n" + "\n".join(lines)


def tag_input_sha256(messages: list[dict[str, Any]]) -> str:
    """Fingerprint the normalized message content that determines U-Corr tags."""
    return hashlib.sha256(tag_input_text(messages).encode()).hexdigest()


def user_correction(msg_tags) -> float:
    """msg_tags: iterable of per-message tag collections for ONE trial.
    Returns #correction + W_NUDGE·#nudge (the canonical User Correction score)."""
    corr = sum(1 for t in msg_tags if "correction" in t)
    nud = sum(1 for t in msg_tags if "nudge" in t)
    return corr + W_NUDGE * nud


def metrics_from_rows(trial_msg_tags) -> dict:
    """Derive the per-trial User Correction metric from `trial_msg_tags` rows
    ([{tags, ...}, ...]). SINGLE SOURCE OF TRUTH for both what gets persisted into
    intent_coverage_verdict.json (by tag_messages.py) and what eval/run_eval.py
    aggregates — so stored and recomputed values can never diverge.
    ``None`` means the trial was not tagged. An explicitly persisted empty list
    means tagging completed and there were no simulator follow-up messages, so
    the canonical correction cost is zero (it must not be dropped from means)."""
    if trial_msg_tags is None:
        return {"user_correction": None, "n_tagged_msgs": 0,
                "n_correction": None, "n_nudge": None}
    try:
        rows = list(trial_msg_tags)
    except TypeError:
        return {"user_correction": None, "n_tagged_msgs": 0,
                "n_correction": None, "n_nudge": None}
    if not rows:
        return {"user_correction": 0.0, "n_tagged_msgs": 0,
                "n_correction": 0, "n_nudge": 0}
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("tags"), list)
        or not all(isinstance(tag, str) for tag in row.get("tags", []))
        for row in rows
    ):
        return {"user_correction": None, "n_tagged_msgs": 0,
                "n_correction": None, "n_nudge": None}
    tags = [r.get("tags", []) for r in rows]
    return {
        "user_correction": round(user_correction(tags), 4),
        "n_tagged_msgs": len(rows),
        "n_correction": sum(1 for t in tags if "correction" in t),
        "n_nudge": sum(1 for t in tags if "nudge" in t),
    }


def normalized_model_id(value: Any) -> str:
    """Normalize provider-qualified model IDs for provenance comparisons."""
    if not isinstance(value, str) or not value.strip():
        return ""
    model = value.strip().lower().rstrip("/")
    # Provider/router prefixes are transport details; provenance pins the final
    # model component (for example openrouter/anthropic/claude-opus-4-6).
    return model.rsplit("/", 1)[-1].replace("_", "-")


def tagging_provenance(model: str) -> dict[str, Any]:
    """Return the versioned provenance envelope written by a single tagger."""
    return {
        "schema_version": TAGGING_SCHEMA_VERSION,
        "model": model,
        "temperature": 0.0,
        "prompt_sha256": TAG_PROMPT_SHA256,
    }


def tagging_provenance_issues(
    value: Any, expected_model: str
) -> list[str]:
    """Validate one single-tagger provenance envelope."""
    if not isinstance(value, dict):
        return ["missing_tagging_provenance"]
    issues: list[str] = []
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != TAGGING_SCHEMA_VERSION
    ):
        issues.append("tagging_schema_version")
    if normalized_model_id(value.get("model")) != normalized_model_id(expected_model):
        issues.append(f"tagging_model:{value.get('model') or 'missing'}")
    raw_temperature = value.get("temperature")
    try:
        temperature = (
            float("nan")
            if isinstance(raw_temperature, bool)
            else float(raw_temperature)
        )
    except (TypeError, ValueError):
        temperature = float("nan")
    if temperature != 0.0:
        issues.append(f"tagging_temperature:{value.get('temperature')!r}")
    if value.get("prompt_sha256") != TAG_PROMPT_SHA256:
        issues.append("tagging_prompt_sha256")
    return issues


def rows_from_verdict(verdict: Any, source: str = "single") -> list[dict] | None:
    """Select the requested U-Corr rows without silently mixing protocols."""
    if not isinstance(verdict, dict):
        return None
    field = "trial_msg_tags_3way" if source == "threeway" else "trial_msg_tags"
    rows = verdict.get(field)
    if not isinstance(rows, list):
        return None
    return rows


def metrics_from_verdict(verdict: Any, source: str = "single") -> dict:
    """Derive User Correction from the selected rows, never a cached scalar."""
    return metrics_from_rows(rows_from_verdict(verdict, source))


def user_correction_provenance_issues(
    verdict: Any,
    *,
    source: str,
    expected_tag_model: str | None = None,
    expected_judge_b_model: str = ENSEMBLE_TAG_JUDGE_B,
    expected_arbiter_model: str = ENSEMBLE_TAG_ARBITER,
    expected_trial_indices: set[int] | None = None,
) -> list[str]:
    """Return gaps that make a requested U-Corr source non-reproducible.

    Metadata is checked whenever an expected model is supplied. The optional
    ``threeway`` source additionally requires complete A/B/arbiter provenance.
    """
    if source not in {"single", "threeway"}:
        return [f"unknown_u_corr_source:{source}"]
    if not isinstance(verdict, dict):
        return ["missing_u_corr_verdict"]

    rows = rows_from_verdict(verdict, source)
    if rows is None:
        field = "trial_msg_tags_3way" if source == "threeway" else "trial_msg_tags"
        return [f"missing_{field}"]

    issues: list[str] = []
    indices = [row.get("trial_idx") for row in rows if isinstance(row, dict)]
    if (
        len(indices) != len(rows)
        or any(type(index) is not int for index in indices)
        or len(indices) != len(set(indices))
        or any(
            not isinstance(row.get("tags"), list)
            or not all(isinstance(tag, str) for tag in row.get("tags", []))
            or any(tag not in ALL_TAGS for tag in row.get("tags", []))
            or not (set(row.get("tags", [])) & BASE)
            or len(row.get("tags", [])) != len(set(row.get("tags", [])))
            or type(row.get("frustration")) is not int
            or row.get("frustration") not in (0, 1)
            for row in rows
            if isinstance(row, dict)
        )
    ):
        issues.append("invalid_tag_rows")
    if (
        expected_trial_indices is not None
        and all(type(index) is int for index in indices)
        and set(indices) != expected_trial_indices
    ):
        issues.append(
            "tag_indices_mismatch:"
            f"observed_{sorted(indices)}!=expected_{sorted(expected_trial_indices)}"
        )
    if source == "single":
        if expected_tag_model:
            issues.extend(
                tagging_provenance_issues(
                    verdict.get("message_tagging"), expected_tag_model
                )
            )
        return issues

    provenance = verdict.get("message_tagging_3way")
    if not isinstance(provenance, dict):
        return ["missing_threeway_provenance"]
    if (
        type(provenance.get("schema_version")) is not int
        or provenance.get("schema_version") != THREEWAY_SCHEMA_VERSION
    ):
        issues.append("threeway_schema_version")
    if provenance.get("source_field") != "trial_msg_tags_3way":
        issues.append("threeway_source_field")
    if provenance.get("legacy_unverified"):
        issues.append("threeway_legacy_unverified")

    judge_a_model = expected_tag_model or CANONICAL_TAG_MODEL
    issues.extend(
        f"judge_a_{issue}"
        for issue in tagging_provenance_issues(
            provenance.get("judge_a"), judge_a_model
        )
    )
    issues.extend(
        f"judge_b_{issue}"
        for issue in tagging_provenance_issues(
            provenance.get("judge_b"), expected_judge_b_model
        )
    )

    arbiter = provenance.get("arbiter")
    if not isinstance(arbiter, dict):
        issues.append("missing_arbiter_provenance")
    else:
        if normalized_model_id(arbiter.get("model")) != normalized_model_id(
            expected_arbiter_model
        ):
            issues.append(f"arbiter_model:{arbiter.get('model') or 'missing'}")
        if arbiter.get("prompt_sha256") != TAG_PROMPT_SHA256:
            issues.append("arbiter_prompt_sha256")

    sidecar_hash = provenance.get("judge_b_sidecar_sha256")
    if not isinstance(sidecar_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", sidecar_hash
    ):
        issues.append("judge_b_sidecar_sha256")

    disputed = provenance.get("disputed_trial_indices")
    votes = provenance.get("arbiter_votes")
    if not isinstance(disputed, list) or not all(type(i) is int for i in disputed):
        issues.append("invalid_disputed_trial_indices")
    elif not isinstance(votes, dict):
        issues.append("missing_arbiter_votes")
    else:
        for index in disputed:
            if str(index) not in votes:
                issues.append(f"missing_arbiter_vote:{index}")
    return issues


# ── helpers ───────────────────────────────────────────────────────────────────
def primary_kind(tags) -> str:
    """Back-compat single label (when something needs one): correction > nudge > base act."""
    if "correction" in tags:
        return "correction"
    if "nudge" in tags:
        return "nudge"
    for t in tags:
        if t in BASE:
            return t
    return "request"


def validate(tags, frustration=0) -> list[str]:
    """Soft schema warnings (non-fatal). Used by the tagger."""
    w = []
    tags = list(tags)
    if not (set(tags) & BASE):
        w.append("no base act")
    bad = set(tags) - ALL_TAGS
    if bad:
        w.append(f"unknown tags: {sorted(bad)}")
    if frustration not in (0, 1):
        w.append(f"frustration={frustration!r} not in {{0,1}}")
    return w
