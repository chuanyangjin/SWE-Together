"""Message tagger — the production tagging step (Stage 3).

One judge call per trial (default Gemini-3.1-Pro):
  - tag-call (prompts/tag_messages_system.md) → per-message {tags, frustration}
written per trial_idx into intent_coverage_verdict.json :: trial_msg_tags.

trial_msg_tags drives the User Correction metric (user_metrics owns the weight):
  - User Correction = #correction + 0.2·#nudge   (user_metrics.user_correction)
It is derived (user_metrics.metrics_from_rows) and persisted into the verdict as
top-level user_correction — the same deriver eval/run_eval.py aggregates, so stored
and recomputed values can never diverge.

Reproducibility: pinned model @ temp 0, versioned prompt
(prompts/tag_messages_system.md), versioned taxonomy (user_metrics.py).

RUN WITH .venv/bin/python3 (bare python3 = anaconda, no harbor → silent crash).
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
from pathlib import Path

import httpx

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "harbor" / "src"))
from eval.user_behavior.coverage_one import (   # noqa: E402
    _make_llm, load_dotenv, load_trial_sim_msgs, parse_json, DEFAULT_MODEL,
)
from eval.user_behavior import user_metrics as kg  # noqa: E402

TAG_SYS = (_HERE.parent / "prompts" / "tag_messages_system.md").read_text()    # tags + frustration
TAG_PROMPT_SHA256 = kg.TAG_PROMPT_SHA256

# Appended only after the model returns an invalid response.  Keep this outside
# the canonical system-prompt file: the first request and its recorded prompt
# hash remain identical to the released protocol, while deterministic retries
# receive enough generic feedback to correct schema-only mistakes.  Never quote
# or repair the model's labels here.
_TAG_RETRY_VALIDATION_FEEDBACK = (
    "Validation correction: return the complete JSON object again with exactly "
    "one unique row for every requested trial_idx and no extra rows. In every "
    "row, tags must be a duplicate-free list containing only allowed tags and "
    "at least one base speech-act tag (request, question, verification, workflow, "
    "approval, or context); frustration must be true/false or integer 0/1."
)

VERTEX_GATEWAY_BACKEND = "vertex-gateway"
_VERTEX_GATEWAY_HOST = "vertex.ai-gateway.x2p.facebook.net"
_VERTEX_GATEWAY_PROJECT = "devai-mea-egeit"
_VERTEX_GATEWAY_LOCATION = "global"
_VERTEX_GATEWAY_API_VERSION = "v1beta1"
_VERTEX_ATTRIBUTION_HEADERS = {
    "X-Meta-AI-Gateway-Calling-Product",
    "X-Meta-AI-Gateway-App-Instance-Id",
    "X-Meta-AI-Gateway-Trace-Id",
}

_TAG_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "results": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "trial_idx": {"type": "INTEGER"},
                    "tags": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "frustration": {"type": "INTEGER"},
                },
                "required": ["trial_idx", "tags", "frustration"],
            },
        }
    },
    "required": ["results"],
}


class _TextResponse:
    def __init__(self, content: str):
        self.content = content


def _vertex_attribution_headers() -> dict[str, str]:
    """Read only the gateway attribution headers from Claude's header bundle."""
    parsed: dict[str, str] = {}
    for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in _VERTEX_ATTRIBUTION_HEADERS and value.strip():
            parsed[key] = value.strip()
    missing = sorted(_VERTEX_ATTRIBUTION_HEADERS - parsed.keys())
    if missing:
        raise RuntimeError(
            "Vertex gateway attribution is unavailable; missing header names: "
            + ", ".join(missing)
        )
    return parsed


class _VertexGatewayLLM:
    """Small Harbor-LLM-compatible client for Vertex ``generateContent``."""

    def __init__(self, model: str, temperature: float):
        model_id = model.rsplit("/", 1)[-1].strip()
        if model_id != "gemini-3.1-pro-preview":
            raise ValueError(
                "the internal Vertex tagger is pinned to gemini-3.1-pro-preview"
            )
        self._url = (
            f"http://{_VERTEX_GATEWAY_HOST}/{_VERTEX_GATEWAY_API_VERSION}/projects/"
            f"{_VERTEX_GATEWAY_PROJECT}/locations/{_VERTEX_GATEWAY_LOCATION}/"
            f"publishers/google/models/{model_id}:generateContent"
        )
        self._temperature = temperature
        self._client = httpx.AsyncClient(
            trust_env=True,
            timeout=300.0,
            headers=_vertex_attribution_headers(),
        )

    async def call(self, prompt: str, **_kwargs) -> _TextResponse:
        response = await self._client.post(
            self._url,
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": prompt}]},
                ],
                "generationConfig": {
                    "temperature": self._temperature,
                    # Gemini 3.1 Pro may spend part of this budget on internal
                    # reasoning before emitting the structured response.  The
                    # 8k ceiling reproducibly truncated otherwise tiny JSON
                    # for long (13-message) trials, so leave enough headroom
                    # for a complete schema-constrained answer.
                    "maxOutputTokens": 32768,
                    # Native JSON mode makes quote escaping deterministic for
                    # long code-heavy user messages.  Prompt-only JSON requests
                    # occasionally returned almost-valid objects that failed
                    # after all three parser retries.
                    "responseMimeType": "application/json",
                    "responseSchema": _TAG_RESPONSE_SCHEMA,
                },
            },
        )
        if response.status_code != 200:
            # Deliberately omit the response body: internal gateway errors can
            # echo request metadata that should not enter batch logs.
            raise RuntimeError(f"Vertex gateway HTTP {response.status_code}")
        try:
            payload = response.json()
            candidate = (payload.get("candidates") or [])[0]
            parts = (candidate.get("content") or {}).get("parts") or []
            text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid Vertex generateContent response") from exc
        if not text.strip():
            raise RuntimeError("empty Vertex generateContent response")
        return _TextResponse(text)

    async def aclose(self) -> None:
        await self._client.aclose()


def _make_tag_llm(model: str, temperature: float, backend: str):
    if backend == VERTEX_GATEWAY_BACKEND:
        return _VertexGatewayLLM(model, temperature)
    return _make_llm(model, temperature)


def _tag_transport_provenance(backend: str) -> dict[str, str]:
    provenance = {"backend": backend}
    if backend == VERTEX_GATEWAY_BACKEND:
        provenance.update(
            {
                "protocol": "vertex-generateContent",
                "api_version": _VERTEX_GATEWAY_API_VERSION,
                "gateway_host": _VERTEX_GATEWAY_HOST,
                "project": _VERTEX_GATEWAY_PROJECT,
                "location": _VERTEX_GATEWAY_LOCATION,
            }
        )
    return provenance


async def _close_llm(llm) -> None:
    close = getattr(llm, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _build_user(msgs) -> str:
    return kg.tag_input_text(msgs)


def _tag_row_schema_issue(row: dict) -> str | None:
    """Return why a model row is unusable, without repairing model output."""
    if type(row.get("trial_idx")) is not int:
        return "trial_idx must be an integer"

    tags = row.get("tags")
    if not isinstance(tags, list):
        return "tags must be a list"
    if not all(isinstance(tag, str) for tag in tags):
        return "tags must contain only strings"
    if len(tags) != len(set(tags)):
        return "tags must not contain duplicates"
    unknown = set(tags) - kg.ALL_TAGS
    if unknown:
        return f"tags contain unknown values: {sorted(unknown)}"
    if not (set(tags) & kg.BASE):
        return "tags must contain at least one base speech act"

    frustration = row.get("frustration")
    if not (
        type(frustration) is bool
        or (type(frustration) is int and frustration in (0, 1))
    ):
        return "frustration must be a boolean or integer 0/1"
    return None


def _normalize(r: dict) -> dict | None:
    """Canonicalize a structurally valid row; never synthesize missing labels."""
    if _tag_row_schema_issue(r) is not None:
        return None
    return {
        "trial_idx": r["trial_idx"],
        "tags": sorted(r["tags"]),
        "frustration": int(r["frustration"]),
    }


async def _ask(
    llm,
    system: str,
    user: str,
    expected_indices: set[int],
    attempts: int = 3,
) -> dict:
    """Return one exact row per requested message, retrying partial responses."""
    base_prompt = system + "\n\n" + user
    add_validation_feedback = False
    for att in range(attempts):
        prompt = (
            base_prompt + "\n\n" + _TAG_RETRY_VALIDATION_FEEDBACK
            if add_validation_feedback
            else base_prompt
        )
        try:
            resp = await llm.call(prompt)
        except Exception:
            # A transport/provider failure supplied no model output to correct,
            # so retry the unchanged request.  Validation feedback begins only
            # after a response was received and failed parsing or validation.
            if att == attempts - 1:
                raise
            await asyncio.sleep(4)
            continue
        try:
            obj = parse_json(resp.content)
            raw_rows = obj.get("results", [])
            if not isinstance(raw_rows, list):
                raise ValueError("tagger results must be a list")
            rows = [
                row
                for row in raw_rows
                if isinstance(row, dict)
                and type(row.get("trial_idx")) is int
            ]
            indices = [row["trial_idx"] for row in rows]
            if (
                len(rows) != len(raw_rows)
                or len(indices) != len(set(indices))
                or set(indices) != expected_indices
            ):
                raise ValueError(
                    "tagger returned incomplete, duplicate, or unexpected trial_idx rows"
                )
            schema_issues = [
                f"trial_idx={row['trial_idx']}: {issue}"
                for row in rows
                if (issue := _tag_row_schema_issue(row)) is not None
            ]
            if schema_issues:
                raise ValueError(
                    "tagger returned structurally invalid rows: "
                    + "; ".join(schema_issues)
                )
            normalized = [_normalize(row) for row in rows]
            # Schema validation above guarantees normalization cannot discard a
            # row. Keep this assertion local so future normalizer changes fail
            # closed instead of silently shrinking the response.
            if any(row is None for row in normalized):
                raise ValueError("tagger row normalization failed")
            return {row["trial_idx"]: row for row in normalized if row is not None}
        except Exception:
            add_validation_feedback = True
            if att == attempts - 1:
                raise
            await asyncio.sleep(4)


async def tag_one(llm, trial_dir: Path, task_dir: Path | None) -> list[dict]:
    """Tag every (non-instruction) sim message of one trial: one judge call →
    {tags, frustration} per message."""
    sim = load_trial_sim_msgs(trial_dir, task_dir)
    msgs = [m for m in sim if m["trial_idx"] != 0]
    if not msgs:
        return []
    user = _build_user(msgs)
    tag_rows = await _ask(
        llm,
        TAG_SYS,
        user,
        {message["trial_idx"] for message in msgs},
    )
    rows = []
    for m in msgs:
        i = m["trial_idx"]
        row = tag_rows.get(i) or {}
        nr = _normalize({"trial_idx": i, "tags": row.get("tags"),
                         "frustration": row.get("frustration")})
        if nr:
            rows.append(nr)
    return rows


def _resolve_dirs(verdict_path: str, verdict: dict):
    td = verdict.get("trial_dir")
    if not td or not os.path.isdir(td):
        td = os.path.dirname(verdict_path)
    tk = verdict.get("task_dir")
    if not tk or not os.path.isdir(tk):
        task = os.path.basename(os.path.dirname(verdict_path)).split("__")[0]
        cand = str(REPO / "tasks" / task)
        if os.path.isdir(cand):
            tk = cand
        else:
            # Harbor truncates trial-name task prefixes to 32 characters.
            matches = sorted(
                p for p in (REPO / "tasks").iterdir()
                if p.is_dir() and p.name.startswith(task)
            )
            tk = str(matches[0]) if len(matches) == 1 else None
    return Path(td), (Path(tk) if tk else None)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


async def _tag_into_verdict(
    llm,
    sem,
    vp: str,
    force: bool,
    model: str,
    require_provenance: bool,
):
    async with sem:
        try:
            with open(vp) as handle:
                v = json.load(handle)
        except (OSError, json.JSONDecodeError):
            v = {}
        # Presence, not truthiness, distinguishes a tagged no-follow-up trial
        # ([]) from a trial that has never been tagged.
        if "trial_msg_tags" in v and not force:
            provenance_ok = not kg.tagging_provenance_issues(
                v.get("message_tagging"), model
            )
            if provenance_ok or not require_provenance:
                return "skip"
        try:
            rows = await tag_one(llm, *_resolve_dirs(vp, v))
        except Exception as e:
            return f"err: {type(e).__name__}: {e}"[:160]
        v["trial_msg_tags"] = rows
        m = kg.metrics_from_rows(rows)          # persist the derived User Correction metric
        v["user_correction"] = m["user_correction"]
        v["message_tagging"] = kg.tagging_provenance(model)
        v.pop("user_effort", None)             # drop legacy effort field if present
        v.pop("trial_msg_specificity", None)   # drop the old single-label kind_hint/tier block
        _write_json_atomic(Path(vp), v)
        return "ok"


def _discover_verdict_paths(trials_roots) -> list[str]:
    return sorted(
        str(trial / "intent_coverage_verdict.json")
        for root in trials_roots
        for trial in Path(root).glob("*__*")
        if trial.is_dir() and (trial / "result.json").exists()
    )


async def run_batch(
    trials_roots,
    model=DEFAULT_MODEL,
    workers=50,
    force=False,
    require_provenance=False,
    backend="litellm",
):
    load_dotenv(REPO)
    if (
        backend != VERTEX_GATEWAY_BACKEND
        and model.startswith("gemini/")
        and not os.environ.get("GEMINI_API_KEY")
    ):
        sys.exit("!! GEMINI_API_KEY missing")
    # Tagging is independent of intent-coverage matching. Discover completed
    # trial directories directly and create the verdict container when the
    # coverage step was intentionally skipped or unavailable.
    vps = _discover_verdict_paths(trials_roots)
    llm = _make_tag_llm(model, 0.0, backend)
    sem = asyncio.Semaphore(workers)
    print(f"tagging {len(vps)} trials with {model}, workers={workers}", flush=True)
    res = {}
    B = 200
    try:
        for i in range(0, len(vps), B):
            out = await asyncio.gather(
                *[
                    _tag_into_verdict(
                        llm, sem, vp, force, model, require_provenance
                    )
                    for vp in vps[i : i + B]
                ]
            )
            for o in out:
                res[o.split(":")[0]] = res.get(o.split(":")[0], 0) + 1
            print(f"  {min(i+B,len(vps))}/{len(vps)}  {res}", flush=True)
    finally:
        await _close_llm(llm)
    print(f"DONE {res}", flush=True)
    return res


async def _tag_sidecar_one(llm, sem, vp: str) -> tuple[str, str, dict | str]:
    trial_name = Path(vp).parent.name
    async with sem:
        try:
            try:
                with open(vp) as handle:
                    verdict = json.load(handle)
            except (OSError, json.JSONDecodeError):
                verdict = {}
            trial_dir, task_dir = _resolve_dirs(vp, verdict)
            sim_messages = load_trial_sim_msgs(trial_dir, task_dir)
            tag_messages = [
                message for message in sim_messages if message["trial_idx"] != 0
            ]
            rows = await tag_one(llm, trial_dir, task_dir)
            tags = {str(row["trial_idx"]): list(row.get("tags") or []) for row in rows}
            return "ok", trial_name, {
                "tags": tags,
                "rows": rows,
                "input_sha256": kg.tag_input_sha256(tag_messages),
            }
        except Exception as exc:  # noqa: BLE001 - batch must report every failed trial
            return "err", trial_name, f"{type(exc).__name__}: {exc}"[:300]


async def run_sidecar_batch(
    trials_roots,
    *,
    model: str,
    workers: int,
    output: Path,
    force: bool = False,
    backend: str = "litellm",
) -> dict[str, int]:
    """Tag into a resumable sidecar without modifying the source trials."""
    load_dotenv(REPO)
    if (
        backend != VERTEX_GATEWAY_BACKEND
        and model.startswith("gemini/")
        and not os.environ.get("GEMINI_API_KEY")
    ):
        raise RuntimeError("GEMINI_API_KEY missing")

    expected_provenance = kg.tagging_provenance(model)
    expected_transport = _tag_transport_provenance(backend)
    sidecar: dict = {
        "schema_version": kg.TAGGING_SCHEMA_VERSION,
        "message_tagging": expected_provenance,
        "tag_transport": expected_transport,
        "trials": {},
        "trial_rows": {},
        "trial_input_sha256": {},
    }
    if output.exists() and not force:
        try:
            existing = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid tag sidecar {output}: {exc}") from exc
        if not isinstance(existing, dict) or not isinstance(existing.get("trials"), dict):
            raise RuntimeError(
                f"legacy/unversioned tag sidecar requires --force: {output}"
            )
        if "trial_rows" in existing and not isinstance(existing["trial_rows"], dict):
            raise RuntimeError(f"invalid trial_rows in tag sidecar: {output}")
        if "trial_input_sha256" in existing and not isinstance(
            existing["trial_input_sha256"], dict
        ):
            raise RuntimeError(f"invalid trial_input_sha256 in tag sidecar: {output}")
        if existing.get("tag_transport") != expected_transport:
            raise RuntimeError(
                f"tag sidecar transport mismatch; use --force to regenerate {output}"
            )
        provenance_issues = kg.tagging_provenance_issues(
            existing.get("message_tagging"), model
        )
        if existing.get("schema_version") != kg.TAGGING_SCHEMA_VERSION:
            provenance_issues.append("sidecar_schema_version")
        if provenance_issues:
            raise RuntimeError(
                f"tag sidecar provenance mismatch ({', '.join(provenance_issues)}); "
                "use --force to regenerate it"
            )
        sidecar = existing
        sidecar.setdefault("trial_rows", {})
        sidecar.setdefault("trial_input_sha256", {})

    vps = _discover_verdict_paths(trials_roots)
    names = [Path(vp).parent.name for vp in vps]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate trial directory names across --trials-root values")
    existing_trials = sidecar["trials"]
    existing_rows = sidecar["trial_rows"]
    existing_input_hashes = sidecar["trial_input_sha256"]
    current_input_hashes: dict[str, str] = {}
    for vp in vps:
        trial_name = Path(vp).parent.name
        try:
            with open(vp) as handle:
                verdict = json.load(handle)
        except (OSError, json.JSONDecodeError):
            verdict = {}
        trial_dir, task_dir = _resolve_dirs(vp, verdict)
        messages = [
            message
            for message in load_trial_sim_msgs(trial_dir, task_dir)
            if message["trial_idx"] != 0
        ]
        current_input_hashes[trial_name] = kg.tag_input_sha256(messages)
    pending = [
        vp
        for vp in vps
        if force
        or Path(vp).parent.name not in existing_trials
        or Path(vp).parent.name not in existing_rows
        or existing_input_hashes.get(Path(vp).parent.name)
        != current_input_hashes[Path(vp).parent.name]
    ]
    llm = _make_tag_llm(model, 0.0, backend)
    sem = asyncio.Semaphore(workers)
    counts = {"ok": 0, "skip": len(vps) - len(pending), "err": 0}
    batch_size = 200
    try:
        for start in range(0, len(pending), batch_size):
            results = await asyncio.gather(
                *[
                    _tag_sidecar_one(llm, sem, vp)
                    for vp in pending[start : start + batch_size]
                ]
            )
            for status, trial_name, payload in results:
                counts[status] += 1
                if status == "ok" and isinstance(payload, dict):
                    existing_trials[trial_name] = payload["tags"]
                    existing_rows[trial_name] = payload["rows"]
                    existing_input_hashes[trial_name] = payload["input_sha256"]
                else:
                    print(f"ERROR {trial_name}: {payload}", file=sys.stderr)
            _write_json_atomic(output, sidecar)
            print(
                f"  {min(start + batch_size, len(pending))}/{len(pending)} "
                f"sidecar trials; {counts}",
                flush=True,
            )
    finally:
        await _close_llm(llm)
    if not pending:
        _write_json_atomic(output, sidecar)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-root", action="append", required=True,
                    help="cohort dir(s) holding <trial>/intent_coverage_verdict.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--backend",
        choices=("litellm", VERTEX_GATEWAY_BACKEND),
        default="litellm",
        help="Tagger transport (vertex-gateway uses internal Vertex generateContent).",
    )
    ap.add_argument("--workers", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="re-tag even if trial_msg_tags exists")
    ap.add_argument(
        "--require-provenance",
        action="store_true",
        help="Re-tag legacy/mismatched trial_msg_tags instead of accepting them.",
    )
    ap.add_argument(
        "--output-sidecar",
        type=Path,
        default=None,
        help="Write a versioned second-judge tag sidecar instead of modifying verdicts.",
    )
    a = ap.parse_args()
    if a.output_sidecar:
        try:
            results = asyncio.run(
                run_sidecar_batch(
                    a.trials_root,
                    model=a.model,
                    workers=a.workers,
                    output=a.output_sidecar,
                    force=a.force,
                    backend=a.backend,
                )
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        return 2 if results.get("err") else 0
    results = asyncio.run(
        run_batch(
            a.trials_root,
            a.model,
            a.workers,
            a.force,
            a.require_provenance,
            a.backend,
        )
    )
    return 2 if any(key.startswith("err") for key in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
