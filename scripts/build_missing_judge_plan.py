#!/usr/bin/env python3
"""Build a Phase-2 plan for substantive patches without a valid verdict."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.patch_utils import patch_file_has_changes  # noqa: E402
from eval.correctness.run_batch import _phase2_verdict_issues  # noqa: E402
from src.eval_infra_sentinel import classify_trial  # noqa: E402


def _valid_completed_result(trial: Path) -> bool:
    """Require a finite verifier reward and a fresh protocol-valid infra audit."""
    try:
        result = json.loads((trial / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    verifier = result.get("verifier_result") if isinstance(result, dict) else None
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or not math.isfinite(float(reward))
    ):
        return False
    # Recompute instead of trusting a possibly stale sidecar written before a
    # late user-simulator or backend failure was observable.
    return classify_trial(trial).status == "ok"


def _reusable_verdict(
    old: object,
    expected_judge_model: str | None,
    rubric: object | None = None,
    *,
    task_name: str | None = None,
    trial_id: str | None = None,
) -> bool:
    """Require exact Phase-2 evidence, score derivation, and provenance."""
    if rubric is None or task_name is None or trial_id is None:
        return False
    return not _phase2_verdict_issues(
        rubric,
        old,
        task_name=task_name,
        trial_id=trial_id,
        expected_judge_model=expected_judge_model,
    )


def build_plan(
    trials_root: Path,
    tasks_root: Path,
    verdict_name: str,
    expected_judge_model: str | None = None,
    *,
    completed_only: bool = False,
    valid_completed_only: bool = False,
) -> list[dict[str, str]]:
    tasks = {path.name: path for path in tasks_root.iterdir() if path.is_dir()}
    plan: list[dict[str, str]] = []
    for trial in sorted(trials_root.glob("*__*")):
        if not trial.is_dir():
            continue
        # Active Harbor trials can already have an interim final.patch. Rolling
        # judges must never snapshot those mutable workspaces; result.json is
        # written only after agent execution and verification complete.
        if valid_completed_only and not _valid_completed_result(trial):
            continue
        if completed_only and not (trial / "result.json").is_file():
            continue
        if not patch_file_has_changes(trial / "agent" / "final.patch"):
            continue
        prefix = trial.name.rsplit("__", 1)[0]
        task = tasks.get(prefix) or next(
            (path for name, path in tasks.items() if name.startswith(prefix)), None
        )
        if not task or not (task / "canonical_goals.json").exists():
            continue
        try:
            rubric = json.loads((task / "canonical_goals.json").read_text())
        except (OSError, json.JSONDecodeError):
            rubric = None

        verdict = trial / verdict_name
        if verdict.exists():
            try:
                old = json.loads(verdict.read_text())
            except (OSError, json.JSONDecodeError):
                old = {}
            if _reusable_verdict(
                old,
                expected_judge_model,
                rubric,
                task_name=task.name,
                trial_id=trial.name,
            ):
                continue
        plan.append(
            {
                "trial_dir": str(trial.resolve()),
                "task_dir": str(task.resolve()),
                "out_name": verdict_name,
            }
        )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--verdict-name", required=True)
    parser.add_argument(
        "--expected-judge-model",
        default=None,
        help="Also repair verdicts produced by a different judge model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the deterministic missing plan into this many disjoint shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to emit (default: 0).",
    )
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Include only immutable trials that already have result.json.",
    )
    parser.add_argument(
        "--valid-completed-only",
        action="store_true",
        help=(
            "Also require a finite verifier reward and a fresh infra-ok audit; "
            "intended for rolling judges while repair producers are active."
        ),
    )
    args = parser.parse_args()
    if Path(args.verdict_name).name != args.verdict_name:
        parser.error("--verdict-name must be a filename")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    plan = build_plan(
        args.trials_root.resolve(),
        args.tasks_root.resolve(),
        args.verdict_name,
        args.expected_judge_model,
        completed_only=args.completed_only,
        valid_completed_only=args.valid_completed_only,
    )
    total = len(plan)
    plan = plan[args.shard_index :: args.num_shards]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"{args.verdict_name}: shard {args.shard_index + 1}/{args.num_shards} "
        f"contains {len(plan)} of {total} missing substantive verdict(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
