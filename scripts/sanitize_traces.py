#!/usr/bin/env python3
"""Remove credentials from trial artifacts before local use or upload.

The sanitizer is deliberately silent about secret values. It recursively
redacts credential-named JSON fields, replaces credentials known from the
process environment or repository ``.env``, and handles common key assignment
and authorization-header forms in UTF-8 text files. Binary files and symlinks
are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
HARBOR_SRC = REPO_ROOT / "external" / "harbor" / "src"
sys.path.insert(0, str(HARBOR_SRC))

from harbor.utils.redaction import (  # noqa: E402
    REDACTED_VALUE,
    is_secret_key,
)


_MIN_GLOBAL_REPLACEMENT_LENGTH = 8
_KEY_ASSIGNMENT_RE = re.compile(
    r"(?im)(?P<prefix>\b(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*)"
    r"(?:(?P<quote>[\"'])(?P<value>[^\r\n]*?)(?P=quote)|"
    r"(?P<bare>[^\s;#]+))"
)
_KEY_HEADER_RE = re.compile(
    r"(?im)(?P<prefix>^\s*(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*)"
    r"(?P<value>[^\r\n]+)"
)
_AUTH_HEADER_RE = re.compile(
    r"(?im)(?P<prefix>\b(?:authorization|proxy-authorization)\s*:\s*"
    r"(?:bearer|basic)\s+)(?P<value>\S+)"
)


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read .env values without logging them, with a dependency-free fallback."""

    if not path.is_file():
        return {}
    try:
        from dotenv import dotenv_values

        return {
            str(key): str(value)
            for key, value in dotenv_values(path).items()
            if key and value is not None
        }
    except ImportError:
        values: dict[str, str] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").lstrip()
            key, separator, value = line.partition("=")
            if not separator:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key.strip()] = value
        return values


def known_secret_values() -> tuple[str, ...]:
    """Collect credential values without ever returning names or logging values."""

    candidates = dict(_read_dotenv(REPO_ROOT / ".env"))
    candidates.update(os.environ)
    secrets = {
        value
        for key, value in candidates.items()
        if is_secret_key(key)
        and isinstance(value, str)
        and len(value) >= _MIN_GLOBAL_REPLACEMENT_LENGTH
        and value != REDACTED_VALUE
    }
    return tuple(sorted(secrets, key=len, reverse=True))


def _replace_known_secrets(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        value = value.replace(secret, REDACTED_VALUE)
    return value


def _sanitize_json_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                REDACTED_VALUE
                if is_secret_key(key)
                else _sanitize_json_value(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_json_value(item, secrets) for item in value]
    if isinstance(value, str):
        return _replace_known_secrets(value, secrets)
    return value


def _redact_assignment(match: re.Match[str]) -> str:
    if not is_secret_key(match.group("key")):
        return match.group(0)
    bare = match.groupdict().get("bare") or ""
    # Preserve executable expressions such as
    # ``API_KEY = os.environ.get("PROXY_API_KEY", "")``. Exact known secret
    # values are already replaced before this structural pass; this rule is for
    # literal assignments, not identifiers or environment lookups.
    if bare and (
        "(" in bare
        or bare.startswith(("$", "os.", "env.", "process.", "settings."))
    ):
        return match.group(0)
    quote = match.groupdict().get("quote") or ""
    return f"{match.group('prefix')}{quote}{REDACTED_VALUE}{quote}"


def _sanitize_text(text: str, secrets: tuple[str, ...]) -> str:
    text = _replace_known_secrets(text, secrets)
    text = _KEY_ASSIGNMENT_RE.sub(_redact_assignment, text)
    text = _KEY_HEADER_RE.sub(_redact_assignment, text)
    return _AUTH_HEADER_RE.sub(
        lambda match: f"{match.group('prefix')}{REDACTED_VALUE}", text
    )


def _decode_text(raw: bytes) -> str | None:
    if b"\x00" in raw[:8192]:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text:
        return text
    disallowed_controls = sum(
        ord(char) < 32 and char not in "\t\n\r\f\b" for char in text[:8192]
    )
    if disallowed_controls / min(len(text), 8192) > 0.01:
        return None
    return text


def _atomic_write_text(path: Path, text: str) -> None:
    original_mode = path.stat(follow_symlinks=False).st_mode & 0o777
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.sanitize-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(original_mode)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def sanitize_file(path: Path, secrets: tuple[str, ...]) -> bool:
    """Sanitize one regular UTF-8 file and return whether it changed."""

    if path.is_symlink() or not path.is_file():
        return False
    raw = path.read_bytes()
    text = _decode_text(raw)
    if text is None:
        return False

    sanitized = text
    if path.suffix.casefold() == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            sanitized = _sanitize_text(text, secrets)
        else:
            redacted = _sanitize_json_value(parsed, secrets)
            if redacted != parsed:
                sanitized = json.dumps(redacted, indent=2, ensure_ascii=False)
                if text.endswith("\n"):
                    sanitized += "\n"
    else:
        # Patch bytes are correctness-judge input. Preserve code-like literals
        # and redact only exact credentials known from the host/.env; generic
        # assignment rewriting here could change an otherwise valid solution
        # before the separate judge stage runs.
        if path.suffix.casefold() in {".patch", ".diff"}:
            sanitized = _replace_known_secrets(text, secrets)
        else:
            sanitized = _sanitize_text(text, secrets)

    if sanitized == text:
        return False
    _atomic_write_text(path, sanitized)
    return True


def _selected_trial_dirs(
    root: Path,
    task_names: set[str] | None,
    completed_only: bool,
) -> list[Path]:
    if task_names is None and not completed_only:
        return [root]
    if task_names is not None and not task_names:
        return []
    selected: list[Path] = []
    for trial in sorted(root.iterdir()):
        if not trial.is_dir() or trial.is_symlink():
            continue
        if completed_only and not (trial / "result.json").is_file():
            continue
        try:
            config = json.loads((trial / "config.json").read_text())
            task = config.get("task")
            task_name = Path(str(task.get("path") or "")).name
        except (AttributeError, OSError, json.JSONDecodeError):
            continue
        if task_names is not None and task_name not in task_names:
            continue
        selected.append(trial)
    return selected


def sanitize_tree(
    trials_dir: Path,
    *,
    task_names: set[str] | None = None,
    completed_only: bool = False,
    workers: int = 1,
) -> tuple[int, int]:
    """Sanitize selected regular files without following links.

    The default remains the original serial traversal. Opt-in workers process
    independent files concurrently while ``executor.map`` preserves the
    deterministic traversal/result order and propagates worker exceptions.
    """

    root = trials_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if workers < 1:
        raise ValueError("workers must be positive")
    secrets = known_secret_values()

    def selected_files() -> Iterator[Path]:
        """Yield exactly the regular files visited by the serial implementation."""

        for selected_root in _selected_trial_dirs(
            root, task_names, completed_only
        ):
            for path in sorted(selected_root.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    continue
                yield path

    if workers == 1:
        scanned = 0
        changed = 0
        for path in selected_files():
            scanned += 1
            changed += sanitize_file(path, secrets)
        return scanned, changed

    sanitize = partial(sanitize_file, secrets=secrets)
    scanned = 0
    changed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for was_changed in executor.map(sanitize, selected_files()):
            scanned += 1
            changed += was_changed
    return scanned, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=REPO_ROOT / "trials",
        help="Trial output directory to sanitize recursively",
    )
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel file readers (default: 1, preserving serial behavior).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    scanned, changed = sanitize_tree(
        args.trials_dir,
        task_names=set(args.task) if args.task is not None else None,
        completed_only=args.completed_only,
        workers=args.workers,
    )
    print(f"Sanitized {changed} of {scanned} trial artifact files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
