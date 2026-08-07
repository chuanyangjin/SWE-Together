#!/usr/bin/env python3
"""Run the canonical Gemini tagger with privately sourced gateway attribution.

The attribution bundle is consumed in this process only.  Header values are
never added to argv, printed, or written to the tag sidecar.  Pass tagger
arguments after ``--``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.user_behavior import tag_messages  # noqa: E402

SOURCE_ENV = "SWE_TOGETHER_VERTEX_ATTRIBUTION_SOURCE"
_HEADER_PATTERN = re.compile(
    r"(X-Meta-AI-Gateway-[A-Za-z0-9-]+):[ \t]*([^\r\n]+)"
)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)


def _validate_private_source(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"attribution source is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("attribution source must be a regular non-symlink file")
    if metadata.st_uid != os.getuid():
        raise RuntimeError("attribution source must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RuntimeError("attribution source must not be group/world accessible")


def load_attribution_source(path: Path) -> dict[str, str]:
    """Extract one complete allowlisted header bundle from text or JSONL."""
    path = path.expanduser()
    _validate_private_source(path)
    text = path.read_text(errors="replace")
    candidates: list[str] = [text]
    for raw in text.splitlines():
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates.extend(_strings(record))

    expected = tag_messages._VERTEX_ATTRIBUTION_HEADERS
    for candidate in candidates:
        found = {
            key: value.strip()
            for key, value in _HEADER_PATTERN.findall(candidate)
            if key in expected and value.strip()
        }
        if expected.issubset(found):
            return {key: found[key] for key in expected}
    raise RuntimeError("private source has no complete Vertex attribution bundle")


def _install_attribution(source: Path | None) -> None:
    try:
        tag_messages._vertex_attribution_headers()
        return
    except RuntimeError:
        pass
    if source is None:
        raise RuntimeError(
            f"set ANTHROPIC_CUSTOM_HEADERS or {SOURCE_ENV}, or pass "
            "--attribution-source"
        )
    headers = load_attribution_source(source)
    os.environ["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(
        f"{key}: {headers[key]}" for key in sorted(headers)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Private-attribution wrapper for the Vertex Gemini tagger."
    )
    parser.add_argument(
        "--attribution-source",
        type=Path,
        default=(Path(os.environ[SOURCE_ENV]) if os.environ.get(SOURCE_ENV) else None),
        help=f"Private text/JSONL source (or {SOURCE_ENV}); values are never logged.",
    )
    parser.add_argument("tagger_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = list(args.tagger_args)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    if not forwarded:
        parser.error("pass tagger arguments after --")
    if "--backend" in forwarded:
        parser.error("the wrapper owns --backend")

    try:
        _install_attribution(args.attribution_source)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    sys.argv = [
        "eval.user_behavior.tag_messages",
        "--backend",
        tag_messages.VERTEX_GATEWAY_BACKEND,
        *forwarded,
    ]
    return tag_messages.main()


if __name__ == "__main__":
    raise SystemExit(main())
