"""Format-aware helpers for deciding whether a captured patch has edits."""

from __future__ import annotations

import re
from pathlib import Path

_CHANGE_MARKER = re.compile(
    r"^(?:diff --git |@@ |GIT binary patch\s*$|Binary files .+ differ\s*$)",
    re.MULTILINE,
)


def patch_text_has_changes(text: str) -> bool:
    """Return true for a unified or binary git patch with an actual change.

    The repo-diff collector emits ``=== ... ===`` repository headings even when
    nothing changed. A byte threshold mistakes those headings for a patch and
    can also discard genuinely tiny edits; structural markers avoid both cases.
    """
    return bool(_CHANGE_MARKER.search(text))


def patch_file_has_changes(path: Path) -> bool:
    try:
        return patch_text_has_changes(path.read_text(errors="replace"))
    except OSError:
        return False


__all__ = ["patch_file_has_changes", "patch_text_has_changes"]
