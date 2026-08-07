#!/usr/bin/env python3
"""Remove the live Qwen deployment bearer from one trial-artifact root.

The bearer is read in-process from the deployment's private ``proxy_info``
file.  It is never accepted on argv and is never printed.  This is a narrow
incident-containment tool for the Qwen3.5-4B lane; future artifacts must also
be prevented from persisting the value at capture time.
"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path
import json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFO = Path(
    "/checkpoint/ram/shared/vllm_deployments_v2/"
    "swe-qwen35-4b-48/proxy_info.json"
)
REPLACEMENT = b"<redacted-qwen-deployment-bearer>"


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def redact(root: Path, info_path: Path) -> tuple[int, int]:
    root = root.resolve()
    allowed = (REPO_ROOT / "trials").resolve()
    if not root.is_dir() or not _inside(root, allowed):
        raise RuntimeError(f"artifact root must be a directory below {allowed}")

    info = json.loads(info_path.read_text())
    secret_text = info.get("api_key") if isinstance(info, dict) else None
    if not isinstance(secret_text, str) or not secret_text:
        raise RuntimeError("deployment info has no nonempty api_key")
    secret = secret_text.encode()

    scanned = 0
    changed = 0
    for path in root.rglob("*"):
        try:
            metadata = path.lstat()
        except OSError:
            continue
        if not stat.S_ISREG(metadata.st_mode):
            continue
        scanned += 1
        try:
            original = path.read_bytes()
        except OSError:
            continue
        if secret not in original:
            continue
        replacement = original.replace(secret, REPLACEMENT)
        temporary = path.with_name(f".{path.name}.qwen-redact-{os.getpid()}")
        try:
            temporary.write_bytes(replacement)
            os.chmod(temporary, stat.S_IMODE(metadata.st_mode))
            os.replace(temporary, path)
            os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        finally:
            if temporary.exists():
                temporary.unlink()
        changed += 1
    return scanned, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--proxy-info", type=Path, default=DEFAULT_INFO)
    args = parser.parse_args()
    scanned, changed = redact(args.trials_root, args.proxy_info)
    print(f"Qwen bearer redaction: scanned={scanned} changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
