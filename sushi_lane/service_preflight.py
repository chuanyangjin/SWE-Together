#!/usr/bin/env python3
"""Authenticated identity/health preflight for the Sushi vLLM route."""

from __future__ import annotations

import argparse
import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path


def load_private_key(path: Path) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("API key file must be a regular non-symlink file")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RuntimeError("API key file must be owner-only and owned by the current user")
    value = path.read_text().strip()
    if len(value) < 16:
        raise RuntimeError("API key file is empty or implausibly short")
    return value


def get(opener, url: str, key: str | None = None) -> tuple[int, bytes]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener.open(request, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        # Deliberately do not log response bodies; gateway errors may echo data.
        return exc.code, b""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-checkpoint", required=True)
    args = parser.parse_args()

    key = load_private_key(args.api_key_file)
    base = args.base_url.rstrip("/")
    if base.endswith("/v1"):
        origin = base[:-3]
        models_url = base + "/models"
    else:
        origin = base
        models_url = base + "/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    health_status, _ = get(opener, origin + "/health", key)
    if health_status != 200:
        raise RuntimeError(f"service health failed with HTTP {health_status}")
    unauth_status, _ = get(opener, models_url)
    if unauth_status not in {401, 403}:
        raise RuntimeError(f"service did not reject unauthenticated model lookup: HTTP {unauth_status}")
    auth_status, body = get(opener, models_url, key)
    if auth_status != 200:
        raise RuntimeError(f"authenticated model lookup failed with HTTP {auth_status}")
    payload = json.loads(body)
    cards = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(cards, list) or len(cards) != 1:
        raise RuntimeError("service model list is not exactly one model")
    card = cards[0]
    if card.get("id") != args.expected_model:
        raise RuntimeError("service model identity mismatch")
    if card.get("root") != args.expected_checkpoint:
        raise RuntimeError("service checkpoint identity mismatch")
    print(
        "SUSHI_SERVICE_PREFLIGHT_OK"
        f" model={args.expected_model} checkpoint={args.expected_checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
