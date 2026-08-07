#!/usr/bin/env python3
"""Rotate the benchmark Qwen deployment bearer without exposing either value."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit


REPO = Path(__file__).resolve().parents[1]
DEPLOYMENT = Path(
    "/checkpoint/ram/shared/vllm_deployments_v2/swe-qwen35-4b-48"
)
PROXY_INFO = DEPLOYMENT / "proxy_info.json"
MASTER_KEY = DEPLOYMENT / "proxy_master_key"
PROXY_CONFIG = DEPLOYMENT / "proxy_litellm_config.yaml"
PROXY_BATCH_SOURCE = DEPLOYMENT / "src/serve_api_v2/proxy/proxy.sbatch"
CONFIG_GENERATOR_SOURCE = DEPLOYMENT / "src/serve_api_v2/proxy/litellm_config.sh"
PRODUCER_LOCK = REPO / "pipeline_logs/qwen35_4b_producer.lock"
ROTATION_LOCK = DEPLOYMENT / ".proxy_master_key.rotation.lock"
EVIDENCE = REPO / "pipeline_logs/qwen35_4b_bearer_rotation.json"
EXPECTED_MODEL = "Qwen3.5-4B"
# LiteLLM's current auth middleware maps an invalid bearer to a
# ``ProxyException`` with HTTP 400, while other releases use 401/403.  These
# are all explicit authentication rejections; transport failures and 5xx
# responses remain fail-closed.
REJECTED_AUTH_STATUSES = frozenset({400, 401, 403})


def _secure_read(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RuntimeError(f"private file validation failed: {path.name}")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(raw) > limit:
        raise RuntimeError(f"private file is unexpectedly large: {path.name}")
    return raw


def _load_proxy_info() -> dict[str, Any]:
    try:
        payload = json.loads(_secure_read(PROXY_INFO, 64 * 1024))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("proxy_info is invalid") from exc
    if not isinstance(payload, dict) or payload.get("model") != EXPECTED_MODEL:
        raise RuntimeError("proxy_info deployment identity mismatch")
    key = payload.get("api_key")
    job_id = str(payload.get("proxy_jobid", ""))
    raw_url = payload.get("url")
    if (
        not isinstance(key, str)
        or len(key) < 16
        or any(character.isspace() for character in key)
        or not job_id.isdigit()
        or not isinstance(raw_url, str)
    ):
        raise RuntimeError("proxy_info is missing required fields")
    parsed = urlsplit(raw_url)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("proxy_info URL is not a credential-free HTTP origin")
    return payload


def _atomic_private_bytes(path: Path, value: bytes) -> None:
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_master_key(key: str) -> None:
    _atomic_private_bytes(MASTER_KEY, (key + "\n").encode())


def _write_proxy_info(payload: dict[str, Any], key: str) -> None:
    updated = dict(payload)
    updated["api_key"] = key
    _atomic_private_bytes(
        PROXY_INFO, (json.dumps(updated, separators=(",", ":")) + "\n").encode()
    )


def _signal_reload(job_id: str) -> None:
    result = subprocess.run(
        ["scancel", "--signal=HUP", "--batch", job_id],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("could not signal the Qwen proxy reload")


def _models_status(base_url: str, key: str) -> int | None:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/models",
        headers={"Authorization": "Bearer " + key},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=10) as response:
            response.read(1024)
            return response.status
    except urllib.error.HTTPError as exc:
        exc.read(1024)
        return exc.code
    except (OSError, TimeoutError, urllib.error.URLError):
        return None


def _wait_for_rotation(base_url: str, old_key: str, new_key: str) -> tuple[int, int]:
    deadline = time.monotonic() + 240
    consecutive = 0
    old_status: int | None = None
    new_status: int | None = None
    while time.monotonic() < deadline:
        new_status = _models_status(base_url, new_key)
        old_status = _models_status(base_url, old_key)
        if new_status == 200 and old_status in REJECTED_AUTH_STATUSES:
            consecutive += 1
            if consecutive >= 3:
                return old_status, new_status
        else:
            consecutive = 0
        time.sleep(2)
    raise RuntimeError("proxy did not converge to the rotated credential")


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RuntimeError("rotation lock is not private")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"lock is held: {path.name}") from exc
        yield
    finally:
        os.close(descriptor)


def _old_key_hits(old_key: str) -> tuple[int, int]:
    roots = [
        DEPLOYMENT,
        REPO / "pipeline_logs",
        REPO / "trials/qwen35_4b_repro",
        REPO / "trials/qwen35_4b_repro_quarantine",
        REPO / "trials/qwen35_4b_secure_sigterm_pilot",
        REPO / "trials/qwen_k2",
    ]
    needle = old_key.encode()
    scanned = 0
    hits = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                metadata = path.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            scanned += 1
            try:
                hits += needle in path.read_bytes()
            except OSError:
                continue
    return scanned, hits


def _write_evidence(payload: dict[str, Any]) -> None:
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{EVIDENCE.name}.", dir=EVIDENCE.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, EVIDENCE)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_persistent_key_source() -> None:
    for path in (PROXY_BATCH_SOURCE, CONFIG_GENERATOR_SOURCE):
        source = path.read_text()
        if "read_proxy_master_key" not in source or "sk-model-proxy-key" in source:
            raise RuntimeError(
                f"deployment source is not pinned to the private key file: {path.name}"
            )


def main() -> int:
    with _exclusive_lock(PRODUCER_LOCK), _exclusive_lock(ROTATION_LOCK):
        _validate_persistent_key_source()
        info = _load_proxy_info()
        old_key = str(info["api_key"])
        existing = _secure_read(MASTER_KEY, 4096).decode().strip()
        if existing != old_key:
            raise RuntimeError("master-key file and proxy_info are out of sync")
        new_key = "sk-qwen-" + secrets.token_urlsafe(48)
        _write_master_key(new_key)
        try:
            _signal_reload(str(info["proxy_jobid"]))
            old_status, new_status = _wait_for_rotation(
                str(info["url"]), old_key, new_key
            )
        except Exception:
            _write_master_key(old_key)
            _signal_reload(str(info["proxy_jobid"]))
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if _models_status(str(info["url"]), old_key) == 200:
                    break
                time.sleep(2)
            else:
                raise RuntimeError("rotation failed and rollback did not recover")
            raise

        _write_proxy_info(info, new_key)
        os.chmod(PROXY_CONFIG, 0o600)
        reloaded = _load_proxy_info()
        if reloaded.get("api_key") != new_key:
            raise RuntimeError("proxy_info did not persist the rotated credential")
        scanned, hits = _old_key_hits(old_key)
        evidence = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "deployment": DEPLOYMENT.name,
            "model": EXPECTED_MODEL,
            "proxy_jobid": str(info["proxy_jobid"]),
            "old_credential_rejected": old_status in REJECTED_AUTH_STATUSES,
            "new_credential_accepted": new_status == 200,
            "proxy_info_mode": oct(stat.S_IMODE(PROXY_INFO.stat().st_mode)),
            "master_key_mode": oct(stat.S_IMODE(MASTER_KEY.stat().st_mode)),
            "proxy_config_mode": oct(stat.S_IMODE(PROXY_CONFIG.stat().st_mode)),
            "persistent_private_key_source": True,
            "old_credential_scan_files": scanned,
            "old_credential_scan_hits": hits,
        }
        _write_evidence(evidence)
        if hits:
            raise RuntimeError("revoked credential remains in persisted files")
        print("Qwen deployment bearer rotation verified; evidence written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
