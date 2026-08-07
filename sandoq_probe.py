#!/usr/bin/env python3
"""Smoke tests for SWE-Together's Sandoq OCI-runner transport.

The default full probe exercises the real two-layer contract: lease the fixed
outer runner, use its bearer-authenticated ``/v1/exec`` API, then start the
requested task image with Podman + gVisor and execute inside it.  Exit code 0
from that mode proves the task rootfs and authenticated command path both work.
``--control-plane-only`` instead proves only lease, health, the unauthenticated
401 boundary, and confirmed deletion; it deliberately does not read a token.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import signal
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external" / "harbor" / "src"))

from sandoq_env import (  # noqa: E402
    _delete_outer_sync,
    _duration_seconds,
    _normalize_command_response,
    _read_token,
    _request,
    _validated_exec_url,
)

BASE = (
    os.environ.get("OCI_RUNNER_BASE_URL")
    or os.environ.get("SANDOQ_BASE_URL")
    or "https://sandoq.eks-prod.cf.aws.metafb.cloud"
).rstrip("/")
ENVIRONMENT = (
    os.environ.get("OCI_RUNNER_ENVIRONMENT")
    or os.environ.get("SANDOQ_OCI_ENV")
    or "oci-runner"
)
IMAGE = os.environ.get(
    "SANDOQ_TEST_IMAGE",
    "ghcr.io/togetherbench/multi-user-turn-codebench/agent-swarm-task-4a881b:a61b000174ea",
)
OWNER = os.environ.get("SANDOQ_OWNER") or os.environ.get("USER") or "sandoq-probe"
LEASE = os.environ.get("OCI_RUNNER_LEASE_DURATION") or "20m"
DEADLINE = _duration_seconds(os.environ.get("OCI_RUNNER_CREATE_DEADLINE"), 300.0)
PULL_TIMEOUT = int(
    _duration_seconds(os.environ.get("OCI_RUNNER_PULL_TIMEOUT"), 1200.0)
)
_ACTIVE_SESSION_ID: str | None = None


def cleanup_probe(timeout: float = 15.0) -> None:
    """Best-effort cleanup for abnormal interpreter exit or Slurm SIGTERM."""
    global _ACTIVE_SESSION_ID
    session_id = _ACTIVE_SESSION_ID
    if not session_id:
        return
    _delete_outer_sync(BASE, session_id, timeout)
    _ACTIVE_SESSION_ID = None


def install_cleanup_handlers() -> None:
    atexit.register(cleanup_probe)
    previous = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum, frame):
        try:
            cleanup_probe(timeout=10.0)
        finally:
            if callable(previous):
                previous(signum, frame)
            elif previous != signal.SIG_IGN:
                raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_sigterm)


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_read_token()}"}


def outer_exec(exec_url: str, command: str, timeout: int = 60) -> tuple[str, str, int]:
    status, body = _request(
        "POST",
        exec_url + "v1/exec",
        {"command": ["bash", "-lc", command], "timeout": timeout},
        timeout + 30,
        auth_headers(),
    )
    if status != 200:
        raise RuntimeError(f"authenticated /v1/exec -> HTTP {status}: {body}")
    stdout, stderr, exit_code, timed_out = _normalize_command_response(body)
    if timed_out:
        raise RuntimeError(f"outer command timed out after {timeout}s")
    return stdout, stderr, exit_code


def create_session() -> dict:
    request_id = uuid.uuid4().hex
    payload = {
        "leaseDuration": LEASE,
        "owner": OWNER,
        "requestId": request_id,
    }
    deadline = time.monotonic() + DEADLINE
    attempt = 0
    while True:
        attempt += 1
        status, body = _request(
            "POST",
            f"{BASE}/api/v1/environments/{ENVIRONMENT}/sessions",
            payload,
            30,
        )
        if status in (200, 201):
            return body
        if status == 429 and time.monotonic() < deadline:
            print(f"lease attempt {attempt}: pool warming (HTTP 429)", flush=True)
            time.sleep(min(float(attempt), 10.0))
            continue
        raise RuntimeError(f"lease failed: HTTP {status}: {json.dumps(body)[:500]}")


def wait_health(exec_url: str) -> None:
    deadline = time.monotonic() + DEADLINE
    while time.monotonic() < deadline:
        status, body = _request("GET", exec_url + "healthz", timeout=5)
        if status == 200 and body.get("status") == "ok":
            return
        time.sleep(1)
    raise RuntimeError("outer command server did not become healthy")


def probe_authenticated_data_plane(exec_url: str) -> None:
    """Exercise bearer exec, nested gVisor state, and private file transfer."""
    stdout, stderr, rc = outer_exec(exec_url, "echo AUTHENTICATED", 10)
    if rc != 0 or "AUTHENTICATED" not in stdout:
        raise RuntimeError(f"authenticated probe failed rc={rc}: {stderr}")
    print("Authenticated /v1/exec: OK")

    image_q = shlex.quote(IMAGE)
    bootstrap = f"""
set -eu
umask 077
mkdir -p /home/runner/.swe-together-transfer
chmod 700 /home/runner/.swe-together-transfer
podman pull --quiet {image_q} >/dev/null
digest=$(podman image inspect {image_q} --format '{{{{.Digest}}}}')
case "$digest" in sha256:*) ;; *) exit 70 ;; esac
podman rm -f task >/dev/null 2>&1 || true
podman run -d --name task --runtime runsc --user 0:0 \
  --entrypoint /bin/bash {image_q} \
  -lc 'trap : TERM INT; sleep infinity & wait' >/dev/null
printf 'OCI_RESOLVED_DIGEST=%s\n' "$digest"
podman exec --user 0:0 task bash -lc \
  'printf "NESTED_OK user=%s cwd=%s\\n" "$(id -u)" "$PWD"; git --version'
"""
    stdout, stderr, rc = outer_exec(exec_url, bootstrap, PULL_TIMEOUT)
    if rc != 0 or "NESTED_OK user=0" not in stdout or "sha256:" not in stdout:
        raise RuntimeError(
            f"nested image bootstrap failed rc={rc}: {(stderr or stdout)[-2000:]}"
        )
    print(stdout.strip())

    stdout, stderr, rc = outer_exec(
        exec_url,
        "podman exec --user 0:0 task bash -lc "
        + shlex.quote("echo roundtrip > /tmp/sandoq-probe && cat /tmp/sandoq-probe"),
        30,
    )
    if rc != 0 or stdout.strip() != "roundtrip":
        raise RuntimeError(f"nested state roundtrip failed rc={rc}: {stderr}")
    print("Nested persistent exec: OK")

    transfer = """
set -eu
printf upload > /home/runner/.swe-together-transfer/probe-in
podman cp /home/runner/.swe-together-transfer/probe-in task:/tmp/probe-in
podman exec --user 0:0 task bash -lc 'cat /tmp/probe-in > /tmp/probe-out; printf %s -download >> /tmp/probe-out'
podman cp task:/tmp/probe-out /home/runner/.swe-together-transfer/probe-out
cat /home/runner/.swe-together-transfer/probe-out
rm -f /home/runner/.swe-together-transfer/probe-in /home/runner/.swe-together-transfer/probe-out
"""
    stdout, stderr, rc = outer_exec(exec_url, transfer, 30)
    if rc != 0 or stdout.strip() != "upload-download":
        raise RuntimeError(f"private podman-cp transfer failed rc={rc}: {stderr}")
    print("Private file transfer: OK")

    stdout, stderr, rc = outer_exec(
        exec_url,
        "podman rm -f task >/dev/null 2>&1 || true; "
        "if podman inspect task >/dev/null 2>&1; then exit 72; fi",
        30,
    )
    if rc != 0:
        raise RuntimeError(
            f"nested task stop was not confirmed rc={rc}: {stderr or stdout}"
        )
    print("Nested task stop: OK")


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_SESSION_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-plane-only",
        action="store_true",
        help="Lease, check health/401, and delete without requiring a bearer token. "
        "This does not validate authenticated exec or nested gVisor.",
    )
    args = parser.parse_args(argv)
    token_path = Path(
        os.environ.get("OCI_RUNNER_TOKEN_FILE", "~/.config/oci-runner/token")
    ).expanduser()
    if not args.control_plane_only:
        try:
            _read_token(token_path)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    print(f"Sandoq OCI probe: base={BASE} environment={ENVIRONMENT}")
    print(f"Task image: {IMAGE}")
    if args.control_plane_only:
        print("Mode: control plane only (authenticated exec is NOT tested)")
    else:
        print(f"Token contract: {token_path} (contents not displayed)")
    install_cleanup_handlers()

    session = create_session()
    session_id = session.get("sessionId")
    exec_url = (session.get("portUrls") or {}).get("exec") or ""
    if not session_id or not exec_url:
        print(f"ERROR: malformed lease response: {json.dumps(session)[:500]}")
        if session_id:
            _delete_outer_sync(BASE, session_id, 60)
        return 3
    _ACTIVE_SESSION_ID = session_id
    print(f"Leased outer session {session_id}")

    try:
        exec_url = _validated_exec_url(exec_url)
        wait_health(exec_url)
        unauth_status, _ = _request(
            "POST",
            exec_url + "v1/exec",
            {"command": ["bash", "-lc", "true"], "timeout": 5},
            10,
        )
        if unauth_status != 401:
            raise RuntimeError(
                f"unauthenticated /v1/exec returned {unauth_status}, expected 401"
            )
        if args.control_plane_only:
            print("Health and unauthenticated /v1/exec=401 boundary: OK")
        else:
            probe_authenticated_data_plane(exec_url)
    finally:
        cleanup_probe(timeout=60)
        print(f"Deleted outer session {session_id}; HTTP 404 confirmed")

    if args.control_plane_only:
        print("SANDOQ CONTROL PLANE: PASS (authenticated data plane untested)")
    else:
        print("SANDOQ OCI-RUNNER END-TO-END: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
