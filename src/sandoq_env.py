"""Sandoq OCI-runner environment for SWE-Together.

Sandoq's ``oci-runner`` is an outer, authenticated command sandbox.  It does
not replace its root filesystem when a session is created.  A task session is
therefore built in two layers:

1. lease the fixed ``oci-runner`` environment;
2. authenticate to its ``/v1/exec`` API with a rotatable bearer token;
3. pull the task image inside the runner and start it with Podman + gVisor;
4. route Harbor exec/file operations into that persistent nested container.

This matches the validated OCI transport in ram_prime_rl's
``feat/sandoq-oci-one-example`` branch.  In particular, task images are never
sent as a create-session field, and the public legacy ``/exec`` endpoint is not
used.

Required configuration:

``OCI_RUNNER_TOKEN_FILE``
    Regular, non-symlink, mode-0600 file containing one bearer token line.

Optional configuration (the ``SANDOQ_*`` aliases remain for compatibility):

``OCI_RUNNER_BASE_URL`` / ``SANDOQ_BASE_URL``
    Sandoq control-plane URL.
``OCI_RUNNER_ENVIRONMENT`` / ``SANDOQ_OCI_ENV``
    Outer environment name (default: ``oci-runner``).
``OCI_RUNNER_LEASE_DURATION`` / ``SANDOQ_LEASE_DURATION``
    Session lease duration (default: ``3h``).
``OCI_RUNNER_CREATE_DEADLINE`` / ``SANDOQ_CREATE_DEADLINE``
    Seconds allowed for leasing/readiness (default: 300).
``OCI_RUNNER_PULL_TIMEOUT`` / ``SANDOQ_PULL_TIMEOUT``
    Seconds allowed for task-image pull/bootstrap (default: 1200).
``OCI_RUNNER_EXEC_TIMEOUT``
    Server-side fallback when Harbor passes no command timeout (default: 3600).
``SANDOQ_FORWARD_ENV``
    Comma-separated host variables to bake into the nested task container.
``SANDOQ_HTTP_PROXY``
    Explicit proxy for Sandoq HTTP requests.  By default requests are direct
    and ignore the host's generic proxy variables.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import errno
import io
import json
import logging
import math
import os
import re
import shlex
import signal
import ssl
import stat
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

_DEFAULT_BASE_URL = "https://sandoq.eks-prod.cf.aws.metafb.cloud"
_DEFAULT_ENVIRONMENT = "oci-runner"
_TRANSFER_ROOT = "/home/runner/.swe-together-transfer"
_STAGING_CHUNK_CHARS = 512 * 1024
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_LOG = logging.getLogger(__name__)


def _env(primary: str, alias: str, default: str) -> str:
    return os.environ.get(primary) or os.environ.get(alias) or default


def _duration_seconds(value: str | None, default: float) -> float:
    text = (value or "").strip().lower()
    if not text:
        return default
    for suffix, multiplier in (("ms", 0.001), ("s", 1.0), ("m", 60.0), ("h", 3600.0)):
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * multiplier
    return float(text)


def _token_path() -> Path:
    return Path(
        os.environ.get("OCI_RUNNER_TOKEN_FILE")
        or os.environ.get("SANDOQ_TOKEN_FILE")
        or "~/.config/oci-runner/token"
    ).expanduser()


def _read_token(path: Path | None = None) -> str:
    """Read a rotatable token from the same file descriptor that was checked."""
    path = path or _token_path()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"OCI runner token file does not exist: {path}. Obtain a token from "
            "the Sandoq/oci-runner owner, store one line in this file, and chmod 0600."
        ) from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError(
                f"OCI runner token path must not be a symlink: {path}"
            ) from exc
        raise RuntimeError(f"Unable to open OCI runner token file {path}: {exc}") from exc
    try:
        file_stat = os.fstat(descriptor)
        mode = stat.S_IMODE(file_stat.st_mode)
        if not stat.S_ISREG(file_stat.st_mode) or mode != 0o600:
            raise RuntimeError(
                f"OCI runner token file must be a regular mode-0600 file: {path} "
                f"(mode={mode:04o})"
            )
        raw = os.read(descriptor, 64 * 1024 + 1)
    finally:
        os.close(descriptor)
    if len(raw) > 64 * 1024:
        raise RuntimeError(f"OCI runner token file is unexpectedly large: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"OCI runner token file is not valid UTF-8: {path}") from exc
    token = text[:-1] if text.endswith("\n") else text
    if not token or token != token.strip() or "\n" in token or "\r" in token:
        raise RuntimeError(
            f"OCI runner token file must contain exactly one nonempty line: {path}"
        )
    return token


def _validated_exec_url(value: object) -> str:
    """Validate a lease-provided endpoint before any bearer token is sent."""
    if not isinstance(value, str):
        raise RuntimeError("Sandoq lease response contained no exec portUrl")
    parsed = urlsplit(value)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "Sandoq exec portUrl must be an HTTPS URL without credentials, query, "
            f"or fragment: {value!r}"
        )
    return value if value.endswith("/") else value + "/"


_client: httpx.Client | None = None
_client_key: tuple[str | None, str | None] | None = None
# SIGTERM is delivered on the main thread between Python bytecodes.  If it
# interrupts that thread while it is constructing/reconfiguring the shared
# client, the cleanup handler re-enters ``_get_client`` in the same thread.
# A plain Lock self-deadlocks in that case.
_client_lock = threading.RLock()


def _get_client() -> httpx.Client:
    """Return a process-shared client, direct unless SANDOQ_HTTP_PROXY is explicit."""
    global _client, _client_key
    proxy = os.environ.get("SANDOQ_HTTP_PROXY")
    ca_file = os.environ.get("SANDOQ_CA_FILE")
    key = (proxy, ca_file)
    if _client is not None and _client_key == key:
        return _client
    with _client_lock:
        if _client is not None and _client_key == key:
            return _client
        if _client is not None:
            _client.close()
        verify: ssl.SSLContext | bool = True
        if ca_file:
            verify = ssl.create_default_context(cafile=ca_file)
        _client = httpx.Client(
            proxy=proxy,
            verify=verify,
            trust_env=False,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        _client_key = key
        return _client


def _request(
    method: str,
    url: str,
    body: dict | None = None,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    request_headers = dict(headers or {})
    if body is not None:
        request_headers.setdefault("Content-Type", "application/json")
    try:
        response = _get_client().request(
            method, url, json=body, headers=request_headers, timeout=timeout
        )
    except (httpx.HTTPError, OSError) as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}
    try:
        payload = response.json() if response.content else {}
    except ValueError:
        payload = {"raw": response.text}
    return response.status_code, payload


def _normalize_command_response(body: dict) -> tuple[str, str, int, bool]:
    result = body.get("result") if isinstance(body.get("result"), dict) else body
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    exit_code = result.get("exit_code", result.get("exitCode", result.get("code")))
    timed_out = bool(result.get("timed_out", result.get("timedOut", False)))
    timed_out = timed_out or exit_code == -1
    if not isinstance(exit_code, int):
        raise RuntimeError(
            f"OCI command response has no integer exit code (keys={sorted(result)})"
        )
    return stdout, stderr, exit_code, timed_out


def _wrap_command(command: str, cwd: str | None, env: dict[str, str] | None) -> str:
    prefix: list[str] = []
    if cwd:
        prefix.append(f"cd {shlex.quote(cwd)} || exit 127")
    for key, value in (env or {}).items():
        if not _ENV_NAME_RE.fullmatch(key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
        prefix.append(f"export {key}={shlex.quote(str(value))}")
    return "\n".join([*prefix, command]) if prefix else command


def _delete_outer_sync(base_url: str, session_id: str, timeout: float = 60.0) -> None:
    """Delete an outer session and verify teardown via a subsequent HTTP 404."""
    status, body = _request(
        "DELETE", f"{base_url}/api/v1/sessions/{session_id}", timeout=timeout
    )
    if status not in (200, 202, 204, 404):
        raise RuntimeError(
            f"Sandoq session delete failed for {session_id}: HTTP {status}: {body}"
        )
    deadline = time.monotonic() + timeout
    observed = 404 if status == 404 else None
    while observed != 404 and time.monotonic() < deadline:
        observed, _ = _request(
            "GET",
            f"{base_url}/api/v1/sessions/{session_id}",
            timeout=min(5.0, timeout),
        )
        if observed != 404:
            time.sleep(0.5)
    if observed != 404:
        raise RuntimeError(
            f"Sandoq session {session_id} deletion was not confirmed by HTTP 404"
        )


_renewer_started = False
_atexit_registered = False
_sigterm_registered = False
_previous_sigterm_handler = None
_renewer_lock = threading.Lock()
# Python delivers signal handlers on the main thread between bytecode
# instructions.  SIGTERM can therefore interrupt that same thread while it is
# registering/unregistering a lease; the cleanup handler immediately snapshots
# this registry.  A plain Lock would self-deadlock in that case, so this lock
# must be re-entrant.
_sessions_lock = threading.RLock()


@dataclass
class _ActiveSession:
    base_url: str
    lease: str
    renewal_interval: float
    next_renewal: float


_active_sessions: dict[str, _ActiveSession] = {}
_renewer_wakeup = threading.Event()


def _lease_renewal_interval(lease: str) -> float:
    """Renew with at least two thirds of the requested lease still remaining."""
    seconds = _duration_seconds(lease, 0.0)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"OCI runner lease duration must be positive: {lease!r}")
    # Five minutes avoids unnecessary control-plane load for normal multi-hour
    # leases.  The lower bound only matters for synthetic/unit-test leases; real
    # production leases are measured in minutes or hours.
    return max(0.1, min(300.0, seconds / 3.0))


def _register_session(session_id: str, base_url: str, lease: str) -> None:
    interval = _lease_renewal_interval(lease)
    with _sessions_lock:
        _active_sessions[session_id] = _ActiveSession(
            base_url=base_url,
            lease=lease,
            renewal_interval=interval,
            next_renewal=time.monotonic() + interval,
        )
    _renewer_wakeup.set()


def _unregister_session(session_id: str) -> None:
    with _sessions_lock:
        _active_sessions.pop(session_id, None)
    _renewer_wakeup.set()


def _renew_sessions_once(session_ids: set[str] | None = None) -> bool:
    """Renew all active leases and return whether every request succeeded."""
    with _sessions_lock:
        sessions = [
            (session_id, state)
            for session_id, state in _active_sessions.items()
            if session_ids is None or session_id in session_ids
        ]
    succeeded = True
    for session_id, state in sessions:
        status, body = _request(
            "PATCH",
            f"{state.base_url}/api/v1/sessions/{session_id}",
            {"leaseDuration": state.lease},
            timeout=15.0,
        )
        request_succeeded = status in (200, 202, 204)
        now = time.monotonic()
        with _sessions_lock:
            current = _active_sessions.get(session_id)
            if current is state:
                current.next_renewal = now + (
                    current.renewal_interval
                    if request_succeeded
                    else min(60.0, max(0.1, current.renewal_interval / 3.0))
                )
        if not request_succeeded:
            succeeded = False
            _LOG.warning(
                "Sandoq lease renewal failed for %s: HTTP %s: %s",
                session_id,
                status,
                str(body)[:300],
            )
    return succeeded


def _renew_loop() -> None:
    while True:
        # Clear before taking the registry snapshot. A registration that races
        # after the clear sets the event and forces an immediate recalculation;
        # a registration before it is already visible in the snapshot.
        _renewer_wakeup.clear()
        now = time.monotonic()
        with _sessions_lock:
            sessions = list(_active_sessions.items())
        if not sessions:
            _renewer_wakeup.wait(300.0)
            continue

        due = {
            session_id
            for session_id, state in sessions
            if state.next_renewal <= now
        }
        if due:
            _renew_sessions_once(due)
            continue

        delay = min(state.next_renewal for _session_id, state in sessions) - now
        _renewer_wakeup.wait(max(0.0, delay))


def _ensure_renewer() -> None:
    global _renewer_started
    with _renewer_lock:
        if _renewer_started:
            return
        _renewer_started = True
        threading.Thread(
            target=_renew_loop, daemon=True, name="sandoq-lease-renewer"
        ).start()


def _cleanup_registered_sessions(timeout: float = 15.0) -> None:
    """Best-effort cleanup with one total deadline, including SIGTERM paths."""
    with _sessions_lock:
        sessions = list(_active_sessions.items())
    deadline = time.monotonic() + timeout
    for session_id, state in sessions:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            _delete_outer_sync(state.base_url, session_id, remaining)
        except Exception as exc:
            _LOG.warning("Sandoq exit cleanup failed for %s: %s", session_id, exc)
            continue
        _unregister_session(session_id)


def _ensure_exit_cleanup() -> None:
    global _atexit_registered, _previous_sigterm_handler, _sigterm_registered
    with _renewer_lock:
        if not _atexit_registered:
            atexit.register(_cleanup_registered_sessions)
            _atexit_registered = True
        if _sigterm_registered:
            return
        # Slurm cancellation delivers SIGTERM, for which Python does not run
        # atexit handlers by default. Install once, preserve any existing
        # orchestrator handler, and keep all cleanup work globally bounded.
        try:
            previous_handler = signal.getsignal(signal.SIGTERM)

            def _handle_sigterm(signum, frame):
                _cleanup_registered_sessions(timeout=10.0)
                previous = _previous_sigterm_handler
                if callable(previous):
                    return previous(signum, frame)
                if previous == signal.SIG_IGN:
                    return None
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, _handle_sigterm)
        except ValueError:
            # signal.signal is main-thread-only. Leave the flag false so a later
            # construction on the main thread can retry installation.
            return
        _previous_sigterm_handler = previous_handler
        _sigterm_registered = True


class SandoqEnvironment(BaseEnvironment):
    """Harbor environment backed by a nested gVisor task container in Sandoq."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args,
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )
        self._base = _env("OCI_RUNNER_BASE_URL", "SANDOQ_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        self._outer_environment = _env(
            "OCI_RUNNER_ENVIRONMENT", "SANDOQ_OCI_ENV", _DEFAULT_ENVIRONMENT
        )
        self._lease = _env("OCI_RUNNER_LEASE_DURATION", "SANDOQ_LEASE_DURATION", "3h")
        self._owner = os.environ.get("SANDOQ_OWNER") or os.environ.get("USER") or "swe-together"
        self._create_deadline = _duration_seconds(
            os.environ.get("OCI_RUNNER_CREATE_DEADLINE")
            or os.environ.get("SANDOQ_CREATE_DEADLINE"),
            300.0,
        )
        self._pull_timeout = int(
            _duration_seconds(
                os.environ.get("OCI_RUNNER_PULL_TIMEOUT")
                or os.environ.get("SANDOQ_PULL_TIMEOUT"),
                1200.0,
            )
        )
        self._default_exec_timeout = int(
            _duration_seconds(os.environ.get("OCI_RUNNER_EXEC_TIMEOUT"), 3600.0)
        )
        self._sandbox_id: str | None = None
        self._exec_url: str | None = None
        self._resolved_digest: str | None = None
        self._nested_ready = False

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.SANDOQ

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return True

    def _nested_resource_flags(self) -> tuple[list[str], int]:
        """Translate Harbor's requested resources to nested Podman settings.

        CPU and memory are enforceable OCI/cgroup limits. Podman's writable-layer
        quota support depends on the outer runner's storage driver and backing
        filesystem, so blindly adding ``--storage-opt size=`` can make every
        production container fail on otherwise valid vfs/overlay installations.
        Treat Harbor's storage value as a capacity requirement instead and verify
        that much free space from inside the started task container.
        """

        resources = {
            "cpus": self.task_env_config.cpus,
            "memory_mb": self.task_env_config.memory_mb,
            "storage_mb": self.task_env_config.storage_mb,
        }
        for name, value in resources.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"Sandoq task resource {name} must be a positive integer: {value!r}"
                )
        return (
            [
                "--cpus",
                str(resources["cpus"]),
                "--memory",
                f"{resources['memory_mb']}m",
            ],
            resources["storage_mb"],
        )

    def _validate_definition(self) -> None:
        _read_token()

    def _auth_headers(self) -> dict[str, str]:
        # Read for every call so token rotation does not require a process restart.
        return {"Authorization": f"Bearer {_read_token()}"}

    def _nested_container_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        requested = {
            name.strip()
            for name in os.environ.get("SANDOQ_FORWARD_ENV", "").split(",")
            if name.strip()
        }
        if os.environ.get("SANDOQ_FORWARD_HOST_PROXY", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            requested.update(
                {"http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"}
            )
        for name in requested:
            if not _ENV_NAME_RE.fullmatch(name):
                raise RuntimeError(
                    f"Invalid variable name in SANDOQ_FORWARD_ENV: {name!r}"
                )
            if value := os.environ.get(name):
                env[name] = value
        return env

    async def start(self, force_build: bool) -> None:
        image = self.task_env_config.docker_image
        if not image:
            raise RuntimeError(
                f"Task {self.environment_name} has no docker_image; Sandoq OCI runner "
                "requires a prebuilt image."
            )
        if force_build:
            self.logger.warning(
                "force_build is ignored by SandoqEnvironment; pulling %s", image
            )

        # Validate every local configuration value before a remote lease exists.
        # In particular, a short-but-valid lease must schedule its first renewal
        # before expiry rather than inheriting the normal five-minute cadence.
        _lease_renewal_interval(self._lease)
        self._nested_resource_flags()

        request_id = uuid.uuid4().hex
        body = {
            "leaseDuration": self._lease,
            "owner": self._owner,
            "requestId": request_id,
        }
        deadline = time.monotonic() + self._create_deadline
        attempt = 0
        while True:
            attempt += 1
            status, response = await asyncio.to_thread(
                _request,
                "POST",
                f"{self._base}/api/v1/environments/{self._outer_environment}/sessions",
                body,
                30.0,
            )
            if status in (200, 201):
                break
            if status == 429 and time.monotonic() < deadline:
                await asyncio.sleep(min(float(attempt), 10.0))
                continue
            raise RuntimeError(
                f"Sandoq OCI runner lease failed for {self._outer_environment!r}: "
                f"HTTP {status}: {json.dumps(response)[:500]}"
            )

        self._sandbox_id = response.get("sessionId")
        exec_url = (response.get("portUrls") or {}).get("exec")
        if not self._sandbox_id or not exec_url:
            if self._sandbox_id:
                await asyncio.to_thread(
                    _delete_outer_sync, self._base, self._sandbox_id, 30.0
                )
            raise RuntimeError(
                "Sandoq lease response contained no sessionId/exec portUrl: "
                f"{json.dumps(response)[:500]}"
            )
        _register_session(self._sandbox_id, self._base, self._lease)
        _ensure_renewer()
        _ensure_exit_cleanup()

        try:
            self._exec_url = _validated_exec_url(exec_url)
            await self._wait_for_command_server()
            await self._verify_authentication()
            self._resolved_digest = await self._bootstrap_nested(image)
            self._nested_ready = True
        except BaseException:
            try:
                await asyncio.shield(self.stop(delete=True))
            except BaseException as cleanup_error:
                self.logger.error(
                    "failed to clean up Sandoq session after startup failure: %s",
                    cleanup_error,
                )
            raise

        self.logger.info(
            "Sandoq nested task ready: session=%s image=%s digest=%s",
            self._sandbox_id,
            image,
            self._resolved_digest,
        )

    async def _wait_for_command_server(self) -> None:
        deadline = time.monotonic() + self._create_deadline
        while time.monotonic() < deadline:
            status, body = await asyncio.to_thread(
                _request, "GET", self._exec_url + "healthz", None, 5.0
            )
            if status == 200 and body.get("status") == "ok":
                return
            await asyncio.sleep(1.0)
        raise RuntimeError(
            f"Sandoq OCI command server for {self._sandbox_id} was not ready in time"
        )

    async def _verify_authentication(self) -> None:
        probe = {"command": ["bash", "-lc", "true"], "timeout": 5}
        status, _ = await asyncio.to_thread(
            _request, "POST", self._exec_url + "v1/exec", probe, 10.0
        )
        if status != 401:
            raise RuntimeError(
                f"Unauthenticated OCI /v1/exec returned HTTP {status}; expected 401"
            )
        result = await self._outer_exec("true", timeout_sec=5)
        if result.return_code != 0:
            raise RuntimeError(
                f"Authenticated OCI command probe failed: {result.stderr}"
            )

    async def _outer_exec(self, command: str, timeout_sec: int = 60) -> ExecResult:
        if not self._exec_url:
            raise RuntimeError("Sandoq outer exec called before session creation")
        status, body = await asyncio.to_thread(
            _request,
            "POST",
            self._exec_url + "v1/exec",
            {"command": ["bash", "-lc", command], "timeout": int(timeout_sec)},
            float(timeout_sec) + 30.0,
            self._auth_headers(),
        )
        if status == 0 and "timeout" in str(body.get("error", "")).lower():
            raise TimeoutError(
                f"Sandoq outer command transport timed out after {timeout_sec}s"
            )
        if status in (408, 504):
            raise TimeoutError(f"Sandoq outer command timed out after {timeout_sec}s")
        if status != 200:
            raise RuntimeError(
                f"Sandoq /v1/exec failed on {self._sandbox_id}: HTTP {status}: "
                f"{json.dumps(body)[:500]}"
            )
        stdout, stderr, exit_code, timed_out = _normalize_command_response(body)
        if timed_out:
            raise TimeoutError(f"Sandoq outer command timed out after {timeout_sec}s")
        return ExecResult(
            stdout=stdout or None, stderr=stderr or None, return_code=exit_code
        )

    async def _bootstrap_nested(self, image: str) -> str:
        image_q = shlex.quote(image)
        run_flags: list[str] = ["--detach", "--name", "task", "--runtime", "runsc"]
        run_flags += ["--user", "0:0"]
        resource_flags, storage_mb = self._nested_resource_flags()
        run_flags += resource_flags
        if not self.task_env_config.allow_internet:
            run_flags += ["--network", "none"]
        for key, value in self._nested_container_env().items():
            run_flags += ["--env", f"{key}={value}"]
        flags_q = " ".join(shlex.quote(part) for part in run_flags)
        bootstrap = f"""
set -eu
umask 077
mkdir -p {_TRANSFER_ROOT}
chmod 700 {_TRANSFER_ROOT}
podman pull --quiet {image_q} >/dev/null
digest=$(podman image inspect {image_q} --format '{{{{.Digest}}}}')
case "$digest" in sha256:*) ;; *) echo "invalid image digest: $digest" >&2; exit 70 ;; esac
podman rm -f task >/dev/null 2>&1 || true
podman run {flags_q} --entrypoint /bin/bash {image_q} -lc 'trap : TERM INT; sleep infinity & wait' >/dev/null
actual=$(podman image inspect "$(podman inspect task --format '{{{{.Image}}}}')" --format '{{{{.Digest}}}}')
test "$actual" = "$digest"
printf 'OCI_RESOLVED_DIGEST=%s\n' "$digest"
"""
        result = await self._outer_exec(bootstrap, timeout_sec=self._pull_timeout)
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "")[-2000:]
            raise RuntimeError(f"OCI image pull/nested startup failed: {detail}")
        matches = _DIGEST_RE.findall(result.stdout or "")
        if len(matches) != 1:
            raise RuntimeError(
                "OCI bootstrap returned an invalid resolved digest: "
                f"{(result.stdout or '')[-1000:]}"
            )
        validation = await self._nested_exec(
            "test \"$(id -u)\" = 0 && command -v bash >/dev/null && "
            "available_mb=$(df -Pm / | awk 'NR == 2 { print $4 }') && "
            "case \"$available_mb\" in ''|*[!0-9]*) exit 71;; esac && "
            f"test \"$available_mb\" -ge {storage_mb} && "
            "mkdir -p /logs/agent /logs/verifier /logs/artifacts /tests /installed-agent && "
            "{ git config --global --add safe.directory '*' 2>/dev/null || true; }",
            timeout_sec=60,
            require_ready=False,
        )
        if validation.return_code != 0:
            raise RuntimeError(
                "Nested task validation failed: "
                f"{(validation.stderr or validation.stdout or '')[-2000:]}"
            )
        return matches[0]

    async def _nested_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int = 60,
        *,
        require_ready: bool = True,
    ) -> ExecResult:
        if require_ready and not self._nested_ready:
            raise RuntimeError("Sandoq nested task container is not ready")
        script = _wrap_command(command, cwd, env)
        marker = f"/tmp/.harbor-exec-{uuid.uuid4().hex}.pid"
        cancel_marker = marker + ".cancel"
        marker_q = shlex.quote(marker)
        marker_tmp_q = shlex.quote(marker + ".tmp")
        cancel_q = shlex.quote(cancel_marker)
        # ``asyncio.to_thread`` cannot cancel an HTTP request that is already in
        # flight. A cleanup request can therefore arrive before the original
        # /v1/exec starts its nested process. The cancellation tombstone closes
        # that race: a late command exits before launching the payload, while a
        # command which has already launched publishes its PGID atomically and
        # checks the tombstone again before waiting.
        wrapped = (
            f"if [ -e {cancel_q} ]; then "
            f"rm -f {marker_q} {marker_tmp_q} {cancel_q}; exit 124; fi; "
            "set -m; "
            f"bash -lc {shlex.quote(script)} & harbor_child=$!; "
            f"printf '%s\\n' \"$harbor_child\" > {marker_tmp_q}; "
            f"mv -f {marker_tmp_q} {marker_q}; "
            f"if [ -e {cancel_q} ]; then "
            "kill -TERM -- \"-$harbor_child\" 2>/dev/null || true; "
            "harbor_i=0; while [ \"$harbor_i\" -lt 20 ] && "
            "kill -0 -- \"-$harbor_child\" 2>/dev/null; do "
            "sleep 0.1; harbor_i=$((harbor_i + 1)); done; "
            "kill -KILL -- \"-$harbor_child\" 2>/dev/null || true; "
            "fi; "
            "wait \"$harbor_child\"; harbor_rc=$?; "
            f"rm -f {marker_q} {marker_tmp_q} {cancel_q}; exit \"$harbor_rc\""
        )
        outer_command = (
            "podman exec --user 0:0 task bash -lc " + shlex.quote(wrapped)
        )
        try:
            return await self._outer_exec(outer_command, timeout_sec=timeout_sec)
        except TimeoutError:
            await self._terminate_nested_exec_group(marker, cancel_marker)
            raise
        except asyncio.CancelledError:
            await asyncio.shield(
                self._terminate_nested_exec_group(marker, cancel_marker)
            )
            raise

    async def _terminate_nested_exec_group(
        self, marker: str, cancel_marker: str
    ) -> None:
        marker_q = shlex.quote(marker)
        marker_tmp_q = shlex.quote(marker + ".tmp")
        cancel_q = shlex.quote(cancel_marker)
        cleanup = (
            # Publish cancellation before looking for a PID. If the in-flight
            # request starts later, its wrapper observes this tombstone and does
            # not execute the caller's payload.
            f": > {cancel_q}; "
            "harbor_wait=0; "
            f"while [ ! -s {marker_q} ] && [ \"$harbor_wait\" -lt 50 ]; do "
            "sleep 0.1; harbor_wait=$((harbor_wait + 1)); done; "
            f"if [ -s {marker_q} ]; then "
            f"read -r harbor_pid < {marker_q}; "
            "case \"$harbor_pid\" in ''|*[!0-9]*) harbor_pid=;; esac; "
            "if [ -n \"$harbor_pid\" ]; then "
            "kill -TERM -- \"-$harbor_pid\" 2>/dev/null || true; "
            "harbor_i=0; while [ \"$harbor_i\" -lt 20 ] && "
            "kill -0 -- \"-$harbor_pid\" 2>/dev/null; do "
            "sleep 0.1; harbor_i=$((harbor_i + 1)); done; "
            "kill -KILL -- \"-$harbor_pid\" 2>/dev/null || true; "
            "fi; "
            f"rm -f {marker_q} {marker_tmp_q} {cancel_q}; fi"
        )
        outer_command = (
            "podman exec --user 0:0 task bash -lc " + shlex.quote(cleanup)
        )
        try:
            result = await self._outer_exec(outer_command, timeout_sec=10)
            if result.return_code != 0:
                self.logger.warning(
                    "nested exec cleanup failed for %s (rc=%s): %s",
                    self.environment_name,
                    result.return_code,
                    result.stderr,
                )
        except Exception as exc:
            self.logger.warning(
                "nested exec cleanup failed for %s: %s",
                self.environment_name,
                exc,
            )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        return await self._nested_exec(
            command,
            cwd=cwd,
            env=self._merge_env(env),
            # Harbor deliberately passes None for agent installation and
            # verifier commands; those can take many minutes. A 60s transport
            # default would pre-empt Harbor's outer setup/verifier timeout.
            timeout_sec=(
                int(timeout_sec)
                if timeout_sec is not None
                else self._default_exec_timeout
            ),
        )

    async def _stage_upload(
        self, outer_path: str, data: bytes, timeout_sec: int = 120
    ) -> None:
        encoded = base64.b64encode(data).decode()
        path_q = shlex.quote(outer_path)
        parent_q = shlex.quote(os.path.dirname(outer_path))
        setup = await self._outer_exec(
            f"umask 077; mkdir -p {parent_q} && : > {path_q}.b64", timeout_sec=30
        )
        if setup.return_code != 0:
            raise RuntimeError(f"OCI staging setup failed: {setup.stderr}")
        for offset in range(0, len(encoded), _STAGING_CHUNK_CHARS):
            chunk = encoded[offset : offset + _STAGING_CHUNK_CHARS]
            append = await self._outer_exec(
                f"printf %s {shlex.quote(chunk)} >> {path_q}.b64",
                timeout_sec=timeout_sec,
            )
            if append.return_code != 0:
                raise RuntimeError(f"OCI staging append failed: {append.stderr}")
        decoded = await self._outer_exec(
            f"base64 -d {path_q}.b64 > {path_q} && rm -f {path_q}.b64",
            timeout_sec=timeout_sec,
        )
        if decoded.return_code != 0:
            raise RuntimeError(f"OCI staging decode failed: {decoded.stderr}")

    async def _stage_download(self, outer_path: str, timeout_sec: int = 120) -> bytes:
        result = await self._outer_exec(
            f"base64 -w0 {shlex.quote(outer_path)}", timeout_sec=timeout_sec
        )
        if result.return_code != 0:
            raise FileNotFoundError(f"Staged OCI file not found: {outer_path}")
        try:
            return base64.b64decode(result.stdout or "", validate=True)
        except ValueError as exc:
            raise RuntimeError("OCI staging returned invalid base64") from exc

    async def _upload_bytes(self, data: bytes, target_path: str) -> None:
        transfer_id = uuid.uuid4().hex
        outer_path = f"{_TRANSFER_ROOT}/{transfer_id}"
        await self._stage_upload(outer_path, data)
        parent = os.path.dirname(target_path) or "."
        try:
            prepared = await self._nested_exec(
                f"mkdir -p {shlex.quote(parent)}", timeout_sec=120
            )
            if prepared.return_code != 0:
                raise RuntimeError(
                    f"Nested upload destination setup failed for {target_path}: "
                    f"{prepared.stderr}"
                )
            copied = await self._outer_exec(
                "podman cp "
                f"{shlex.quote(outer_path)} {shlex.quote('task:' + target_path)}",
                timeout_sec=120,
            )
            if copied.return_code != 0:
                raise RuntimeError(
                    f"Nested upload copy failed for {target_path}: {copied.stderr}"
                )
        finally:
            await self._outer_exec(
                f"rm -f -- {shlex.quote(outer_path)}", timeout_sec=30
            )

    async def _download_bytes(self, source_path: str, timeout_sec: int = 120) -> bytes:
        transfer_id = uuid.uuid4().hex
        outer_path = f"{_TRANSFER_ROOT}/{transfer_id}"
        copied = await self._outer_exec(
            "podman cp "
            f"{shlex.quote('task:' + source_path)} {shlex.quote(outer_path)}",
            timeout_sec=timeout_sec,
        )
        if copied.return_code != 0:
            raise FileNotFoundError(f"Sandoq nested file not found: {source_path}")
        try:
            return await self._stage_download(outer_path, timeout_sec=timeout_sec)
        finally:
            await self._outer_exec(
                f"rm -f -- {shlex.quote(outer_path)}", timeout_sec=30
            )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        data = await asyncio.to_thread(Path(source_path).read_bytes)
        await self._upload_bytes(data, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)

        def make_tar() -> bytes:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for item in sorted(source.iterdir()):
                    archive.add(item, arcname=item.name)
            return buffer.getvalue()

        data = await asyncio.to_thread(make_tar)
        temp_path = f"/tmp/_sandoq_upload_{uuid.uuid4().hex}.tgz"
        await self._upload_bytes(data, temp_path)
        result = await self._nested_exec(
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"tar -xzf {shlex.quote(temp_path)} -C {shlex.quote(target_dir)} && "
            f"rm -f {shlex.quote(temp_path)}",
            timeout_sec=300,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Sandoq upload_dir extraction failed for {target_dir}: {result.stderr}"
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        data = await self._download_bytes(source_path)
        destination = Path(target_path)

        def write() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

        await asyncio.to_thread(write)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        temp_path = f"/tmp/_sandoq_download_{uuid.uuid4().hex}.tgz"
        result = await self._nested_exec(
            f"tar -czf {shlex.quote(temp_path)} -C {shlex.quote(source_dir)} .",
            timeout_sec=300,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Sandoq download_dir archive failed for {source_dir}: {result.stderr}"
            )
        try:
            data = await self._download_bytes(temp_path, timeout_sec=300)
        finally:
            await self._nested_exec(
                f"rm -f {shlex.quote(temp_path)}", timeout_sec=30
            )
        destination = Path(target_dir)

        def extract() -> None:
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                archive.extractall(destination, filter="data")

        await asyncio.to_thread(extract)

    async def stop(self, delete: bool) -> None:
        del delete  # a remote lease must always be released
        session_id = self._sandbox_id
        if not session_id:
            return
        await asyncio.to_thread(_delete_outer_sync, self._base, session_id, 60.0)
        _unregister_session(session_id)
        self._sandbox_id = None
        self._exec_url = None
        self._nested_ready = False


__all__ = [
    "SandoqEnvironment",
    "_delete_outer_sync",
    "_duration_seconds",
    "_lease_renewal_interval",
    "_normalize_command_response",
    "_read_token",
    "_request",
    "_validated_exec_url",
    "_wrap_command",
]
