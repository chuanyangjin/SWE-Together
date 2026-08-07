"""Local rootless-podman environment for SWE-Together's on-cluster harness.

A Harbor ``BaseEnvironment`` that runs each trial in a podman container instead
of an E2B cloud sandbox or a docker-compose stack. It is selected per-trial via
``EnvironmentConfig.import_path = "podman_env:PodmanEnvironment"`` (see
``src/run_eval.py``), so Harbor needs no factory registration.

Why a bespoke environment (not Harbor's ``DockerEnvironment``):
  * There is no docker daemon here; ``DockerEnvironment`` shells to ``docker
    compose``. We shell to ``podman`` directly.
  * podman 3.4.4 in this pod (``claude-sandbox-pod``, no CAP_SYS_ADMIN) only
    works through one fragile incantation — a single ``unshare -Urm`` user+mount
    namespace, a **tmpfs** image store, the ``vfs`` storage driver with
    ``vfs.ignore_chown_errors=true`` (the userns maps a single uid), and a
    pre-created ``/tmp/podman-run-<uid>`` runtime dir. That session is
    established by ``run_local.sh``; this class assumes it is already inside it
    and reads the store/runroot/tmpdir locations from ``HARBOR_PODMAN_*`` env.

Image policy: **we never build.** Building the task Dockerfiles here is
impossible (107/109 ``apt-get install`` and the Ubuntu archive is not in the
egress allowlist). Instead every ``task.toml`` pins a prebuilt image on
``ghcr.io`` (anonymously pullable, and ghcr.io *is* allowlisted), so ``start``
uses ``task_env_config.docker_image`` — pulling it if not already present.

Networking: the container runs with ``--network host`` because the Qwen action
endpoint is only reachable via the host's direct route to the internal 10/8
network. podman's automatic host-proxy passthrough is disabled
(``--http-proxy=false``) and replaced with corrected values: the host filtering
proxy for external egress (pip/npm/ghcr installs) plus an **exact-IP**
``no_proxy`` for the model endpoint (the container's curl is too old to
CIDR-match ``10.0.0.0/8``), so in-container model calls stay direct.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess
import os
import re
import shlex
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.utils.redaction import is_secret_key


_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ENV_FILE_PLACEHOLDER = "__SWE_PODMAN_ENV_FILE__"
_QWEN_LOOPBACK_PLACEHOLDER = "qwen-loopback-placeholder"


def _validate_secure_qwen_env() -> None:
    if os.environ.get("SWE_QWEN_LOOPBACK_PROXY") != "1":
        return
    parsed = urlparse(os.environ.get("OPENAI_BASE_URL", ""))
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("secure Qwen loopback URL has an invalid port") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "secure Qwen mode requires OPENAI_BASE_URL=http://127.0.0.1:<port>/v1"
        )
    if os.environ.get("OPENAI_API_KEY") != _QWEN_LOOPBACK_PLACEHOLDER:
        raise RuntimeError(
            "secure Qwen mode refuses to forward a non-placeholder API key"
        )


class PodmanEnvironment(BaseEnvironment):
    # Serialize every operation that can add/remove an image reference.  This
    # is broader than a pull lock: with k>1, one trial can finish while another
    # trial of the same task is between ``image exists`` and ``run``.  An
    # overlapping rmi must not invalidate that successful existence check.
    _image_pull_locks: dict[str, asyncio.Lock] = {}

    # Cap concurrent pulls across ALL trials. Many 1GB+ images pulling at once
    # over the shared filtering proxy (plus vfs unpack) starves bandwidth and can
    # blow the per-trial env-start timeout. Containers still run fully
    # concurrently; only the pull step is throttled. Tune via HARBOR_PODMAN_MAX_PULLS.
    _pull_semaphore: asyncio.Semaphore = asyncio.Semaphore(
        int(os.environ.get("HARBOR_PODMAN_MAX_PULLS", "3"))
    )

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

        # podman store lives on tmpfs inside the unshare session (set by
        # run_local.sh). Fall back to /mnt/* so a manual invocation that already
        # set up the mounts still works; warn loudly if the env is absent since
        # the c/storage ops SIGSEGV on overlayfs/NFS stores.
        self._store = os.environ.get("HARBOR_PODMAN_STORE", "/mnt/store")
        self._runroot = os.environ.get("HARBOR_PODMAN_RUNROOT", "/mnt/run")
        self._tmpdir = os.environ.get("HARBOR_PODMAN_TMPDIR", "/mnt/tmp")
        if "HARBOR_PODMAN_STORE" not in os.environ:
            self.logger.warning(
                "HARBOR_PODMAN_STORE unset — assuming defaults under /mnt. "
                "PodmanEnvironment must run inside run_local.sh's unshare session."
            )

        # Container name: unique per trial, DNS-safe.
        self._container = "hb-" + session_id.lower().replace(".", "-").replace(
            "_", "-"
        )[:64]
        self._container_id: str | None = None

    # ── podman command helpers ────────────────────────────────────────────────

    @staticmethod
    def _nofile_ulimit() -> str:
        """`<soft>:<hard>` for --ulimit nofile, capped at our current hard limit."""
        override = os.environ.get("HARBOR_PODMAN_NOFILE")
        if override:
            return override if ":" in override else f"{override}:{override}"
        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            return f"{soft}:{hard}"
        except Exception:
            return "500000:500000"

    def _podman(self, *args: str) -> list[str]:
        return [
            "podman",
            "--root",
            self._store,
            "--runroot",
            self._runroot,
            "--storage-driver",
            "vfs",
            "--storage-opt",
            "vfs.ignore_chown_errors=true",
            "--tmpdir",
            self._tmpdir,
            *args,
        ]

    async def _run(
        self,
        cmd: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        container_env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run podman without exposing container credentials in argv/environ.

        Container variables travel through an anonymous, inherited env-file
        descriptor.  They never overlay PATH/LD_PRELOAD/CONTAINERS_* in the
        host-side Podman client and are absent from its command line.
        """
        command = list(cmd)
        env_file = None
        pass_fds: tuple[int, ...] = ()
        if container_env:
            invalid = [key for key in container_env if not _ENV_NAME_RE.fullmatch(key)]
            if invalid:
                raise ValueError(f"invalid container environment name: {invalid[0]!r}")
            for key, value in container_env.items():
                if "\x00" in str(value) or "\n" in str(value) or "\r" in str(value):
                    raise ValueError(
                        f"container environment value for {key!r} contains a newline/NUL"
                    )
            try:
                placeholder_index = command.index(_ENV_FILE_PLACEHOLDER)
            except ValueError as exc:
                raise ValueError(
                    "container_env requires an env-file placeholder in podman argv"
                ) from exc
            env_file = tempfile.TemporaryFile(mode="w+b")
            env_file.write(
                "".join(f"{key}={value}\n" for key, value in container_env.items()).encode()
            )
            env_file.flush()
            env_file.seek(0)
            descriptor = env_file.fileno()
            command[placeholder_index] = f"/proc/self/fd/{descriptor}"
            pass_fds = (descriptor,)

        # Avoid inheriting host credentials into an otherwise unrelated Podman
        # client. Variables explicitly destined for the container are already
        # delivered through the anonymous descriptor above.
        client_env = {
            key: value
            for key, value in os.environ.items()
            if not is_secret_key(key)
        }
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                env=client_env,
                pass_fds=pass_fds,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            if timeout_sec:
                stdout_b, stderr_b = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_b, stderr_b = await process.communicate()
        except asyncio.TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.communicate(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
            # Preserve timeout semantics for wrapper-level cap rescue.
            # ``exec_with_budget`` catches TimeoutError and can recover the
            # OpenCode session/partial patch; a generic RuntimeError instead
            # aborts the whole Harbor trial and loses verifier output.
            raise TimeoutError(
                f"podman command timed out after {timeout_sec}s: {' '.join(command[:6])}…"
            )
        finally:
            if env_file is not None:
                env_file.close()

        result = ExecResult(
            stdout=stdout_b.decode(errors="replace") if stdout_b else None,
            stderr=stderr_b.decode(errors="replace") if stderr_b else None,
            return_code=process.returncode or 0,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"podman command failed (rc={result.return_code}) for "
                f"{self.environment_name}: {' '.join(command)}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result

    # ── BaseEnvironment metadata ──────────────────────────────────────────────

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.PODMAN

    @property
    def is_mounted(self) -> bool:
        # We copy logs/artifacts out with `podman cp` rather than bind-mounting,
        # so Harbor's verifier/trial download paths fire (matches E2B semantics).
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        # Approximated by withholding the external proxy env (see start); the
        # in-cluster model route stays reachable regardless.
        return True

    def _validate_definition(self):
        if not self.environment_dir.is_dir():
            raise FileNotFoundError(
                f"environment dir not found: {self.environment_dir}"
            )

    # ── container env (proxy / no_proxy) ──────────────────────────────────────

    def _container_env(self) -> dict[str, str]:
        """Env baked into the container at ``podman run`` time.

        Three concerns, all read from the host env at runtime so a rotated proxy
        port or model IP is picked up automatically:

        1. ``no_proxy`` — the model endpoint host by exact match (old curl on
           22.04 cannot CIDR-match ``10.0.0.0/8``) so in-container model calls
           bypass the proxy and go direct.
        2. proxy — the host filtering proxy for external egress (pip/npm/ghcr
           agent installs). Withheld when ``allow_internet`` is False to
           approximate network isolation (the direct model route still works).
        3. action-model env — ``OPENAI_BASE_URL``/``OPENAI_API_KEY`` (+ any
           ``HARBOR_PODMAN_FORWARD_ENV`` extras) forwarded so the coding agent
           reaches the model however Harbor execs it. ``OPENAI_BASE_URL`` is the
           critical one: Harbor's OpenCode agent forwards ``OPENAI_API_KEY`` but
           NOT the base URL, so without baking it here opencode's openai provider
           would hit api.openai.com instead of the Qwen endpoint.
        """
        _validate_secure_qwen_env()
        secure_qwen = os.environ.get("SWE_QWEN_LOOPBACK_PROXY") == "1"
        no_proxy = self._no_proxy_hosts()
        env = {"no_proxy": no_proxy, "NO_PROXY": no_proxy}

        if self.task_env_config.allow_internet:
            proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or ""
            if proxy:
                env.update(
                    {
                        "http_proxy": proxy,
                        "https_proxy": proxy,
                        "HTTP_PROXY": proxy,
                        "HTTPS_PROXY": proxy,
                    }
                )

        forward = ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
        extra = os.environ.get("HARBOR_PODMAN_FORWARD_ENV", "")
        forward += [
            value.strip()
            for value in extra.split(",")
            if value.strip()
            and not (secure_qwen and is_secret_key(value.strip()))
        ]
        for var in forward:
            val = os.environ.get(var)
            if val:
                env[var] = val

        return env

    def _no_proxy_hosts(self) -> str:
        """localhost + the model endpoint host(s), by exact match.

        Auto-includes the host from OPENAI_BASE_URL / ANTHROPIC_BASE_URL (the
        action model may be pointed at either), plus any HARBOR_PODMAN_NO_PROXY
        extras.
        """
        if os.environ.get("SWE_QWEN_LOOPBACK_PROXY") == "1":
            _validate_secure_qwen_env()
            return "127.0.0.1,localhost"

        hosts = {"localhost", "127.0.0.1"}
        for var in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
            # A proxied Anthropic-compatible route (metagen, ARK, GLM, etc.)
            # must traverse the filtering/relay proxy. Adding its real target
            # to no_proxy makes curl in the Node fallback attempt an unreachable
            # direct connection. The local agent endpoint itself is localhost,
            # already covered above.
            if var == "ANTHROPIC_BASE_URL" and os.environ.get(
                "LITELLM_PROXY_MODEL"
            ):
                continue
            url = os.environ.get(var)
            if url:
                host = urlparse(url).hostname
                if host:
                    hosts.add(host)
        extra = os.environ.get("HARBOR_PODMAN_NO_PROXY", "")
        for h in extra.split(","):
            if h.strip():
                hosts.add(h.strip())
        return ",".join(sorted(hosts))

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, force_build: bool):
        image = self.task_env_config.docker_image
        if not image:
            raise RuntimeError(
                f"Task {self.environment_name} has no [environment].docker_image. "
                "The podman harness does not build images locally (apt egress is "
                "blocked); point docker_image at a prebuilt (e.g. ghcr.io) image."
            )
        if force_build:
            self.logger.warning(
                "force_build is ignored by PodmanEnvironment — using prebuilt "
                "image %s (local builds unsupported).",
                image,
            )

        # Keep the existence check, optional pull, and container creation
        # atomic with respect to stop()'s image removal for this image.
        lock = self._image_pull_locks.setdefault(image, asyncio.Lock())
        async with lock:
            exists = await self._run(
                self._podman("image", "exists", image), check=False
            )
            if exists.return_code != 0:
                async with self._pull_semaphore:
                    self.logger.info("pulling %s", image)
                    await self._run(self._podman("pull", image), timeout_sec=1800)

            # Remove any stale container from a previous run with this session id.
            await self._run(self._podman("rm", "-f", self._container), check=False)

            run_cmd = self._podman(
                "run",
                "-d",
                "--name",
                self._container,
                "--network",
                "host",
                "--ipc=host",
                "--uts=host",
                "--http-proxy=false",
                # Cap RLIMIT_NOFILE at our own hard limit. The OCI runtime otherwise
                # tries to raise NOFILE to podman's default (1048576); inside the
                # nested `unshare -Urm` userns that exceeds the parent hard limit and
                # fails ("setrlimit RLIMIT_NOFILE: Operation not permitted", OCI
                # permission denied) on hosts (e.g. login nodes) whose hard limit is
                # lower. Matching the current hard limit means no raise is attempted.
                # Override with HARBOR_PODMAN_NOFILE=<soft>:<hard> (or a single int).
                "--ulimit",
                f"nofile={self._nofile_ulimit()}",
                # Force root. `unshare -Urm` maps a SINGLE uid (0→our real uid), so
                # the image's non-root default USER (e.g. `agent`=1001) can't be
                # entered — the OCI runtime fails with "cannot setresgid to 1001".
                # Root (0) is the one mapped uid and is what this repo's E2B judge
                # also uses (it applies patches/tests as root for the same reason).
                "--user",
                "0:0",
            )
            container_env = self._container_env()
            if container_env:
                run_cmd += ["--env-file", _ENV_FILE_PLACEHOLDER]
            run_cmd += [image, "sleep", "infinity"]

            result = await self._run(run_cmd, container_env=container_env)
        self._container_id = (result.stdout or "").strip()
        self.logger.info(
            "started container %s (%s) from %s",
            self._container,
            self._container_id[:12],
            image,
        )

        # We run as root against repos the image chowned to a non-root user;
        # preempt git's "dubious ownership" refusal (mirrors the judge's
        # `git -c safe.directory='*'`). Best-effort — git may be absent.
        await self.exec(
            "git config --global --add safe.directory '*' 2>/dev/null || true",
            timeout_sec=30,
        )

    async def stop(self, delete: bool):
        image = self.task_env_config.docker_image
        remove_image = bool(
            delete and image and os.environ.get("HARBOR_PODMAN_RMI") == "1"
        )

        async def remove_container() -> None:
            res = await self._run(
                self._podman("rm", "-f", self._container), check=False
            )
            if res.return_code != 0:
                self.logger.warning(
                    "podman rm -f %s failed: %s", self._container, res.stderr
                )

        if remove_image:
            # A k>1 trial can share this image with a live container.  Serialize
            # against start(), and deliberately omit rmi --force: non-forced rmi
            # safely refuses while another container uses the image, whereas
            # Podman 3.x --force can remove those containers too.
            lock = self._image_pull_locks.setdefault(image, asyncio.Lock())
            async with lock:
                await remove_container()
                rmi = await self._run(
                    self._podman("rmi", image), check=False
                )
                if rmi.return_code != 0:
                    self.logger.debug(
                        "podman rmi %s failed (non-fatal): %s", image, rmi.stderr
                    )
        else:
            # Always remove the per-trial container, even when images are kept.
            await remove_container()

    # ── exec ──────────────────────────────────────────────────────────────────

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        env = self._merge_env(env)
        # Killing the host-side ``podman exec`` client does not kill its
        # in-container process (verified against Podman 3.4.4).  Put every
        # command in a dedicated job-control process group and persist its PGID
        # so a transport timeout can terminate the entire inner tree before a
        # multi-turn wrapper resumes the session.
        marker = f"/tmp/.harbor-exec-{uuid.uuid4().hex}.pid"
        marker_q = shlex.quote(marker)
        payload_q = shlex.quote(command)
        wrapped_command = (
            "set -m; "
            f"bash -c {payload_q} & harbor_child=$!; "
            f"printf '%s\\n' \"$harbor_child\" > {marker_q}; "
            "wait \"$harbor_child\"; harbor_rc=$?; "
            f"rm -f {marker_q}; exit \"$harbor_rc\""
        )
        # Exec as root for the same single-uid-userns reason as `run` (above).
        exec_cmd = self._podman("exec", "--user", "0:0")
        if cwd:
            exec_cmd += ["-w", cwd]
        if env:
            exec_cmd += ["--env-file", _ENV_FILE_PLACEHOLDER]
        exec_cmd += [self._container, "bash", "-c", wrapped_command]
        try:
            return await self._run(
                exec_cmd,
                check=False,
                timeout_sec=timeout_sec,
                container_env=env,
            )
        except TimeoutError:
            await self._terminate_exec_group(marker)
            raise
        except asyncio.CancelledError:
            # Harbor may cancel the coroutine at its outer deadline. Shield the
            # bounded cleanup so cancellation cannot leave an orphaned agent.
            await asyncio.shield(self._terminate_exec_group(marker))
            raise

    async def _terminate_exec_group(self, marker: str) -> None:
        marker_q = shlex.quote(marker)
        cleanup = (
            f"if [ -r {marker_q} ]; then "
            f"read -r harbor_pid < {marker_q}; "
            "case \"$harbor_pid\" in ''|*[!0-9]*) harbor_pid=;; esac; "
            "if [ -n \"$harbor_pid\" ]; then "
            "kill -TERM -- \"-$harbor_pid\" 2>/dev/null || true; "
            "harbor_i=0; while [ \"$harbor_i\" -lt 20 ] && "
            "kill -0 -- \"-$harbor_pid\" 2>/dev/null; do "
            "sleep 0.1; harbor_i=$((harbor_i + 1)); done; "
            "kill -KILL -- \"-$harbor_pid\" 2>/dev/null || true; "
            "fi; "
            f"rm -f {marker_q}; fi"
        )
        cleanup_cmd = self._podman(
            "exec", "--user", "0:0", self._container, "bash", "-c", cleanup
        )
        try:
            result = await self._run(
                cleanup_cmd, check=False, timeout_sec=10
            )
            if result.return_code != 0:
                self.logger.warning(
                    "inner exec cleanup failed for %s (rc=%s): %s",
                    self.environment_name,
                    result.return_code,
                    result.stderr,
                )
        except Exception as exc:
            self.logger.warning(
                "inner exec cleanup failed for %s: %s",
                self.environment_name,
                exc,
            )

    # ── file transfer (podman cp) ─────────────────────────────────────────────

    async def upload_file(self, source_path: Path | str, target_path: str):
        await self._run(
            self._podman("cp", str(source_path), f"{self._container}:{target_path}")
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        # Ensure the destination exists, then copy the *contents* of source_dir
        # into it (the trailing "/." mirrors docker/E2B semantics).
        await self.exec(f"mkdir -p {shlex.quote(target_dir)}")
        await self._run(
            self._podman(
                "cp", f"{source_dir}/.", f"{self._container}:{target_dir}"
            )
        )

    async def download_file(self, source_path: str, target_path: Path | str):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        await self._run(
            self._podman("cp", f"{self._container}:{source_path}", str(target_path))
        )

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        await self._run(
            self._podman(
                "cp", f"{self._container}:{source_dir}/.", str(target_dir)
            )
        )
