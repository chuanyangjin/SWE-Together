#!/usr/bin/env python3
"""In-userns bootstrap for the SWE-Together local podman harness.

``run_local.sh`` (run stage) and ``judge_local.sh`` (judge stage) exec this
script *inside* ``unshare -Urm`` — a user+mount namespace where our single real
uid is mapped to root (uid 0) and we hold a full capability set over the
namespace's own resources. Its job is to prepare the fragile podman
prerequisites and then hand off to whatever command it was given (``python
<args>``, e.g. ``src/run_eval.py … --env-type podman`` or ``-m
eval.correctness.run_batch …``):

  1. A **tmpfs image store at /mnt** — c/storage SIGSEGVs on overlayfs/NFS
     stores, so the only combination that works is the ``vfs`` driver on a tmpfs.
  2. A **writable /run** — podman's root-mode runtime paths (``/run/libpod``)
     must be creatable. On hosts where ``/run`` is owned by an unmapped uid
     (e.g. a Slurm login node) our namespace-root cannot write it, so we shadow
     it with a tmpfs. Skipped when ``/run`` is already writable (e.g. the
     original ``claude-sandbox-pod``), keeping behaviour identical there.
  3. ``HARBOR_PODMAN_*`` / ``XDG_RUNTIME_DIR`` / ``TMPDIR`` so
     ``PodmanEnvironment`` finds the store and temp files land on tmpfs.

Mounts go through the ``mount(2)`` syscall via ctypes rather than the setuid
``/usr/bin/mount`` binary: on some hosts, execve of a setuid-root binary inside
the userns returns EPERM even though we hold CAP_SYS_ADMIN in the namespace and
the raw syscall succeeds. Calling the syscall directly sidesteps that entirely.
"""

from __future__ import annotations

import ctypes
import os
import sys

MS_BIND = 0x1000

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _mount(
    source: str | None,
    target: str,
    fstype: str | None,
    flags: int = 0,
    data: str = "",
) -> None:
    rc = _libc.mount(
        source.encode() if source else None,
        target.encode(),
        fstype.encode() if fstype else None,
        flags,
        data.encode() if data else None,
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(
            err,
            f"mount({source!r} -> {target!r}, {fstype!r}, flags={flags:#x}, "
            f"data={data!r}) failed: {os.strerror(err)}",
        )


def _writable(path: str) -> bool:
    """True if we can create (and remove) a subdirectory under ``path``."""
    probe = os.path.join(path, ".hb-write-probe")
    try:
        os.mkdir(probe)
        os.rmdir(probe)
        return True
    except OSError:
        return False


def main() -> None:
    repo = os.environ["SWE_REPO"]
    py = os.environ.get("SWE_PY", sys.executable)
    tmpfs_size = os.environ.get("SWE_TMPFS_SIZE", "120G")
    outer_uid = os.environ.get("SWE_OUTER_UID", str(os.getuid()))

    # Snapshot resolv.conf up front. It is a real file on the hosts we run on, so
    # a /run tmpfs never touches it — but if some host symlinks it into /run we
    # bind-mount the snapshot back so DNS survives (podman pull / gateway calls).
    resolv = "/etc/resolv.conf"
    try:
        with open(resolv, "rb") as f:
            resolv_bytes: bytes | None = f.read()
    except OSError:
        resolv_bytes = None

    # 1. tmpfs image store at /mnt.
    _mount("none", "/mnt", "tmpfs", data=f"size={tmpfs_size}")
    for sub in ("store", "run", "tmp", "xdg"):
        os.makedirs(f"/mnt/{sub}", exist_ok=True)

    # 2. Ensure podman's root-mode /run/libpod is creatable.
    if not _writable("/run"):
        _mount("none", "/run", "tmpfs")
        if resolv_bytes is not None:
            try:
                with open(resolv, "rb") as f:
                    still_ok = f.read() == resolv_bytes
            except OSError:
                still_ok = False
            if not still_ok:
                good = "/mnt/tmp/resolv.conf"
                with open(good, "wb") as f:
                    f.write(resolv_bytes)
                _mount(good, resolv, None, MS_BIND)

    # podman 3.4.4 stats this runtime dir even when --runroot is given.
    try:
        os.makedirs(f"/tmp/podman-run-{outer_uid}", exist_ok=True)
    except OSError:
        pass

    # 3. Point PodmanEnvironment (and temp I/O) at the tmpfs.
    os.environ["HARBOR_PODMAN_STORE"] = "/mnt/store"
    os.environ["HARBOR_PODMAN_RUNROOT"] = "/mnt/run"
    os.environ["HARBOR_PODMAN_TMPDIR"] = "/mnt/tmp"
    os.environ["XDG_RUNTIME_DIR"] = "/mnt/xdg"
    os.environ["TMPDIR"] = "/mnt/tmp"

    os.chdir(repo)
    # Hand off to the command we were given, run under SWE_PY. Callers pass the
    # full python arg vector, e.g. `src/run_eval.py … --env-type podman`
    # (run_local.sh) or `-m eval.correctness.run_batch …` (judge_local.sh).
    if len(sys.argv) < 2:
        raise SystemExit("podman_bootstrap: no command given to exec")
    os.execv(py, [py, *sys.argv[1:]])


if __name__ == "__main__":
    main()
