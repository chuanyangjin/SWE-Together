#!/usr/bin/env python3
"""Atomically add/remove one labelled exact IP in the secure relay registry."""

from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import os
import socket
import stat
import tempfile
from pathlib import Path
from typing import Any

from secure_tcp_relay import LABEL_RE, REGISTRY_VERSION


def _canonical_ip(raw: str) -> str:
    address = ipaddress.ip_address(raw)
    if str(address) != raw:
        raise ValueError("IP address must use canonical notation")
    return raw


def _source_ip_for(relay_host: str, relay_port: int) -> str:
    candidates = socket.getaddrinfo(
        relay_host, relay_port, type=socket.SOCK_DGRAM
    )
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in candidates:
        try:
            with socket.socket(family, socktype, proto) as probe:
                probe.connect(sockaddr)
                return str(ipaddress.ip_address(probe.getsockname()[0]))
        except OSError as exc:
            last_error = exc
    raise RuntimeError("could not determine relay-route source IP") from last_error


def _validate_existing(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError("registry is not a regular file")
    if metadata.st_uid != os.getuid():
        os.close(descriptor)
        raise RuntimeError("registry is not owned by the current user")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        os.close(descriptor)
        raise RuntimeError("registry is group/world accessible")
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("version") != REGISTRY_VERSION:
        raise RuntimeError("registry schema/version is invalid")
    clients = payload.get("clients")
    if not isinstance(clients, dict):
        raise RuntimeError("registry clients field is invalid")
    for label, raw_ip in clients.items():
        if not isinstance(label, str) or not LABEL_RE.fullmatch(label):
            raise RuntimeError("registry contains an invalid label")
        if not isinstance(raw_ip, str):
            raise RuntimeError("registry contains an invalid IP")
        _canonical_ip(raw_ip)
    return payload


def _secure_lock(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        os.close(descriptor)
        raise RuntimeError("registry lock is not an owner-controlled regular file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        os.close(descriptor)
        raise RuntimeError("registry lock is group/world accessible")
    return descriptor


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temp)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_registry(path: Path, label: str, ip: str | None) -> None:
    # Make the path absolute without following a possibly malicious symlink.
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_descriptor = _secure_lock(lock_path)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if path.exists() or path.is_symlink():
            payload = _validate_existing(path)
        else:
            payload = {"version": REGISTRY_VERSION, "clients": {}}
        clients = dict(payload["clients"])
        if ip is None:
            clients.pop(label, None)
        else:
            clients[label] = _canonical_ip(ip)
        _atomic_write(path, {"version": REGISTRY_VERSION, "clients": clients})
    finally:
        os.close(lock_descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--relay-host")
    parser.add_argument("--relay-port", type=int, default=48836)
    parser.add_argument("--ip")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    if not LABEL_RE.fullmatch(args.label):
        parser.error("--label must match [A-Za-z0-9_.-]{1,96}")
    if args.remove and (args.ip or args.relay_host):
        parser.error("--remove cannot be combined with --ip/--relay-host")
    if not args.remove and bool(args.ip) == bool(args.relay_host):
        parser.error("registration requires exactly one of --ip or --relay-host")

    ip = None
    if not args.remove:
        ip = (
            _canonical_ip(args.ip)
            if args.ip
            else _source_ip_for(args.relay_host, args.relay_port)
        )
    update_registry(args.registry, args.label, ip)
    print(f"relay client {'removed' if args.remove else 'registered'}: {args.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
