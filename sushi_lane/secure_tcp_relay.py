#!/usr/bin/env python3
"""Fail-closed TCP relay with an owner-only exact-source-IP registry.

The registry is reloaded for every accepted connection so Slurm jobs can add
and remove their assigned node IP atomically without restarting the relay.
Malformed, missing, symlinked, non-owner, or group/world-readable registries
deny the connection before the upstream socket is opened.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


REGISTRY_VERSION = 1
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")


def _validate_registry(path: Path) -> dict[str, str]:
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
        raise RuntimeError("registry is not owned by the relay user")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        os.close(descriptor)
        raise RuntimeError("registry is group/world accessible")

    with os.fdopen(descriptor, encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict) or payload.get("version") != REGISTRY_VERSION:
        raise RuntimeError("registry schema/version is invalid")
    clients = payload.get("clients")
    if not isinstance(clients, dict):
        raise RuntimeError("registry clients field is invalid")

    validated: dict[str, str] = {}
    for label, raw_ip in clients.items():
        if not isinstance(label, str) or not LABEL_RE.fullmatch(label):
            raise RuntimeError("registry contains an invalid client label")
        if not isinstance(raw_ip, str):
            raise RuntimeError("registry contains a non-string client IP")
        address = ipaddress.ip_address(raw_ip)
        if str(address) != raw_ip:
            raise RuntimeError("registry client IP is not canonical")
        validated[label] = raw_ip
    return validated


def _canonical_peer(raw: Any) -> str | None:
    if not isinstance(raw, tuple) or not raw:
        return None
    try:
        address = ipaddress.ip_address(str(raw[0]))
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return str(address)


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


class Relay:
    def __init__(self, registry: Path, target_host: str, target_port: int) -> None:
        self.registry = registry
        self.target_host = target_host
        self.target_port = target_port

    async def handle(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        peer = _canonical_peer(client_writer.get_extra_info("peername"))
        try:
            clients = _validate_registry(self.registry)
            allowed = peer is not None and peer in set(clients.values())
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"relay deny: invalid registry ({type(exc).__name__})", flush=True)
            allowed = False

        if not allowed:
            print(f"relay deny: unregistered source {peer or 'unknown'}", flush=True)
            client_writer.close()
            await client_writer.wait_closed()
            return

        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                self.target_host, self.target_port
            )
        except (OSError, ConnectionError) as exc:
            print(f"relay upstream unavailable ({type(exc).__name__})", flush=True)
            client_writer.close()
            await client_writer.wait_closed()
            return
        await asyncio.gather(
            _pipe(client_reader, upstream_writer),
            _pipe(upstream_reader, client_writer),
        )


async def _main(args: argparse.Namespace) -> None:
    clients = _validate_registry(args.registry)
    relay = Relay(args.registry, args.target_host, args.target_port)
    server = await asyncio.start_server(relay.handle, args.listen_host, args.listen_port)
    print(
        "secure relay listening "
        f"on {args.listen_host}:{args.listen_port}; "
        f"registered_clients={len(clients)}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
