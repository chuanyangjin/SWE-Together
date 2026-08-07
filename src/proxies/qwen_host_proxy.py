#!/usr/bin/env python3
"""Loopback-only authenticated bridge to the internal Qwen deployment.

The real deployment bearer is read from ``proxy_info.json`` in this process
and injected only into the upstream request.  Sandboxes receive a harmless
placeholder key and connect to this proxy over the Podman host network.  The
bearer is never accepted on argv, placed in the process environment, logged,
or returned downstream.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import signal
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web


REDACTED_BEARER = b"<redacted-upstream-bearer>"
PLACEHOLDER_KEY = "qwen-loopback-placeholder"
EXPECTED_MODEL = "Qwen3.5-4B"
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class DeploymentInfo:
    base_url: str
    api_key: str
    model: str


def _read_regular_file(path: Path, limit: int = 64 * 1024) -> bytes:
    """Read one owner-only regular file without following a swapped symlink."""

    path = path.expanduser()
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeError("deployment info is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise RuntimeError("deployment info must be an owner-only regular file")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("deployment info could not be opened securely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) & 0o077
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError("deployment info changed during validation")
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
        raise RuntimeError("deployment info is unexpectedly large")
    return raw


def load_deployment_info(path: Path) -> DeploymentInfo:
    try:
        payload = json.loads(_read_regular_file(path))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("deployment info is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("deployment info must be a JSON object")
    base_url = payload.get("url")
    api_key = payload.get("api_key")
    model = payload.get("model")
    if not all(isinstance(value, str) and value for value in (base_url, api_key, model)):
        raise RuntimeError("deployment info is missing url/api_key/model")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise RuntimeError("deployment URL must be a credential-free HTTP origin")
    if not re.fullmatch(r"[A-Za-z0-9._~+/=-]{8,4096}", api_key):
        raise RuntimeError("deployment api_key has an invalid shape")
    if model != EXPECTED_MODEL:
        raise RuntimeError(f"deployment model must be {EXPECTED_MODEL}")
    return DeploymentInfo(base_url=base_url.rstrip("/"), api_key=api_key, model=model)


def _request_headers(request: web.Request, api_key: str) -> dict[str, str]:
    connection_headers = {
        value.strip().lower()
        for value in request.headers.get("Connection", "").split(",")
        if value.strip()
    }
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower()
        not in _HOP_BY_HOP
        | connection_headers
        | {"authorization", "x-api-key", "host", "content-length"}
    }
    headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _redaction_replacement(secret: bytes) -> bytes:
    """Return a marker that can never reproduce ``secret`` downstream."""

    if not secret:
        raise ValueError("secret must be nonempty")
    return REDACTED_BEARER if secret not in REDACTED_BEARER else b""


class _StreamingSecretRedactor:
    """Incrementally remove an exact byte string across arbitrary chunks."""

    def __init__(self, secret: bytes) -> None:
        if not secret:
            raise ValueError("secret must be nonempty")
        self._secret = secret
        self._replacement = _redaction_replacement(secret)
        if secret in self._replacement:  # Defensive invariant for future edits.
            raise ValueError("redaction replacement contains the secret")
        self._pending = b""

    @property
    def replacement(self) -> bytes:
        return self._replacement

    def feed(self, chunk: bytes) -> bytes:
        self._pending += chunk
        emitted: list[bytes] = []
        while len(self._pending) >= len(self._secret):
            match = self._pending.find(self._secret)
            if match < 0:
                # Only the final len(secret)-1 bytes can begin a match that
                # finishes in a future chunk.
                safe_length = len(self._pending) - len(self._secret) + 1
                emitted.append(self._pending[:safe_length])
                self._pending = self._pending[safe_length:]
                break
            emitted.extend((self._pending[:match], self._replacement))
            self._pending = self._pending[match + len(self._secret) :]
        return b"".join(emitted)

    def finish(self) -> bytes:
        remaining = self._pending.replace(self._secret, self._replacement)
        self._pending = b""
        return remaining


def _response_headers(
    response: aiohttp.ClientResponse, secret: str
) -> dict[str, str]:
    connection_headers = {
        value.strip().lower()
        for value in response.headers.get("Connection", "").split(",")
        if value.strip()
    }
    return {
        key: value
        for key, value in response.headers.items()
        if (
            key.lower()
            not in _HOP_BY_HOP
            | connection_headers
            | {"content-length", "content-encoding", "set-cookie", "location"}
            and secret not in key
            and secret not in value
        )
    }


async def _write_redacted_stream(
    downstream: web.StreamResponse,
    upstream: aiohttp.ClientResponse,
    secret: bytes,
) -> None:
    """Copy a streaming response while catching secrets split across chunks."""

    redactor = _StreamingSecretRedactor(secret)
    async for chunk in upstream.content.iter_chunked(64 * 1024):
        output = redactor.feed(chunk)
        if output:
            await downstream.write(output)
    output = redactor.finish()
    if output:
        await downstream.write(output)


def create_app(proxy_info: Path) -> web.Application:
    app = web.Application(client_max_size=64 * 1024**2)
    session_key: web.AppKey[aiohttp.ClientSession] = web.AppKey(
        "upstream_session", aiohttp.ClientSession
    )

    async def session_context(application: web.Application):
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=600)
        application[session_key] = aiohttp.ClientSession(
            timeout=timeout,
            trust_env=False,
            auto_decompress=True,
        )
        yield
        await application[session_key].close()

    app.cleanup_ctx.append(session_context)

    @web.middleware
    async def require_placeholder(
        request: web.Request, handler
    ) -> web.StreamResponse:
        scheme, separator, candidate = request.headers.get(
            "Authorization", ""
        ).partition(" ")
        if not (
            separator == " "
            and scheme.lower() == "bearer"
            and secrets.compare_digest(candidate, PLACEHOLDER_KEY)
        ):
            return web.json_response(
                {"error": "local proxy authentication required"}, status=401
            )
        return await handler(request)

    app.middlewares.append(require_placeholder)

    async def health(_request: web.Request) -> web.Response:
        try:
            info = load_deployment_info(proxy_info)
            async with app[session_key].get(
                f"{info.base_url}/health",
                headers={"Authorization": f"Bearer {info.api_key}"},
            ) as response:
                ok = response.status == 200
        except Exception:  # Never reflect exception text; it may contain credentials.
            ok = False
        return web.json_response(
            {"ok": ok, "service": "qwen-host-proxy"}, status=200 if ok else 503
        )

    async def forward(request: web.Request) -> web.StreamResponse:
        try:
            info = load_deployment_info(proxy_info)
            upstream_url = f"{info.base_url}{request.rel_url}"
            body = await request.read()
            async with app[session_key].request(
                request.method,
                upstream_url,
                headers=_request_headers(request, info.api_key),
                data=body,
                allow_redirects=False,
            ) as upstream:
                if 300 <= upstream.status < 400:
                    return web.json_response(
                        {"error": "qwen upstream redirect refused"}, status=502
                    )
                downstream = web.StreamResponse(
                    status=upstream.status,
                    headers=_response_headers(upstream, info.api_key),
                )
                await downstream.prepare(request)
                await _write_redacted_stream(
                    downstream, upstream, info.api_key.encode()
                )
                await downstream.write_eof()
                return downstream
        except web.HTTPException:
            raise
        except Exception:  # Fail closed without reflecting upstream exception details.
            return web.json_response(
                {"error": "qwen host proxy upstream failure"}, status=502
            )

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", forward)
    app.router.add_post("/v1/chat/completions", forward)
    return app


def _write_ready_file(path: Path, port: int) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.ready-"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "host": "127.0.0.1",
                    "port": port,
                    "pid": os.getpid(),
                    "model": EXPECTED_MODEL,
                },
                handle,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


async def serve(proxy_info: Path, port: int, ready_file: Path) -> None:
    runner = web.AppRunner(create_app(proxy_info), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=port)
    await site.start()
    sockets = getattr(site, "_server", None).sockets  # aiohttp exposes no public port API.
    actual_port = int(sockets[0].getsockname()[1])
    _write_ready_file(ready_file, actual_port)
    print(f"qwen host proxy ready on 127.0.0.1:{actual_port}", flush=True)

    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopped.set)
        except NotImplementedError:  # pragma: no cover - non-POSIX fallback.
            pass
    try:
        await stopped.wait()
    finally:
        ready_file.unlink(missing_ok=True)
        await runner.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-info", required=True, type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    # Preflight before binding or writing the readiness file.
    load_deployment_info(args.proxy_info)
    asyncio.run(serve(args.proxy_info, args.port, args.ready_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
