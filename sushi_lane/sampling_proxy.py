#!/usr/bin/env python3
"""Streaming reverse proxy that pins Sushi/Qwen action-model sampling defaults."""

from __future__ import annotations

import argparse
import hmac
import http.client
import json
import logging
import os
import socketserver
import stat
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROFILE = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.05,
    "max_tokens": 32768,
    "chat_template_kwargs": {"enable_thinking": False},
}
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream: str
    upstream_api_key: str
    client_api_key = "sushi-local-placeholder-key"
    request_count = 0

    def log_message(self, fmt: str, *args) -> None:
        logging.info("client=%s " + fmt, self.client_address[0], *args)

    def _profile_response(self) -> None:
        payload = json.dumps({"status": "ok", "sampling": PROFILE}, sort_keys=True).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _forward(self) -> None:
        if self.path == "/sampling-profile":
            self._profile_response()
            return

        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.client_api_key}"
        if not hmac.compare_digest(supplied, expected):
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        parsed = urlparse(self.upstream)
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length) if length else b""
        modified = False
        stream = False
        if self.command == "POST" and self.path.rstrip("/").endswith("chat/completions"):
            try:
                payload = json.loads(request_body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.send_error(400, f"invalid JSON: {exc}")
                return
            if not isinstance(payload, dict):
                self.send_error(400, "chat-completions payload must be an object")
                return
            stream = bool(payload.get("stream"))
            payload.update(PROFILE)
            # Avoid sending two competing output-limit fields.
            payload.pop("max_completion_tokens", None)
            request_body = json.dumps(payload, separators=(",", ":")).encode()
            modified = True

        upstream_path = self.path
        base_path = parsed.path.rstrip("/")
        if upstream_path != "/health" and base_path and not (
            upstream_path == base_path or upstream_path.startswith(base_path + "/")
        ):
            upstream_path = base_path + "/" + upstream_path.lstrip("/")
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP
            and key.lower() not in {"host", "content-length", "authorization"}
        }
        headers["Authorization"] = f"Bearer {self.upstream_api_key}"
        if request_body:
            headers["Content-Length"] = str(len(request_body))
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=7200)
        try:
            connection.request(self.command, upstream_path, body=request_body or None, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP and key.lower() != "content-length":
                    self.send_header(key, value)
            if self.command == "HEAD":
                self.end_headers()
                return
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                # read1 returns currently available bytes instead of waiting
                # for a full 64 KiB buffer, preserving OpenCode's SSE stream.
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            type(self).request_count += 1
            logging.info(
                "request=%d method=%s path=%s status=%d sampling_pinned=%s stream=%s",
                type(self).request_count,
                self.command,
                self.path,
                response.status,
                modified,
                stream,
            )
        except (BrokenPipeError, ConnectionResetError):
            logging.warning("downstream disconnected path=%s", self.path)
        except Exception:
            logging.exception("proxy failure method=%s path=%s", self.command, self.path)
            if not self.wfile.closed:
                try:
                    self.send_error(502, "upstream request failed")
                except Exception:
                    pass
        finally:
            connection.close()

    do_GET = _forward
    do_HEAD = _forward
    do_POST = _forward
    do_OPTIONS = _forward


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--upstream-api-key-file", type=Path, required=True)
    args = parser.parse_args()
    parsed = urlparse(args.upstream)
    if parsed.scheme != "http" or not parsed.hostname:
        parser.error("--upstream must be an http:// URL")
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("sampling proxy must listen on loopback")
    metadata = args.upstream_api_key_file.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        parser.error("upstream API key file must be an owner-only regular file")
    upstream_api_key = args.upstream_api_key_file.read_text().strip()
    if len(upstream_api_key) < 32:
        parser.error("upstream API key is implausibly short")
    Handler.upstream = args.upstream.rstrip("/")
    Handler.upstream_api_key = upstream_api_key
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(
        "sampling proxy listening=%s:%d upstream=%s profile=%s",
        args.host,
        args.port,
        Handler.upstream,
        json.dumps(PROFILE, sort_keys=True),
    )
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
