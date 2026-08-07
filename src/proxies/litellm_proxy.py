"""In-sandbox LiteLLM-compat reverse proxy launcher.

Used by `user_enabled_{claude_code,mini_swe_agent,opencode}.py` to spin up the
`localhost:4210` proxy that translates an Anthropic-compat client (CC, LiteLLM,
opencode's anthropic provider) onto a real upstream like MiniMax, GLM, ARK,
DeepSeek, or OpenRouter.

The proxy itself is a stdlib http.server that:
  - Reads `LITELLM_PROXY_MODEL` / `PROXY_TARGET_URL` / `PROXY_API_KEY` / etc.
    from the env that `src/run_eval.py:build_agent_env` already injected.
  - Rewrites the `model` field in POST bodies to the real target.
  - Streams SSE chunks via Transfer-Encoding: chunked (CC's parser requires
    real streaming, not buffered Content-Length).
  - Has a fallback path to OpenRouter on 429 (and ARK Bearer auth detection).
  - z.ai silent-throttle detection + retry.

Behaviour matches what `user_enabled_claude_code.py` shipped historically;
extracting here so mini-swe-agent + opencode can use the same proxy and route
`minimaxd/`, `glmd/`, `ark/`, `deepseek/`, etc. without each wrapper duplicating
~280 lines of proxy-script generation.

Usage:

    from proxies.litellm_proxy import launch_litellm_proxy

    async def setup(self, environment):
        await self._inner.setup(environment)
        await launch_litellm_proxy(environment, self.logs_dir)
        # … rest of setup
"""
from __future__ import annotations

import logging
import hashlib
import json
import os
import secrets
import socket
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
_RESERVED_HOST_PORTS: set[int] = set()


def allocate_litellm_proxy_port(environment: Any) -> int:
    """Choose a collision-free host port for rootless Podman trials.

    Podman uses ``--network host`` for internal model reachability, so the
    historical fixed 4210 port was shared by every concurrent container.  A
    health check could therefore accept another trial's proxy.  Other sandbox
    backends have isolated networks and retain the configured/default port.
    """
    configured = int(os.environ.get("LITELLM_PROXY_PORT", "4210"))
    kind = environment.type()
    env_type = getattr(kind, "value", kind)
    if str(env_type).lower() != "podman":
        return configured

    identity = str(getattr(environment, "session_id", id(environment)))
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
    low, span = 20000, 30000
    for offset in range(span):
        port = low + ((seed + offset) % span)
        if port in _RESERVED_HOST_PORTS:
            continue
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        except OSError:
            continue
        _RESERVED_HOST_PORTS.add(port)
        return port
    raise RuntimeError("No free host port available for the in-sandbox model proxy")


def _proxy_script_source(*, proxy_port: str, target_url: str,
                         proxy_model: str, is_openrouter_target: bool,
                         fallback_url: str, fallback_model: str,
                         instance_id: str = "") -> str:
    """Generate the proxy.py source from per-trial config.

    Kept verbatim from `user_enabled_claude_code.py`'s historical inline
    string so behaviour is identical across all callers — the only change is
    that the format-string substitutions now come from helper args instead
    of being inlined in the caller. Any future tuning to the proxy (e.g.
    new fallback strategies, more provider quirks) should happen here.
    """
    return f'''#!/usr/bin/env python3
"""Reverse proxy: remaps model, forwards to target API, falls back to OpenRouter on 429."""
import http.server, urllib.request, ssl, json, os, sys, threading, time

TARGET = "{target_url}"
PORT = {proxy_port}
# Credentials are injected only into this process environment by the launcher.
# Never interpolate them into this file: model_proxy.py is retained as a trial
# artifact and is commonly uploaded for benchmark inspection.
API_KEY = os.environ.get("PROXY_API_KEY", "")
REMAP_MODEL = "{proxy_model}"
IS_OPENROUTER = {is_openrouter_target}

FALLBACK_URL = "{fallback_url}"
FALLBACK_KEY = os.environ.get("PROXY_FALLBACK_KEY", "")
FALLBACK_MODEL = "{fallback_model}"
INSTANCE_ID = "{instance_id}"
MAX_RETRIES = 2
RETRY_DELAY = 5
UPSTREAM_TIMEOUT = 600

class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _build_request(self, url, body, is_or):
        is_anthropic_route = REMAP_MODEL.startswith("anthropic/")
        is_zai_route = REMAP_MODEL.startswith("z-ai/")
        strip_beta = is_or and not is_anthropic_route and not is_zai_route
        is_ark = "volces.com" in TARGET
        headers = {{}}
        for k, v in self.headers.items():
            k_lower = k.lower()
            # accept-encoding: NEVER forward. litellm/httpx advertises gzip;
            # MiniMax compresses non-streaming JSON bodies; we strip the
            # content-encoding RESPONSE header below but stream the raw
            # (compressed) bytes — the client then dies with "'utf-8' codec
            # can't decode byte 0x8b in position 1" (gzip magic) and the
            # whole agent turn fails. 71% of mini_mm27 lite70 trials zeroed
            # this way (2026-06-05). Force identity so upstream never
            # compresses. Streaming SSE (claude-code, opencode) was immune —
            # event-streams don't get gzipped — which is why this hid for
            # months.
            if k_lower in ("host", "content-length", "accept-encoding"):
                continue
            if strip_beta and k_lower == "anthropic-beta":
                continue
            headers[k] = v
        headers["Accept-Encoding"] = "identity"
        if is_or:
            headers["Authorization"] = f"Bearer {{FALLBACK_KEY}}"
            headers["HTTP-Referer"] = "https://togetherbench.com"
            headers["X-Title"] = "togetherbench-eval"
            for h in ("x-api-key", "X-Api-Key"):
                headers.pop(h, None)
        elif is_ark:
            headers["Authorization"] = f"Bearer {{API_KEY}}"
            for h in ("x-api-key", "X-Api-Key"):
                headers.pop(h, None)
        else:
            headers["x-api-key"] = API_KEY
        return urllib.request.Request(url, data=body, headers=headers, method="POST")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)

        body_primary = raw_body
        if REMAP_MODEL:
            try:
                data = json.loads(raw_body)
                data["model"] = REMAP_MODEL
                if IS_OPENROUTER and REMAP_MODEL.startswith("z-ai/"):
                    data["provider"] = {{"only": ["z-ai"]}}
                body_primary = json.dumps(data).encode()
            except (json.JSONDecodeError, KeyError):
                pass

        url = TARGET + self.path
        ctx = ssl.create_default_context()

        for attempt in range(MAX_RETRIES + 1):
            req = self._build_request(url, body_primary, IS_OPENROUTER)
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=UPSTREAM_TIMEOUT) as resp:
                    first_chunk = resp.read1(8192)
                    if (b"event: error" in first_chunk
                            and (b'"code":"1302"' in first_chunk
                                 or b"Rate limit" in first_chunk
                                 or b"rate limit" in first_chunk)):
                        if attempt < MAX_RETRIES:
                            print(f"[proxy] z.ai silent throttle (attempt {{attempt+1}}/{{MAX_RETRIES+1}}), retrying in {{RETRY_DELAY}}s...", flush=True)
                            time.sleep(RETRY_DELAY)
                            continue
                        elif FALLBACK_URL and FALLBACK_MODEL:
                            print(f"[proxy] z.ai silent throttle exhausted, falling back to OpenRouter/{{FALLBACK_MODEL}}", flush=True)
                            break
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() in ("content-encoding", "content-length", "transfer-encoding"):
                            continue
                        self.send_header(k, v)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    if first_chunk:
                        self.wfile.write(f"{{len(first_chunk):x}}\\r\\n".encode() + first_chunk + b"\\r\\n")
                        self.wfile.flush()
                    while True:
                        chunk = resp.read1(8192)
                        if not chunk:
                            self.wfile.write(b"0\\r\\n\\r\\n")
                            break
                        self.wfile.write(f"{{len(chunk):x}}\\r\\n".encode() + chunk + b"\\r\\n")
                        self.wfile.flush()
                    return
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < MAX_RETRIES:
                    print(f"[proxy] 429 from primary (attempt {{attempt+1}}), retrying in {{RETRY_DELAY}}s...", flush=True)
                    e.read()
                    time.sleep(RETRY_DELAY)
                    continue
                elif e.code == 429 and FALLBACK_URL and FALLBACK_MODEL:
                    print(f"[proxy] 429 from primary, falling back to OpenRouter/{{FALLBACK_MODEL}}", flush=True)
                    e.read()
                    break
                else:
                    resp_body = e.read()
                    self.send_response(e.code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
                    return
            except Exception as e:
                err = json.dumps({{"error": {{"message": str(e), "type": "proxy_error"}}}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
                return

        if FALLBACK_URL and FALLBACK_MODEL:
            try:
                data = json.loads(raw_body)
                data["model"] = FALLBACK_MODEL
                body_fb = json.dumps(data).encode()
            except:
                body_fb = raw_body
            fb_url = FALLBACK_URL + self.path
            req = self._build_request(fb_url, body_fb, True)
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=UPSTREAM_TIMEOUT) as resp:
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() in ("content-encoding", "content-length", "transfer-encoding"):
                            continue
                        self.send_header(k, v)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = resp.read1(8192)
                        if not chunk:
                            self.wfile.write(b"0\\r\\n\\r\\n")
                            break
                        self.wfile.write(f"{{len(chunk):x}}\\r\\n".encode() + chunk + b"\\r\\n")
                        self.wfile.flush()
                    return
            except urllib.error.HTTPError as e:
                resp_body = e.read()
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
            except Exception as e:
                err = json.dumps({{"error": {{"message": str(e), "type": "fallback_error"}}}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            body = json.dumps({{"status": "ok", "instance": INSTANCE_ID}}).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        pass

server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Proxy)
server.daemon_threads = True
print(f"Proxy listening on port {{PORT}}")
server.serve_forever()
'''


def _node_proxy_script_source(
    *,
    proxy_port: int,
    target_url: str,
    proxy_model: str,
    is_openrouter_target: bool,
    instance_id: str,
) -> str:
    """Dependency-free Node fallback for images without Python.

    Curl performs the upstream request because it reliably honors the cluster's
    HTTP(S) proxy variables; Node/undici does not.  Response headers are parsed
    once and the body is streamed to the local Anthropic-compatible client.
    """
    config = json.dumps(
        {
            "port": proxy_port,
            "target": target_url,
            "model": proxy_model,
            "openrouter": is_openrouter_target,
            "instance": instance_id,
        }
    )
    template = r'''#!/usr/bin/env node
const http = require("http");
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const cfg = __CONFIG__;

function sendJson(res, status, value) {
  const body = Buffer.from(JSON.stringify(value));
  res.writeHead(status, {"content-type": "application/json", "content-length": body.length});
  res.end(body);
}

const server = http.createServer((req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    sendJson(res, 200, {status: "ok", instance: cfg.instance});
    return;
  }
  if (req.method !== "POST") {
    sendJson(res, 404, {error: {message: "not found"}});
    return;
  }

  const chunks = [];
  let size = 0;
  req.on("data", (chunk) => {
    size += chunk.length;
    if (size > 16 * 1024 * 1024) req.destroy(new Error("request too large"));
    else chunks.push(chunk);
  });
  req.on("error", (err) => {
    if (!res.headersSent) sendJson(res, 400, {error: {message: err.message, type: "proxy_error"}});
  });
  req.on("end", () => {
    let body = Buffer.concat(chunks);
    try {
      const value = JSON.parse(body.toString("utf8"));
      value.model = cfg.model;
      if (cfg.openrouter && cfg.model.startsWith("z-ai/")) value.provider = {only: ["z-ai"]};
      body = Buffer.from(JSON.stringify(value));
    } catch (_) {}

    const headers = {};
    for (const [name, value] of Object.entries(req.headers)) {
      const lower = name.toLowerCase();
      if (["host", "content-length", "accept-encoding", "connection"].includes(lower)) continue;
      headers[lower] = Array.isArray(value) ? value.join(", ") : String(value);
    }
    headers["accept-encoding"] = "identity";
    const apiKey = process.env.PROXY_API_KEY || "";
    if (cfg.openrouter) {
      headers["authorization"] = "Bearer " + (process.env.PROXY_FALLBACK_KEY || apiKey);
      headers["http-referer"] = "https://togetherbench.com";
      headers["x-title"] = "togetherbench-eval";
      delete headers["x-api-key"];
    } else if (cfg.target.includes("volces.com")) {
      headers["authorization"] = "Bearer " + apiKey;
      delete headers["x-api-key"];
    } else {
      headers["x-api-key"] = apiKey;
    }

    // Never place header values (notably x-api-key/Authorization) in curl's
    // argv, where any same-host process can read them through /proc or `ps`.
    // Curl supports `-H @file`. Create an empty private inode, unlink it BEFORE
    // writing credentials, then inherit a read descriptor as fd 3. Thus even a
    // SIGKILL at spawn can leave no named secret-bearing file. Node already
    // rejected CR/LF in parsed HTTP headers, but normalize again.
    const headerPath = path.join(
      os.tmpdir(),
      ".swe-proxy-headers-" + process.pid + "-" + crypto.randomBytes(12).toString("hex")
    );
    const headerText = Object.entries(headers)
      .map(([name, value]) => name + ": " + String(value).replace(/[\r\n]/g, " "))
      .join("\n") + "\n";
    const args = ["--silent", "--show-error", "--no-buffer", "--suppress-connect-headers",
                  "--request", "POST", "--dump-header", "-", "--data-binary", "@-",
                  "--header", "@/proc/self/fd/3"];
    args.push(cfg.target.replace(/\/$/, "") + req.url);
    let headerWriteFd = null;
    let headerReadFd = null;
    let curl;
    try {
      headerWriteFd = fs.openSync(headerPath, "wx+", 0o600);
      headerReadFd = fs.openSync(headerPath, "r");
      // Do not write unless unlink succeeded. Both descriptors currently point
      // at an empty inode; after unlink, only these descriptors can reach it.
      fs.unlinkSync(headerPath);
      fs.writeFileSync(headerWriteFd, headerText, {encoding: "utf8"});
      fs.closeSync(headerWriteFd);
      headerWriteFd = null;
      curl = spawn("curl", args, {
        env: process.env,
        stdio: ["pipe", "pipe", "pipe", headerReadFd],
      });
    } finally {
      if (headerReadFd !== null) {
        try { fs.closeSync(headerReadFd); } catch (_) {}
      }
      if (headerWriteFd !== null) {
        try { fs.closeSync(headerWriteFd); } catch (_) {}
      }
      // Best-effort removal of the empty inode if failure occurred before the
      // mandatory unlink. Credentials are never written in that state.
      try { fs.unlinkSync(headerPath); } catch (_) {}
    }
    let pending = Buffer.alloc(0);
    let sent = false;
    let finished = false;
    let stderr = "";
    const finishError = (message) => {
      if (finished || res.writableEnded || res.destroyed) return;
      finished = true;
      sendJson(res, 502, {error: {message, type: "proxy_error"}});
    };
    curl.stderr.on("data", (chunk) => { stderr = (stderr + chunk.toString()).slice(-4096); });
    curl.stdout.on("data", (chunk) => {
      if (finished) return;
      if (sent) { res.write(chunk); return; }
      pending = Buffer.concat([pending, chunk]);
      let split = pending.indexOf("\r\n\r\n");
      let width = 4;
      if (split < 0) { split = pending.indexOf("\n\n"); width = 2; }
      if (split < 0) return;
      const rawHeaders = pending.subarray(0, split).toString("latin1").split(/\r?\n/);
      const match = /^HTTP\/\S+\s+(\d+)/.exec(rawHeaders.shift() || "");
      if (!match) { finishError("invalid upstream response"); curl.kill(); return; }
      const responseHeaders = {};
      for (const line of rawHeaders) {
        const colon = line.indexOf(":");
        if (colon <= 0) continue;
        const name = line.slice(0, colon).trim().toLowerCase();
        if (["content-length", "transfer-encoding", "content-encoding", "connection"].includes(name)) continue;
        responseHeaders[name] = line.slice(colon + 1).trim();
      }
      res.writeHead(Number(match[1]), responseHeaders);
      sent = true;
      const rest = pending.subarray(split + width);
      if (rest.length) res.write(rest);
      pending = Buffer.alloc(0);
    });
    curl.on("error", (err) => {
      finishError(err.message);
    });
    curl.on("close", (code) => {
      if (finished) return;
      if (!sent) finishError(stderr || ("curl exited " + code));
      else { finished = true; res.end(); }
    });
    curl.stdin.on("error", (err) => { finishError(err.message); });
    res.on("close", () => {
      finished = true;
      if (!curl.killed) curl.kill("SIGTERM");
    });
    curl.stdin.end(body);
  });
});
server.listen(cfg.port, "0.0.0.0", () => console.log("Proxy listening on port " + cfg.port));
'''
    return template.replace("__CONFIG__", config)


async def launch_litellm_proxy(
    environment: Any, logs_dir: Path, *, proxy_port: int | None = None
) -> bool:
    """Launch the LiteLLM-compat proxy in the sandbox when LITELLM_PROXY_MODEL
    is set. Returns True iff the proxy started AND responded to /health.

    No-op when LITELLM_PROXY_MODEL is unset (caller uses native upstream).
    Safe to call from any wrapper's setup() — the env-var gate makes it a
    cheap pass-through for direct-Anthropic runs.
    """
    proxy_model = os.environ.get("LITELLM_PROXY_MODEL")
    if not proxy_model:
        return False

    proxy_port = proxy_port or int(os.environ.get("LITELLM_PROXY_PORT", "4210"))
    target_url = os.environ.get("PROXY_TARGET_URL", "https://openrouter.ai/api")
    proxy_api_key = os.environ.get("PROXY_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    is_openrouter_target = "openrouter" in target_url
    fallback_url = os.environ.get("PROXY_FALLBACK_URL", "")
    fallback_key = os.environ.get("PROXY_FALLBACK_KEY", "")
    fallback_model = os.environ.get("PROXY_FALLBACK_MODEL", "")
    instance_id = secrets.token_hex(12)

    log.info(
        "Starting LiteLLM proxy in sandbox: model=%s port=%s target=%s fallback=%s",
        proxy_model, proxy_port, target_url, fallback_url or "none",
    )

    proxy_script = _proxy_script_source(
        proxy_port=proxy_port, target_url=target_url,
        proxy_model=proxy_model,
        is_openrouter_target=is_openrouter_target,
        fallback_url=fallback_url,
        fallback_model=fallback_model,
        instance_id=instance_id,
    )
    node_proxy_script = _node_proxy_script_source(
        proxy_port=proxy_port,
        target_url=target_url,
        proxy_model=proxy_model,
        is_openrouter_target=is_openrouter_target,
        instance_id=instance_id,
    )

    proxy_path = logs_dir / "model_proxy.py"
    proxy_path.write_text(proxy_script)
    await environment.upload_file(
        source_path=proxy_path, target_path="/tmp/model_proxy.py",
    )
    node_proxy_path = logs_dir / "model_proxy.js"
    node_proxy_path.write_text(node_proxy_script)
    await environment.upload_file(
        source_path=node_proxy_path, target_path="/tmp/model_proxy.js",
    )

    setup_cmd = (
        f"if command -v python3 >/dev/null 2>&1; then "
        f"  nohup python3 /tmp/model_proxy.py > /tmp/proxy.log 2>&1 & "
        f"else "
        f"  nohup node /tmp/model_proxy.js > /tmp/proxy.log 2>&1 & "
        f"fi; "
        f"for i in $(seq 1 15); do "
        f"  sleep 1; "
        f"  health=$(curl -s http://localhost:{proxy_port}/health 2>/dev/null || true); "
        f"  printf '%s' \"$health\" | grep -F '{instance_id}' >/dev/null && "
        f"  echo 'Proxy ready on port {proxy_port}' && exit 0; "
        f"done; "
        f"echo 'WARNING: proxy not healthy after 15s' >&2; "
        f"cat /tmp/proxy.log >&2; exit 1"
    )
    result = await environment.exec(
        command=setup_cmd,
        env={
            "PROXY_API_KEY": proxy_api_key,
            "PROXY_FALLBACK_KEY": fallback_key,
        },
    )
    if result.return_code != 0:
        log.warning("LiteLLM proxy start failed: %s", result.stderr or result.stdout)
        return False
    log.info("LiteLLM proxy started successfully on port %s", proxy_port)
    return True


# ──────────────────────────────────────────────────────────────────────
# Model-name remap (Harbor's validator rejects our `minimaxd/`, `glmd/`,
# `ark/` etc. prefixes because they're not in PROVIDER_MODEL_NAMES).
# Wrappers that bake in a Harbor agent (mini-swe-agent, opencode) call
# `mask_proxied_model_name()` BEFORE constructing the inner agent so the
# Harbor validator sees a placeholder it accepts. The proxy then rewrites
# the model field at the network layer (build_agent_env already wired the
# real target into PROXY_TARGET_URL + LITELLM_PROXY_MODEL).
# ──────────────────────────────────────────────────────────────────────

# Provider prefixes that route through our in-sandbox proxy. Anything in
# this list gets masked to a Harbor-recognized placeholder for the inner
# agent; the proxy handles the real routing.
PROXIED_PROVIDER_PREFIXES: tuple[str, ...] = (
    "minimaxd/",
    "glmd/",
    "ark/",
    "fireworks/",
    # NOTE: openrouter/ and deepseek/ removed — both are Harbor-recognized
    # providers (see external/harbor/.../agents/utils.py PROVIDER_API_KEY_VARS)
    # that LiteLLM, Harbor, and opencode all route natively via OPENROUTER_API_KEY
    # / DEEPSEEK_API_KEY. Masking them produced "anthropic/claude-sonnet-4-6",
    # which LiteLLM's anthropic provider dispatched to api.anthropic.com (it
    # reads ANTHROPIC_API_BASE, NOT ANTHROPIC_BASE_URL — so the proxy at
    # localhost:4210 was bypassed) → 401 invalid x-api-key on every trial.
    # See new29-diverse pilot diagnosis (2026-05-29): same failure mode on
    # mini-Opus pilot10 reruns first surfaced this for openrouter/.
    "chutes/",
    "glm/",
    # metagen x2p gateway (Opus 4.8). Masked to the placeholder so opencode's
    # anthropic provider points at localhost:4210; the proxy rewrites the body
    # model to claude-opus-4-8, injects x-api-key: mg-api-…, and forwards to the
    # gateway via the relay (python honors http_proxy; node/undici does not).
    "metagen/",
)

# Placeholder model name that Harbor's get_api_key_var_names_from_model_name
# accepts, and that LiteLLM / opencode's anthropic provider know how to dispatch
# (they'll then hit ANTHROPIC_BASE_URL=localhost:4210, where the proxy rewrites
# to the real model).
PROXY_PLACEHOLDER_MODEL = "anthropic/claude-sonnet-4-6"


def mask_proxied_model_name(model_name: str | None) -> str | None:
    """If `model_name` uses one of our proxied prefixes, return the placeholder.
    Otherwise return the input unchanged.

    Use case: `super().__init__(model_name=mask_proxied_model_name(model_name))`
    in user_enabled_{mini_swe_agent,opencode}.UserEnabled* — keeps Harbor's
    model-name validator happy while the proxy does the real routing.
    """
    if not model_name:
        return model_name
    if any(model_name.startswith(p) for p in PROXIED_PROVIDER_PREFIXES):
        return PROXY_PLACEHOLDER_MODEL
    return model_name
