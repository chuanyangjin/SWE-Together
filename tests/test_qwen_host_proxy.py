from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from proxies.qwen_host_proxy import (  # noqa: E402
    PLACEHOLDER_KEY,
    REDACTED_BEARER,
    _StreamingSecretRedactor,
    create_app,
    load_deployment_info,
)


class DeploymentInfoTests(unittest.TestCase):
    def test_requires_owner_only_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            info = root / "proxy_info.json"
            info.write_text(
                json.dumps(
                    {
                        "url": "http://qwen.internal:8100",
                        "api_key": "fixture-deployment-bearer",
                        "model": "Qwen3.5-4B",
                    }
                )
            )
            info.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "owner-only"):
                load_deployment_info(info)

            info.chmod(0o600)
            loaded = load_deployment_info(info)
            self.assertEqual(loaded.model, "Qwen3.5-4B")

            link = root / "linked.json"
            link.symlink_to(info)
            with self.assertRaisesRegex(RuntimeError, "owner-only"):
                load_deployment_info(link)

            info.write_text(
                json.dumps(
                    {
                        "url": "http://qwen.internal:8100",
                        "api_key": "fixture-deployment-bearer",
                        "model": "wrong-model",
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "Qwen3.5-4B"):
                load_deployment_info(info)


class QwenHostProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.secret = "fixture-live-deployment-bearer"
        self.auth = {"Authorization": f"Bearer {PLACEHOLDER_KEY}"}
        self.seen_authorization: list[str] = []
        self.seen_queries: list[str] = []

        async def health(request: web.Request) -> web.Response:
            self.seen_authorization.append(request.headers.get("Authorization", ""))
            return web.json_response({"healthy": True, "debug": self.secret})

        async def models(request: web.Request) -> web.Response:
            self.seen_authorization.append(request.headers.get("Authorization", ""))
            self.seen_queries.append(request.query_string)
            if request.query.get("redirect"):
                raise web.HTTPFound(location=f"http://elsewhere.invalid/{self.secret}")
            if request.query.get("headers"):
                return web.Response(
                    body=b"safe-body",
                    status=418,
                    reason=self.secret,
                    headers={"X-Debug": self.secret, "X-Safe": "kept"},
                )
            return web.json_response(
                {
                    "data": [{"id": "Qwen3.5-4B"}],
                    "must_not_escape": self.secret,
                }
            )

        async def stream(request: web.Request) -> web.StreamResponse:
            self.seen_authorization.append(request.headers.get("Authorization", ""))
            response = web.StreamResponse(
                headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            split = len(self.secret) // 2
            await response.write(b"data: before-" + self.secret[:split].encode())
            await asyncio.sleep(0.01)
            await response.write(self.secret[split:].encode() + b"-after\n\n")
            await response.write_eof()
            return response

        upstream_app = web.Application()
        upstream_app.router.add_get("/health", health)
        upstream_app.router.add_get("/v1/models", models)
        upstream_app.router.add_post("/v1/chat/completions", stream)
        self.upstream = TestServer(upstream_app)
        await self.upstream.start_server()

        self.info_path = self.root / "proxy_info.json"
        self.info_path.write_text(
            json.dumps(
                {
                    "url": str(self.upstream.make_url("/")).rstrip("/"),
                    "api_key": self.secret,
                    "model": "Qwen3.5-4B",
                }
            )
        )
        self.info_path.chmod(0o600)

        self.proxy = TestServer(create_app(self.info_path))
        self.client = TestClient(self.proxy)
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.upstream.close()
        self._temporary.cleanup()

    async def test_injects_upstream_bearer_and_never_returns_it(self) -> None:
        response = await self.client.get(
            "/v1/models?probe=1",
            headers=self.auth,
        )
        body = await response.read()
        self.assertEqual(response.status, 200)
        self.assertNotIn(self.secret.encode(), body)
        self.assertIn(REDACTED_BEARER, body)
        self.assertEqual(self.seen_queries, ["probe=1"])
        self.assertEqual(
            self.seen_authorization,
            [f"Bearer {self.secret}"],
        )

    async def test_redacts_bearer_split_across_stream_chunks(self) -> None:
        response = await self.client.post(
            "/v1/chat/completions",
            json={"model": "Qwen3.5-4B", "messages": []},
            headers=self.auth,
        )
        body = await response.read()
        self.assertEqual(response.status, 200)
        self.assertNotIn(self.secret.encode(), body)
        self.assertIn(REDACTED_BEARER, body)

    async def test_health_checks_upstream_without_reflecting_body(self) -> None:
        response = await self.client.get("/health", headers=self.auth)
        payload = await response.json()
        self.assertEqual(response.status, 200)
        self.assertEqual(payload, {"ok": True, "service": "qwen-host-proxy"})
        self.assertNotIn(self.secret, json.dumps(payload))

    async def test_reloads_rotated_bearer_from_private_file(self) -> None:
        rotated = "fixture-rotated-deployment-bearer"
        self.info_path.write_text(
            json.dumps(
                {
                    "url": str(self.upstream.make_url("/")).rstrip("/"),
                    "api_key": rotated,
                    "model": "Qwen3.5-4B",
                }
            )
        )
        os.chmod(self.info_path, 0o600)
        response = await self.client.get("/v1/models", headers=self.auth)
        await response.read()
        self.assertEqual(self.seen_authorization[-1], f"Bearer {rotated}")

    async def test_rejects_missing_or_wrong_placeholder_and_unknown_routes(self) -> None:
        missing = await self.client.get("/v1/models")
        wrong = await self.client.get(
            "/v1/models", headers={"Authorization": "Bearer wrong"}
        )
        unknown = await self.client.post("/v1/responses", headers=self.auth)
        self.assertEqual(missing.status, 401)
        self.assertEqual(wrong.status, 401)
        self.assertEqual(unknown.status, 404)
        self.assertEqual(self.seen_authorization, [])

    async def test_refuses_upstream_redirect_without_reflecting_location(self) -> None:
        response = await self.client.get(
            "/v1/models?redirect=1", headers=self.auth
        )
        body = await response.read()
        self.assertEqual(response.status, 502)
        self.assertNotIn("Location", response.headers)
        self.assertNotIn(self.secret.encode(), body)

    async def test_drops_secret_headers_and_never_reflects_reason(self) -> None:
        response = await self.client.get(
            "/v1/models?headers=1", headers=self.auth
        )
        body = await response.read()
        self.assertEqual(response.status, 418)
        self.assertEqual(response.headers.get("X-Safe"), "kept")
        self.assertNotIn("X-Debug", response.headers)
        self.assertNotIn(self.secret, response.reason)
        self.assertNotIn(self.secret.encode(), body)


class StreamingRedactorTests(unittest.TestCase):
    def test_all_two_boundary_splits_and_replacement_substrings(self) -> None:
        for secret in (
            b"fixture-live-key",
            b"redacted",
            b"upstream",
            b"bearer",
            b"aaaaaaaa",
        ):
            payload = b"prefix-" + secret + b"-middle-" + secret + b"-suffix"
            expected_redactor = _StreamingSecretRedactor(secret)
            expected = payload.replace(secret, expected_redactor.replacement)
            for first in range(len(payload) + 1):
                for second in range(first, len(payload) + 1):
                    redactor = _StreamingSecretRedactor(secret)
                    actual = b"".join(
                        (
                            redactor.feed(payload[:first]),
                            redactor.feed(payload[first:second]),
                            redactor.feed(payload[second:]),
                            redactor.finish(),
                        )
                    )
                    self.assertEqual(actual, expected)
                    self.assertNotIn(secret, actual)


if __name__ == "__main__":
    unittest.main()
