#!/usr/bin/env python3
"""Cross-node OpenAI-compatible tool-call smoke for the Sushi checkpoint."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ENDPOINT_ENV = Path(
    os.environ.get(
        "BENCH_ENDPOINT_ENV",
        str(REPO / "sushi_lane" / "state" / "endpoint.env"),
    )
)


def load_endpoint() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENDPOINT_ENV.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main() -> int:
    endpoint = load_endpoint()
    base_url = endpoint["SUSHI_OPENAI_BASE_URL"]
    model = endpoint["SUSHI_SERVED_MODEL"]
    api_key = Path(endpoint["SUSHI_SERVICE_KEY_FILE"]).read_text().strip()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Use the calculator tool to add 17 and 25. Do not answer directly.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
        # These are the checkpoint's validated anti-loop sampling settings.
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.05,
        # Qwen3.5 emits a reasoning segment before its structured tool call.
        "max_tokens": int(os.environ.get("SUSHI_SMOKE_MAX_TOKENS", "8192")),
    }
    if os.environ.get("SUSHI_SMOKE_ENABLE_THINKING") == "0":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    # Bypass login-shell proxy settings for the private compute-node endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        result = json.load(response)

    choice = result["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if not calls:
        message = choice["message"]
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning") or "")
        reasoning_lines = [" ".join(line.split()) for line in reasoning.splitlines() if line.strip()]
        line_counts: dict[str, int] = {}
        for line in reasoning_lines:
            line_counts[line] = line_counts.get(line, 0) + 1
        max_line_repeats = max(line_counts.values(), default=0)
        diagnostic = {
            "finish_reason": choice.get("finish_reason"),
            "content_chars": len(content),
            "content_tail": content[-600:],
            "reasoning_chars": len(reasoning),
            "reasoning_tail": reasoning[-1200:],
            "reasoning_lines": len(reasoning_lines),
            "reasoning_unique_lines": len(line_counts),
            "reasoning_max_line_repeats": max_line_repeats,
            "message_keys": sorted(message),
            "usage": result.get("usage"),
        }
        raise RuntimeError("expected tool_calls; diagnostic=" + json.dumps(diagnostic, sort_keys=True))
    function = calls[0]["function"]
    arguments = json.loads(function["arguments"])
    if function["name"] != "calculator" or arguments != {"a": 17, "b": 25}:
        raise RuntimeError(f"unexpected tool call: {function['name']} {arguments}")
    print(
        f"{os.environ.get('BENCH_SMOKE_LABEL', 'SUSHI')}_TOOL_SMOKE_OK"
        f" model={result.get('model')} finish_reason={choice.get('finish_reason')}"
        f" tool={function['name']} arguments={json.dumps(arguments, sort_keys=True)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
