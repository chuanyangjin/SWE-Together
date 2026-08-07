#!/usr/bin/env python3
"""Pinned-sampling structured tool-call smoke for Olmo step 500."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


EXPECTED_MODEL = "Olmo-0716-step500"
EXPECTED_CHECKPOINT = (
    "/checkpoint/ram/chuanyang/autodata/run_0716_single_turn/weights/step_500"
)


def load_endpoint(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    required = {
        "OLMO_OPENAI_BASE_URL",
        "OLMO_SERVED_MODEL",
        "OLMO_SERVICE_KEY_FILE",
        "OLMO_CHECKPOINT",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise RuntimeError(f"endpoint file is missing fields: {', '.join(missing)}")
    if values["OLMO_SERVED_MODEL"] != EXPECTED_MODEL:
        raise RuntimeError("endpoint model identity mismatch")
    if values["OLMO_CHECKPOINT"] != EXPECTED_CHECKPOINT:
        raise RuntimeError("endpoint checkpoint identity mismatch")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-file", type=Path, required=True)
    args = parser.parse_args()

    endpoint = load_endpoint(args.endpoint_file)
    base_url = endpoint["OLMO_OPENAI_BASE_URL"]
    api_key = Path(endpoint["OLMO_SERVICE_KEY_FILE"]).read_text().strip()
    payload = {
        "model": EXPECTED_MODEL,
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
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.05,
        "max_tokens": 32768,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=600) as response:
        result = json.load(response)

    choice = result["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if not calls:
        message = choice["message"]
        diagnostic = {
            "finish_reason": choice.get("finish_reason"),
            "content_chars": len(str(message.get("content") or "")),
            "reasoning_chars": len(str(message.get("reasoning") or "")),
            "message_keys": sorted(message),
            "usage": result.get("usage"),
        }
        raise RuntimeError(
            "expected tool_calls; diagnostic=" + json.dumps(diagnostic, sort_keys=True)
        )
    function = calls[0]["function"]
    arguments = json.loads(function["arguments"])
    if function["name"] != "calculator" or arguments != {"a": 17, "b": 25}:
        raise RuntimeError(f"unexpected tool call: {function['name']} {arguments}")
    if result.get("model") != EXPECTED_MODEL:
        raise RuntimeError(f"response model identity mismatch: {result.get('model')!r}")
    print(
        "OLMO_TOOL_SMOKE_OK"
        f" model={result['model']} finish_reason={choice.get('finish_reason')}"
        f" tool={function['name']} arguments={json.dumps(arguments, sort_keys=True)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
